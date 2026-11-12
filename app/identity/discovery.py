"""Cold-start experimental discovery from fixed raw evidence, never from templates.

Positive edges require reciprocal multiple-observation support. Only disjoint
complete-support groups survive bridge and rival vetoes; no target population K.
"""

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from itertools import combinations
from statistics import median
from typing import Any, Literal

import numpy as np

from app.contracts.discovery import (
    AutomaticTemplateSupport,
    DiscoveredGroup,
    DiscoveryComparison,
    DiscoveryDirection,
    DiscoveryPair,
    DiscoveryPolicy,
    DiscoveryReport,
    UnknownSegment,
)
from app.contracts.identity import MatchingPolicy
from app.identity.constraints import SpatioTemporalConstraintEngine
from app.identity.distance import validate_feature
from app.identity.fingerprints import evidence_fingerprint
from app.identity.prepare import active_risks, evidence_keys, resolve_feature


def _hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def _id(prefix, *values):
    return f"{prefix}_{_hash(values)[:32]}"


def _canonical_batch(batch):
    data = batch.model_dump(mode="json", exclude={"batch_id"})
    for key, value in data.items():
        if isinstance(value, list):
            data[key] = sorted(value, key=lambda v: json.dumps(v, sort_keys=True))
    if data.get("identity_evidence"):
        for key, value in data["identity_evidence"].items():
            if isinstance(value, list):
                data["identity_evidence"][key] = sorted(
                    value, key=lambda v: json.dumps(v, sort_keys=True)
                )
    return data


def _track_sort_key(track_id: str | int) -> tuple[int, int | str]:
    """Stable sort key for mixed numeric and string track identifiers."""
    s = str(track_id)
    if s.isdigit():
        return (0, int(s))
    return (1, s)


def _components(samples, assets):
    parents = {s.sample_id: s.sample_id for s in samples}

    def find(sid):
        while parents[sid] != sid:
            parents[sid] = parents[parents[sid]]
            sid = parents[sid]
        return sid

    known = {}
    for s in sorted(samples, key=lambda s: s.sample_id):
        for key in evidence_keys(s, assets):
            if key in known:
                a, b = find(s.sample_id), find(known[key])
                parents[max(a, b)] = min(a, b)
            known[key] = s.sample_id
    return {sid: find(sid) for sid in parents}


class _Distances:
    def __init__(self, samples, space, values, policy):
        self.features, self.cache = {}, {}
        self.policy = policy
        mp = MatchingPolicy(
            feature_space_id=space.feature_space_id, min_common_parts=policy.min_common_parts
        )
        for s in samples:
            f = resolve_feature(s, space, values)
            if f is None:
                continue
            validate_feature(space, f, mp)
            self.features[s.sample_id] = (
                np.asarray(f.values).reshape(space.shape),
                np.asarray(f.visibility),
            )

    def pair(self, a, b):
        key = tuple(sorted((a.sample_id, b.sample_id)))
        if key not in self.cache:
            fa, va = self.features[key[0]]
            fb, vb = self.features[key[1]]
            common = (
                (va >= self.policy.min_part_visibility)
                & (va > 0)
                & (vb >= self.policy.min_part_visibility)
                & (vb > 0)
            )
            count = int(common.sum())
            distance = None
            if count >= self.policy.min_common_parts:
                w = np.sqrt(va[common] * vb[common])
                distance = float(
                    np.dot(w, np.linalg.norm(fa[common] - fb[common], axis=1)) / (2 * w.sum())
                )
            self.cache[key] = DiscoveryComparison(
                left_sample_id=key[0],
                right_sample_id=key[1],
                distance=distance,
                common_parts=count,
                reason=None if distance is not None else "insufficient_reliable_common_parts",
            )
        return self.cache[key]


def _separated_observations(left, right, evidence, policy):
    if left.clip_id != right.clip_id or left.track_id == right.track_id:
        return []

    def observations(s):
        return {
            o.frame_idx if o.frame_idx is not None else o.time_s: o
            for o in evidence.observations
            if o.clip_id == s.clip_id
            and o.track_id == s.track_id
            and s.start_s <= o.time_s < s.end_s
        }

    a, b = observations(left), observations(right)
    witnesses = defaultdict(list)
    for key in sorted(a.keys() & b.keys()):
        x, y = a[key], b[key]
        x0, t, r, d = x.bbox_xyxy
        u, v, w, z = y.bbox_xyxy
        # Time overlap alone, overlapping boxes and duplicate detections never veto.
        disjoint = r < u or w < x0 or d < v or z < t
        height = ((d - t) + (z - v)) / 2
        separation = math.hypot((x0 + r - u - w) / 2, (t + d - v - z) / 2) / height
        if disjoint and separation >= policy.cooccurrence_separation_heights:
            witnesses[math.floor(x.time_s / policy.bucket_s)].extend(
                [x.observation_id, y.observation_id]
            )
    return (
        sorted({oid for ids in witnesses.values() for oid in ids})
        if (len(witnesses) >= policy.cooccurrence_min_buckets)
        else []
    )


def _qualify_active_segments(
    active,
    proposals,
    evidence,
    distances,
    components,
    bad_duplicates,
    policy,
    manual_constraints=None,
    allow_short_match: bool = False,
    allow_external_anchor: bool = False,
):
    reasons = defaultdict(list)
    selected = {}
    all_samples = {}
    manual_constraints = manual_constraints or []
    blocked_by_manual = {
        getattr(c, "segment_id", None)
        for c in manual_constraints
        if getattr(c, "kind", None) == "block_automatic"
    }

    for sid, segment in sorted(active.items()):
        if sid in blocked_by_manual:
            reasons[sid].append("blocked_by_manual_constraint")
            continue

        proposal = proposals.get(sid)
        external_anchor = allow_external_anchor and getattr(
            proposal, "requires_external_anchor", False
        )
        if proposal is None or (proposal.status != "supported" and not external_anchor):
            reasons[sid].extend(["segment_not_supported", *(proposal.reasons if proposal else [])])
            continue

        can_match = getattr(proposal, "can_match_existing", True)
        can_create = getattr(proposal, "can_create_new_identity", True)
        if not allow_short_match and not can_create:
            reasons[sid].append("short_segment_limited_match_support")
            continue
        if not can_match and not can_create:
            reasons[sid].extend(["segment_not_supported", *(proposal.reasons if proposal else [])])
            continue

        risks = [
            r
            for r in active_risks(evidence, segment)
            if not (r.kind == "quality_review" and r.basis == "raw_track_continuity_unverified")
            and not (
                external_anchor
                and r.kind == "quality_review"
                and r.basis == "external_anchor_required"
                and r.segment_id == sid
                and r.source.producer == proposal.source.producer
                and r.source.evidence_version == proposal.source.evidence_version
            )
        ]
        if risks:
            reasons[sid].append("unresolved_identity_risk")
            continue

        samples = sorted(
            [s for s in evidence.samples if s.segment_id == sid],
            key=lambda s: (s.time_s, s.sample_id),
        )
        usable = []
        for sample in samples:
            q = sample.tracking_quality
            if components[sample.sample_id] in bad_duplicates:
                reasons[sid].append("duplicate_descriptor_conflict")
            if q and (
                q.occluded
                or q.mixed_identity
                or q.usability == "unusable"
                or q.clean_review == "rejected"
            ):
                continue
            if sample.sample_id in distances.features:
                usable.append(sample)

        if reasons[sid]:
            continue

        unique, buckets = {}, {}
        for s in usable:
            unique.setdefault(components[s.sample_id], s)
        for s in sorted(unique.values(), key=lambda s: (s.time_s, s.sample_id)):
            b_idx = math.floor((s.time_s - segment.start_s) / policy.bucket_s)
            buckets.setdefault(b_idx, s)
        representatives = list(buckets.values())
        min_b = (
            getattr(policy, "min_match_buckets", 2)
            if (allow_short_match and can_match)
            else policy.min_support_buckets
        )
        if len(representatives) < min_b:
            reasons[sid].append("insufficient_nonduplicate_buckets")
            continue

        is_v3_supported = (
            proposal is not None
            and proposal.status == "supported"
            and any(
                r in proposal.reasons
                for r in (
                    "local_continuous_appearance_with_medoid_support",
                    "local_all_pairs_consistent_with_independent_temporal_support",
                )
            )
        )
        if is_v3_supported or external_anchor:
            if external_anchor and (
                set(proposal.support_sample_ids) != {s.sample_id for s in usable}
                or any(distances.pair(a, b).distance is None for a, b in combinations(usable, 2))
            ):
                reasons[sid].append("external_anchor_evidence_incomplete")
                continue
            if any(
                (d := distances.pair(a, b).distance) is not None and d >= policy.far_distance
                for a, b in combinations(usable, 2)
            ):
                reasons[sid].append("within_segment_conflict_or_incomparability")
                continue
        else:
            if any(
                (d := distances.pair(a, b).distance) is None
                or d > policy.intra_segment_near_distance
                for a, b in combinations(usable, 2)
            ):
                reasons[sid].append("within_segment_conflict_or_incomparability")
                continue

        all_samples[sid], selected[sid] = usable, representatives

    return reasons, selected, all_samples


qualify_active_segments = _qualify_active_segments


def _evaluate_segment_pair(
    left: str,
    right: str,
    active: dict[str, Any],
    all_samples: dict[str, list[Any]],
    selected: dict[str, list[Any]],
    seg_reps_by_id: dict[str, Any],
    distances: _Distances,
    components: dict[str, str],
    evidence: Any,
    policy: DiscoveryPolicy,
    engine: Any | None,
    fsid: str,
    match_id: str,
) -> DiscoveryPair:
    from app.identity.representation import compute_track_distance

    pid = _id("dsp", match_id, fsid, left, right)
    seg_left = active[left]
    seg_right = active[right]
    conflict_obs = _separated_observations(seg_left, seg_right, evidence, policy)
    why: list[str] = []
    is_conflict = False

    # 1. Physical co-occurrence overlap in the same clip
    if seg_left.clip_id == seg_right.clip_id:
        if str(seg_left.track_id) != str(seg_right.track_id):
            s1, e1 = float(seg_left.start_s), float(seg_left.end_s)
            s2, e2 = float(seg_right.start_s), float(seg_right.end_s)
            if max(s1, s2) < min(e1, e2) and not (
                engine is not None and hasattr(engine, "pair_relation")
            ):
                is_conflict = True
                why.append("spatio_temporal_overlap")
        if conflict_obs:
            is_conflict = True
            why.append("spatially_separated_cooccurrence")

    if engine is not None:
        idx_left = next((i for i, t in enumerate(engine.tracks) if t.segment_id == left), None)
        idx_right = next((i for i, t in enumerate(engine.tracks) if t.segment_id == right), None)
        if (
            idx_left is not None
            and idx_right is not None
            and not engine.can_merge_clusters({idx_left}, {idx_right})
        ):
            is_conflict = True
            why.append(
                "spatio_temporal_overlap"
                if engine.is_forbidden(idx_left, idx_right)
                else "unresolved_cooccurrence_overlap"
            )

    # 2. Pairwise sample comparisons & duplicate component tracking
    comparisons: dict[tuple[str, str], DiscoveryComparison] = {}
    samples_left = all_samples.get(left, [])
    samples_right = all_samples.get(right, [])

    left_comps = {components[s.sample_id] for s in samples_left if s.sample_id in components}
    right_comps = {components[s.sample_id] for s in samples_right if s.sample_id in components}
    shared_comps = left_comps & right_comps

    for a in samples_left:
        for b in samples_right:
            if components.get(a.sample_id) == components.get(b.sample_id):
                continue
            cmp = distances.pair(a, b)
            comparisons[(a.sample_id, b.sample_id)] = cmp
            if cmp.distance is not None and cmp.distance >= policy.far_distance:
                is_conflict = True
                why.append("contradictory_sample_pair")

    # 3. Directional sample coverage & support
    directions: list[DiscoveryDirection] = []
    reps_left = selected.get(left, [])
    reps_right = selected.get(right, [])
    for sid, other, query_reps, other_reps, all_query in (
        (left, right, reps_left, reps_right, samples_left),
        (right, left, reps_right, reps_left, samples_right),
    ):
        scores: list[float] = []
        supports: list[str] = []
        for q in query_reps:
            # Shared duplicate components cannot receive independent support
            if components.get(q.sample_id) in shared_comps:
                continue
            valid_targets = [
                t
                for t in other_reps
                if components.get(t.sample_id) not in shared_comps
                and components.get(t.sample_id) != components.get(q.sample_id)
            ]
            ds = sorted(
                d for t in valid_targets if (d := distances.pair(q, t).distance) is not None
            )
            if len(ds) >= 2:
                sc = sum(ds[:2]) / 2
                scores.append(sc)
                if sc <= policy.clustering_threshold:
                    supports.append(q.sample_id)
            elif len(ds) == 1:
                sc = ds[0]
                scores.append(sc)
                if sc <= policy.clustering_threshold:
                    supports.append(q.sample_id)

        for q in all_query:
            valid_targets = [
                t for t in other_reps if components.get(t.sample_id) != components.get(q.sample_id)
            ]
            ds = sorted(
                d for t in valid_targets if (d := distances.pair(q, t).distance) is not None
            )
            if len(ds) >= 2 and sum(ds[:2]) / 2 >= policy.far_distance:
                is_conflict = True
                why.append("minority_sample_disagreement")

        cov = len(supports) / max(1, len(query_reps))
        directions.append(
            DiscoveryDirection(
                query_segment_id=sid,
                query_sample_ids=[s.sample_id for s in query_reps],
                supported_sample_ids=supports,
                coverage=cov,
                score=float(median(scores)) if scores else None,
            )
        )

    # 4. Segment-level pooled representation distance
    rep_left = seg_reps_by_id.get(left)
    rep_right = seg_reps_by_id.get(right)
    seg_dist: float | None = None
    proto_witnesses: list[dict[str, Any]] = []
    if rep_left is not None and rep_right is not None:
        seg_dist = compute_track_distance(
            rep_left,
            rep_right,
            min_common_parts=policy.min_common_parts,
            min_vis=policy.min_part_visibility,
            near_distance=policy.intra_segment_near_distance,
            far_distance=policy.far_distance,
            clustering_threshold=policy.clustering_threshold,
        )
        if rep_left.prototypes and rep_right.prototypes:
            from app.identity.representation import compute_prototype_distance

            for p1 in rep_left.prototypes:
                for p2 in rep_right.prototypes:
                    d_p = compute_prototype_distance(
                        p1,
                        p2,
                        min_common_parts=policy.min_common_parts,
                        min_vis=policy.min_part_visibility,
                    )
                    if d_p is not None and d_p <= policy.clustering_threshold:
                        proto_witnesses.append(
                            {
                                "left_prototype_id": p1.prototype_id,
                                "right_prototype_id": p2.prototype_id,
                                "distance": float(d_p),
                                "left_sample_id": p1.sample_id,
                                "right_sample_id": p2.sample_id,
                            }
                        )

    if seg_dist is not None and seg_dist >= policy.far_distance:
        is_conflict = True
        why.append("contradictory_segment_distance")

    has_independent_support = True
    if seg_left.clip_id != seg_right.clip_id:
        unshared_left = left_comps - shared_comps
        unshared_right = right_comps - shared_comps
        min_independent = policy.min_support_buckets
        if len(unshared_left) < min_independent or len(unshared_right) < min_independent:
            has_independent_support = False
            why.append("insufficient_independent_components_after_dedup")
            if shared_comps:
                why.append("shared_duplicate_components_excluded")

    def _is_direction_coverage_satisfied(d: DiscoveryDirection, min_cov: float) -> bool:
        if d.coverage >= min_cov:
            return True
        n = len(d.query_sample_ids)
        k = len(d.supported_sample_ids)
        if n == 4 and k >= 3:
            return True
        return False

    has_coverage = (
        all(_is_direction_coverage_satisfied(d, policy.min_coverage) for d in directions)
        if directions
        else False
    )
    if not has_coverage:
        why.append("insufficient_coverage")

    if is_conflict:
        status: Literal["supported", "conflict", "insufficient", "ambiguous"] = "conflict"
    elif (
        seg_dist is not None
        and seg_dist <= policy.clustering_threshold
        and has_independent_support
        and has_coverage
    ):
        status = "supported"
    elif (
        seg_dist is not None
        and policy.clustering_threshold < seg_dist < policy.far_distance
        and has_independent_support
        and has_coverage
    ):
        status = "ambiguous"
        why.append("distance_in_ambiguous_zone")
    else:
        status = "insufficient"

    clean_reasons = sorted(set(why))
    if not clean_reasons:
        clean_reasons = (
            ["reciprocal_multisample_support"]
            if status == "supported"
            else ["insufficient_support"]
        )

    return DiscoveryPair(
        pair_id=pid,
        left_segment_id=left,
        right_segment_id=right,
        status=status,
        score=seg_dist,
        reasons=clean_reasons,
        directions=directions,
        comparisons=list(comparisons.values()),
        conflict_observation_ids=conflict_obs,
        prototype_witnesses=proto_witnesses,
    )


def _select_group_templates(
    member_sids: Sequence[str],
    active: dict[str, Any],
    selected: dict[str, list[Any]],
    all_samples: dict[str, list[Any]],
    seg_reps_by_id: dict[str, Any],
    distances: _Distances,
    components: dict[str, str],
    cluster_pairs_map: dict[tuple[str, str], DiscoveryPair],
    policy: DiscoveryPolicy,
    fsid: str,
) -> list[AutomaticTemplateSupport]:
    """Select verifiable multi-prototype automatic templates spanning distinct clips."""
    template_candidates_by_clip: dict[str, list[tuple[float, AutomaticTemplateSupport]]] = (
        defaultdict(list)
    )

    sample_segments = {
        sample.sample_id: sid for sid in member_sids for sample in selected.get(sid, [])
    }
    for sid in member_sids:
        for sample in selected.get(sid, []):
            witnesses: list[str] = []
            witness_pairs: list[str] = []
            sample_comp = components.get(sample.sample_id)
            for other in member_sids:
                if active[other].clip_id == active[sid].clip_id:
                    continue
                pair = cluster_pairs_map.get(tuple(sorted((sid, other))))
                if pair is None or pair.status != "supported":
                    continue
                shared_comps_pair = {
                    components.get(s.sample_id) for s in all_samples.get(sid, [])
                } & {components.get(s.sample_id) for s in all_samples.get(other, [])}
                if sample_comp in shared_comps_pair:
                    continue
                close = [
                    s.sample_id
                    for s in selected.get(other, [])
                    if components.get(s.sample_id) not in shared_comps_pair
                    and components.get(s.sample_id) != sample_comp
                    and (d := distances.pair(sample, s).distance) is not None
                    and d <= policy.clustering_threshold
                ]
                if close:
                    witnesses.extend(close)
                    witness_pairs.append(pair.pair_id)

            unique_witnesses_map: dict[str, str] = {}
            for wid in witnesses:
                comp = components.get(wid, wid)
                if comp == sample_comp:
                    continue
                other_sid = sample_segments[wid]
                if any(
                    s.sample_id != sample.sample_id and components.get(s.sample_id) == sample_comp
                    for s in selected.get(other_sid, [])
                ):
                    continue
                if comp not in unique_witnesses_map:
                    unique_witnesses_map[comp] = wid
            unique_witnesses = sorted(unique_witnesses_map.values())

            if len(unique_witnesses) >= 2 and witness_pairs:
                mean_vis = float(np.mean(distances.features[sample.sample_id][1]))
                # Locate matching prototype if available
                rep_obj = seg_reps_by_id.get(sid)
                matched_proto = next(
                    (
                        p
                        for p in getattr(rep_obj, "prototypes", ())
                        if p.sample_id == sample.sample_id
                    ),
                    None,
                )
                if matched_proto is not None:
                    p_id = matched_proto.prototype_id
                    t_bucket = matched_proto.time_bucket
                    r_parts = list(matched_proto.reliable_parts)
                    q_sc = matched_proto.quality_score
                    p_risks = list(matched_proto.risks)
                elif getattr(policy, "representation_mode", "multi_prototype") == "multi_prototype":
                    continue
                else:
                    p_id = None
                    t_bucket = None
                    r_parts = []
                    q_sc = None
                    p_risks = []

                template_cand = AutomaticTemplateSupport(
                    sample_id=sample.sample_id,
                    segment_id=sid,
                    feature_space_id=fsid,
                    support_pair_ids=sorted(set(witness_pairs)),
                    witness_sample_ids=unique_witnesses,
                    prototype_id=p_id,
                    time_bucket=t_bucket,
                    reliable_parts=r_parts,
                    quality_score=float(q_sc) if q_sc is not None else None,
                    risks=p_risks,
                    source_component=sample_comp,
                )
                template_candidates_by_clip[active[sid].clip_id].append((mean_vis, template_cand))

    if len(template_candidates_by_clip) < 2:
        return []

    templates: list[AutomaticTemplateSupport] = []
    seen_segs: set[str] = set()
    seen_comps: set[str] = set()
    clips_sorted = sorted(template_candidates_by_clip.keys())
    for cid in clips_sorted:
        template_candidates_by_clip[cid].sort(key=lambda item: item[0], reverse=True)

    target_templates = min(policy.max_templates_per_group, 3)
    # First pass: one template per clip, avoiding duplicate components
    for cid in clips_sorted:
        for _, t in template_candidates_by_clip[cid]:
            t_comp = t.source_component or components.get(t.sample_id)
            if (
                t.segment_id not in seen_segs
                and (t_comp is None or t_comp not in seen_comps)
                and len(templates) < target_templates
            ):
                templates.append(t)
                seen_segs.add(t.segment_id)
                if t_comp is not None:
                    seen_comps.add(t_comp)
                break

    # Fill remaining capacity with independent views after the cross-clip pass.
    if len(templates) < target_templates:
        for cid in clips_sorted:
            for _, t in template_candidates_by_clip[cid]:
                t_comp = t.source_component or components.get(t.sample_id)
                if (t_comp is None or t_comp not in seen_comps) and len(
                    templates
                ) < target_templates:
                    templates.append(t)
                    if t_comp is not None:
                        seen_comps.add(t_comp)
                    if len(templates) >= target_templates:
                        break

    if len({active[t.segment_id].clip_id for t in templates}) < 2:
        return []

    return templates


def generate_candidate_clusters(
    qualified_reps: list[Any],
    constraints: Any,
    policy: DiscoveryPolicy,
) -> list[Any]:
    """Generate candidate identity clusters using constrained hierarchical clustering.

    Separates candidate generation from final positive relation and group acceptance.
    """
    from app.identity.clustering import constrained_hierarchical_clustering

    linkage = getattr(policy, "linkage", "average")
    return constrained_hierarchical_clustering(
        tracks=qualified_reps,
        constraints=constraints,
        threshold=policy.clustering_threshold,
        linkage=linkage,
        min_common_parts=policy.min_common_parts,
        min_vis=policy.min_part_visibility,
        max_pair_threshold=policy.far_distance,
    )


def accept_candidate_groups(
    candidate_clusters: list[Any],
    all_qualified_sids: list[str],
    active: dict[str, Any],
    all_samples: dict[str, list[Any]],
    selected: dict[str, list[Any]],
    seg_reps_by_id: dict[str, Any],
    distances: _Distances,
    components: dict[str, str],
    evidence: Any,
    policy: DiscoveryPolicy,
    engine: Any | None,
    fsid: str,
    match_id: str,
    proposals: dict[str, Any] | None = None,
) -> tuple[list[DiscoveredGroup], dict[str, DiscoveryPair], dict[str, list[str]]]:
    """Accept positive candidate groups through rigorous structural and evidence gates.

    Enforces:
    1. Spatio-temporal hard mutual exclusion and zero-far counter-evidence (conflict gate).
    2. Rival margin against outside competing candidates.
    3. Structural support graph connectivity and single-bridge chain pruning.
    4. Independent cross-clip support spanning at least 2 distinct clips.
    5. Non-duplicate template witnesses spanning at least 2 distinct clips.
    6. Deferral of unproven/ambiguous segments to unknown without forced grouping.
    """
    from app.identity.clustering import connected_components, find_graph_bridges

    groups: list[DiscoveredGroup] = []
    all_report_pairs: dict[str, DiscoveryPair] = {}
    pair_cache: dict[tuple[str, str], DiscoveryPair] = {}
    reasons: dict[str, list[str]] = defaultdict(list)
    proposals = proposals or {}

    def get_pair(u: str, v: str) -> DiscoveryPair:
        key = tuple(sorted((u, v)))
        if key not in pair_cache:
            p = _evaluate_segment_pair(
                left=key[0],
                right=key[1],
                active=active,
                all_samples=all_samples,
                selected=selected,
                seg_reps_by_id=seg_reps_by_id,
                distances=distances,
                components=components,
                evidence=evidence,
                policy=policy,
                engine=engine,
                fsid=fsid,
                match_id=match_id,
            )
            pair_cache[key] = p
            all_report_pairs[p.pair_id] = p
        return pair_cache[key]

    for cluster in candidate_clusters:
        member_sids = sorted({t.segment_id for t in cluster.tracks if t.segment_id is not None})
        if not member_sids:
            continue

        # Initial cross-clip check
        cluster_clips = {active[sid].clip_id for sid in member_sids}
        if len(cluster_clips) < 2:
            for sid in member_sids:
                reasons[sid].append("no_independent_cross_clip_group")
            continue

        # Evaluate all intra-cluster pairs
        for left, right in combinations(member_sids, 2):
            get_pair(left, right)

        # Gate 1: Intra-cluster Conflict Pruning (zero far, zero spatio-temporal clash)
        surviving = set(member_sids)
        while True:
            conflicts = [
                get_pair(u, v)
                for u, v in combinations(sorted(surviving), 2)
                if get_pair(u, v).status == "conflict"
            ]
            if not conflicts:
                break
            conflict_counts = Counter()
            for p in conflicts:
                conflict_counts[p.left_segment_id] += 1
                conflict_counts[p.right_segment_id] += 1
            worst = conflict_counts.most_common(1)[0][0]
            surviving.remove(worst)
            reasons[worst].append("within_group_conflict")

        if len(surviving) < 2:
            for sid in surviving:
                reasons[sid].append("no_independent_cross_clip_group")
            continue

        # Gate 2: Rival Margin Check against Outside Competitors
        if getattr(policy, "rival_margin_check", True):
            rival_pruned = set()
            for sid in sorted(surviving):
                in_pairs = [
                    get_pair(sid, other)
                    for other in surviving
                    if other != sid and get_pair(sid, other).status == "supported"
                ]
                best_in = min([p.score for p in in_pairs if p.score is not None], default=None)
                if best_in is None:
                    continue

                # Find any outside rival in all_qualified_sids
                outside_rival_scores = []
                for out_sid in all_qualified_sids:
                    if out_sid not in surviving:
                        is_competitor = policy.support_mode == "absolute" or any(
                            get_pair(out_sid, member).status == "conflict" for member in surviving
                        )
                        if not is_competitor:
                            continue
                        p_out = get_pair(sid, out_sid)
                        if p_out.status == "supported" and p_out.score is not None:
                            outside_rival_scores.append(p_out.score)

                if outside_rival_scores:
                    best_rival = min(outside_rival_scores)
                    margin = best_rival - best_in
                    for p in in_pairs:
                        p.rival_margins[sid] = float(margin)
                    if margin < policy.rival_margin:
                        rival_pruned.add(sid)
                        reasons[sid].append("unresolved_competing_segment_margin")

            surviving -= rival_pruned

        if len(surviving) < 2:
            for sid in surviving:
                reasons[sid].append("no_independent_cross_clip_group")
            continue

        # Gate 3: Graph Connectivity and Single Bridge Chain Pruning
        supported_edges = [
            tuple(sorted((u, v)))
            for u, v in combinations(sorted(surviving), 2)
            if get_pair(u, v).status == "supported"
        ]
        adj: dict[str, set[str]] = defaultdict(set)
        for u, v in supported_edges:
            adj[u].add(v)
            adj[v].add(u)

        isolated = {u for u in surviving if len(adj[u]) == 0}
        for u in isolated:
            reasons[u].append("no_independent_cross_clip_group")
        surviving -= isolated

        if not surviving:
            continue

        comps = connected_components(sorted(surviving), supported_edges)
        valid_subgroups: list[set[str]] = []

        for comp_nodes in comps:
            if len(comp_nodes) < 2:
                for u in comp_nodes:
                    reasons[u].append("no_independent_cross_clip_group")
                continue

            comp_edges = [e for e in supported_edges if e[0] in comp_nodes and e[1] in comp_nodes]
            active_comp = set(comp_nodes)

            # Bridge detection only applies to multi-node structures (>= 3 nodes)
            if len(comp_nodes) >= 3 and getattr(policy, "bridge_pruning", True):
                bridges = find_graph_bridges(sorted(comp_nodes), comp_edges)
                if bridges:
                    for u, v in bridges:
                        if u not in active_comp or v not in active_comp:
                            continue
                        edges_without = [
                            e
                            for e in comp_edges
                            if e != (u, v) and e[0] in active_comp and e[1] in active_comp
                        ]
                        sub_split = connected_components(sorted(active_comp), edges_without)
                        if len(sub_split) == 2:
                            s1, s2 = sub_split[0], sub_split[1]
                            clips_s1 = {active[x].clip_id for x in s1}
                            clips_s2 = {active[x].clip_id for x in s2}
                            # Concatenation of two multi-node subclusters: split to separate
                            # only when BOTH subclusters independently have cross-clip support (>= 2 clips)
                            if len(clips_s1) >= 2 and len(clips_s2) >= 2:
                                active_comp = set()
                                valid_subgroups.append(s1)
                                valid_subgroups.append(s2)
                                break

            if active_comp and len(active_comp) >= 2:
                valid_subgroups.append(active_comp)

        # Gate 4: Multi-clip & Template Witness Verification for each valid subgroup
        for sub_members in valid_subgroups:
            sub_sids = sorted(sub_members)
            sub_clips = {active[sid].clip_id for sid in sub_sids}
            if len(sub_clips) < 2:
                for sid in sub_sids:
                    reasons[sid].append("no_independent_cross_clip_group")
                continue

            sub_pairs_map = {
                tuple(sorted((u, v))): get_pair(u, v) for u, v in combinations(sub_sids, 2)
            }

            templates = _select_group_templates(
                member_sids=sub_sids,
                active=active,
                selected=selected,
                all_samples=all_samples,
                seg_reps_by_id=seg_reps_by_id,
                distances=distances,
                components=components,
                cluster_pairs_map=sub_pairs_map,
                policy=policy,
                fsid=fsid,
            )

            if len({active[t.segment_id].clip_id for t in templates}) < 2:
                for sid in sub_sids:
                    reasons[sid].append("insufficient_independent_template_support")
                continue

            anchor = sub_sids[0]
            gid = _id("dsg", match_id, anchor)
            support_pids = [
                sub_pairs_map[tuple(sorted((u, v)))].pair_id
                for u, v in combinations(sub_sids, 2)
                if sub_pairs_map[tuple(sorted((u, v)))].status == "supported"
            ]
            rejected = sorted(set(member_sids) - set(sub_sids))

            member_ev: dict[str, Any] = {}
            for sid in sub_sids:
                conns = [
                    other
                    for other in sub_sids
                    if other != sid
                    and sub_pairs_map[tuple(sorted((sid, other)))].status == "supported"
                ]
                member_ev[sid] = {
                    "clip_id": active[sid].clip_id,
                    "track_id": active[sid].track_id,
                    "supported_connections": conns,
                    "connection_count": len(conns),
                }

            groups.append(
                DiscoveredGroup(
                    group_id=gid,
                    anchor_segment_id=anchor,
                    member_segment_ids=sub_sids,
                    support_pair_ids=support_pids,
                    templates=templates,
                    rejected_member_ids=rejected,
                    member_evidence=member_ev,
                    acceptance_status="accepted",
                    acceptance_reasons=[
                        "multi_clip_verified",
                        "rival_margin_verified",
                        "bridge_screened",
                    ],
                )
            )

    return groups, all_report_pairs, reasons


def group_and_accept_candidates(
    qualified_sids: list[str],
    active: dict[str, Any],
    all_samples: dict[str, list[Any]],
    selected: dict[str, list[Any]],
    seg_reps_by_id: dict[str, Any],
    distances: _Distances,
    components: dict[str, str],
    evidence: Any,
    policy: DiscoveryPolicy,
    engine: Any | None,
    fsid: str,
    match_id: str,
    manual_constraints: list[Any] | None = None,
    proposals: dict[str, Any] | None = None,
) -> tuple[list[DiscoveredGroup], dict[str, DiscoveryPair], dict[str, list[str]]]:
    """Candidate generation followed by group acceptance.

    Shared interface between cold-start discovery and incremental planning.
    """
    sorted_sids = sorted(
        qualified_sids,
        key=lambda sid: (
            str(active[sid].clip_id),
            _track_sort_key(active[sid].track_id),
            active[sid].start_s,
            sid,
        ),
    )
    qualified_reps = [seg_reps_by_id[sid] for sid in sorted_sids]

    clustering_engine = SpatioTemporalConstraintEngine(
        tracks=qualified_reps,
        observations=getattr(engine, "observations", getattr(evidence, "observations", [])),
        manual_constraints=manual_constraints or getattr(engine, "manual_constraints", None),
        player_assignments=getattr(engine, "player_assignments", None),
        segments_by_id=getattr(engine, "segments_by_id", active),
    )

    candidate_clusters = generate_candidate_clusters(
        qualified_reps=qualified_reps,
        constraints=clustering_engine,
        policy=policy,
    )

    return accept_candidate_groups(
        candidate_clusters=candidate_clusters,
        all_qualified_sids=sorted_sids,
        active=active,
        all_samples=all_samples,
        selected=selected,
        seg_reps_by_id=seg_reps_by_id,
        distances=distances,
        components=components,
        evidence=evidence,
        policy=policy,
        engine=clustering_engine,
        fsid=fsid,
        match_id=match_id,
        proposals=proposals,
    )


def evaluate_existing_player_match(
    segment: Any,
    usable_samples: list[Any],
    representatives: list[Any],
    existing_players: dict[str, Any],
    player_templates: dict[str, list[tuple[Any, Any]]],
    distances: _Distances,
    components: dict[str, str],
    policy: DiscoveryPolicy,
    spatio_temporal_engine: Any = None,
    forbidden_players: set[str] | None = None,
    can_match: bool = True,
    match_threshold: float | None = None,
    require_all_samples: bool = False,
    diagnostics: dict[str, Any] | None = None,
) -> tuple[str | None, float | None, list[str]]:
    """Rank comparable identities before acceptance; retain every sample's counter-evidence."""
    if not can_match:
        return None, None, ["segment_cannot_match_existing"]
    thresh = match_threshold
    if thresh is None:
        thresh = policy.existing_player_match_threshold
    if thresh is None:
        thresh = policy.clustering_threshold

    forbidden = set(forbidden_players or ())
    sample_scores: dict[str, dict[str, float]] = {}
    player_scores: dict[str, float] = {}
    counter_evidence: set[str] = set()
    mature_players: set[str] = set()
    queries = {s.sample_id: s for s in [*usable_samples, *representatives]}
    query_components = {components[sid] for sid in queries}
    for pid, tlist in sorted(player_templates.items()):
        if pid not in existing_players:
            continue
        if (
            spatio_temporal_engine is not None
            and spatio_temporal_engine.is_segment_forbidden_for_player(segment.segment_id, pid)
        ):
            forbidden.add(pid)
        templates = {}
        for _, sample in tlist:
            component = components[sample.sample_id]
            if component not in query_components:
                templates.setdefault(component, sample)
        if len({getattr(s, "clip_id", None) for s in templates.values()} - {None}) >= 2:
            mature_players.add(pid)
        scores = {}
        for sid, query in queries.items():
            ds = sorted(
                d
                for template in templates.values()
                if (d := distances.pair(query, template).distance) is not None
            )
            if ds and ds[0] >= policy.far_distance:
                counter_evidence.add(pid)
            matching_ds = [d for d in ds if d <= thresh]
            if len(matching_ds) >= 2:
                scores[sid] = sum(matching_ds[:2]) / 2
            elif len(matching_ds) == 1:
                scores[sid] = matching_ds[0]
            elif len(ds) >= 2:
                scores[sid] = sum(ds[:2]) / 2
            elif ds:
                scores[sid] = ds[0]
        sample_scores[pid] = scores
        rep_scores = [scores[s.sample_id] for s in representatives if s.sample_id in scores]
        if len(rep_scores) >= policy.min_match_buckets and pid not in forbidden:
            player_scores[pid] = float(median(rep_scores))

    if diagnostics is not None:
        diagnostics.update(
            player_scores=player_scores,
            sample_scores=sample_scores,
            counter_evidence_player_ids=sorted(counter_evidence),
            has_existing_support=any(
                s <= thresh for scores in sample_scores.values() for s in scores.values()
            ),
            has_unresolved_existing=any(
                pid not in forbidden and len(scores) < len(queries)
                for pid, scores in sample_scores.items()
            ),
        )
    if not player_scores:
        return None, None, ["no_matching_existing_player"]
    ranked = sorted(player_scores.items(), key=lambda item: (item[1], item[0]))
    best_pid, best_dist = ranked[0]
    if best_dist > thresh:
        return None, best_dist, ["best_match_exceeds_threshold"]
    margin = getattr(policy, "existing_player_rival_margin", 0.04)
    if margin is None:
        margin = policy.rival_margin
    if len(ranked) > 1 and ranked[1][1] - best_dist < margin:
        return None, best_dist, ["unresolved_competing_player_margin"]
    if best_pid in counter_evidence:
        return None, best_dist, ["minority_sample_far_from_existing_player"]

    scores = sample_scores[best_pid]
    for sid, own_score in scores.items():
        if any(
            pid != best_pid
            and sid in rival
            and rival[sid] <= thresh
            and own_score - rival[sid] >= policy.counter_evidence_margin
            for pid, rival in sample_scores.items()
        ):
            return None, best_dist, ["minority_sample_supports_competing_player"]
    if require_all_samples and (
        len(scores) != len(queries) or any(s > thresh for s in scores.values())
    ):
        return None, best_dist, ["external_anchor_requires_all_sample_support"]

    rep_scores = [scores[s.sample_id] for s in representatives if s.sample_id in scores]
    supported = sum(s <= thresh for s in rep_scores)
    coverage = supported / len(representatives) if representatives else 0.0
    small_sample_tolerance = (
        not require_all_samples
        and len(representatives) == 4
        and len(rep_scores) == 4
        and supported == 3
        and len(scores) == len(queries)
        and all(s < policy.far_distance for s in scores.values())
        and sum(rep_scores) / 4 <= thresh
    )
    if coverage < policy.min_coverage and not small_sample_tolerance:
        return None, best_dist, ["insufficient_existing_player_coverage"]
    return (
        best_pid,
        best_dist,
        [
            "matched_unique_existing_player",
            *(
                ["small_sample_uncertainty_tolerated"]
                if small_sample_tolerance and coverage < policy.min_coverage
                else []
            ),
        ],
    )


def qualify_for_new_identity(
    candidate_sids: Sequence[str],
    proposals: dict[str, Any],
    active: dict[str, Any],
    policy: DiscoveryPolicy,
) -> tuple[bool, list[str]]:
    """Evaluate whether candidate segments qualify to establish a brand-new identity.

    Evidence Threshold for New Identity Creation:
    1. Every member must have can_create_new_identity=True (e.g. >= 3 buckets, solid support).
       Weak/short segments (can_create_new_identity=False) CANNOT independently create a new identity!
    2. Must belong to an accepted candidate group spanning >= 2 distinct clips.
    3. Unmatched, distant, or single appearance does NOT equal new identity.
    """
    reasons: list[str] = []
    clips = set()
    for sid in candidate_sids:
        prop = proposals.get(sid)
        if prop is not None and not getattr(prop, "can_create_new_identity", True):
            reasons.append(f"short_segment_limited_match_support:{sid}")
        seg = active.get(sid)
        if seg is not None:
            clips.add(seg.clip_id)

    if len(clips) < 2:
        reasons.append("no_independent_cross_clip_group")

    if reasons:
        return False, sorted(set(reasons))

    return True, ["qualified_for_new_identity"]


def discover_identities(
    batch,
    feature_values=None,
    *,
    segmentation_report,
    policy=None,
    manual_constraints=None,
):
    """Pure plan. Evidence/proposals are fixed before scoring; no gallery feedback.

    At least two clips must support a group. Single appearances remain unknown.
    Each template has two nonduplicate witnesses in another member segment;
    this is a raw comparison witness, not a claim that either template was approved.
    """
    policy = DiscoveryPolicy.model_validate((policy or DiscoveryPolicy()).model_dump())
    evidence = batch.identity_evidence
    if evidence is None:
        raise ValueError("discovery requires identity evidence")
    spaces = {s.feature_space_id: s for s in evidence.feature_spaces}
    fsid = policy.feature_space_id
    if fsid is None:
        if len(spaces) != 1:
            raise ValueError("discovery needs one explicit feature_space_id for multi-space input")
        fsid = next(iter(spaces))
    if fsid not in spaces:
        raise ValueError("discovery feature space missing")
    if policy.calibration_profile is not None:
        if policy.calibration_profile.feature_space_id != fsid:
            raise ValueError(
                f"Discovery policy calibration profile space {policy.calibration_profile.feature_space_id!r} "
                f"does not match evidence feature space {fsid!r}"
            )
        if (
            policy.calibration_fingerprint is not None
            and policy.calibration_fingerprint != policy.calibration_profile.fingerprint
        ):
            raise ValueError("calibration_fingerprint mismatch with attached calibration_profile")
        prof = policy.calibration_profile
        for attr in ("intra_segment_near_distance", "inter_segment_tau", "tau"):
            if abs(getattr(policy, attr) - prof.near_distance) > 1e-6:
                raise ValueError(
                    f"Discovery policy {attr}={getattr(policy, attr)} conflicts with "
                    f"calibration_profile.near_distance={prof.near_distance}"
                )
        for attr in ("candidate_far_distance", "negative_evidence_distance", "far_distance"):
            if abs(getattr(policy, attr) - prof.far_distance) > 1e-6:
                raise ValueError(
                    f"Discovery policy {attr}={getattr(policy, attr)} conflicts with "
                    f"calibration_profile.far_distance={prof.far_distance}"
                )
    if segmentation_report.derived_evidence_fingerprint != evidence_fingerprint(
        batch, feature_values
    ):
        raise ValueError("stale segmentation report: evidence/features differ")
    proposals = {p.segment.segment_id: p for t in segmentation_report.tracks for p in t.proposals}
    active = {s.segment_id: s for s in evidence.segments if s.active}
    if set(active) != set(proposals) or any(active[k] != p.segment for k, p in proposals.items()):
        raise ValueError("segmentation report does not describe this active evidence")

    distances = _Distances(evidence.samples, spaces[fsid], feature_values, policy)
    components = _components(evidence.samples, batch.assets)

    # A descriptor contradiction within a duplicate component is never extra support.
    duplicate_sets = defaultdict(list)
    for sample in evidence.samples:
        if sample.sample_id in distances.features:
            duplicate_sets[components[sample.sample_id]].append(sample)
    bad_duplicates = set()
    for key, samples in duplicate_sets.items():
        if any(
            (d := distances.pair(a, b).distance) is None or d > policy.intra_segment_near_distance
            for a, b in combinations(samples, 2)
        ):
            bad_duplicates.add(key)

    reasons, selected, all_samples = _qualify_active_segments(
        active=active,
        proposals=proposals,
        evidence=evidence,
        distances=distances,
        components=components,
        bad_duplicates=bad_duplicates,
        policy=policy,
        manual_constraints=manual_constraints,
    )

    if policy.support_mode == "constrained_manifold":
        from app.identity.representation import pool_segment_features

        all_report_pairs: dict[str, DiscoveryPair] = {}
        groups: list[DiscoveredGroup] = []

        if selected:
            # 1. Pool features strictly for qualified segments
            seg_reps_by_id = pool_segment_features(
                segments=[active[sid] for sid in selected],
                samples=all_samples,
                space=spaces[fsid],
                values=feature_values,
                min_vis=policy.min_part_visibility,
                bucket_s=policy.bucket_s,
                near_distance=policy.intra_segment_near_distance,
                far_distance=policy.far_distance,
                max_prototypes=getattr(policy, "max_prototypes_per_segment", 4),
                components=components,
            )
            sorted_sids = sorted(
                selected.keys(),
                key=lambda sid: (
                    str(active[sid].clip_id),
                    _track_sort_key(active[sid].track_id),
                    active[sid].start_s,
                    sid,
                ),
            )
            qualified_reps = [seg_reps_by_id[sid] for sid in sorted_sids]

            # 2. Spatio-temporal hard mutual exclusion constraint engine
            engine = SpatioTemporalConstraintEngine(
                qualified_reps,
                evidence.observations,
                manual_constraints=manual_constraints,
                segments_by_id=active,
                bucket_s=policy.bucket_s,
                cooccurrence_min_buckets=policy.cooccurrence_min_buckets,
                cooccurrence_separation_heights=policy.cooccurrence_separation_heights,
            )

            # 3. Candidate generation and group acceptance
            groups, all_report_pairs, group_reasons = group_and_accept_candidates(
                qualified_sids=sorted_sids,
                active=active,
                all_samples=all_samples,
                selected=selected,
                seg_reps_by_id=seg_reps_by_id,
                distances=distances,
                components=components,
                evidence=evidence,
                policy=policy,
                engine=engine,
                fsid=fsid,
                match_id=batch.match_id,
                manual_constraints=manual_constraints,
                proposals=proposals,
            )
            for sid, r_list in group_reasons.items():
                reasons[sid].extend(r_list)

        assigned = {sid for group in groups for sid in group.member_segment_ids}
        unknown = [
            UnknownSegment(
                segment_id=sid,
                reasons=sorted(set(reasons[sid]))
                or ["isolated_or_insufficient_cross_clip_support"],
            )
            for sid in sorted(set(active) - assigned)
        ]

        policy_fp = _hash(policy.model_dump(mode="json"))
        segmentation_fp = _hash(
            segmentation_report.model_dump(mode="json", exclude={"input_fingerprint"})
        )
        input_fp = _hash(
            {
                "batch": _canonical_batch(batch),
                "resolved_features": {
                    sid: [f.tolist(), v.tolist()]
                    for sid, (f, v) in sorted(distances.features.items())
                },
            }
        )

        report_pairs = list(all_report_pairs.values())

        return DiscoveryReport(
            report_id=_id("dsr", batch.match_id, input_fp, policy_fp, segmentation_fp),
            match_id=batch.match_id,
            policy=policy,
            policy_fingerprint=policy_fp,
            input_fingerprint=input_fp,
            segmentation_fingerprint=segmentation_fp,
            feature_space_id=fsid,
            groups=groups,
            pairs=report_pairs,
            unknown=unknown,
            summary={
                "groups": len(groups),
                "member_segments": len(assigned),
                "active_segments": len(active),
                "qualified_segments": len(selected),
                "unknown_segments": len(unknown),
                "templates": sum(len(g.templates) for g in groups),
                "pair_statuses": dict(Counter(p.status for p in report_pairs)),
                "unknown_reasons": dict(Counter(r for u in unknown for r in u.reasons)),
                "calibration": "unvalidated",
                "status": "experimental",
                "completion": "partial"
                if groups and unknown
                else "discovered"
                if groups
                else "no_supported_identity_groups",
                "human_interventions": 0,
            },
        )

    pairs, by_pair = [], {}
    geometry = {
        (left, right): _separated_observations(active[left], active[right], evidence, policy)
        for left, right in combinations(sorted(selected), 2)
    }
    for left, right in combinations(sorted(selected), 2):
        if policy.support_mode in ("cooccurrence_contrastive", "compensated_contrastive"):
            from app.identity.contrastive import contrastive_pair

            pair = contrastive_pair(
                left,
                right,
                selected,
                all_samples,
                active,
                evidence,
                policy.model_copy(update={"feature_space_id": fsid}),
                distances,
                components,
                geometry,
            )
            pairs.append(pair)
            by_pair[left, right] = pair
            continue
        pid = _id("dsp", batch.match_id, fsid, left, right)
        conflict_obs = _separated_observations(active[left], active[right], evidence, policy)
        # Entire duplicated replays cannot cross-support by comparing shifted frames.
        shared = {components[s.sample_id] for s in all_samples[left]} & {
            components[s.sample_id] for s in all_samples[right]
        }
        queries = {
            sid: [s for s in selected[sid] if components[s.sample_id] not in shared]
            for sid in (left, right)
        }
        directions, comparisons, why = [], {}, []
        veto = bool(conflict_obs)
        if veto:
            why.append("spatially_separated_cooccurrence")
        for a in all_samples[left]:
            for b in all_samples[right]:
                if components[a.sample_id] == components[b.sample_id]:
                    continue
                pair = distances.pair(a, b)
                comparisons[a.sample_id, b.sample_id] = pair
                if pair.distance is not None and pair.distance >= policy.far_distance:
                    veto = True
                    why.append("contradictory_sample_pair")
        for sid, other in ((left, right), (right, left)):
            scores, supports = [], []
            for q in queries[sid]:
                ds = sorted(
                    d for t in queries[other] if (d := distances.pair(q, t).distance) is not None
                )
                if len(ds) >= 2:
                    score = sum(ds[:2]) / 2
                    scores.append(score)
                    if score <= policy.inter_segment_tau:
                        supports.append(q.sample_id)
            # Inspect even nonrepresentative raw samples for disagreement.
            for q in all_samples[sid]:
                if components[q.sample_id] in shared:
                    continue
                ds = sorted(
                    d for t in queries[other] if (d := distances.pair(q, t).distance) is not None
                )
                if len(ds) >= 2 and sum(ds[:2]) / 2 > policy.max_pair_distance:
                    veto = True
                    why.append("minority_sample_disagreement")
            coverage = len(supports) / len(selected[sid])
            directions.append(
                DiscoveryDirection(
                    query_segment_id=sid,
                    query_sample_ids=[s.sample_id for s in queries[sid]],
                    supported_sample_ids=supports,
                    coverage=coverage,
                    score=float(median(scores)) if scores else None,
                )
            )
        good = all(
            len(d.supported_sample_ids) >= policy.min_support_buckets
            and (
                d.coverage >= policy.min_coverage
                or (len(d.query_sample_ids) == 4 and len(d.supported_sample_ids) >= 3)
            )
            and d.score is not None
            and d.score <= policy.inter_segment_tau
            for d in directions
        )
        score = (
            max(d.score for d in directions)
            if all(d.score is not None for d in directions)
            else None
        )
        status = "conflict" if veto else "supported" if good else "insufficient"
        pair = DiscoveryPair(
            pair_id=pid,
            left_segment_id=left,
            right_segment_id=right,
            status=status,
            score=score,
            reasons=sorted(set(why))
            or (
                ["reciprocal_multisample_support"]
                if good
                else ["insufficient_reciprocal_independent_support"]
            ),
            directions=directions,
            comparisons=list(comparisons.values()),
            conflict_observation_ids=conflict_obs,
        )
        pairs.append(pair)
        by_pair[left, right] = pair

    def edge(a, b):
        return by_pair[tuple(sorted((a, b)))]

    neighbors = {
        sid: {
            other for other in selected if other != sid and edge(sid, other).status == "supported"
        }
        for sid in selected
    }
    blocked = set()
    for sid, ns in neighbors.items():
        if any(edge(a, b).status != "supported" for a, b in combinations(ns, 2)):
            blocked.add(sid)
            reasons[sid].append("incompatible_positive_neighbors_bridge")
    # After bridge removal, surviving graph is a disjoint union of cliques.
    candidates, visited = [], set(blocked)
    for sid in sorted(selected):
        if sid in visited:
            continue
        group = {sid} | (neighbors[sid] - blocked)
        visited.update(group)
        if any(edge(a, b).status != "supported" for a, b in combinations(group, 2)):
            for member in group:
                reasons[member].append("incomplete_group_support")
            continue
        candidates.append(group)
    groups = []
    for members in candidates:
        uncertain = set()
        for sid in sorted(members):
            scores = [edge(sid, t).score for t in members if t != sid]
            if not scores:
                continue
            best = min(scores)
            rivals = [
                edge(sid, t)
                for t in selected
                if t not in members
                and (
                    policy.support_mode == "absolute"
                    or any(
                        t != member and geometry.get(tuple(sorted((t, member))))
                        for member in members
                    )
                )
            ]
            # In contrastive mode an outside appearance is a distinct competitor
            # only with independent exclusion evidence; otherwise it may simply
            # be another unassigned appearance of this same unknown identity.
            rival_limit = (
                policy.inter_segment_tau
                if policy.support_mode == "absolute"
                else policy.candidate_far_distance
            )
            if any(
                p.score is not None
                and p.score <= rival_limit
                and p.score - best <= policy.rival_margin
                for p in rivals
            ):
                uncertain.add(sid)
                reasons[sid].append("unresolved_competing_segment_margin")
        members = members - uncertain
        if len({active[s].clip_id for s in members}) < 2:
            for sid in members:
                reasons[sid].append("no_independent_cross_clip_group")
            continue
        member_ids = sorted(members)
        support_pairs = [edge(a, b) for a, b in combinations(member_ids, 2)]
        template_candidates = defaultdict(list)
        for sid in member_ids:
            for sample in selected[sid]:
                witnesses, witness_pairs = [], []
                for other in member_ids:
                    if active[other].clip_id == active[sid].clip_id:
                        continue
                    shared = {components[s.sample_id] for s in all_samples[sid]} & {
                        components[s.sample_id] for s in all_samples[other]
                    }
                    if components[sample.sample_id] in shared:
                        continue
                    if policy.support_mode in (
                        "cooccurrence_contrastive",
                        "compensated_contrastive",
                    ):
                        direction = next(
                            d for d in edge(sid, other).directions if d.query_segment_id == sid
                        )
                        if sample.sample_id not in direction.supported_sample_ids:
                            continue
                        contrast = next(
                            c
                            for c in direction.sample_contrasts
                            if c["sample_id"] == sample.sample_id
                        )
                        close = contrast["candidate_sample_ids"]
                    else:
                        close = [
                            s.sample_id
                            for s in selected[other]
                            if components[s.sample_id] not in shared
                            and (d := distances.pair(sample, s).distance) is not None
                            and d <= policy.inter_segment_tau
                        ]
                    if len(close) >= 2:
                        witnesses.extend(close)
                        witness_pairs.append(edge(sid, other).pair_id)
                if witnesses:
                    # Other member clips may repeat each other's images; retain
                    # one raw witness per global duplicate component as well.
                    unique_witnesses = {}
                    for witness_id in sorted(set(witnesses)):
                        unique_witnesses.setdefault(components[witness_id], witness_id)
                    witnesses = list(unique_witnesses.values())
                    template_candidates[sid].append(
                        AutomaticTemplateSupport(
                            sample_id=sample.sample_id,
                            segment_id=sid,
                            feature_space_id=fsid,
                            support_pair_ids=sorted(witness_pairs),
                            witness_sample_ids=sorted(set(witnesses)),
                        )
                    )
        # Round-robin across clips/segments avoids a single segment monopolizing the gallery.
        templates, seen_components = [], set()
        ranked = sorted(member_ids, key=lambda s: (active[s].clip_id, s))
        while any(template_candidates.values()) and len(templates) < policy.max_templates_per_group:
            for sid in ranked:
                if template_candidates[sid] and len(templates) < policy.max_templates_per_group:
                    template = template_candidates[sid].pop(0)
                    comp = components[template.sample_id]
                    if comp not in seen_components:
                        templates.append(template)
                        seen_components.add(comp)
        if len({active[t.segment_id].clip_id for t in templates}) < 2:
            for sid in members:
                reasons[sid].append("insufficient_independent_template_support")
            continue
        anchor = member_ids[0]
        groups.append(
            DiscoveredGroup(
                group_id=_id("dsg", batch.match_id, anchor),
                anchor_segment_id=anchor,
                member_segment_ids=member_ids,
                support_pair_ids=[p.pair_id for p in support_pairs],
                templates=templates,
            )
        )
    assigned = {sid for group in groups for sid in group.member_segment_ids}
    unknown = [
        UnknownSegment(
            segment_id=sid,
            reasons=sorted(set(reasons[sid])) or ["isolated_or_insufficient_cross_clip_support"],
        )
        for sid in sorted(set(active) - assigned)
    ]
    policy_fp = _hash(policy.model_dump(mode="json"))
    segmentation_fp = _hash(
        segmentation_report.model_dump(mode="json", exclude={"input_fingerprint"})
    )
    input_fp = _hash(
        {
            "batch": _canonical_batch(batch),
            "resolved_features": {
                sid: [f.tolist(), v.tolist()] for sid, (f, v) in sorted(distances.features.items())
            },
        }
    )
    return DiscoveryReport(
        report_id=_id("dsr", batch.match_id, input_fp, policy_fp, segmentation_fp),
        match_id=batch.match_id,
        policy=policy,
        policy_fingerprint=policy_fp,
        input_fingerprint=input_fp,
        segmentation_fingerprint=segmentation_fp,
        feature_space_id=fsid,
        groups=groups,
        pairs=pairs,
        unknown=unknown,
        summary={
            "groups": len(groups),
            "member_segments": len(assigned),
            "active_segments": len(active),
            "qualified_segments": len(selected),
            "unknown_segments": len(unknown),
            "templates": sum(len(g.templates) for g in groups),
            "pair_statuses": dict(Counter(p.status for p in pairs)),
            "unknown_reasons": dict(Counter(r for u in unknown for r in u.reasons)),
            "calibration": "unvalidated",
            "status": "experimental",
            "completion": "partial"
            if groups and unknown
            else "discovered"
            if groups
            else "no_supported_identity_groups",
            "human_interventions": 0,
        },
    )
