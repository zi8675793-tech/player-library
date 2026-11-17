"""Pure open-set incremental identity matching, cross-batch discovery, and gallery maintenance.

Distinguishes three mutually exclusive decisions per segment:
1. matched_existing: supported segment uniquely matches an existing player in the fixed snapshot.
2. created_new_identity: supported segment joins a newly discovered cross-clip group (>= 2 clips)
   that is fully separated from all existing players.
3. insufficient_evidence: segment lacks sufficient cross-clip support, has risks,桥接,
   rival ambiguity, minority contradiction, or insufficient witnesses; stays unassigned.

No distance alone creates a new identity. Templates are updated only after batch completion.
Only support_mode='constrained_manifold' is supported; fallback is strictly prohibited.
"""

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any, Literal
from uuid import NAMESPACE_URL, uuid5

from app.contracts.common import ContractModel, Identifier
from app.contracts.discovery import (
    DiscoveredGroup,
    DiscoveryPair,
    DiscoveryPolicy,
    DiscoveryReport,
    UnknownSegment,
)
from app.contracts.evidence import EvidenceBatch, InlineFeature
from app.contracts.identity import (
    Assignment,
    AssignmentEvidence,
    ManualConstraint,
    Player,
    Template,
)
from app.contracts.segmentation import SegmentationReport
from app.identity.constraints import SpatioTemporalConstraintEngine
from app.identity.discovery import (
    _components,
    _Distances,
    _hash,
    _id,
    _separated_observations,
    evaluate_existing_player_match,
    group_and_accept_candidates,
    qualify_active_segments,
    qualify_for_new_identity,
)
from app.identity.fingerprints import evidence_fingerprint, sample_binding
from app.identity.representation import pool_segment_features


class IncrementalSegmentDecision(ContractModel):
    segment_id: Identifier
    clip_id: Identifier
    track_id: Identifier
    category: Literal["matched_existing", "created_new_identity", "insufficient_evidence"]
    player_id: Identifier | None = None
    assignment: Assignment
    reasons: list[Identifier] = []
    distance: float | None = None
    matched_existing_player_id: Identifier | None = None


class IncrementalIdentityPlan(ContractModel):
    match_id: Identifier
    feature_space_id: Identifier
    decisions: list[IncrementalSegmentDecision]
    new_players: list[Player] = []
    new_templates: list[Template] = []
    deactivated_template_ids: list[Identifier] = []
    discovery_report: DiscoveryReport
    player_map: dict[str, str] = {}
    sample_sources: dict[str, dict] = {}
    summary: dict = {}
    matching_diagnostics: dict[str, dict] = {}


def _usable_samples(samples, components, bad_duplicates, feature_ids):
    usable = []
    for s in samples:
        q = s.tracking_quality
        if components[s.sample_id] in bad_duplicates:
            continue
        if q and (
            q.occluded
            or q.mixed_identity
            or q.usability == "unusable"
            or q.clean_review == "rejected"
        ):
            continue
        if s.sample_id in feature_ids:
            usable.append(s)
    return usable


def _bucket_representatives(usable_samples, segment_start_s, bucket_s, components):
    unique = {}
    for s in usable_samples:
        unique.setdefault(components[s.sample_id], s)
    buckets = {}
    for s in sorted(unique.values(), key=lambda s: (s.time_s, s.sample_id)):
        buckets.setdefault(math.floor((s.time_s - segment_start_s) / bucket_s), s)
    return list(buckets.values())


def _is_protected_assignment(a: Assignment) -> bool:
    """Human decisions and stable confirmations outrank any automatic re-evaluation."""
    return (
        a.status == "confirmed"
        or a.method in {"manual", "seed", "imported"}
        or a.unassigned_reason in {"other_person", "rejected"}
    )


def _is_machine_tentative(a: Assignment) -> bool:
    """An automatic proposal (not yet confirmed) stays re-evaluable as new evidence arrives."""
    return a.player_id is not None and a.status == "proposed" and a.method == "automatic"


def plan_incremental_identities(
    snapshot: Any,
    batch: EvidenceBatch,
    *,
    segmentation_report: SegmentationReport,
    feature_values: Mapping[tuple[str, int], InlineFeature],
    policy: DiscoveryPolicy,
    constraints: Sequence[ManualConstraint] = (),
    association_only: bool = False,
) -> IncrementalIdentityPlan:
    """Compute pure open-set incremental identity decisions against a fixed snapshot.

    1. Mode enforcement: strictly requires support_mode='constrained_manifold'.
       Falling back to other modes or silently running other algorithms is forbidden.
    2. Fixed snapshot isolation: existing players and templates are fixed at Revision R.
       Existing players serve strictly as read-only matching targets, never manifold clustering nodes.
       Protected assignments (human manual/seed/imported, any confirmed, or rejected/other-person)
       are never re-clustered, renumbered, or automatically changed. Machine-tentative proposals
       (proposed/automatic) are re-evaluated against new evidence with leave-one-out matching: they
       may be versioned-reassigned to a different existing player that passes every gate, but are
       never auto-revoked to unassigned and never re-clustered into a new identity.
    3. Candidate pool qualification:
       - Incoming supported segments (from segmentation_report);
       - Historical unassigned OR machine-tentative segments whose persisted proposal status is
         'supported'. Historical segments without persisted supported proposals cannot restore
         qualification and safely remain deferred (unassigned) or keep their current tentative player.
    4. Two-step resolution:
       - Step A: Attempt match against existing players under hard spatio-temporal and margin constraints.
       - Step B: Shared manifold candidate grouping on remaining unmatched segments.
    5. Anti-duplicate player deduplication:
       Candidate groups must verify full separation against all Revision R existing players.
    6. Three clean, mutually exclusive segment decisions:
       - matched_existing
       - created_new_identity
       - insufficient_evidence
    """
    if policy.support_mode != "constrained_manifold":
        raise ValueError(
            f"unsupported support_mode for incremental identity planning: {policy.support_mode!r}. "
            "Explicitly configure support_mode='constrained_manifold'; falling back to other modes is strictly prohibited."
        )

    if hasattr(snapshot, "gallery"):
        gallery = snapshot.gallery
        stored_proposals = getattr(snapshot, "segment_proposals", {})
    else:
        gallery = snapshot
        stored_proposals = {}

    evidence = batch.identity_evidence
    if evidence is None:
        raise ValueError("incremental identity plan requires identity evidence")

    spaces = {s.feature_space_id: s for s in evidence.feature_spaces}
    fsid = policy.feature_space_id
    if fsid is None:
        if len(spaces) != 1:
            raise ValueError("multi-space input requires explicit feature_space_id in policy")
        fsid = next(iter(spaces))
    if fsid not in spaces:
        raise ValueError(f"feature space {fsid!r} not declared in evidence")
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
    space = spaces[fsid]

    all_assets = list({a.asset_id: a for a in [*gallery.assets, *batch.assets]}.values())
    combined_samples = {s.sample_id: s for s in [*gallery.evidence.samples, *evidence.samples]}
    all_samples_list = sorted(combined_samples.values(), key=lambda s: s.sample_id)
    components = _components(all_samples_list, all_assets)
    distances = _Distances(all_samples_list, space, feature_values, policy)

    duplicate_sets = defaultdict(list)
    for sample in all_samples_list:
        if sample.sample_id in distances.features:
            duplicate_sets[components[sample.sample_id]].append(sample)
    bad_duplicates = set()
    for comp_key, samples in duplicate_sets.items():
        if any(
            (d := distances.pair(a, b).distance) is None or d > policy.intra_segment_near_distance
            for a, b in combinations(samples, 2)
        ):
            bad_duplicates.add(comp_key)

    # Combined observations, tracks, segments for physical constraint validation
    combined_observations = list(
        {
            o.observation_id: o for o in [*gallery.evidence.observations, *evidence.observations]
        }.values()
    )
    combined_tracks = list(
        {(t.clip_id, t.track_id): t for t in [*gallery.evidence.tracks, *evidence.tracks]}.values()
    )
    combined_segments = {s.segment_id: s for s in [*gallery.evidence.segments, *evidence.segments]}
    combined_risks = list(
        {r.risk_id: r for r in [*gallery.evidence.risks, *evidence.risks]}.values()
    )
    combined_evidence_obj = type(evidence)(
        tracks=combined_tracks,
        observations=combined_observations,
        segments=list(combined_segments.values()),
        samples=all_samples_list,
        feature_spaces=[space],
        risks=combined_risks,
    )

    # Manual constraints
    forbidden_players = defaultdict(set)
    cannot_link = defaultdict(set)
    for c in constraints:
        if c.kind == "forbid_player" and c.player_id is not None:
            forbidden_players[c.segment_id].add(c.player_id)
        elif c.kind == "cannot_link" and c.related_segment_id is not None:
            cannot_link[c.segment_id].add(c.related_segment_id)
            cannot_link[c.related_segment_id].add(c.segment_id)

    # Existing players and active templates in snapshot (fixed read-only reference)
    existing_players = {p.player_id: p for p in gallery.players if p.alias_of is None}
    player_assignments = {
        a.segment_id: a.player_id for a in gallery.assignments if a.player_id is not None
    }

    # Group snapshot active templates by player
    player_templates = defaultdict(list)
    for t in gallery.templates:
        if not t.active or t.feature_space_id != fsid:
            continue
        if t.player_id not in existing_players:
            continue
        tsample = combined_samples.get(t.sample_id)
        if tsample is None or tsample.sample_id not in distances.features:
            continue
        q = tsample.tracking_quality
        if q and (q.occluded or q.usability == "unusable" or q.clean_review == "rejected"):
            continue
        player_templates[t.player_id].append((t, tsample))

    # Deduplicate player templates by component up to max_templates_per_group
    deduped_player_templates = {}
    for pid, tlist in player_templates.items():
        seen_comps = set()
        selected_tpls = []
        for t, ts in tlist:
            comp = components[ts.sample_id]
            if comp not in seen_comps:
                seen_comps.add(comp)
                selected_tpls.append((t, ts))
                if len(selected_tpls) >= policy.max_templates_per_group:
                    break
        deduped_player_templates[pid] = selected_tpls

    # Validate segmentation report freshness and proposal binding
    if (
        segmentation_report is not None
        and segmentation_report.derived_evidence_fingerprint is not None
    ):
        expected_fp = evidence_fingerprint(batch, feature_values)
        if segmentation_report.derived_evidence_fingerprint != expected_fp:
            raise ValueError(
                f"stale segmentation report: derived_evidence_fingerprint {segmentation_report.derived_evidence_fingerprint!r} "
                f"does not match current evidence fingerprint {expected_fp!r}"
            )

    # Incoming segmentation proposals
    incoming_proposals = {
        p.segment.segment_id: p for t in segmentation_report.tracks for p in t.proposals
    }
    active_incoming = {s.segment_id: s for s in evidence.segments if s.active}

    for sid, segment in sorted(active_incoming.items()):
        prop = incoming_proposals.get(sid)
        if prop and prop.segment != segment:
            raise ValueError(
                f"stale segmentation proposal for segment {sid!r}: segment definition differs"
            )

    # Evaluate incoming segment qualifications using unified qualify_active_segments
    assigned_history = {a.segment_id: a for a in gallery.assignments}
    unqualified_reasons = defaultdict(list)

    incoming_to_qualify = {}
    for sid, segment in sorted(active_incoming.items()):
        existing_a = assigned_history.get(sid)
        if existing_a and _is_protected_assignment(existing_a):
            unqualified_reasons[sid].append("already_assigned_in_gallery")
            continue
        incoming_to_qualify[sid] = segment

    inc_reasons, inc_representatives, inc_usable_samples = qualify_active_segments(
        active=incoming_to_qualify,
        proposals=incoming_proposals,
        evidence=evidence,
        distances=distances,
        components=components,
        bad_duplicates=bad_duplicates,
        policy=policy,
        manual_constraints=constraints,
        allow_short_match=True,
        allow_external_anchor=association_only,
    )
    for sid, rlist in inc_reasons.items():
        unqualified_reasons[sid].extend(rlist)

    qualified_incoming = {sid: active_incoming[sid] for sid in inc_representatives}
    incoming_usable_samples = {sid: inc_usable_samples[sid] for sid in qualified_incoming}
    incoming_representatives = {sid: inc_representatives[sid] for sid in qualified_incoming}

    # Evaluate historical unassigned segment qualifications using unified qualify_active_segments
    historical_to_qualify = {}
    historical_proposals = {}

    for seg in gallery.evidence.segments:
        if not seg.active:
            continue
        if seg.segment_id in active_incoming:
            continue
        a = assigned_history.get(seg.segment_id)
        if (
            a is not None
            and not _is_protected_assignment(a)
            and (a.status == "unassigned" or _is_machine_tentative(a))
        ):
            stored_prop = stored_proposals.get(seg.segment_id)
            if stored_prop is None or (
                stored_prop.status != "supported"
                and not (
                    association_only and getattr(stored_prop, "requires_external_anchor", False)
                )
            ):
                unqualified_reasons[seg.segment_id].append(
                    "missing_or_unsupported_historical_proposal"
                )
                continue
            if stored_prop.segment != seg:
                unqualified_reasons[seg.segment_id].append(
                    "stale_historical_proposal_segment_mismatch"
                )
                continue
            historical_to_qualify[seg.segment_id] = seg
            historical_proposals[seg.segment_id] = stored_prop

    hist_reasons, hist_representatives, hist_usable_samples = qualify_active_segments(
        active=historical_to_qualify,
        proposals=historical_proposals,
        evidence=combined_evidence_obj,
        distances=distances,
        components=components,
        bad_duplicates=bad_duplicates,
        policy=policy,
        manual_constraints=constraints,
        allow_short_match=True,
        allow_external_anchor=association_only,
    )
    for sid, rlist in hist_reasons.items():
        unqualified_reasons[sid].extend(rlist)

    qualified_historical = {sid: historical_to_qualify[sid] for sid in hist_representatives}
    historical_usable_samples = {sid: hist_usable_samples[sid] for sid in qualified_historical}
    historical_representatives = {sid: hist_representatives[sid] for sid in qualified_historical}

    # Combine all qualified segments for feature pooling and mutual exclusion
    all_qualified_segments = {**qualified_incoming, **qualified_historical}
    all_usable_samples = {**incoming_usable_samples, **historical_usable_samples}
    all_representatives = {**incoming_representatives, **historical_representatives}
    all_proposals = {**stored_proposals, **incoming_proposals}

    # Machine-tentative segments (proposed/automatic) are re-evaluated, not frozen.
    # Leave-one-out: a segment must not match its own contributed templates, so its
    # samples are excluded from every player while it is being re-assessed.
    reeval_tentative_sids = {
        sid
        for sid in all_qualified_segments
        if assigned_history.get(sid) is not None and _is_machine_tentative(assigned_history[sid])
    }
    loo_templates_by_sid = {
        sid: {
            pid: [(t, ts) for (t, ts) in tlist if ts.segment_id != sid]
            for pid, tlist in deduped_player_templates.items()
        }
        for sid in reeval_tentative_sids
    }

    # Pool track-level representations strictly for qualified segments
    seg_reps_by_id = pool_segment_features(
        segments=list(all_qualified_segments.values()),
        samples=all_usable_samples,
        space=space,
        values=feature_values,
        min_vis=policy.min_part_visibility,
        bucket_s=policy.bucket_s,
        near_distance=policy.intra_segment_near_distance,
        far_distance=policy.far_distance,
        max_prototypes=getattr(policy, "max_prototypes_per_segment", 4),
        components=components,
    )

    # Physical spatio-temporal conflict checker
    def are_segments_conflicting(s1, s2):
        if s2.segment_id in cannot_link[s1.segment_id]:
            return True
        if s1.clip_id == s2.clip_id:
            if engine is not None and hasattr(engine, "pair_relation"):
                if engine.pair_relation(s1, s2) != "clear":
                    return True
            elif str(s1.track_id) != str(s2.track_id):
                if max(s1.start_s, s2.start_s) < min(s1.end_s, s2.end_s):
                    return True
            if _separated_observations(s1, s2, combined_evidence_obj, policy):
                return True
        return False

    # Instantiate spatio-temporal constraint engine with player assignments
    engine = SpatioTemporalConstraintEngine(
        tracks=list(seg_reps_by_id.values()),
        observations=combined_observations,
        manual_constraints=constraints,
        player_assignments=player_assignments,
        segments_by_id=combined_segments,
        bucket_s=policy.bucket_s,
        cooccurrence_min_buckets=policy.cooccurrence_min_buckets,
        cooccurrence_separation_heights=policy.cooccurrence_separation_heights,
    )

    # Step A: Match qualified segments against existing players
    match_threshold = (
        policy.existing_player_match_threshold
        if policy.existing_player_match_threshold is not None
        else policy.clustering_threshold
    )

    matched_existing: dict[str, str] = {}
    matched_diagnostics: dict[str, dict[str, Any]] = {}
    candidate_matches_by_sid: dict[str, tuple[str, float]] = {}

    matching_diagnostics = {}
    for sid, segment in sorted(all_qualified_segments.items()):
        reps = all_representatives[sid]
        usable = all_usable_samples[sid]
        prop = all_proposals.get(sid)
        is_short = getattr(prop, "can_create_new_identity", True) is False or getattr(
            prop, "reasons", []
        ) == ["short_segment_limited_match_support"]
        can_match = getattr(prop, "can_match_existing", True)
        seg_match_threshold = (
            policy.short_segment_match_threshold
            if is_short and policy.short_segment_match_threshold is not None
            else match_threshold
        )

        diagnostic = {}
        best_pid, best_dist, match_reasons = evaluate_existing_player_match(
            segment=segment,
            usable_samples=usable,
            representatives=reps,
            existing_players=existing_players,
            player_templates=loo_templates_by_sid.get(sid, deduped_player_templates),
            distances=distances,
            components=components,
            policy=policy,
            spatio_temporal_engine=engine,
            forbidden_players=forbidden_players[sid],
            can_match=can_match,
            match_threshold=seg_match_threshold,
            require_all_samples=getattr(prop, "requires_external_anchor", False),
            require_independent_anchor=getattr(prop, "independent_anchor_required", False),
            diagnostics=diagnostic,
        )
        matching_diagnostics[sid] = {"existing_match": diagnostic}

        if best_pid is not None and best_dist is not None:
            candidate_matches_by_sid[sid] = (best_pid, best_dist)
            matched_diagnostics[sid] = {
                "player_id": best_pid,
                "distance": best_dist,
                "reasons": match_reasons,
            }
        elif "unresolved_competing_player_margin" in match_reasons:
            unqualified_reasons[sid].append("ambiguous_competing_existing_players")
        elif match_reasons not in (
            ["no_matching_existing_player"],
            ["best_match_exceeds_threshold"],
        ):
            unqualified_reasons[sid].extend(match_reasons)

    # Archive matches of individual exposures need independent local corroboration.
    # This never propagates identity to intervening samples or observations: each
    # accepted image has already passed every archive, far and rival gate itself.
    if policy.require_local_archive_corroboration:
        atomic = {
            sid: match
            for sid, match in candidate_matches_by_sid.items()
            if getattr(all_proposals.get(sid), "independent_anchor_required", False)
        }
        rejected_atomic = set()
        for sid, (pid, _) in atomic.items():
            prop = all_proposals[sid]
            query = all_usable_samples[sid][0]
            witnesses = []
            for other_sid, (other_pid, _) in atomic.items():
                other_prop = all_proposals[other_sid]
                other = all_usable_samples[other_sid][0]
                if (
                    sid == other_sid
                    or pid != other_pid
                    or prop.parent_segment_id != other_prop.parent_segment_id
                    or components[query.sample_id] == components[other.sample_id]
                ):
                    continue
                start, end = sorted((query.time_s, other.time_s))
                if not (
                    policy.bucket_s
                    <= end - start
                    <= segmentation_report.policy.max_continuous_sample_gap_s
                ):
                    continue
                # No path may bridge a hole, an unusable image, an incomparable
                # observation or a single far intruder, even with clean endpoints.
                if any(
                    p.parent_segment_id == prop.parent_segment_id
                    and p.segment.start_s < end
                    and p.segment.end_s > start
                    and p.reasons == ["unobserved_or_unreliable_interval"]
                    for p in all_proposals.values()
                ):
                    continue
                between = [
                    s
                    for s in all_samples_list
                    if (s.clip_id, s.track_id) == (query.clip_id, query.track_id)
                    and start <= s.time_s <= end
                ]
                if any(
                    s.sample_id not in distances.features
                    or (
                        s.tracking_quality is not None
                        and (
                            s.tracking_quality.occluded
                            or s.tracking_quality.mixed_identity
                            or s.tracking_quality.usability == "unusable"
                            or s.tracking_quality.clean_review == "rejected"
                        )
                    )
                    for s in between
                ):
                    continue
                if any(
                    (d := distances.pair(a, b).distance) is None or d >= policy.far_distance
                    for a, b in combinations(between, 2)
                ):
                    continue
                witnesses.append(
                    {
                        "segment_id": other_sid,
                        "sample_id": other.sample_id,
                        "elapsed_s": end - start,
                        "distance": distances.pair(query, other).distance,
                    }
                )
            matching_diagnostics[sid]["local_archive_corroboration"] = {
                "status": "pass" if witnesses else "fail",
                "witnesses": witnesses,
                "propagation_boundary": "observed_sample_exposure_only",
                "continuity_is_identity_evidence": False,
            }
            if not witnesses:
                rejected_atomic.add(sid)
                unqualified_reasons[sid].append("insufficient_local_archive_corroboration")
        for sid in rejected_atomic:
            del candidate_matches_by_sid[sid]
            del matched_diagnostics[sid]

    # Resolve mutual exclusion among segments matching the same existing player
    matched_players_grouped = defaultdict(list)
    for sid, (pid, dist) in candidate_matches_by_sid.items():
        matched_players_grouped[pid].append((sid, dist))

    for pid, sids_dists in matched_players_grouped.items():
        sids = [s for s, d in sids_dists]
        dists = {s: d for s, d in sids_dists}
        conflicted_out = set()
        if len(sids) > 1:
            for s1, s2 in combinations(sids, 2):
                if are_segments_conflicting(all_qualified_segments[s1], all_qualified_segments[s2]):
                    d1, d2 = dists[s1], dists[s2]
                    if abs(d1 - d2) < policy.rival_margin:
                        conflicted_out.update((s1, s2))
                        unqualified_reasons[s1].append(
                            "ambiguous_spatio_temporal_conflict_for_player"
                        )
                        unqualified_reasons[s2].append(
                            "ambiguous_spatio_temporal_conflict_for_player"
                        )
                    else:
                        rejected = s2 if d1 < d2 else s1
                        conflicted_out.add(rejected)
                        unqualified_reasons[rejected].append(
                            "spatio_temporal_conflict_with_closer_match"
                        )
        for sid, dist in sids_dists:
            matching_diagnostics[sid]["batch_spatiotemporal"] = {
                "status": "fail" if sid in conflicted_out else "pass",
                "reasons": list(unqualified_reasons[sid]),
            }
            if sid not in conflicted_out:
                matched_existing[sid] = pid
                matched_diagnostics[sid].update(player_id=pid, distance=dist)

    if policy.require_local_archive_corroboration:
        # A witness removed by the batch physical conflict resolver cannot certify
        # another exposure. Prune to a fixed point, without selecting a winner.
        changed = True
        while changed:
            changed = False
            for sid in list(matched_existing):
                gate = matching_diagnostics[sid].get("local_archive_corroboration")
                if gate is None:
                    continue
                surviving = [
                    w
                    for w in gate["witnesses"]
                    if matched_existing.get(w["segment_id"]) == matched_existing[sid]
                ]
                if not surviving:
                    gate.update(status="fail", surviving_witnesses=[])
                    unqualified_reasons[sid].append("local_archive_witness_removed_by_conflict")
                    del matched_existing[sid]
                    del matched_diagnostics[sid]
                    changed = True
                else:
                    gate["surviving_witnesses"] = surviving

    for sid, reasons in unqualified_reasons.items():
        if sid not in matching_diagnostics:
            matching_diagnostics[sid] = {
                "existing_match": {
                    "status": "not_evaluated",
                    "prequalification_reasons": list(reasons),
                }
            }

    # Step B: Open-set constrained manifold clustering on remaining unmatched segments
    for sid in all_qualified_segments:
        if sid not in matched_existing and not getattr(
            all_proposals.get(sid),
            "can_create_new_identity",
            True,
        ):
            if not unqualified_reasons[sid]:
                unqualified_reasons[sid].append("short_segment_limited_match_support")

    unmatched_sids = [
        sid
        for sid in sorted(all_qualified_segments)
        if not association_only
        and sid not in matched_existing
        and sid not in reeval_tentative_sids
        and not unqualified_reasons[sid]
        and getattr(
            all_proposals.get(sid),
            "can_create_new_identity",
            True,
        )
    ]

    discovered_groups: list[DiscoveredGroup] = []
    all_report_pairs: dict[str, DiscoveryPair] = {}
    new_players: list[Player] = []
    player_map: dict[str, str] = {}
    new_identity_assignment_map: dict[str, str] = {}

    if len(unmatched_sids) >= 2:
        candidate_groups, group_pairs, group_reasons = group_and_accept_candidates(
            qualified_sids=unmatched_sids,
            active=all_qualified_segments,
            all_samples=all_usable_samples,
            selected=all_representatives,
            seg_reps_by_id=seg_reps_by_id,
            distances=distances,
            components=components,
            evidence=combined_evidence_obj,
            policy=policy,
            engine=engine,
            fsid=fsid,
            match_id=batch.match_id,
            manual_constraints=constraints,
            proposals=all_proposals,
        )
        all_report_pairs.update(group_pairs)
        for sid, r_list in group_reasons.items():
            unqualified_reasons[sid].extend(r_list)

        # Step 5: Group deduplication against existing players in Revision R
        for g in candidate_groups:
            member_sids = g.member_segment_ids
            anchor_segment_id = g.anchor_segment_id
            group_id = g.group_id

            unresolved_existing = False
            for m_id in member_sids:
                diagnostics: dict[str, Any] = {}
                matched_pid, _, _ = evaluate_existing_player_match(
                    segment=all_qualified_segments[m_id],
                    usable_samples=all_usable_samples[m_id],
                    representatives=all_representatives[m_id],
                    existing_players=existing_players,
                    player_templates=deduped_player_templates,
                    distances=distances,
                    components=components,
                    policy=policy,
                    spatio_temporal_engine=engine,
                    forbidden_players=forbidden_players[m_id],
                    match_threshold=match_threshold,
                    diagnostics=diagnostics,
                )
                matching_diagnostics.setdefault(m_id, {})["new_group_existing_check"] = diagnostics
                if matched_pid is not None or any(
                    score <= match_threshold
                    for score in diagnostics.get("player_scores", {}).values()
                ):
                    unresolved_existing = True
            if unresolved_existing:
                for sid in member_sids:
                    unqualified_reasons[sid].append("group_not_separated_from_existing_players")
                continue
            else:
                ok, qual_reasons = qualify_for_new_identity(
                    member_sids, all_proposals, all_qualified_segments, policy
                )
                if (
                    not ok
                    or len(g.templates) < 2
                    or len({all_qualified_segments[t.segment_id].clip_id for t in g.templates}) < 2
                ):
                    for sid in member_sids:
                        unqualified_reasons[sid].extend(
                            qual_reasons or ["insufficient_profile_evidence_or_templates"]
                        )
                    continue

                assigned_pid = (
                    "pl_"
                    + uuid5(
                        NAMESPACE_URL, f"{batch.match_id}\0auto-discovery\0{anchor_segment_id}"
                    ).hex
                )
                player_map[group_id] = assigned_pid
                new_players.append(Player(match_id=batch.match_id, player_id=assigned_pid))
                for m_id in member_sids:
                    new_identity_assignment_map[m_id] = assigned_pid
                discovered_groups.append(g)
    else:
        for sid in unmatched_sids:
            unqualified_reasons[sid].append("no_independent_cross_clip_group")

    # Step C: Template admission and decision construction
    input_fp = evidence_fingerprint(batch, feature_values)
    seg_fp = _hash(segmentation_report.model_dump(mode="json", exclude={"input_fingerprint"}))
    policy_fp = _hash(policy.model_dump(mode="json"))
    report_id = _id(
        "dsr",
        batch.match_id,
        input_fp,
        policy_fp,
        fsid,
        seg_fp,
        gallery.revision,
        association_only,
        "safe-matching-v3-diagnostics",
    )

    decisions: list[IncrementalSegmentDecision] = []
    new_templates: list[Template] = []
    admitted_template_keys = set()

    # Pre-add templates from discovered groups
    for g in discovered_groups:
        pid = player_map[g.group_id]
        for support in g.templates:
            key = (pid, support.feature_space_id, support.sample_id)
            if key not in admitted_template_keys:
                admitted_template_keys.add(key)
                tpl_id = (
                    "tpl_auto_"
                    + hashlib.sha256(
                        f"{batch.match_id}\0{pid}\0{support.feature_space_id}\0{support.sample_id}".encode()
                    ).hexdigest()[:24]
                )
                new_templates.append(
                    Template(
                        template_id=tpl_id,
                        match_id=batch.match_id,
                        player_id=pid,
                        sample_id=support.sample_id,
                        feature_space_id=support.feature_space_id,
                        review_status="unreviewed",
                        active=True,
                        admission="automatic_experimental",
                        approval=None,
                        automatic_support_id=report_id,
                        revision=gallery.revision + 1,
                        prototype_id=support.prototype_id,
                        time_bucket=support.time_bucket,
                        reliable_parts=support.reliable_parts,
                        quality_score=support.quality_score,
                        risks=support.risks,
                        source_component=support.source_component,
                    )
                )

    # Versioned re-evaluation of machine-tentative assignments. A tentative segment
    # is kept unless leave-one-out matching finds a different existing player that
    # passes every gate; it is never auto-revoked and never re-clustered as new.
    deactivated_template_ids: list[str] = []
    reassigned_tentative = 0
    kept_tentative = 0

    def _tentative_reeval_decision(sid, segment):
        """Return a reassignment decision, or None when the current player is kept.

        A kept tentative assignment is a no-op: it emits no decision and never
        revokes to unassigned, even when the segment failed to re-qualify.
        """
        nonlocal reassigned_tentative, kept_tentative
        current_pid = assigned_history[sid].player_id
        new_pid = matched_existing.get(sid, current_pid)
        diag = matched_diagnostics.get(sid)
        if new_pid == current_pid:
            kept_tentative += 1
            matching_diagnostics.setdefault(sid, {})["reassessment"] = {
                "current_player_id": current_pid,
                "leave_one_out": sid in loo_templates_by_sid,
                "outcome": "kept",
            }
            return None
        reassigned_tentative += 1
        dist = diag["distance"] if diag else None
        stale = [
            t.template_id
            for t in gallery.templates
            if t.active
            and t.player_id == current_pid
            and combined_samples.get(t.sample_id) is not None
            and combined_samples[t.sample_id].segment_id == sid
        ]
        deactivated_template_ids.extend(stale)
        matching_diagnostics.setdefault(sid, {})["reassessment"] = {
            "current_player_id": current_pid,
            "leave_one_out": True,
            "outcome": "reassigned",
            "new_player_id": new_pid,
            "deactivated_template_ids": stale,
        }
        assignment = Assignment(
            match_id=batch.match_id,
            segment_id=sid,
            player_id=new_pid,
            status="proposed",
            method="automatic",
            evidence=[
                AssignmentEvidence(
                    kind="automatic_reassessment_reassignment",
                    reference_id=report_id,
                    feature_space_id=fsid,
                    distance=dist,
                ),
                AssignmentEvidence(
                    kind="automatic_reassessment_superseded_player",
                    reference_id=current_pid,
                ),
            ],
            revision=gallery.revision + 1,
        )
        return IncrementalSegmentDecision(
            segment_id=sid,
            clip_id=segment.clip_id,
            track_id=segment.track_id,
            category="matched_existing",
            player_id=new_pid,
            assignment=assignment,
            distance=dist,
            matched_existing_player_id=new_pid,
        )

    # Decisions for incoming batch segments
    for sid, segment in sorted(active_incoming.items()):
        existing_a = assigned_history.get(sid)
        if existing_a and _is_protected_assignment(existing_a):
            decisions.append(
                IncrementalSegmentDecision(
                    segment_id=sid,
                    clip_id=segment.clip_id,
                    track_id=segment.track_id,
                    category="matched_existing"
                    if existing_a.player_id
                    else "insufficient_evidence",
                    player_id=existing_a.player_id,
                    assignment=existing_a,
                )
            )
            continue
        if existing_a and _is_machine_tentative(existing_a):
            reeval = _tentative_reeval_decision(sid, segment)
            if reeval is not None:
                decisions.append(reeval)
            continue
        elif sid in matched_existing:
            assigned_pid = matched_existing[sid]
            diag = matched_diagnostics.get(sid)
            dist = diag["distance"] if diag else None
            assignment = Assignment(
                match_id=batch.match_id,
                segment_id=sid,
                player_id=assigned_pid,
                status="proposed",
                method="automatic",
                evidence=[
                    AssignmentEvidence(
                        kind="automatic_incremental_match",
                        reference_id=report_id,
                        feature_space_id=fsid,
                        distance=dist,
                    )
                ],
                revision=gallery.revision + 1,
            )
            decisions.append(
                IncrementalSegmentDecision(
                    segment_id=sid,
                    clip_id=segment.clip_id,
                    track_id=segment.track_id,
                    category="matched_existing",
                    player_id=assigned_pid,
                    assignment=assignment,
                    distance=dist,
                    matched_existing_player_id=assigned_pid,
                )
            )

        elif sid in new_identity_assignment_map:
            assigned_pid = new_identity_assignment_map[sid]
            assignment = Assignment(
                match_id=batch.match_id,
                segment_id=sid,
                player_id=assigned_pid,
                status="proposed",
                method="automatic",
                evidence=[
                    AssignmentEvidence(
                        kind="automatic_discovery_incremental",
                        reference_id=report_id,
                        feature_space_id=fsid,
                    )
                ],
                revision=gallery.revision + 1,
            )
            decisions.append(
                IncrementalSegmentDecision(
                    segment_id=sid,
                    clip_id=segment.clip_id,
                    track_id=segment.track_id,
                    category="created_new_identity",
                    player_id=assigned_pid,
                    assignment=assignment,
                )
            )
        else:
            reasons = unqualified_reasons[sid] or ["insufficient_cross_clip_support"]
            unassigned_reason = (
                "ambiguous" if "ambiguous_competing_existing_players" in reasons else "unknown"
            )
            assignment = Assignment(
                match_id=batch.match_id,
                segment_id=sid,
                player_id=None,
                status="unassigned",
                method="automatic",
                unassigned_reason=unassigned_reason,
                evidence=[
                    AssignmentEvidence(
                        kind="insufficient_evidence",
                        description="; ".join(reasons),
                    )
                ],
                revision=gallery.revision + 1,
            )
            decisions.append(
                IncrementalSegmentDecision(
                    segment_id=sid,
                    clip_id=segment.clip_id,
                    track_id=segment.track_id,
                    category="insufficient_evidence",
                    player_id=None,
                    assignment=assignment,
                    reasons=reasons,
                )
            )

    # Decisions for historical segments resolved into groups or re-evaluated tentative
    resolved_historical_sids = {
        sid
        for sid in qualified_historical
        if sid in matched_existing
        or sid in new_identity_assignment_map
        or sid in reeval_tentative_sids
    }
    for sid in sorted(resolved_historical_sids - set(active_incoming)):
        hist_seg = qualified_historical[sid]
        if sid in reeval_tentative_sids:
            reeval = _tentative_reeval_decision(sid, hist_seg)
            if reeval is not None:
                decisions.append(reeval)
        elif sid in matched_existing:
            assigned_pid = matched_existing[sid]
            diag = matched_diagnostics.get(sid)
            dist = diag["distance"] if diag else None
            assignment = Assignment(
                match_id=batch.match_id,
                segment_id=sid,
                player_id=assigned_pid,
                status="proposed",
                method="automatic",
                evidence=[
                    AssignmentEvidence(
                        kind="automatic_incremental_match",
                        reference_id=report_id,
                        feature_space_id=fsid,
                        distance=dist,
                    )
                ],
                revision=gallery.revision + 1,
            )
            decisions.append(
                IncrementalSegmentDecision(
                    segment_id=sid,
                    clip_id=hist_seg.clip_id,
                    track_id=hist_seg.track_id,
                    category="matched_existing",
                    player_id=assigned_pid,
                    assignment=assignment,
                    distance=dist,
                    matched_existing_player_id=assigned_pid,
                )
            )
        else:
            assigned_pid = new_identity_assignment_map[sid]
            assignment = Assignment(
                match_id=batch.match_id,
                segment_id=sid,
                player_id=assigned_pid,
                status="proposed",
                method="automatic",
                evidence=[
                    AssignmentEvidence(
                        kind="automatic_discovery_incremental",
                        reference_id=report_id,
                        feature_space_id=fsid,
                    )
                ],
                revision=gallery.revision + 1,
            )
            decisions.append(
                IncrementalSegmentDecision(
                    segment_id=sid,
                    clip_id=hist_seg.clip_id,
                    track_id=hist_seg.track_id,
                    category="created_new_identity",
                    player_id=assigned_pid,
                    assignment=assignment,
                )
            )

    # Unknown segments for discovery report. Machine-tentative incoming segments keep
    # their current player even when re-evaluation emits no decision, so they are
    # assigned, not unknown.
    tentative_incoming = {
        sid
        for sid in active_incoming
        if assigned_history.get(sid) is not None and _is_machine_tentative(assigned_history[sid])
    }
    assigned_sids = {
        d.segment_id for d in decisions if d.player_id is not None
    } | tentative_incoming
    unknown_report_segments = [
        UnknownSegment(
            segment_id=sid,
            reasons=unqualified_reasons[sid] or ["isolated_or_insufficient_cross_clip_support"],
        )
        for sid in sorted(set(active_incoming) - assigned_sids)
    ]

    policy_fp = _hash(policy.model_dump(mode="json"))
    report = DiscoveryReport(
        report_id=report_id,
        match_id=batch.match_id,
        policy=policy,
        policy_fingerprint=policy_fp,
        input_fingerprint=input_fp,
        segmentation_fingerprint=seg_fp,
        feature_space_id=fsid,
        groups=discovered_groups,
        pairs=list(all_report_pairs.values()),
        unknown=unknown_report_segments,
        summary={
            "snapshot_revision": gallery.revision,
            "association_only": association_only,
            "matching_algorithm": "safe-matching-v3-diagnostics",
            "groups": len(discovered_groups),
            "new_players": len(new_players),
            "matched_existing": sum(d.category == "matched_existing" for d in decisions),
            "created_new_identity": sum(d.category == "created_new_identity" for d in decisions),
            "insufficient_evidence": sum(d.category == "insufficient_evidence" for d in decisions),
            "active_segments": len(active_incoming),
            "templates": len(new_templates),
            "completion": "discovered" if discovered_groups else "no_supported_identity_groups",
        },
    )

    support_ids = {
        sid
        for group in discovered_groups
        for support in group.templates
        for sid in [support.sample_id, *support.witness_sample_ids]
    }
    support_pair_ids = {
        pair_id
        for group in discovered_groups
        for support in group.templates
        for pair_id in support.support_pair_ids
    }
    support_ids.update(
        sample_id
        for pair in all_report_pairs.values()
        if pair.pair_id in support_pair_ids
        for comparison in pair.comparisons
        for sample_id in (comparison.left_sample_id, comparison.right_sample_id)
    )
    sample_ids_to_bind = support_ids.union(s.sample_id for s in batch.identity_evidence.samples)
    sample_sources = {
        s_id: sample_binding(combined_samples[s_id], all_assets)
        for s_id in sample_ids_to_bind
        if s_id in combined_samples
    }

    from collections import Counter

    deferred_counts = Counter(
        reason for d in decisions if d.category == "insufficient_evidence" for reason in d.reasons
    )

    summary = {
        "batch_id": batch.batch_id,
        "input_fingerprint": input_fp,
        "policy_fingerprint": policy_fp,
        "segmentation_fingerprint": seg_fp,
        "matched_existing_count": sum(d.category == "matched_existing" for d in decisions),
        "created_new_identity_count": sum(d.category == "created_new_identity" for d in decisions),
        "insufficient_evidence_count": sum(
            d.category == "insufficient_evidence" for d in decisions
        ),
        "new_players_count": len(new_players),
        "new_templates_count": len(new_templates),
        "active_players_count": len(existing_players) + len(new_players),
        "reassessed_tentative_count": len(reeval_tentative_sids),
        "reassigned_tentative_count": reassigned_tentative,
        "kept_tentative_count": kept_tentative,
        "deactivated_templates_count": len(deactivated_template_ids),
        "deferred_reasons": dict(deferred_counts),
    }

    return IncrementalIdentityPlan(
        match_id=batch.match_id,
        feature_space_id=fsid,
        decisions=decisions,
        new_players=new_players,
        new_templates=new_templates,
        deactivated_template_ids=deactivated_template_ids,
        discovery_report=report,
        player_map=player_map,
        sample_sources=sample_sources,
        summary=summary,
        matching_diagnostics=matching_diagnostics,
    )
