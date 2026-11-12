"""Conservative, explainable within-track consistency from machine evidence only.

Distance alone can veto a whole-track proposal but cannot assert a switch. A
suspected switch needs persistent appearance separation AND a local spatial
break. Unobserved transition intervals remain unknown. No clustering of players,
file reads, upstream review-label interpretation, or segment-average embeddings.
"""

import hashlib
import json
import math
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np

from app.contracts.common import Provenance
from app.contracts.evidence import EvidenceBatch, IdentityEvidence
from app.contracts.identity import IdentityRisk, MatchingPolicy, SampleExclusion, Segment
from app.contracts.segmentation import (
    ConsistencyFinding,
    ConsistencyPair,
    SegmentationPolicy,
    SegmentationReport,
    SegmentationResult,
    SegmentProposal,
    TrackConsistencyReport,
)
from app.identity.distance import FeatureValidationError, validate_feature
from app.identity.fingerprints import evidence_fingerprint
from app.identity.prepare import FeatureValues, evidence_keys, resolve_feature


def _hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def _id(prefix, *values):
    return f"{prefix}_{_hash(values)[:32]}"


def _iou(a, b):
    x1, y1, x2, y2 = a
    u1, v1, u2, v2 = b
    intersection = max(0, min(x2, u2) - max(x1, u1)) * max(0, min(y2, v2) - max(y1, v1))
    return intersection / ((x2 - x1) * (y2 - y1) + (u2 - u1) * (v2 - v1) - intersection)


def _jump(a, b, policy):
    dt = b.time_s - a.time_s
    if dt <= 0:
        return False, 0.0
    x, y, z, w = a.bbox_xyxy
    u, v, s, t = b.bbox_xyxy
    height = max((w - y + t - v) / 2, 1e-12)
    distance = math.hypot((x + z - u - s) / 2, (y + w - v - t) / 2) / height
    return distance >= policy.jump_heights and distance / dt >= policy.jump_heights_per_s, distance


def _end_of(sample, clip, end):
    # One observed frame exposure in CFR; otherwise no invented VFR duration.
    if clip is not None and clip.frame_timing == "cfr" and clip.fps is not None:
        return min(end, sample.time_s + 1 / clip.fps)
    return min(end, math.nextafter(sample.time_s, math.inf))


def _source(parent, fingerprint, details, observations, *, version="track-consistency-v1"):
    # Endpoints locate the source rows; per-proposal observation IDs retain the full set.
    endpoints = observations[:1] + observations[-1:]
    locations = {loc.model_dump_json(): loc for obs in endpoints for loc in obs.source.locations}
    return Provenance(
        producer=version.replace("-", "_"),
        record_type="segment_consistency",
        source_record_id=parent.segment_id + ":" + _hash(details)[:24],
        evidence_version=fingerprint,
        content_hash=_hash(details),
        locations=[locations[key] for key in sorted(locations)],
    )


class _Distances:
    def __init__(self, samples, spaces, values, policy):
        self.policy = policy
        self.features = {}
        self.comparisons = {}
        self.spaces = spaces
        self.thresholds = {p.feature_space_id: p for p in policy.spaces}
        for prof in getattr(policy, "calibration_profiles", []):
            self.thresholds[prof.feature_space_id] = prof
        for sample in samples:
            resolved = {}
            for fsid, space in spaces.items():
                feature = resolve_feature(sample, space, values)
                if feature is None:
                    continue
                mp = MatchingPolicy(feature_space_id=fsid, min_common_parts=policy.min_common_parts)
                validate_feature(space, feature, mp)
                resolved[fsid] = (
                    np.asarray(feature.values).reshape(space.shape),
                    np.asarray(feature.visibility),
                )
            self.features[sample.sample_id] = resolved

    def pair(self, a, b):
        key = tuple(sorted((a.sample_id, b.sample_id)))
        if key in self.comparisons:
            return self.comparisons[key]
        result = []
        for fsid in sorted(set(self.features[key[0]]) & set(self.features[key[1]])):
            fa, va = self.features[key[0]][fsid]
            fb, vb = self.features[key[1]][fsid]
            common = (va >= self.policy.min_part_visibility) & (
                vb >= self.policy.min_part_visibility
            )
            common &= (va > 0) & (vb > 0)
            count = int(common.sum())
            comparable = count >= self.policy.min_common_parts
            weight = np.sqrt(va[common] * vb[common])
            distance = (
                float(
                    np.dot(weight, np.linalg.norm(fa[common] - fb[common], axis=1))
                    / (2 * weight.sum())
                )
                if comparable
                else None
            )
            result.append(
                ConsistencyPair(
                    left_sample_id=key[0],
                    right_sample_id=key[1],
                    feature_space_id=fsid,
                    comparable=comparable,
                    common_parts=count,
                    distance=distance,
                    reason=None if comparable else "insufficient_reliable_common_parts",
                )
            )
        self.comparisons[key] = result
        return result

    def relation(self, a, b):
        pairs = self.pair(a, b)
        valid = [p for p in pairs if p.comparable]
        if not valid:
            return "uncomparable"
        near, far = [], []
        for pair in valid:
            if pair.feature_space_id in self.thresholds:
                thresholds = self.thresholds[pair.feature_space_id]
            elif self.thresholds or getattr(self.policy, "strict_space_calibration", False):
                raise FeatureValidationError(
                    f"Missing calibration profile for feature space {pair.feature_space_id!r}; "
                    "silent reuse of other space thresholds is forbidden"
                )
            else:
                thresholds = self.policy
            near.append(pair.distance <= thresholds.near_distance)
            far.append(pair.distance >= thresholds.far_distance)
        # Spaces never get averaged; disagreement blocks a supported proposal.
        if all(near):
            return "near"
        if all(far):
            return "far"
        return "ambiguous"


def _is_coherent_local_run(run, distances, policy):
    """Check if a contiguous chronological sample run is locally coherent in appearance.

    A coherent local run requires:
    1. Consecutive sample gap <= policy.max_sample_gap_s.
    2. Every adjacent pair has near appearance relation (relation == 'near').
    3. Zero far pairs (no strong counter-evidence within the run).
    4. Bounded radius: There exists at least one sample m in run that serves as
       a medoid anchor, such that relation(m, s) == 'near' for every s in run.
    """
    if len(run) <= 1:
        return True
    if any(b.time_s - a.time_s > policy.max_sample_gap_s for a, b in zip(run, run[1:])):
        return False
    if any(distances.relation(a, b) == "far" for a, b in combinations(run, 2)):
        return False
    if any(distances.relation(a, b) != "near" for a, b in zip(run, run[1:])):
        return False
    return any(all(distances.relation(m, s) == "near" for s in run) for m in run)


def _is_supported_window_v3(run, distances, policy):
    """Check if a coherent run has sufficient independent temporal support.

    Distinguishes:
    - Full support (>= min_support_buckets, typically 3): can_create_new_identity=True, can_match_existing=True.
    - Short segment support (>= min_match_buckets, typically 2): can_create_new_identity=False, can_match_existing=True.
    """
    if not _is_coherent_local_run(run, distances, policy):
        return False, "appearance_or_visibility_unresolved"
    rel_buckets = {math.floor((s.time_s - run[0].time_s) / policy.bucket_s) for s in run}
    min_match = getattr(policy, "min_match_buckets", 2)
    if len(rel_buckets) < min_match:
        return False, "insufficient_independent_buckets"
    if len(rel_buckets) >= policy.min_support_buckets:
        return True, "local_continuous_appearance_with_medoid_support"
    return True, "short_segment_limited_match_support"


def _supported(samples, distances, policy):
    if not samples:
        return False, "insufficient_independent_buckets"
    buckets = {math.floor(s.time_s / policy.bucket_s) for s in samples}
    rel_buckets = {math.floor((s.time_s - samples[0].time_s) / policy.bucket_s) for s in samples}
    if policy.version == "track-consistency-v3":
        min_buckets = getattr(policy, "min_support_buckets", 3)
        min_match = getattr(policy, "min_match_buckets", 2)
        if len(rel_buckets) < min_match:
            return False, "insufficient_independent_buckets"
        if any(b.time_s - a.time_s > policy.max_sample_gap_s for a, b in zip(samples, samples[1:])):
            return False, "sparse_feature_sampling"
        if any(distances.relation(a, b) == "far" for a, b in combinations(samples, 2)):
            return False, "appearance_conflict_including_minority"
        if any(distances.relation(a, b) != "near" for a, b in zip(samples, samples[1:])):
            return False, "appearance_or_visibility_unresolved"
        has_medoid = any(all(distances.relation(m, s) == "near" for s in samples) for m in samples)
        if not has_medoid:
            return False, "appearance_or_visibility_unresolved"
        if len(rel_buckets) >= min_buckets:
            return True, "local_continuous_appearance_with_medoid_support"
        return True, "short_segment_limited_match_support"
    else:
        # v1 and v2: all-pairs near requirement
        if len(buckets) < policy.min_support_buckets:
            return False, "insufficient_independent_buckets"
        if getattr(policy, "ensure_relative_buckets", False):
            if len(rel_buckets) < policy.min_support_buckets:
                return False, "insufficient_independent_buckets"
        if any(b.time_s - a.time_s > policy.max_sample_gap_s for a, b in zip(samples, samples[1:])):
            return False, "sparse_feature_sampling"
        relations = [distances.relation(a, b) for a, b in combinations(samples, 2)]
        if any(relation == "far" for relation in relations):
            return False, "appearance_conflict_including_minority"
        if any(relation != "near" for relation in relations):
            return False, "appearance_or_visibility_unresolved"
        return True, "all_pairs_consistent_with_independent_temporal_support"


def _local_support_windows_v3(samples, distances, policy):
    """Select disjoint chronological runs supported by local continuous appearance.

    Each window must be a coherent local run anchored by a medoid sample,
    with adjacent steps near and zero far pairs, spanning at least min_match_buckets.
    """
    windows, current = [], []
    for sample in samples:
        candidate = current + [sample]
        compatible = not current or _is_coherent_local_run(candidate, distances, policy)
        if not compatible:
            if current and _is_supported_window_v3(current, distances, policy)[0]:
                windows.append(current)
                current = []
            else:
                while current and not _is_coherent_local_run(current + [sample], distances, policy):
                    current = current[1:]
        current.append(sample)
    if current and _is_supported_window_v3(current, distances, policy)[0]:
        windows.append(current)
    return windows


def _local_support_windows(samples, distances, policy):
    """Select disjoint chronological runs without skipping an incompatible image.

    Grow each run while every pair remains near. An unsupported prefix can slide
    past incompatible samples, but those samples stay outside every returned run.
    Once a run has enough support, close it at the first incompatible observation;
    never concatenate its two sides or infer a physical identity boundary.
    """
    windows, current = [], []
    for sample in samples:
        compatible = not current or (
            sample.time_s - current[-1].time_s <= policy.max_sample_gap_s
            and all(distances.relation(previous, sample) == "near" for previous in current)
        )
        if not compatible:
            if _supported(current, distances, policy)[0]:
                windows.append(current)
                current = []
            else:
                # Keep only a contiguous suffix compatible with the next image.
                # A foreign minority image cannot be skipped inside an accepted run.
                while current and (
                    sample.time_s - current[-1].time_s > policy.max_sample_gap_s
                    or any(distances.relation(previous, sample) != "near" for previous in current)
                ):
                    current = current[1:]
        current.append(sample)
    if _supported(current, distances, policy)[0]:
        windows.append(current)
    return windows


def _anchor_samples_usable(samples, excluded_ids):
    return all(
        sample.sample_id not in excluded_ids
        and (
            sample.tracking_quality is None
            or (
                sample.tracking_quality.usability != "unusable"
                and not sample.tracking_quality.occluded
                and not sample.tracking_quality.mixed_identity
                and sample.tracking_quality.clean_review != "rejected"
            )
        )
        for sample in samples
    )


def _anchor_pairs_compatible(samples, distances, policy):
    for a, b in combinations(samples, 2):
        if distances.relation(a, b) in {"far", "uncomparable"}:
            return False
        for pair in distances.pair(a, b):
            thresholds = distances.thresholds.get(pair.feature_space_id, policy)
            if not pair.comparable or pair.distance >= thresholds.far_distance:
                return False
    return True


def segment_tracks(
    batch: EvidenceBatch,
    feature_values: FeatureValues | None = None,
    *,
    policy: SegmentationPolicy | None = None,
) -> SegmentationResult:
    """Return a fresh standard EvidenceBatch and typed, reproducible comparison report.

    Re-run on the *raw* batch, never on a prior proposal. The caller supplies file
    features in memory. Raw evidence is copied and retains inactive parent Segments;
    derived active Segments partition each parent completely into supported/unknown.
    """
    policy = policy or SegmentationPolicy()
    evidence = batch.identity_evidence
    if evidence is None:
        raise ValueError("track consistency requires identity evidence")
    if any(not s.active or s.supersedes_segment_id for s in evidence.segments):
        raise ValueError("track consistency expects raw active segments, not prior proposals")
    fingerprint = _hash(policy.model_dump(mode="json"))
    input_fingerprint = _hash(batch.model_dump(mode="json", exclude={"batch_id"}))
    spaces = {s.feature_space_id: s for s in evidence.feature_spaces}
    clips = {c.clip_id: c for c in batch.clips or []}
    by_track, samples_by_track = defaultdict(list), defaultdict(list)
    by_frame = defaultdict(list)
    for observation in evidence.observations:
        by_track[observation.clip_id, observation.track_id].append(observation)
        if observation.frame_idx is not None:
            by_frame[observation.clip_id, observation.frame_idx].append(observation)
    for sample in evidence.samples:
        samples_by_track[sample.clip_id, sample.track_id].append(sample)
    for observations in by_track.values():
        observations.sort(key=lambda o: (o.time_s, o.observation_id))
    duplicates = []
    for (clip_id, frame), observations in sorted(by_frame.items()):
        for a, b in combinations(observations, 2):
            overlap = _iou(a.bbox_xyxy, b.bbox_xyxy)
            if a.track_id != b.track_id and overlap >= policy.duplicate_iou:
                duplicates.append(
                    {
                        "clip_id": clip_id,
                        "frame_idx": frame,
                        "observation_ids": sorted([a.observation_id, b.observation_id]),
                        "iou": overlap,
                        "kind": "duplicate_box_candidate",
                        "cannot_link": False,
                    }
                )
    # Simultaneous jumps in many tracks are possible camera motion, not person switches.
    transition_tracks, transition_jumps = defaultdict(set), defaultdict(set)
    for key, observations in by_track.items():
        for a, b in zip(observations, observations[1:]):
            transition = (key[0], a.time_s, b.time_s)
            transition_tracks[transition].add(key[1])
            if _jump(a, b, policy)[0]:
                transition_jumps[transition].add(key[1])
    camera = {
        key
        for key, tracks in transition_tracks.items()
        if len(tracks) >= policy.camera_min_tracks
        and len(transition_jumps[key]) / len(tracks) >= policy.camera_jump_fraction
    }
    reports, segments, risks = [], [], [r.model_copy(deep=True) for r in evidence.risks]
    sample_segments = {}
    for parent in sorted(
        evidence.segments, key=lambda s: (s.clip_id, s.track_id, s.start_s, s.segment_id)
    ):
        track_key = (parent.clip_id, parent.track_id)
        observations = [o for o in by_track[track_key] if parent.start_s <= o.time_s < parent.end_s]
        samples = sorted(
            [s for s in samples_by_track[track_key] if parent.start_s <= s.time_s < parent.end_s],
            key=lambda s: (s.time_s, s.sample_id),
        )
        clip = clips.get(parent.clip_id)
        report = TrackConsistencyReport(parent_segment=parent.model_copy(deep=True))
        distances = _Distances(samples, spaces, feature_values, policy)
        scope_fingerprint = _hash(
            {
                "parent": parent.model_dump(mode="json"),
                "observations": [o.model_dump(mode="json") for o in observations],
                "samples": [s.model_dump(mode="json") for s in samples],
                "features": {
                    sid: {fsid: [f.tolist(), v.tolist()] for fsid, (f, v) in features.items()}
                    for sid, features in distances.features.items()
                },
            }
        )
        # Every sample is examined, even when it is not a temporal representative.
        eligible, candidates = [], []
        blocked_sample_ids = set()
        for sample in samples:
            quality = sample.tracking_quality
            reason = None
            actual_observation = any(
                abs(o.time_s - sample.time_s) <= 1e-6
                and (sample.frame_idx is None or o.frame_idx == sample.frame_idx)
                for o in observations
            )
            if not actual_observation:
                reason = "missing_actual_observation"
            elif quality is not None and quality.usability == "unusable":
                reason = "sample_unusable"
            elif quality is not None and quality.occluded:
                reason = "occluded_sample"
            elif not distances.features[sample.sample_id]:
                reason = "missing_feature"
            elif not any(
                np.count_nonzero((v >= policy.min_part_visibility) & (v > 0))
                >= policy.min_common_parts
                for _, v in distances.features[sample.sample_id].values()
            ):
                reason = "insufficient_reliable_visible_parts"
            if reason:
                blocked_sample_ids.add(sample.sample_id)
                report.exclusions.append(SampleExclusion(sample_id=sample.sample_id, reason=reason))
                report.findings.append(
                    ConsistencyFinding(
                        kind="occlusion_or_low_visibility"
                        if reason in {"occluded_sample", "insufficient_reliable_visible_parts"}
                        else reason,
                        start_s=sample.time_s,
                        end_s=_end_of(sample, clip, parent.end_s),
                        sample_ids=[sample.sample_id],
                    )
                )
                continue
            eligible.append(sample)
        # Build transitive duplicate components before selecting representatives.
        parents = list(range(len(eligible)))

        def find(index):
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        known = {}
        for index, sample in enumerate(eligible):
            for key in evidence_keys(sample, batch.assets):
                if key in known:
                    parents[find(index)] = find(known[key])
                known[key] = index
        groups = defaultdict(list)
        for index, sample in enumerate(eligible):
            groups[find(index)].append(sample)
        for group in groups.values():
            ordered = sorted(group, key=lambda s: (s.time_s, s.sample_id))
            representative = ordered[0]
            conflict = any(distances.relation(a, b) != "near" for a, b in combinations(ordered, 2))
            if conflict:
                blocked_sample_ids.update(s.sample_id for s in ordered)
                report.findings.append(
                    ConsistencyFinding(
                        kind="duplicate_descriptor_conflict",
                        start_s=ordered[0].time_s,
                        end_s=max(_end_of(s, clip, parent.end_s) for s in ordered),
                        sample_ids=sorted(s.sample_id for s in ordered),
                    )
                )
            else:
                candidates.append(representative)
            for sample in ordered[1:]:
                report.exclusions.append(
                    SampleExclusion(
                        sample_id=sample.sample_id,
                        reason="duplicate_evidence",
                        duplicate_of=representative.sample_id,
                    )
                )
        candidates.sort(key=lambda s: (s.time_s, s.sample_id))
        # Diagnostics cover every usable pair, including sparse/short unknown spans.
        for a, b in combinations(candidates, 2):
            distances.pair(a, b)
        # Breaks are observational/spatial facts, never feature jumps alone.
        breaks = []
        for a, b in zip(observations, observations[1:]):
            gap = b.time_s - a.time_s > policy.max_observation_gap_s
            jump, displacement = _jump(a, b, policy)
            if not (gap or jump):
                continue
            left = _end_of(a, clip, parent.end_s)
            right = b.time_s
            # Adjacent CFR frames do not locate a switch at an exact instant.
            if right <= left + 1e-9:
                left = a.time_s
            kind = (
                "observation_interruption"
                if gap
                else "possible_camera_motion"
                if (parent.clip_id, a.time_s, b.time_s) in camera
                else "motion_discontinuity"
            )
            breaks.append((left, right, kind))
            report.findings.append(
                ConsistencyFinding(
                    kind=kind,
                    start_s=left,
                    end_s=right,
                    observation_ids=[a.observation_id, b.observation_id],
                    details={
                        "elapsed_s": b.time_s - a.time_s,
                        "center_displacement_heights": displacement,
                        "identity_switch_proven": False,
                    },
                )
            )
        # Excluded observations and long feature gaps get explicit unknown intervals.
        holes = []
        for a, b, _ in breaks:
            left_support = [s for s in candidates if s.time_s <= a]
            right_support = [s for s in candidates if s.time_s >= b]
            # Geometry localizes a tracking discontinuity, not an identity boundary.
            # Reserve the entire interval without appearance support around it.
            begin = _end_of(left_support[-1], clip, parent.end_s) if left_support else a
            finish = right_support[0].time_s if right_support else b
            holes.append((min(a, begin), max(b, finish)))
        for sample in samples:
            if sample.sample_id in blocked_sample_ids:
                holes.append((sample.time_s, _end_of(sample, clip, parent.end_s)))
        excluded_ids = blocked_sample_ids | {e.sample_id for e in report.exclusions}
        for a, b in zip(candidates, candidates[1:]):
            if b.time_s - a.time_s > policy.max_sample_gap_s:
                left = _end_of(a, clip, parent.end_s)
                holes.append((left, b.time_s))
                overlapping_breaks = [
                    kind for start, end, kind in breaks if start < b.time_s and end > a.time_s
                ]
                observation_continuous = "observation_interruption" not in overlapping_breaks
                report.findings.append(
                    ConsistencyFinding(
                        kind="sparse_feature_sampling",
                        start_s=left,
                        end_s=b.time_s,
                        sample_ids=[a.sample_id, b.sample_id],
                        details={
                            "elapsed_s": b.time_s - a.time_s,
                            "observation_continuous": observation_continuous,
                            "identity_coverage_preserved": False,
                            "anchor_candidate_only": policy.version == "track-consistency-v3"
                            and not overlapping_breaks
                            and _anchor_samples_usable(
                                [s for s in samples if a.time_s <= s.time_s <= b.time_s],
                                excluded_ids,
                            )
                            and _anchor_pairs_compatible([a, b], distances, policy),
                        },
                    )
                )
        # Merge overlapping holes; all resulting ranges retain explicit uncertainty.
        merged = []
        for a, b in sorted(holes):
            if b <= a:
                continue
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
            else:
                merged.append((a, b))
        edges = sorted({parent.start_s, parent.end_s, *(x for hole in merged for x in hole)})
        proposed = []
        for start, end in zip(edges, edges[1:]):
            inside = [s for s in candidates if start <= s.time_s < end]
            is_hole = any(a < end and b > start for a, b in merged)
            if is_hole:
                proposed.append((start, end, "unknown", "unobserved_or_unreliable_interval", []))
                continue
            ok, reason = _supported(inside, distances, policy)
            # Geometrical context is mandatory for an automatic supported span.
            if not observations:
                ok, reason = False, "missing_structured_observations"
            if not ok:
                if inside and reason.startswith("appearance"):
                    report.findings.append(
                        ConsistencyFinding(
                            kind=reason,
                            start_s=start,
                            end_s=end,
                            sample_ids=[s.sample_id for s in inside],
                            details={"switch_inferred": False},
                        )
                    )
                trigger_reasons = (
                    {
                        "appearance_conflict_including_minority",
                        "appearance_or_visibility_unresolved",
                        "insufficient_independent_buckets",
                    }
                    if policy.version == "track-consistency-v3"
                    else {
                        "appearance_conflict_including_minority",
                        "appearance_or_visibility_unresolved",
                    }
                )
                windows = (
                    (
                        _local_support_windows_v3(inside, distances, policy)
                        if policy.version == "track-consistency-v3"
                        else _local_support_windows(inside, distances, policy)
                    )
                    if policy.local_support_windows and observations and reason in trigger_reasons
                    else []
                )
                cursor = start
                for window in windows:
                    first, last = window[0].time_s, _end_of(window[-1], clip, end)
                    if cursor < first:
                        gap_samples = [s for s in inside if cursor <= s.time_s < first]
                        is_transition = False
                        if getattr(policy, "enable_transition_zones", False) and gap_samples:
                            has_far = any(
                                distances.relation(a, b) == "far"
                                for a in gap_samples
                                for b in window
                            )
                            has_motion_break = any(
                                bk[0] < first and bk[1] > cursor for bk in breaks
                            )
                            if not has_far and not has_motion_break:
                                is_transition = True
                        if is_transition:
                            proposed.append(
                                (
                                    cursor,
                                    first,
                                    "unknown",
                                    "viewpoint_transition_uncertainty",
                                    gap_samples,
                                )
                            )
                            report.findings.append(
                                ConsistencyFinding(
                                    kind="viewpoint_transition_zone",
                                    start_s=cursor,
                                    end_s=first,
                                    sample_ids=[s.sample_id for s in gap_samples],
                                    details={
                                        "parent_interval_reason": reason,
                                        "transition_type": "continuous_perspective_shift",
                                        "switch_inferred": False,
                                        "identity_boundary_known": False,
                                    },
                                )
                            )
                        else:
                            proposed.append((cursor, first, "unknown", reason, []))
                    w_reason = (
                        _is_supported_window_v3(window, distances, policy)[1]
                        if policy.version == "track-consistency-v3"
                        else "local_all_pairs_consistent_with_independent_temporal_support"
                    )
                    proposed.append(
                        (
                            first,
                            last,
                            "supported",
                            w_reason,
                            window,
                        )
                    )
                    report.findings.append(
                        ConsistencyFinding(
                            kind="local_appearance_support_window",
                            start_s=first,
                            end_s=last,
                            sample_ids=[s.sample_id for s in window],
                            details={
                                "parent_interval_reason": reason,
                                "identity_verified": False,
                                "switch_inferred": False,
                                "identity_boundary_known": False,
                                "selection": "contiguous_local_continuous_appearance_with_medoid_support"
                                if policy.version == "track-consistency-v3"
                                else "contiguous_all_pairs_near_no_skipped_samples",
                            },
                        )
                    )
                    cursor = last
                if cursor < end:
                    gap_samples = [s for s in inside if cursor <= s.time_s < end]
                    is_transition = False
                    if (
                        getattr(policy, "enable_transition_zones", False)
                        and gap_samples
                        and windows
                    ):
                        has_far = any(
                            distances.relation(a, b) == "far"
                            for a in gap_samples
                            for b in windows[-1]
                        )
                        has_motion_break = any(bk[0] < end and bk[1] > cursor for bk in breaks)
                        if not has_far and not has_motion_break:
                            is_transition = True
                    if is_transition:
                        proposed.append(
                            (
                                cursor,
                                end,
                                "unknown",
                                "viewpoint_transition_uncertainty",
                                gap_samples,
                            )
                        )
                        report.findings.append(
                            ConsistencyFinding(
                                kind="viewpoint_transition_zone",
                                start_s=cursor,
                                end_s=end,
                                sample_ids=[s.sample_id for s in gap_samples],
                                details={
                                    "parent_interval_reason": reason,
                                    "transition_type": "continuous_perspective_shift",
                                    "switch_inferred": False,
                                    "identity_boundary_known": False,
                                },
                            )
                        )
                    else:
                        proposed.append((cursor, end, "unknown", reason, []))
                continue
            first, last = inside[0].time_s, _end_of(inside[-1], clip, end)
            if start < first:
                proposed.append((start, first, "unknown", "unsupported_leading_extent", []))
            proposed.append((first, last, "supported", reason, inside))
            if last < end:
                proposed.append((last, end, "unknown", "unsupported_trailing_extent", []))
        # Annotate a suspected switch only with independently supported flanks.
        for a, b, kind in breaks:
            left = next(
                (p for p in reversed(proposed) if p[1] <= a + 1e-9 and p[2] == "supported"), None
            )
            right = next((p for p in proposed if p[0] >= b - 1e-9 and p[2] == "supported"), None)
            if (
                left is None
                or right is None
                or left[1] < a - policy.max_sample_gap_s
                or right[0] > b + policy.max_sample_gap_s
            ):
                continue
            relations = [distances.relation(x, y) for x in left[4] for y in right[4]]
            if relations and all(r == "far" for r in relations):
                dup_cooccurring = any(
                    dup["clip_id"] == parent.clip_id
                    and any(
                        a - 0.5 <= o.time_s <= b + 0.5
                        for o in observations
                        if o.observation_id in dup["observation_ids"]
                    )
                    for dup in duplicates
                )
                corroboration = [kind]
                if dup_cooccurring:
                    corroboration.append("duplicate_box_candidate")
                report.findings.append(
                    ConsistencyFinding(
                        kind="suspected_identity_switch"
                        if kind == "motion_discontinuity"
                        else "appearance_change_after_interruption",
                        start_s=left[4][-1].time_s,
                        end_s=right[4][0].time_s,
                        sample_ids=[s.sample_id for s in left[4] + right[4]],
                        details={
                            "appearance_separation": "all_cross_pairs_far",
                            "corroboration": corroboration[0]
                            if len(corroboration) == 1
                            else corroboration,
                            "exact_boundary_known": False,
                            "distinct_identity_confirmed": False,
                            "geometric_break_interval": [a, b],
                        },
                    )
                )
        # Adjacent identical unknown pieces are coalesced without deleting their findings.
        coalesced = []
        for proposal in proposed:
            if coalesced and proposal[2] == "unknown" and coalesced[-1][2:4] == proposal[2:4]:
                previous = coalesced[-1]
                coalesced[-1] = (previous[0], proposal[1], *proposal[2:])
            else:
                coalesced.append(proposal)
        anchor_ranges = set()
        carved = []
        for start, end, status, reason, support in coalesced:
            window = [s for s in candidates if start <= s.time_s < end]
            piece_samples = [s for s in samples if start <= s.time_s < end]
            if (
                policy.version == "track-consistency-v3"
                and status == "unknown"
                and reason
                in {"appearance_or_visibility_unresolved", "viewpoint_transition_uncertainty"}
                and window
                and observations
                and not any(a < end and b > start for a, b, _ in breaks)
                and _anchor_samples_usable(piece_samples, excluded_ids)
                and len(
                    {math.floor((s.time_s - window[0].time_s) / policy.bucket_s) for s in window}
                )
                >= policy.min_match_buckets
                and all(
                    b.time_s - a.time_s <= policy.max_sample_gap_s
                    for a, b in zip(window, window[1:])
                )
                and _anchor_pairs_compatible(window, distances, policy)
            ):
                first, last = window[0].time_s, _end_of(window[-1], clip, end)
                if start < first:
                    carved.append((start, first, "unknown", "unsupported_leading_extent", []))
                carved.append((first, last, status, reason, window))
                anchor_ranges.add((first, last))
                if last < end:
                    carved.append((last, end, "unknown", "unsupported_trailing_extent", []))
                report.findings.append(
                    ConsistencyFinding(
                        kind="external_anchor_candidate",
                        start_s=first,
                        end_s=last,
                        sample_ids=[s.sample_id for s in window],
                        observation_ids=[
                            o.observation_id for o in observations if first <= o.time_s < last
                        ],
                        details={
                            "parent_interval_reason": reason,
                            "requires_external_anchor": True,
                            "can_match_existing": True,
                            "can_create_new_identity": False,
                            "identity_verified": False,
                            "switch_inferred": False,
                            "selection": "independent_usable_samples_without_far_or_uncomparable_pairs",
                        },
                    )
                )
            else:
                carved.append((start, end, status, reason, support))
        segments.append(parent.model_copy(update={"active": False}, deep=True))
        for start, end, status, reason, support in carved:
            requires_anchor = (start, end) in anchor_ranges
            sid = _id("segc", parent.segment_id, fingerprint, scope_fingerprint, start, end)
            frame_start = frame_end = None
            if clip is not None and clip.frame_timing == "cfr" and clip.fps:
                frame_start = round(start * clip.fps)
                frame_end = round(end * clip.fps)
                if frame_end <= frame_start:
                    frame_start = frame_end = None
            segment = Segment(
                segment_id=sid,
                match_id=parent.match_id,
                clip_id=parent.clip_id,
                track_id=parent.track_id,
                start_s=start,
                end_s=end,
                frame_start=frame_start,
                frame_end_exclusive=frame_end,
                supersedes_segment_id=parent.segment_id,
            )
            observed_ids = [o.observation_id for o in observations if start <= o.time_s < end]
            rel_buckets = (
                {math.floor((s.time_s - support[0].time_s) / policy.bucket_s) for s in support}
                if support
                else set()
            )
            is_full = len(rel_buckets) >= policy.min_support_buckets
            can_match = (status == "supported" or requires_anchor) and len(support) > 0
            can_create = (
                status == "supported"
                and is_full
                and reason != "short_segment_limited_match_support"
            )
            proposal = SegmentProposal(
                segment=segment,
                parent_segment_id=parent.segment_id,
                source=_source(
                    parent,
                    fingerprint,
                    {
                        "policy_fingerprint": fingerprint,
                        "input_scope_fingerprint": scope_fingerprint,
                        "parent_segment_id": parent.segment_id,
                        "status": status,
                        "reason": reason,
                        "start_s": start,
                        "end_s": end,
                        "support_sample_ids": [s.sample_id for s in support],
                        **({"requires_external_anchor": True} if requires_anchor else {}),
                    },
                    observations,
                    version=policy.version,
                ),
                status=status,
                reasons=[reason],
                support_sample_ids=[s.sample_id for s in support],
                source_observation_ids=observed_ids,
                can_match_existing=can_match,
                can_create_new_identity=can_create,
                requires_external_anchor=requires_anchor,
            )
            report.proposals.append(proposal)
            segments.append(segment)
            # Uncertainty is represented in the existing structured risk contract.
            if status == "unknown":
                details = {
                    "policy_fingerprint": fingerprint,
                    "parent_segment_id": parent.segment_id,
                    "reason": reason,
                    "start_s": start,
                    "end_s": end,
                    **({"requires_external_anchor": True} if requires_anchor else {}),
                }
                risk = IdentityRisk(
                    risk_id=_id("riskc", sid, reason),
                    basis="external_anchor_required" if requires_anchor else "unspecified",
                    kind="quality_review",
                    match_id=parent.match_id,
                    segment_id=sid,
                    clip_id=parent.clip_id,
                    track_id=parent.track_id,
                    start_s=start,
                    end_s=end,
                    source=_source(
                        parent, fingerprint, details, observations, version=policy.version
                    ),
                )
                risks.append(risk)
                report.risks.append(risk)
            for sample in samples:
                if start <= sample.time_s < end:
                    sample_segments[sample.sample_id] = sid
        # Duplicate detections do not create cannot-link constraints, but must not
        # become independent clean input to automatic library building either.
        observation_index = {o.observation_id: o for o in observations}
        for duplicate in duplicates:
            affected = [
                observation_index[oid]
                for oid in duplicate["observation_ids"]
                if oid in observation_index
            ]
            for observation in affected:
                report.findings.append(
                    ConsistencyFinding(
                        kind="duplicate_box_candidate",
                        start_s=observation.time_s,
                        end_s=_end_of(observation, clip, parent.end_s),
                        observation_ids=duplicate["observation_ids"],
                        details={
                            "cannot_link": False,
                            "independent_support": False,
                            "iou": duplicate["iou"],
                        },
                    )
                )
        # Convert safety findings to existing scoped risks, in addition to the report.
        finding_risks = {
            "suspected_identity_switch": "identity_switch",
            "duplicate_descriptor_conflict": "identity_conflict",
            "duplicate_box_candidate": "duplicate_box",
        }
        for finding in report.findings:
            if finding.kind not in finding_risks:
                continue
            for proposal in report.proposals:
                segment = proposal.segment
                start, end = (
                    max(segment.start_s, finding.start_s),
                    min(segment.end_s, finding.end_s),
                )
                if end <= start:
                    continue
                details = {**finding.model_dump(mode="json"), "segment_id": segment.segment_id}
                risk = IdentityRisk(
                    risk_id=_id("riskc", segment.segment_id, details),
                    kind=finding_risks[finding.kind],
                    match_id=parent.match_id,
                    clip_id=parent.clip_id,
                    track_id=parent.track_id,
                    segment_id=segment.segment_id,
                    start_s=start,
                    end_s=end,
                    source=_source(
                        parent, fingerprint, details, observations, version=policy.version
                    ),
                )
                report.risks.append(risk)
                risks.append(risk)
        # Preserve input risks unchanged on the inactive parent; carry their scope to children.
        # Input tags cannot create or locate a split, but still block unsafe downstream use.
        inherited_risks = {risk.risk_id: risk for risk in evidence.risks}
        for sample in samples:
            if sample.tracking_quality is not None:
                inherited_risks.update(
                    {risk.risk_id: risk for risk in sample.tracking_quality.risks}
                )
        for risk in inherited_risks.values():
            if risk.segment_id != parent.segment_id:
                continue
            for proposal in report.proposals:
                s = proposal.segment
                start = max(s.start_s, risk.start_s if risk.start_s is not None else s.start_s)
                end = min(s.end_s, risk.end_s if risk.end_s is not None else s.end_s)
                if end <= start:
                    continue
                if (
                    getattr(policy, "scope_risks_to_breaks", False)
                    and proposal.status == "supported"
                    and risk.kind == "quality_review"
                    and risk.sample_id is None
                ):
                    # Clean supported segment with continuous observations does not inherit parent's abrupt motion
                    continue
                if risk.sample_id is not None and not any(
                    sample.sample_id == risk.sample_id and start <= sample.time_s < end
                    for sample in samples
                ):
                    continue
                # Machine consistency does not clear any upstream concern or human review.
                copy = risk.model_copy(
                    update={
                        "risk_id": _id("riskc", risk.risk_id, s.segment_id),
                        "segment_id": s.segment_id,
                        "start_s": start,
                        "end_s": end,
                        "source": _source(
                            parent,
                            fingerprint,
                            {
                                "inherited_risk": risk.model_dump(mode="json"),
                                "segment_id": s.segment_id,
                                "start_s": start,
                                "end_s": end,
                            },
                            observations,
                            version=policy.version,
                        ),
                    },
                    deep=True,
                )
                risks.append(copy)
                report.risks.append(copy)
        report.comparisons = [
            p for key in sorted(distances.comparisons) for p in distances.comparisons[key]
        ]
        reports.append(report)
    # Raw sample provenance and features are immutable; segment membership is derived.
    samples = [
        s.model_copy(
            update={"segment_id": sample_segments.get(s.sample_id, s.segment_id)}, deep=True
        )
        for s in evidence.samples
    ]
    derived = IdentityEvidence.model_validate(
        {**evidence.model_dump(), "segments": segments, "samples": samples, "risks": risks}
    )
    result_batch = EvidenceBatch.model_validate(
        {**batch.model_dump(), "identity_evidence": derived}
    )
    all_proposals = [p for r in reports for p in r.proposals]
    total = sum(p.segment.end_s - p.segment.start_s for p in all_proposals)
    unknown = sum(
        p.segment.end_s - p.segment.start_s for p in all_proposals if p.status == "unknown"
    )
    unknown_samples = sum(
        any(p.status == "unknown" and p.segment.segment_id == s.segment_id for p in all_proposals)
        for s in samples
    )
    summary = {
        "raw_tracks": len(evidence.tracks),
        "raw_segments": len(evidence.segments),
        "active_segments": len(all_proposals),
        "supported_segments": sum(p.status == "supported" for p in all_proposals),
        "unknown_segments": sum(p.status == "unknown" for p in all_proposals),
        "can_match_existing_segments": sum(
            getattr(p, "can_match_existing", False) for p in all_proposals
        ),
        "can_create_new_identity_segments": sum(
            getattr(p, "can_create_new_identity", False) for p in all_proposals
        ),
        "requires_external_anchor_segments": sum(p.requires_external_anchor for p in all_proposals),
        "track_extent_seconds": total,
        "unknown_extent_seconds": unknown,
        "insufficient_evidence_duration_fraction": unknown / total if total else None,
        "unknown_samples": unknown_samples,
        "samples": len(samples),
        "insufficient_evidence_sample_fraction": unknown_samples / len(samples)
        if samples
        else None,
        "tracks_without_supported_segments": sum(
            not any(p.status == "supported" for p in r.proposals) for r in reports
        ),
        "finding_counts": dict(Counter(f.kind for r in reports for f in r.findings)),
        "exclusion_counts": dict(Counter(e.reason for r in reports for e in r.exclusions)),
        "duplicate_box_candidates": len(duplicates),
        "identity_verified": False,
        "threshold_calibration": "engineering_gates_not_identity_calibrated",
        "duration_denominator": "sum_of_raw_track_extents_overlapping_tracks_count_separately",
    }
    return SegmentationResult(
        batch=result_batch,
        report=SegmentationReport(
            policy=policy,
            policy_fingerprint=fingerprint,
            input_fingerprint=input_fingerprint,
            derived_evidence_fingerprint=evidence_fingerprint(result_batch, feature_values),
            tracks=reports,
            duplicate_candidates=duplicates,
            summary=summary,
        ),
    )
