"""Track-level representation and multi-prototype appearance modeling.

Extracts high signal-to-noise ratio trajectory representations by filtering out
noisy (low-visibility) observations, computing multi-prototype appearance models
grounded in raw evidence (medoids, time buckets, independent witnesses), and
computing multi-part weighted Euclidean distances with viewpoint compatibility
and one-vote veto on strong counter-evidence.
"""

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.contracts.evidence import FeatureSpace, InlineFeature, Sample
from app.identity.prepare import FeatureValues, resolve_feature


@dataclass(frozen=True)
class AppearancePrototype:
    """A concrete, verifiable appearance prototype anchored in raw sample evidence."""

    prototype_id: str
    sample_id: str
    segment_id: str
    clip_id: str
    track_id: int | str
    time_s: float
    time_bucket: int
    feature_space_id: str
    feature: np.ndarray  # shape: (P, D), float32, L2 normalized per part
    visibility: np.ndarray  # shape: (P,), float32
    reliable_parts: tuple[int, ...]
    quality_score: float | None = None
    risks: tuple[str, ...] = ()
    witness_sample_ids: tuple[str, ...] = ()
    source_component: str | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AppearancePrototype):
            return NotImplemented
        return (
            self.prototype_id == other.prototype_id
            and self.sample_id == other.sample_id
            and self.segment_id == other.segment_id
            and self.clip_id == other.clip_id
            and str(self.track_id) == str(other.track_id)
            and np.isclose(self.time_s, other.time_s)
            and self.time_bucket == other.time_bucket
            and self.feature_space_id == other.feature_space_id
            and np.array_equal(self.feature, other.feature)
            and np.array_equal(self.visibility, other.visibility)
            and self.reliable_parts == other.reliable_parts
            and self.risks == other.risks
            and self.witness_sample_ids == other.witness_sample_ids
            and self.source_component == other.source_component
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.prototype_id,
                self.sample_id,
                self.segment_id,
                self.clip_id,
                str(self.track_id),
                self.time_s,
                self.time_bucket,
                self.feature_space_id,
                self.reliable_parts,
                self.risks,
                self.witness_sample_ids,
                self.source_component,
            )
        )


@dataclass(frozen=True)
class TrackRepresentation:
    clip_id: str
    track_id: int | str
    start_s: float
    end_s: float
    feature: np.ndarray  # shape: (P, D) - representative or medoid feature
    visibility: np.ndarray  # shape: (P,) - representative visibility
    sample_ids: list[str]
    segment_id: str | None = None
    feature_space_id: str | None = None
    prototypes: tuple[AppearancePrototype, ...] = field(default_factory=tuple)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrackRepresentation):
            return NotImplemented
        return (
            self.clip_id == other.clip_id
            and str(self.track_id) == str(other.track_id)
            and np.isclose(self.start_s, other.start_s)
            and np.isclose(self.end_s, other.end_s)
            and np.array_equal(self.feature, other.feature)
            and np.array_equal(self.visibility, other.visibility)
            and self.sample_ids == other.sample_ids
            and self.segment_id == other.segment_id
            and self.feature_space_id == other.feature_space_id
            and self.prototypes == other.prototypes
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.clip_id,
                str(self.track_id),
                self.start_s,
                self.end_s,
                tuple(self.sample_ids),
                self.segment_id,
                self.feature_space_id,
                self.prototypes,
            )
        )


SegmentRepresentation = TrackRepresentation


def _extract_sample_features(
    sample: Any,
    space: FeatureSpace | None,
    values: FeatureValues | Mapping[tuple[str, int], InlineFeature] | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract (feature_matrix, visibility_vector) from a Sample or duck-typed object."""
    # 1. Standard Sample with features and FeatureSpace
    if space is not None and hasattr(sample, "features") and isinstance(sample, Sample):
        inline = resolve_feature(sample, space, values)
        if inline is not None and inline.values and inline.visibility:
            p_dim, d_dim = space.shape if hasattr(space, "shape") else (6, 128)
            f_mat = np.asarray(inline.values, dtype=np.float32).reshape((p_dim, d_dim))
            v_vec = np.asarray(inline.visibility, dtype=np.float32)
            return f_mat, v_vec

    # 2. Object or dict with explicit 'feature' and 'visibility'
    if hasattr(sample, "feature") and hasattr(sample, "visibility"):
        f = getattr(sample, "feature")
        v = getattr(sample, "visibility")
        if f is not None and v is not None:
            return np.asarray(f, dtype=np.float32), np.asarray(v, dtype=np.float32)

    if isinstance(sample, dict) and "feature" in sample and "visibility" in sample:
        f = sample["feature"]
        v = sample["visibility"]
        if f is not None and v is not None:
            return np.asarray(f, dtype=np.float32), np.asarray(v, dtype=np.float32)

    # 3. Object with 'values' and 'visibility'
    if hasattr(sample, "values") and hasattr(sample, "visibility"):
        vals = getattr(sample, "values")
        v = getattr(sample, "visibility")
        if vals is not None and v is not None:
            v_arr = np.asarray(v, dtype=np.float32)
            p_dim = len(v_arr)
            f_mat = np.asarray(vals, dtype=np.float32).reshape((p_dim, -1))
            return f_mat, v_arr

    # 4. Fallback: resolve_feature with duck-typed sample
    if space is not None and hasattr(sample, "features"):
        try:
            inline = resolve_feature(sample, space, values)
            if inline is not None and inline.values and inline.visibility:
                p_dim, d_dim = space.shape if hasattr(space, "shape") else (6, 128)
                f_mat = np.asarray(inline.values, dtype=np.float32).reshape((p_dim, d_dim))
                v_vec = np.asarray(inline.visibility, dtype=np.float32)
                return f_mat, v_vec
        except Exception:
            pass

    return None, None


def compute_prototype_distance(
    p1: AppearancePrototype,
    p2: AppearancePrototype,
    min_common_parts: int = 2,
    min_vis: float = 0.2,
) -> float | None:
    """Compute multi-part weighted Euclidean distance between two appearance prototypes."""
    if p1.feature_space_id and p2.feature_space_id and p1.feature_space_id != p2.feature_space_id:
        return None

    v1 = np.asarray(p1.visibility, dtype=np.float64)
    v2 = np.asarray(p2.visibility, dtype=np.float64)

    min_p = min(len(v1), len(v2))
    v1_sub = v1[:min_p]
    v2_sub = v2[:min_p]

    overlap = (v1_sub >= min_vis) & (v2_sub >= min_vis) & np.isfinite(v1_sub) & np.isfinite(v2_sub)
    common_parts = int(np.count_nonzero(overlap))
    if common_parts < min_common_parts:
        return None

    weights = np.sqrt(v1_sub[overlap] * v2_sub[overlap])
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-12:
        return None

    f1 = np.asarray(p1.feature[:min_p], dtype=np.float64)[overlap]
    f2 = np.asarray(p2.feature[:min_p], dtype=np.float64)[overlap]

    part_diffs = np.linalg.norm(f1 - f2, axis=-1)
    distance = float(np.sum(weights * part_diffs) / (2.0 * weight_sum))
    return distance


def _pairwise_sample_distance(
    f1: np.ndarray,
    v1: np.ndarray,
    f2: np.ndarray,
    v2: np.ndarray,
    min_common_parts: int = 2,
    min_vis: float = 0.2,
) -> float | None:
    min_p = min(len(v1), len(v2))
    v1_sub = v1[:min_p]
    v2_sub = v2[:min_p]
    overlap = (v1_sub >= min_vis) & (v2_sub >= min_vis) & np.isfinite(v1_sub) & np.isfinite(v2_sub)
    if int(np.count_nonzero(overlap)) < min_common_parts:
        return None
    weights = np.sqrt(v1_sub[overlap] * v2_sub[overlap])
    w_sum = float(np.sum(weights))
    if w_sum <= 1e-12:
        return None
    f1_sub = f1[:min_p][overlap]
    f2_sub = f2[:min_p][overlap]
    diffs = np.linalg.norm(f1_sub - f2_sub, axis=-1)
    return float(np.sum(weights * diffs) / (2.0 * w_sum))


def extract_segment_prototypes(
    segment: Any,
    samples: Iterable[Any],
    space: FeatureSpace | None = None,
    values: FeatureValues | Mapping[tuple[str, int], InlineFeature] | None = None,
    min_vis: float = 0.2,
    bucket_s: float = 0.5,
    near_distance: float = 0.18,
    far_distance: float = 0.45,
    max_prototypes: int = 4,
    components: Mapping[str, str] | None = None,
) -> list[AppearancePrototype]:
    """Extract distinct, verifiable appearance prototypes from a segment's usable samples."""
    sid = str(
        getattr(
            segment, "segment_id", segment.get("segment_id") if isinstance(segment, dict) else ""
        )
    )
    clip_id = str(
        getattr(segment, "clip_id", segment.get("clip_id") if isinstance(segment, dict) else "")
    )
    raw_tid = getattr(
        segment, "track_id", segment.get("track_id") if isinstance(segment, dict) else 0
    )
    try:
        track_id: int | str = int(raw_tid)
    except (ValueError, TypeError):
        track_id = str(raw_tid)

    fsid = (
        space.feature_space_id if (space is not None and hasattr(space, "feature_space_id")) else ""
    )

    # 1. Filter usable samples strictly belonging to this segment
    usable_items: list[tuple[Any, np.ndarray, np.ndarray, str, float]] = []
    for s in samples:
        s_sid = str(getattr(s, "segment_id", s.get("segment_id") if isinstance(s, dict) else ""))
        if s_sid != sid:
            continue
        q = getattr(
            s, "tracking_quality", s.get("tracking_quality") if isinstance(s, dict) else None
        )
        if q and (
            getattr(q, "occluded", False)
            or getattr(q, "mixed_identity", False)
            or getattr(q, "usability", "") == "unusable"
            or getattr(q, "clean_review", "") == "rejected"
        ):
            continue
        f_mat, v_vec = _extract_sample_features(s, space, values)
        if f_mat is None or v_vec is None:
            continue
        s_id = str(getattr(s, "sample_id", s.get("sample_id") if isinstance(s, dict) else ""))
        time_s = float(getattr(s, "time_s", s.get("time_s") if isinstance(s, dict) else 0.0))
        usable_items.append((s, f_mat, v_vec, s_id, time_s))

    if not usable_items:
        return []

    # Sort chronologically
    usable_items.sort(key=lambda item: (item[4], item[3]))

    # 2. Partition into non-duplicate temporal buckets
    bucketed: dict[int, list[tuple[Any, np.ndarray, np.ndarray, str, float]]] = defaultdict(list)
    for item in usable_items:
        b_idx = int(math.floor(item[4] / bucket_s))
        bucketed[b_idx].append(item)

    # 3. Compute local medoid for each temporal bucket
    bucket_medoids: list[dict[str, Any]] = []
    for b_idx in sorted(bucketed.keys()):
        b_items = bucketed[b_idx]
        if len(b_items) == 1:
            med_item = b_items[0]
            witness_ids = [med_item[3]]
        else:
            # Find medoid minimizing pairwise distances
            n_b = len(b_items)
            dist_sums = []
            for i in range(n_b):
                d_sum = 0.0
                for j in range(n_b):
                    if i != j:
                        d = _pairwise_sample_distance(
                            b_items[i][1],
                            b_items[i][2],
                            b_items[j][1],
                            b_items[j][2],
                            min_vis=min_vis,
                        )
                        d_sum += d if d is not None else 0.5
                # Secondary preference: higher visibility
                mean_v = float(np.mean(b_items[i][2]))
                dist_sums.append((d_sum, -mean_v, b_items[i][4], i))
            dist_sums.sort()
            best_idx = dist_sums[0][3]
            med_item = b_items[best_idx]

            # Find local witnesses in the bucket
            witness_ids = []
            for item in b_items:
                d = _pairwise_sample_distance(
                    med_item[1], med_item[2], item[1], item[2], min_vis=min_vis
                )
                if d is not None and d <= near_distance:
                    witness_ids.append(item[3])

        comp_key = components.get(med_item[3]) if components else None
        q = getattr(med_item[0], "tracking_quality", None)
        q_score = getattr(q, "score", None) if q else None
        bucket_medoids.append(
            {
                "item": med_item,
                "time_bucket": b_idx,
                "witness_ids": witness_ids,
                "source_component": comp_key,
                "quality_score": float(q_score) if q_score is not None else None,
            }
        )

    # 4. Select diverse viewpoints across candidate bucket medoids
    if len(bucket_medoids) <= max_prototypes:
        selected_candidates = bucket_medoids
    else:
        # Farthest point sampling for multi-viewpoint coverage
        # Start with global medoid
        all_dists = []
        for i, cand_i in enumerate(bucket_medoids):
            d_sum = 0.0
            for j, cand_j in enumerate(bucket_medoids):
                if i != j:
                    d = _pairwise_sample_distance(
                        cand_i["item"][1],
                        cand_i["item"][2],
                        cand_j["item"][1],
                        cand_j["item"][2],
                        min_vis=min_vis,
                    )
                    d_sum += d if d is not None else 0.5
            all_dists.append((d_sum, -float(np.mean(cand_i["item"][2])), i))
        all_dists.sort()
        selected_indices = [all_dists[0][2]]

        while len(selected_indices) < max_prototypes:
            best_cand = None
            max_min_d = -1.0
            for idx, cand in enumerate(bucket_medoids):
                if idx in selected_indices:
                    continue
                min_d = float("inf")
                for s_idx in selected_indices:
                    sel = bucket_medoids[s_idx]
                    d = _pairwise_sample_distance(
                        cand["item"][1],
                        cand["item"][2],
                        sel["item"][1],
                        sel["item"][2],
                        min_vis=min_vis,
                    )
                    d_val = d if d is not None else 0.5
                    if d_val < min_d:
                        min_d = d_val
                if min_d > max_min_d:
                    max_min_d = min_d
                    best_cand = idx
            if best_cand is None:
                break
            selected_indices.append(best_cand)

        selected_indices.sort(key=lambda idx: bucket_medoids[idx]["time_bucket"])
        selected_candidates = [bucket_medoids[idx] for idx in selected_indices]

    prototypes: list[AppearancePrototype] = []
    for cand in selected_candidates:
        s_obj, f_mat, v_vec, s_id, t_s = cand["item"]
        p_dim = len(v_vec)
        reliable = tuple(p for p in range(p_dim) if v_vec[p] >= min_vis and np.isfinite(v_vec[p]))

        # Also find any segment-level witnesses for this prototype
        seg_witnesses = list(cand["witness_ids"])
        for item in usable_items:
            if item[3] not in seg_witnesses:
                d = _pairwise_sample_distance(f_mat, v_vec, item[1], item[2], min_vis=min_vis)
                if d is not None and d <= near_distance:
                    seg_witnesses.append(item[3])

        prototypes.append(
            AppearancePrototype(
                prototype_id=f"proto_{sid}_{s_id}",
                sample_id=s_id,
                segment_id=sid,
                clip_id=clip_id,
                track_id=track_id,
                time_s=t_s,
                time_bucket=cand["time_bucket"],
                feature_space_id=fsid,
                feature=f_mat.astype(np.float32),
                visibility=v_vec.astype(np.float32),
                reliable_parts=reliable,
                quality_score=cand["quality_score"],
                risks=(),
                witness_sample_ids=tuple(sorted(set(seg_witnesses))),
                source_component=cand["source_component"],
            )
        )

    return prototypes


def compute_multi_prototype_distance(
    t1: TrackRepresentation,
    t2: TrackRepresentation,
    min_common_parts: int = 2,
    min_vis: float = 0.2,
    near_distance: float = 0.18,
    far_distance: float = 0.45,
    clustering_threshold: float = 0.28,
) -> float | None:
    """Compute appearance distance between two multi-prototype representations.

    1. Incompatible feature spaces return None (uncomparable).
    2. Counter-evidence rule (One-Vote Veto on Reliable Outliers):
       If ANY prototype in t1 is comparable to t2 but has distance >= far_distance
       to ALL comparable prototypes of t2 (or vice versa), it represents an
       unreconciled identity contradiction/intruder. Returns far_distance + 0.1 (veto).
    3. Viewpoint compatibility:
       Finds compatible prototype pairs (distance <= clustering_threshold).
       If compatible pairs exist, returns min(compatible_distances).
    4. If no compatible pairs exist but no far counter-evidence exists, returns
       min(all_comparable_distances) in the ambiguous band.
    """
    if t1.feature_space_id and t2.feature_space_id and t1.feature_space_id != t2.feature_space_id:
        return None

    if not t1.prototypes or not t2.prototypes:
        return _compute_pooled_distance(
            t1.feature, t1.visibility, t2.feature, t2.visibility, min_common_parts, min_vis
        )

    pair_distances: dict[tuple[int, int], float] = {}
    for i, p1 in enumerate(t1.prototypes):
        for j, p2 in enumerate(t2.prototypes):
            d = compute_prototype_distance(
                p1, p2, min_common_parts=min_common_parts, min_vis=min_vis
            )
            if d is not None:
                pair_distances[(i, j)] = d

    if not pair_distances:
        return None

    # One-vote veto on strong counter-evidence:
    # A prototype is a strong counter-evidence if it is comparable to the other representation
    # but exhibits distance >= far_distance to ALL comparable prototypes of that representation.
    for i in range(len(t1.prototypes)):
        dists_p1 = [d for (idx1, _), d in pair_distances.items() if idx1 == i]
        if dists_p1 and min(dists_p1) >= far_distance:
            return far_distance + 0.1

    for j in range(len(t2.prototypes)):
        dists_p2 = [d for (_, idx2), d in pair_distances.items() if idx2 == j]
        if dists_p2 and min(dists_p2) >= far_distance:
            return far_distance + 0.1

    all_dists = list(pair_distances.values())
    compat = [d for d in all_dists if d <= clustering_threshold]
    if compat:
        return min(compat)

    return min(all_dists)


def _compute_pooled_distance(
    f1: np.ndarray,
    v1: np.ndarray,
    f2: np.ndarray,
    v2: np.ndarray,
    min_common_parts: int = 2,
    min_vis: float = 0.2,
) -> float | None:
    v1_arr = np.asarray(v1, dtype=np.float64)
    v2_arr = np.asarray(v2, dtype=np.float64)

    min_p = min(len(v1_arr), len(v2_arr))
    v1_sub = v1_arr[:min_p]
    v2_sub = v2_arr[:min_p]

    overlap = (v1_sub >= min_vis) & (v2_sub >= min_vis) & np.isfinite(v1_sub) & np.isfinite(v2_sub)
    common_parts = int(np.count_nonzero(overlap))
    if common_parts < min_common_parts:
        return None

    weights = np.sqrt(v1_sub[overlap] * v2_sub[overlap])
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-12:
        return None

    f1_sub = np.asarray(f1[:min_p], dtype=np.float64)[overlap]
    f2_sub = np.asarray(f2[:min_p], dtype=np.float64)[overlap]

    part_diffs = np.linalg.norm(f1_sub - f2_sub, axis=-1)
    distance = float(np.sum(weights * part_diffs) / (2.0 * weight_sum))
    return distance


def compute_track_distance(
    t1: TrackRepresentation,
    t2: TrackRepresentation,
    min_common_parts: int = 2,
    min_vis: float = 0.2,
    near_distance: float = 0.18,
    far_distance: float = 0.45,
    clustering_threshold: float = 0.28,
) -> float | None:
    """Compute distance between two track/segment representations.

    Dispatches to multi-prototype distance when prototypes are present, or falls
    back to classic part-weighted pooling distance for backward compatibility.
    """
    if t1.prototypes and t2.prototypes:
        return compute_multi_prototype_distance(
            t1=t1,
            t2=t2,
            min_common_parts=min_common_parts,
            min_vis=min_vis,
            near_distance=near_distance,
            far_distance=far_distance,
            clustering_threshold=clustering_threshold,
        )

    return _compute_pooled_distance(
        t1.feature,
        t1.visibility,
        t2.feature,
        t2.visibility,
        min_common_parts=min_common_parts,
        min_vis=min_vis,
    )


def pool_track_features(
    samples: Iterable[Any],
    space: FeatureSpace | None = None,
    values: FeatureValues | Mapping[tuple[str, int], InlineFeature] | None = None,
    min_vis: float = 0.2,
) -> dict[tuple[str, int], TrackRepresentation]:
    """Pool track-level representations from frame samples by part-weighted averaging."""
    grouped_samples: dict[tuple[str, int], list[Any]] = defaultdict(list)

    for sample in samples:
        clip_id = str(
            getattr(sample, "clip_id", sample.get("clip_id") if isinstance(sample, dict) else "")
        )
        raw_tid = getattr(
            sample, "track_id", sample.get("track_id") if isinstance(sample, dict) else 0
        )
        try:
            track_id = int(raw_tid)
        except (ValueError, TypeError):
            track_id = raw_tid
        grouped_samples[(clip_id, track_id)].append(sample)

    default_parts = space.shape[0] if (space is not None and hasattr(space, "shape")) else 6
    default_dims = space.shape[1] if (space is not None and hasattr(space, "shape")) else 128
    fsid = (
        space.feature_space_id
        if (space is not None and hasattr(space, "feature_space_id"))
        else None
    )

    results: dict[tuple[str, int], TrackRepresentation] = {}

    for (clip_id, track_id), track_samples in grouped_samples.items():
        track_samples.sort(
            key=lambda s: float(
                getattr(s, "time_s", s.get("time_s") if isinstance(s, dict) else 0.0)
            )
        )

        start_s = min(
            float(
                getattr(
                    s,
                    "start_s",
                    getattr(s, "time_s", s.get("time_s", 0.0) if isinstance(s, dict) else 0.0),
                )
            )
            for s in track_samples
        )
        end_s = max(
            float(
                getattr(
                    s,
                    "end_s",
                    getattr(s, "time_s", s.get("time_s", 0.0) if isinstance(s, dict) else 0.0),
                )
            )
            for s in track_samples
        )
        sample_ids = [
            str(getattr(s, "sample_id", s.get("sample_id") if isinstance(s, dict) else ""))
            for s in track_samples
        ]

        extracted_frames: list[tuple[np.ndarray, np.ndarray]] = []
        for s in track_samples:
            f_mat, v_vec = _extract_sample_features(s, space, values)
            if f_mat is not None and v_vec is not None:
                extracted_frames.append((f_mat, v_vec))

        p_dim = default_parts
        d_dim = default_dims
        if extracted_frames:
            p_dim, d_dim = extracted_frames[0][0].shape

        pooled_feature = np.zeros((p_dim, d_dim), dtype=np.float32)
        pooled_visibility = np.zeros((p_dim,), dtype=np.float32)

        for p in range(p_dim):
            retained_w: list[float] = []
            retained_f: list[np.ndarray] = []

            for f_mat, v_vec in extracted_frames:
                if p < len(v_vec):
                    v = float(v_vec[p])
                    if v >= min_vis and np.isfinite(v):
                        retained_w.append(v)
                        retained_f.append(f_mat[p])

            if retained_w:
                w_arr = np.asarray(retained_w, dtype=np.float64)
                f_arr = np.asarray(retained_f, dtype=np.float64)
                weighted_sum = np.sum(w_arr[:, None] * f_arr, axis=0)
                norm = float(np.linalg.norm(weighted_sum))
                if norm > 1e-12:
                    pooled_feature[p] = (weighted_sum / norm).astype(np.float32)
                else:
                    pooled_feature[p] = np.zeros((d_dim,), dtype=np.float32)
                pooled_visibility[p] = float(np.mean(w_arr))
            else:
                pooled_feature[p] = np.zeros((d_dim,), dtype=np.float32)
                pooled_visibility[p] = 0.0

        results[(clip_id, track_id)] = TrackRepresentation(
            clip_id=clip_id,
            track_id=track_id,
            start_s=start_s,
            end_s=end_s,
            feature=pooled_feature,
            visibility=pooled_visibility,
            sample_ids=sample_ids,
            feature_space_id=fsid,
        )

    return results


def pool_segment_features(
    segments: Mapping[str, Any] | Iterable[Any],
    samples: Iterable[Any] | Mapping[str, list[Any]],
    space: FeatureSpace | None = None,
    values: FeatureValues | Mapping[tuple[str, int], InlineFeature] | None = None,
    min_vis: float = 0.2,
    bucket_s: float = 0.5,
    near_distance: float = 0.18,
    far_distance: float = 0.45,
    max_prototypes: int = 4,
    components: Mapping[str, str] | None = None,
) -> dict[str, TrackRepresentation]:
    """Pool segment-level representations and extract multi-prototype appearance models."""
    if isinstance(segments, Mapping):
        seg_dict = dict(segments)
    else:
        seg_dict = {
            str(getattr(s, "segment_id", s.get("segment_id") if isinstance(s, dict) else "")): s
            for s in segments
        }

    if isinstance(samples, Mapping):
        grouped_samples = {str(k): list(v) for k, v in samples.items()}
    else:
        grouped_samples = defaultdict(list)
        for sample in samples:
            sid = str(
                getattr(
                    sample,
                    "segment_id",
                    sample.get("segment_id") if isinstance(sample, dict) else "",
                )
            )
            grouped_samples[sid].append(sample)

    default_parts = space.shape[0] if (space is not None and hasattr(space, "shape")) else 6
    default_dims = space.shape[1] if (space is not None and hasattr(space, "shape")) else 128
    fsid = (
        space.feature_space_id
        if (space is not None and hasattr(space, "feature_space_id"))
        else None
    )

    results: dict[str, TrackRepresentation] = {}

    for sid, seg in seg_dict.items():
        seg_samples = grouped_samples.get(sid, [])
        seg_samples.sort(
            key=lambda s: float(
                getattr(s, "time_s", s.get("time_s") if isinstance(s, dict) else 0.0)
            )
        )

        clip_id = str(getattr(seg, "clip_id", seg.get("clip_id") if isinstance(seg, dict) else ""))
        raw_tid = getattr(seg, "track_id", seg.get("track_id") if isinstance(seg, dict) else 0)
        try:
            track_id: int | str = int(raw_tid)
        except (ValueError, TypeError):
            track_id = str(raw_tid)

        start_s = float(
            getattr(seg, "start_s", seg.get("start_s") if isinstance(seg, dict) else 0.0)
        )
        end_s = float(getattr(seg, "end_s", seg.get("end_s") if isinstance(seg, dict) else 0.0))

        sample_ids = [
            str(getattr(s, "sample_id", s.get("sample_id") if isinstance(s, dict) else ""))
            for s in seg_samples
        ]

        extracted_frames: list[tuple[np.ndarray, np.ndarray]] = []
        for s in seg_samples:
            f_mat, v_vec = _extract_sample_features(s, space, values)
            if f_mat is not None and v_vec is not None:
                extracted_frames.append((f_mat, v_vec))

        p_dim = default_parts
        d_dim = default_dims
        if extracted_frames:
            p_dim, d_dim = extracted_frames[0][0].shape

        pooled_feature = np.zeros((p_dim, d_dim), dtype=np.float32)
        pooled_visibility = np.zeros((p_dim,), dtype=np.float32)

        for p in range(p_dim):
            retained_w: list[float] = []
            retained_f: list[np.ndarray] = []

            for f_mat, v_vec in extracted_frames:
                if p < len(v_vec):
                    v = float(v_vec[p])
                    if v >= min_vis and np.isfinite(v):
                        retained_w.append(v)
                        retained_f.append(f_mat[p])

            if retained_w:
                w_arr = np.asarray(retained_w, dtype=np.float64)
                f_arr = np.asarray(retained_f, dtype=np.float64)
                weighted_sum = np.sum(w_arr[:, None] * f_arr, axis=0)
                norm = float(np.linalg.norm(weighted_sum))
                if norm > 1e-12:
                    pooled_feature[p] = (weighted_sum / norm).astype(np.float32)
                else:
                    pooled_feature[p] = np.zeros((d_dim,), dtype=np.float32)
                pooled_visibility[p] = float(np.mean(w_arr))
            else:
                pooled_feature[p] = np.zeros((d_dim,), dtype=np.float32)
                pooled_visibility[p] = 0.0

        # Extract multi-prototypes grounded in concrete raw samples
        prototypes = extract_segment_prototypes(
            segment=seg,
            samples=seg_samples,
            space=space,
            values=values,
            min_vis=min_vis,
            bucket_s=bucket_s,
            near_distance=near_distance,
            far_distance=far_distance,
            max_prototypes=max_prototypes,
            components=components,
        )

        results[sid] = TrackRepresentation(
            clip_id=clip_id,
            track_id=track_id,
            start_s=start_s,
            end_s=end_s,
            feature=pooled_feature,
            visibility=pooled_visibility,
            sample_ids=sample_ids,
            segment_id=sid,
            feature_space_id=fsid,
            prototypes=tuple(prototypes),
        )

    return results


__all__ = [
    "AppearancePrototype",
    "SegmentRepresentation",
    "TrackRepresentation",
    "compute_multi_prototype_distance",
    "compute_prototype_distance",
    "compute_track_distance",
    "extract_segment_prototypes",
    "pool_segment_features",
    "pool_track_features",
]
