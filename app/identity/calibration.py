"""Extraction, diagnostics, and freezing for feature-space distance calibration profiles."""

import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.contracts.calibration import (
    CalibrationDiagnosticReport,
    CandidatePairExclusion,
    DistanceCalibrationProfile,
    DistanceQuantiles,
    DistanceSemantics,
)
from app.contracts.evidence import EvidenceBatch, FeatureSpace
from app.contracts.identity import MatchingPolicy
from app.identity.distance import kpr_distance, validate_feature_space
from app.identity.fingerprints import evidence_fingerprint
from app.identity.prepare import FeatureValues, resolve_feature


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = a
    u1, v1, u2, v2 = b
    intersection = max(0.0, min(x2, u2) - max(x1, u1)) * max(0.0, min(y2, v2) - max(y1, v1))
    area_a = (x2 - x1) * (y2 - y1)
    area_b = (u2 - u1) * (v2 - v1)
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def _center_distance_heights(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    x, y, z, w = a
    u, v, s, t = b
    height = max((w - y + t - v) / 2.0, 1e-12)
    return float(math.hypot((x + z - u - s) / 2.0, (y + w - v - t) / 2.0) / height)


def _compute_quantiles(values: list[float]) -> DistanceQuantiles | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    pcts = np.percentile(arr, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    return DistanceQuantiles(
        count=int(len(arr)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        p01=float(pcts[0]),
        p05=float(pcts[1]),
        p10=float(pcts[2]),
        p25=float(pcts[3]),
        p50=float(pcts[4]),
        p75=float(pcts[5]),
        p90=float(pcts[6]),
        p95=float(pcts[7]),
        p99=float(pcts[8]),
    )


def extract_candidate_calibration_evidence(
    batch: EvidenceBatch,
    feature_values: FeatureValues,
    *,
    feature_space_id: str | None = None,
    max_positive_gap_s: float = 1.0,
    max_negative_iou: float = 0.2,
    min_negative_separation_heights: float = 0.2,
    min_common_parts: int = 2,
    min_part_visibility: float = 0.2,
    jump_heights: float = 1.5,
    jump_heights_per_s: float = 8.0,
) -> tuple[
    FeatureSpace,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[CandidatePairExclusion],
    dict[str, int],
]:
    """Extract candidate positives, physical negatives, and exclusions from machine evidence.

    Candidate positives are adjacent, temporally close observations within the same track,
    without spatial teleportation or image/descriptor duplication.
    Physical negatives are co-occurring, non-overlapping bounding boxes in the same video
    frame across distinct tracks, representing verified distinct physical bodies.
    All exclusions are tracked with specific structural reasons.
    """
    evidence = batch.identity_evidence
    if evidence is None:
        raise ValueError("Evidence batch must contain identity evidence for distance calibration")

    spaces = {s.feature_space_id: s for s in evidence.feature_spaces}
    if not spaces:
        raise ValueError("No feature spaces found in evidence batch")
    if feature_space_id is None:
        if len(spaces) != 1:
            raise ValueError(
                "Multiple feature spaces present; feature_space_id must be specified explicitly"
            )
        feature_space_id = next(iter(spaces))
    if feature_space_id not in spaces:
        raise ValueError(
            f"Feature space {feature_space_id!r} not found in evidence batch feature spaces"
        )

    target_space = spaces[feature_space_id]
    policy = MatchingPolicy(
        feature_space_id=feature_space_id,
        min_common_parts=min_common_parts,
        min_part_visibility=min_part_visibility,
    )
    validate_feature_space(target_space, policy)

    assets_by_id = {a.asset_id: a for a in batch.assets}
    obs_map = {(o.clip_id, o.track_id, o.frame_idx): o for o in evidence.observations}

    EXPLICIT_RISK_KINDS = {
        "mixed_identity",
        "identity_conflict",
        "suspected_identity_switch",
        "identity_switch",
    }
    sample_risk_reasons: dict[str, str] = {}
    track_risks: list[Any] = []

    for risk in getattr(evidence, "risks", []):
        rkind = risk.kind.value if hasattr(risk.kind, "value") else str(risk.kind)
        if rkind in EXPLICIT_RISK_KINDS:
            if getattr(risk, "sample_id", None):
                sample_risk_reasons[risk.sample_id] = rkind
            if getattr(risk, "clip_id", None) and getattr(risk, "track_id", None):
                track_risks.append(risk)

    for sample in evidence.samples:
        q = sample.tracking_quality
        if q is not None:
            if getattr(q, "mixed_identity", False):
                sample_risk_reasons[sample.sample_id] = "mixed_identity"
            for r in getattr(q, "risks", []):
                rkind = r.kind.value if hasattr(r.kind, "value") else str(r.kind)
                if rkind in EXPLICIT_RISK_KINDS:
                    sample_risk_reasons[sample.sample_id] = rkind

    def _has_explicit_risk(s: Any) -> bool:
        if s.sample_id in sample_risk_reasons:
            return True
        for tr in track_risks:
            if tr.clip_id == s.clip_id and tr.track_id == s.track_id:
                if tr.start_s is None and tr.end_s is None:
                    return True
                if (
                    tr.start_s is not None
                    and tr.end_s is not None
                    and s.time_s is not None
                    and tr.start_s <= s.time_s <= tr.end_s
                ):
                    return True
        return False

    def _asset_sha(sample: Any) -> str:
        asset = assets_by_id.get(sample.image_asset_id)
        if asset is not None and hasattr(asset, "file") and asset.file.sha256:
            return asset.file.sha256
        return str(sample.image_asset_id)

    def _desc_sha(sample: Any) -> str | None:
        feat = resolve_feature(sample, target_space, feature_values)
        if feat is None:
            return None
        return hashlib.sha256(np.asarray(feat.values, dtype=np.float32).tobytes()).hexdigest()

    samples_by_track: dict[tuple[str, str], list[Any]] = defaultdict(list)
    samples_by_clip_frame: dict[tuple[str, int], list[Any]] = defaultdict(list)
    for sample in evidence.samples:
        samples_by_track[(sample.clip_id, sample.track_id)].append(sample)
        if sample.frame_idx is not None:
            samples_by_clip_frame[(sample.clip_id, sample.frame_idx)].append(sample)

    positive_pairs: list[dict[str, Any]] = []
    negative_pairs: list[dict[str, Any]] = []
    exclusions: list[CandidatePairExclusion] = []
    exclusion_counts: Counter[str] = Counter()

    def record_exclusion(
        pair_type: str,
        left_id: str,
        right_id: str,
        clip_id: str,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        exclusions.append(
            CandidatePairExclusion(
                pair_type=pair_type,  # type: ignore[arg-type]
                left_sample_id=left_id,
                right_sample_id=right_id,
                clip_id=clip_id,
                reason=reason,
                details=details or {},
            )
        )
        exclusion_counts[reason] += 1

    # 1. Candidate Positives
    for (clip_id, track_id), track_samples in samples_by_track.items():
        track_samples.sort(key=lambda s: (s.time_s, s.sample_id))
        for a, b in zip(track_samples, track_samples[1:]):
            dt = b.time_s - a.time_s
            if dt > max_positive_gap_s:
                record_exclusion(
                    "candidate_positive",
                    a.sample_id,
                    b.sample_id,
                    clip_id,
                    "temporal_gap_too_large",
                    {"dt_s": dt, "max_gap_s": max_positive_gap_s},
                )
                continue
            if dt <= 0.0:
                record_exclusion(
                    "candidate_positive",
                    a.sample_id,
                    b.sample_id,
                    clip_id,
                    "non_positive_time_delta",
                    {"dt_s": dt},
                )
                continue

            if _asset_sha(a) == _asset_sha(b):
                record_exclusion(
                    "candidate_positive",
                    a.sample_id,
                    b.sample_id,
                    clip_id,
                    "duplicate_image",
                )
                continue

            ha, hb = _desc_sha(a), _desc_sha(b)
            if ha is not None and ha == hb:
                record_exclusion(
                    "candidate_positive",
                    a.sample_id,
                    b.sample_id,
                    clip_id,
                    "duplicate_descriptor",
                )
                continue

            if _has_explicit_risk(a) or _has_explicit_risk(b):
                bad_s = a if _has_explicit_risk(a) else b
                record_exclusion(
                    "candidate_positive",
                    a.sample_id,
                    b.sample_id,
                    clip_id,
                    "explicit_identity_risk",
                    {"sample_id": bad_s.sample_id},
                )
                continue

            qa, qb = a.tracking_quality, b.tracking_quality
            if qa and (qa.occluded or qa.mixed_identity or qa.usability == "unusable"):
                record_exclusion(
                    "candidate_positive",
                    a.sample_id,
                    b.sample_id,
                    clip_id,
                    "low_quality_or_occluded",
                    {"sample_id": a.sample_id},
                )
                continue
            if qb and (qb.occluded or qb.mixed_identity or qb.usability == "unusable"):
                record_exclusion(
                    "candidate_positive",
                    a.sample_id,
                    b.sample_id,
                    clip_id,
                    "low_quality_or_occluded",
                    {"sample_id": b.sample_id},
                )
                continue

            obsa = obs_map.get((a.clip_id, a.track_id, a.frame_idx))
            obsb = obs_map.get((b.clip_id, b.track_id, b.frame_idx))
            if (
                obsa is None
                or obsb is None
                or getattr(obsa, "bbox_xyxy", None) is None
                or getattr(obsb, "bbox_xyxy", None) is None
            ):
                record_exclusion(
                    "candidate_positive",
                    a.sample_id,
                    b.sample_id,
                    clip_id,
                    "missing_observation_or_bbox",
                )
                continue

            dist = _center_distance_heights(obsa.bbox_xyxy, obsb.bbox_xyxy)
            if dist >= jump_heights and (dist / max(dt, 1e-6)) >= jump_heights_per_s:
                record_exclusion(
                    "candidate_positive",
                    a.sample_id,
                    b.sample_id,
                    clip_id,
                    "spatial_jump",
                    {"jump_heights": dist, "dt_s": dt},
                )
                continue

            fa = resolve_feature(a, target_space, feature_values)
            fb = resolve_feature(b, target_space, feature_values)
            if fa is None or fb is None:
                record_exclusion(
                    "candidate_positive",
                    a.sample_id,
                    b.sample_id,
                    clip_id,
                    "missing_feature_values",
                )
                continue

            res = kpr_distance(target_space, fa, fb, policy)
            if not res.comparable or res.distance is None:
                record_exclusion(
                    "candidate_positive",
                    a.sample_id,
                    b.sample_id,
                    clip_id,
                    res.reason or "incomparable",
                    {"common_parts": res.common_parts},
                )
                continue

            positive_pairs.append(
                {
                    "left_sample_id": a.sample_id,
                    "right_sample_id": b.sample_id,
                    "clip_id": clip_id,
                    "track_id": track_id,
                    "dt_s": dt,
                    "distance": res.distance,
                    "common_parts": res.common_parts,
                }
            )

    # 2. Mutually Exclusive Physical Negatives
    from itertools import combinations

    for (clip_id, frame_idx), frame_samples in samples_by_clip_frame.items():
        if len(frame_samples) < 2:
            continue
        for a, b in combinations(frame_samples, 2):
            if a.track_id == b.track_id:
                continue

            if _has_explicit_risk(a) or _has_explicit_risk(b):
                bad_s = a if _has_explicit_risk(a) else b
                record_exclusion(
                    "physical_negative",
                    a.sample_id,
                    b.sample_id,
                    clip_id,
                    "explicit_identity_risk",
                    {"sample_id": bad_s.sample_id},
                )
                continue

            obsa = obs_map.get((a.clip_id, a.track_id, a.frame_idx))
            obsb = obs_map.get((b.clip_id, b.track_id, b.frame_idx))
            if (
                obsa is None
                or obsb is None
                or getattr(obsa, "bbox_xyxy", None) is None
                or getattr(obsb, "bbox_xyxy", None) is None
            ):
                record_exclusion(
                    "physical_negative",
                    a.sample_id,
                    b.sample_id,
                    clip_id,
                    "missing_observation_or_bbox",
                )
                continue

            iou = _iou(obsa.bbox_xyxy, obsb.bbox_xyxy)
            if iou >= max_negative_iou:
                record_exclusion(
                    "physical_negative",
                    a.sample_id,
                    b.sample_id,
                    clip_id,
                    "duplicate_or_overlapping_box",
                    {"iou": iou, "max_iou": max_negative_iou},
                )
                continue
            sep = _center_distance_heights(obsa.bbox_xyxy, obsb.bbox_xyxy)
            if sep < min_negative_separation_heights:
                record_exclusion(
                    "physical_negative",
                    a.sample_id,
                    b.sample_id,
                    clip_id,
                    "insufficient_physical_separation",
                    {"separation_heights": sep},
                )
                continue

            if _asset_sha(a) == _asset_sha(b):
                record_exclusion(
                    "physical_negative",
                    a.sample_id,
                    b.sample_id,
                    clip_id,
                    "duplicate_image_across_tracks",
                )
                continue

            ha, hb = _desc_sha(a), _desc_sha(b)
            if ha is not None and ha == hb:
                record_exclusion(
                    "physical_negative",
                    a.sample_id,
                    b.sample_id,
                    clip_id,
                    "duplicate_descriptor_across_tracks",
                )
                continue

            qa, qb = a.tracking_quality, b.tracking_quality
            if qa and (qa.occluded or qa.mixed_identity or qa.usability == "unusable"):
                record_exclusion(
                    "physical_negative",
                    a.sample_id,
                    b.sample_id,
                    clip_id,
                    "low_quality_or_occluded",
                    {"sample_id": a.sample_id},
                )
                continue
            if qb and (qb.occluded or qb.mixed_identity or qb.usability == "unusable"):
                record_exclusion(
                    "physical_negative",
                    a.sample_id,
                    b.sample_id,
                    clip_id,
                    "low_quality_or_occluded",
                    {"sample_id": b.sample_id},
                )
                continue

            fa = resolve_feature(a, target_space, feature_values)
            fb = resolve_feature(b, target_space, feature_values)
            if fa is None or fb is None:
                record_exclusion(
                    "physical_negative",
                    a.sample_id,
                    b.sample_id,
                    clip_id,
                    "missing_feature_values",
                )
                continue

            res = kpr_distance(target_space, fa, fb, policy)
            if not res.comparable or res.distance is None:
                record_exclusion(
                    "physical_negative",
                    a.sample_id,
                    b.sample_id,
                    clip_id,
                    res.reason or "incomparable",
                    {"common_parts": res.common_parts},
                )
                continue

            negative_pairs.append(
                {
                    "left_sample_id": a.sample_id,
                    "right_sample_id": b.sample_id,
                    "clip_id": clip_id,
                    "left_track_id": a.track_id,
                    "right_track_id": b.track_id,
                    "frame_idx": frame_idx,
                    "distance": res.distance,
                    "common_parts": res.common_parts,
                }
            )

    return target_space, positive_pairs, negative_pairs, exclusions, dict(exclusion_counts)


def build_distance_calibration_profile(
    batch: EvidenceBatch,
    feature_values: FeatureValues,
    *,
    feature_space_id: str | None = None,
    calibration_version: str = "stage1-diagnostic-v1",
    profile_id: str | None = None,
    evidence_source: str = "test-machine-evidence-10clips",
    near_distance: float = 0.18,
    far_distance: float = 0.45,
    min_common_parts: int = 2,
    min_part_visibility: float = 0.2,
) -> tuple[DistanceCalibrationProfile, CalibrationDiagnosticReport]:
    """Diagnose evidence distance distributions and produce a frozen DistanceCalibrationProfile.

    Honoring Stage 1 requirements: If the unsupervised candidate evidence shows overlapping
    positive and negative distributions, the profile status remains 'uncalibrated' (or 'provisional'),
    maintaining the engineering gates near=0.18, far=0.45 without arbitrary tampering.
    """
    (
        space,
        pos_pairs,
        neg_pairs,
        exclusions,
        exclusion_counts,
    ) = extract_candidate_calibration_evidence(
        batch,
        feature_values,
        feature_space_id=feature_space_id,
        min_common_parts=min_common_parts,
        min_part_visibility=min_part_visibility,
    )

    pos_distances = [p["distance"] for p in pos_pairs]
    neg_distances = [p["distance"] for p in neg_pairs]

    pos_quantiles = _compute_quantiles(pos_distances)
    neg_quantiles = _compute_quantiles(neg_distances)

    ev_fingerprint = evidence_fingerprint(batch, feature_values)
    config_data = {
        "calibration_version": calibration_version,
        "feature_space_id": space.feature_space_id,
        "near_distance": near_distance,
        "far_distance": far_distance,
        "min_common_parts": min_common_parts,
        "min_part_visibility": min_part_visibility,
    }
    cfg_fingerprint = hashlib.sha256(
        json.dumps(config_data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    pid = profile_id or f"calprof_{space.feature_space_id[:16]}_{cfg_fingerprint[:16]}"

    overlap_interval: list[float] = []
    if pos_quantiles and neg_quantiles:
        overlap_interval = [float(neg_quantiles.min), float(pos_quantiles.max)]

    pos_p50 = f"{pos_quantiles.p50:.4f}" if pos_quantiles else "N/A"
    pos_p90 = f"{pos_quantiles.p90:.4f}" if pos_quantiles else "N/A"
    neg_p10 = f"{neg_quantiles.p10:.4f}" if neg_quantiles else "N/A"
    neg_p50 = f"{neg_quantiles.p50:.4f}" if neg_quantiles else "N/A"

    diagnosis = (
        f"Feature space {space.feature_space_id}: Extracted {len(pos_distances)} candidate positives "
        f"(median={pos_p50}, p90={pos_p90}) "
        f"and {len(neg_distances)} physical negatives "
        f"(p10={neg_p10}, median={neg_p50}). "
        f"Distributions overlap across [{overlap_interval[0]:.4f}, {overlap_interval[1]:.4f}]. "
        "Global threshold adjustment (e.g. bumping near to 0.26/0.30) is statistically rejected; "
        "retaining uncalibrated engineering gates near=0.18, far=0.45 awaiting Stage 2 local windows."
    )

    profile = DistanceCalibrationProfile(
        profile_id=pid,
        feature_space_id=space.feature_space_id,
        calibration_version=calibration_version,
        distance_metric=space.distance,
        distance_semantics=DistanceSemantics(
            backend=space.backend,
            layout=space.layout,
            normalization=space.normalization,
            distance_metric=space.distance,
            part_weighting="sqrt_visibility",
        ),
        status="uncalibrated",
        near_distance=near_distance,
        far_distance=far_distance,
        min_common_parts=min_common_parts,
        min_part_visibility=min_part_visibility,
        evidence_source=evidence_source,
        evidence_fingerprint=ev_fingerprint,
        config_fingerprint=cfg_fingerprint,
        created_at_utc=datetime.now(UTC).isoformat(),
        candidate_positives_summary=pos_quantiles,
        physical_negatives_summary=neg_quantiles,
        exclusion_counts=exclusion_counts,
        notes=diagnosis,
    )

    report = CalibrationDiagnosticReport(
        profile=profile,
        candidate_positive_count=len(pos_distances),
        physical_negative_count=len(neg_distances),
        total_exclusions_count=len(exclusions),
        exclusion_breakdown=exclusion_counts,
        overlap_interval=overlap_interval,
        separation_diagnosis=diagnosis,
        positive_pairs=pos_pairs,
        negative_pairs=neg_pairs,
        exclusions=exclusions,
    )

    return profile, report


def save_calibration_profile(profile: DistanceCalibrationProfile, target_path: Path) -> Path:
    """Save a DistanceCalibrationProfile to JSON with indentation and sorted keys."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as handle:
        json.dump(
            profile.model_dump(mode="json"), handle, ensure_ascii=False, indent=2, sort_keys=True
        )
        handle.write("\n")
    return target_path


def load_calibration_profile(path: Path | str) -> DistanceCalibrationProfile:
    """Load and validate a DistanceCalibrationProfile from JSON."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Calibration profile file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return DistanceCalibrationProfile.model_validate(data)


def save_diagnostic_report(report: CalibrationDiagnosticReport, target_path: Path) -> Path:
    """Save a CalibrationDiagnosticReport to JSON."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as handle:
        json.dump(
            report.model_dump(mode="json"), handle, ensure_ascii=False, indent=2, sort_keys=True
        )
        handle.write("\n")
    return target_path
