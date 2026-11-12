"""Strict KPR part distances, independent of storage and input file formats."""

from typing import Literal

import numpy as np

from app.contracts.calibration import DistanceCalibrationProfile
from app.contracts.evidence import FeatureSpace, InlineFeature
from app.contracts.identity import DistanceResult, MatchingPolicy

DistanceRelation = Literal["near", "far", "ambiguous", "uncomparable"]


class FeatureValidationError(ValueError):
    """The supplied feature or backend contract is invalid, rather than uncomparable."""


def validate_feature_space(space: FeatureSpace, policy: MatchingPolicy) -> None:
    """Reject unsupported backend semantics and illegal per-space policy bounds."""
    if space.feature_space_id != policy.feature_space_id:
        raise FeatureValidationError("policy and feature space must use the same feature_space_id")
    if (
        space.backend != "kpr"
        or space.layout != "parts"
        or space.normalization != "l2"
        or space.distance != "euclidean_sqrt_visibility_mean_div2"
        or len(space.shape) != 2
        or space.visibility is None
        or space.visibility.shape != space.shape[:1]
    ):
        raise FeatureValidationError("unsupported KPR backend or semantic feature contract")
    if policy.min_common_parts > space.shape[0]:
        raise FeatureValidationError("min_common_parts exceeds feature space part count")


def validate_feature(space: FeatureSpace, feature: InlineFeature, policy: MatchingPolicy) -> None:
    """Validate the declared shape and visible part L2 norms without repairing values."""
    validate_feature_space(space, policy)
    if feature.feature_space_id != space.feature_space_id:
        raise FeatureValidationError("feature and contract feature_space_id mismatch")
    p, d = space.shape
    values = np.asarray(feature.values)
    visibility = np.asarray(feature.visibility)
    if values.shape != (p * d,):
        raise FeatureValidationError(f"feature values must have flattened shape {(p * d,)}")
    if visibility.shape != (p,):
        raise FeatureValidationError(f"feature visibility must have shape {(p,)}")
    if values.dtype.kind not in "fi" or visibility.dtype.kind not in "fi":
        raise FeatureValidationError("feature values and visibility must be numeric")
    if not np.isfinite(values).all() or not np.isfinite(visibility).all():
        raise FeatureValidationError("feature values and visibility must be finite")
    if np.any((visibility < 0) | (visibility > 1)):
        raise FeatureValidationError("feature visibility must be in [0,1]")
    norms = np.linalg.norm(values.astype(np.float64).reshape(p, d)[visibility > 0], axis=-1)
    if np.any(~np.isclose(norms, 1.0, rtol=policy.l2_rtol, atol=policy.l2_atol)):
        raise FeatureValidationError("visible feature parts must be L2 normalized")


def kpr_distance(
    space: FeatureSpace, a: InlineFeature, b: InlineFeature, policy: MatchingPolicy
) -> DistanceResult:
    """Return a finite weighted distance or an explicit lack of comparability.

    Part weights are sqrt(v_a * v_b), and the weighted Euclidean mean is divided
    by two. No vector or visibility is changed, normalized, clipped or filled.
    """
    if a.feature_space_id != b.feature_space_id or a.feature_space_id != space.feature_space_id:
        return DistanceResult(comparable=False, common_parts=0, reason="feature_space_mismatch")
    validate_feature(space, a, policy)
    validate_feature(space, b, policy)
    va = np.asarray(a.visibility, dtype=np.float64)
    vb = np.asarray(b.visibility, dtype=np.float64)
    min_vis = getattr(policy, "min_part_visibility", 0.0)
    valid = (va >= min_vis) & (vb >= min_vis) & (va > 0) & (vb > 0)
    common_parts = int(np.count_nonzero(valid))
    if common_parts < policy.min_common_parts:
        reason = "insufficient_visible_parts" if min_vis > 0.0 else "insufficient_common_parts"
        return DistanceResult(comparable=False, common_parts=common_parts, reason=reason)
    weights = np.zeros_like(va)
    weights[valid] = np.sqrt(va[valid] * vb[valid])
    weight_sum = float(weights.sum())
    if weight_sum <= policy.visibility_weight_epsilon:
        return DistanceResult(
            comparable=False, common_parts=common_parts, reason="insufficient_visibility_weight"
        )
    fa = np.asarray(a.values, dtype=np.float64).reshape(space.shape)
    fb = np.asarray(b.values, dtype=np.float64).reshape(space.shape)
    visible = valid
    value = float(
        np.dot(weights[visible], np.linalg.norm(fa[visible] - fb[visible], axis=-1))
        / (2 * weight_sum)
    )
    return DistanceResult(comparable=True, distance=value, common_parts=common_parts)


def validate_calibration_profile_space(
    space: FeatureSpace, profile: DistanceCalibrationProfile
) -> None:
    """Ensure the calibration profile matches the exact target feature space and distance semantics."""
    if profile.feature_space_id != space.feature_space_id:
        raise FeatureValidationError(
            f"calibration profile feature_space_id {profile.feature_space_id!r} "
            f"does not match space {space.feature_space_id!r}"
        )
    if profile.distance_metric != space.distance:
        raise FeatureValidationError(
            f"calibration profile distance_metric {profile.distance_metric!r} "
            f"does not match space distance {space.distance!r}"
        )
    if profile.distance_semantics.backend != space.backend:
        raise FeatureValidationError(
            f"calibration profile backend {profile.distance_semantics.backend!r} "
            f"does not match space backend {space.backend!r}"
        )
    if profile.distance_semantics.normalization != space.normalization:
        raise FeatureValidationError(
            f"calibration profile normalization {profile.distance_semantics.normalization!r} "
            f"does not match space normalization {space.normalization!r}"
        )


def classify_distance_relation(
    distance: float | None,
    near_distance: float,
    far_distance: float,
) -> DistanceRelation:
    """Classify a finite distance into explicit relational semantics.

    Semantics:
    - near: Local appearance consistency support. Does NOT assert unconditional identity truth.
    - far: Strong counter-evidence. A single far distance does NOT directly prove an identity switch.
    - ambiguous: Indeterminate region between near and far; lacks strong support and strong counter-evidence.
    - uncomparable: Distance is None due to insufficient common parts or feature space mismatch.
    """
    if distance is None:
        return "uncomparable"
    if distance <= near_distance:
        return "near"
    if distance >= far_distance:
        return "far"
    return "ambiguous"


def evaluate_feature_relation(
    space: FeatureSpace,
    a: InlineFeature,
    b: InlineFeature,
    policy: MatchingPolicy,
    profile: DistanceCalibrationProfile | None = None,
) -> tuple[DistanceRelation, DistanceResult]:
    """Compute KPR distance and classify its relation under a feature space and profile."""
    result = kpr_distance(space, a, b, policy)
    if profile is not None:
        validate_calibration_profile_space(space, profile)
        near = profile.near_distance
        far = profile.far_distance
    else:
        # Require tau / delta or explicit defaults
        near = policy.tau if policy.tau is not None else 0.18
        far = policy.delta if policy.delta is not None else 0.45
    relation = classify_distance_relation(result.distance, near, far)
    return relation, result
