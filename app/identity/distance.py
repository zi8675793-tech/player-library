"""Strict KPR part distances, independent of storage and input file formats."""

import numpy as np

from app.contracts.evidence import FeatureSpace, InlineFeature
from app.contracts.identity import DistanceResult, MatchingPolicy


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
    weights = np.sqrt(va * vb)
    common_parts = int(np.count_nonzero(weights > 0))
    if common_parts < policy.min_common_parts:
        return DistanceResult(
            comparable=False, common_parts=common_parts, reason="insufficient_common_parts"
        )
    weight_sum = float(weights.sum())
    if weight_sum <= policy.visibility_weight_epsilon:
        return DistanceResult(
            comparable=False, common_parts=common_parts, reason="insufficient_visibility_weight"
        )
    fa = np.asarray(a.values, dtype=np.float64).reshape(space.shape)
    fb = np.asarray(b.values, dtype=np.float64).reshape(space.shape)
    visible = weights > 0
    value = float(
        np.dot(weights[visible], np.linalg.norm(fa[visible] - fb[visible], axis=-1))
        / (2 * weight_sum)
    )
    return DistanceResult(comparable=True, distance=value, common_parts=common_parts)
