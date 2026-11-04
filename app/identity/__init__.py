"""Pure identity rules: prepare, associate, resolve, and plan gallery updates."""

from app.identity.associate import associate_segments
from app.identity.distance import FeatureValidationError, kpr_distance
from app.identity.gallery import plan_gallery
from app.identity.prepare import prepare_evidence
from app.identity.resolve import resolve_associations

__all__ = [
    "FeatureValidationError",
    "associate_segments",
    "kpr_distance",
    "plan_gallery",
    "prepare_evidence",
    "resolve_associations",
]
