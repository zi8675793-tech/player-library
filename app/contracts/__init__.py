"""Canonical public contracts; detailed models also live in the named submodules."""

from app.contracts.api import ErrorResponse, ImportRequest, ReviewRequest, SuccessResponse
from app.contracts.common import Provenance, SourceKey, SourceLocation
from app.contracts.events import (
    Event,
    IntervalEventTime,
    PointEventTime,
    ResolvedEvent,
    Subject,
)
from app.contracts.evidence import (
    Clip,
    EvidenceBatch,
    EvidenceWarning,
    FeatureFile,
    FeatureSpace,
    IdentityEvidence,
    MediaAsset,
    Sample,
    SourceMapping,
    Track,
)
from app.contracts.identity import (
    Assignment,
    Candidate,
    PersonRole,
    Player,
    Resolution,
    Segment,
    Template,
)

__all__ = [
    "Assignment",
    "Candidate",
    "Clip",
    "ErrorResponse",
    "Event",
    "EvidenceBatch",
    "EvidenceWarning",
    "FeatureFile",
    "FeatureSpace",
    "IdentityEvidence",
    "ImportRequest",
    "IntervalEventTime",
    "MediaAsset",
    "PersonRole",
    "Player",
    "PointEventTime",
    "Provenance",
    "Resolution",
    "ResolvedEvent",
    "ReviewRequest",
    "Sample",
    "Segment",
    "SourceKey",
    "SourceLocation",
    "SourceMapping",
    "Subject",
    "SuccessResponse",
    "Template",
    "Track",
]
