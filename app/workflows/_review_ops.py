"""Shared current evidence interpretation and formal consistency checks."""

import hashlib
from collections.abc import Mapping, Sequence

from app.contracts.common import SourceKey
from app.contracts.evidence import (
    CleanReviewStatus,
    GallerySnapshot,
    IdentityEvidence,
    TrackingQuality,
)
from app.contracts.identity import AssignmentStatus, ConstraintKind, ManualConstraint
from app.identity.prepare import _within


def stable_player_id(match_id: str, namespace: str, request_id: str) -> str:
    key = f"{match_id}\0{namespace}\0{request_id}".encode()
    return "pl_" + hashlib.sha256(key).hexdigest()[:24]


def current_evidence(
    evidence: IdentityEvidence,
    *,
    clean_reviews: Mapping[str, dict] | None = None,
) -> IdentityEvidence:
    """Project raw samples onto explicit replacement descendants without rewriting evidence.

    Upstream clean labels are descriptive only. Accepted human clean reviews are
    separately persisted and overlay this temporary algorithm input.
    """
    result = evidence.model_copy(deep=True)
    segments = {segment.segment_id: segment for segment in result.segments}
    for sample in result.samples:
        quality = sample.tracking_quality or TrackingQuality()
        review = (clean_reviews or {}).get(sample.sample_id)
        if review and review.get("source_key") != sample.source.model_dump(
            include=set(SourceKey.model_fields)
        ):
            review = None
        quality.clean_review = (
            CleanReviewStatus(review["review_status"]) if review else CleanReviewStatus.UNREVIEWED
        )
        sample.tracking_quality = quality
        origin = segments.get(sample.segment_id)
        if origin is None or origin.active:
            continue
        descendants = []
        for candidate in result.segments:
            if not candidate.active or not _within(sample, candidate):
                continue
            ancestor = candidate
            seen = set()
            while ancestor.supersedes_segment_id and ancestor.segment_id not in seen:
                seen.add(ancestor.segment_id)
                if ancestor.supersedes_segment_id == origin.segment_id:
                    descendants.append(candidate)
                    break
                ancestor = segments.get(ancestor.supersedes_segment_id)
                if ancestor is None:
                    break
        if len(descendants) == 1:
            sample.segment_id = descendants[0].segment_id
    return result


def validate_constraints(
    snapshot: GallerySnapshot, constraints: Sequence[ManualConstraint]
) -> None:
    assignments = {
        assignment.segment_id: assignment
        for assignment in snapshot.assignments
        if assignment.status == AssignmentStatus.CONFIRMED
    }
    active = {segment.segment_id for segment in snapshot.evidence.segments if segment.active}
    for constraint in constraints:
        assignment = assignments.get(constraint.segment_id)
        if assignment is None or constraint.segment_id not in active:
            continue
        if (
            constraint.kind == ConstraintKind.FORBID_PLAYER
            and assignment.player_id == constraint.player_id
        ):
            raise ValueError("confirmed assignment violates forbid_player constraint")
        if constraint.kind == ConstraintKind.CANNOT_LINK:
            other = assignments.get(constraint.related_segment_id)
            if other and other.segment_id in active and other.player_id == assignment.player_id:
                raise ValueError("confirmed assignments violate cannot_link constraint")
