"""Shared, read-only gallery qualification checks.

A template approval is a human sample-level review. It never establishes the
identity of its segment; that must come from a current confirmed Assignment.
"""

from collections.abc import Mapping

from app.contracts.evidence import (
    EvidenceBatch,
    GallerySnapshot,
    InlineFeature,
    PreparedEvidence,
    PreparedSample,
)
from app.contracts.identity import (
    AssignmentStatus,
    MatchingPolicy,
    Template,
    TemplateReviewStatus,
)
from app.identity.prepare import prepare_evidence


def prepare_gallery(
    snapshot: GallerySnapshot,
    policy: MatchingPolicy,
    feature_values: Mapping[tuple[str, int], InlineFeature] | None = None,
) -> PreparedEvidence:
    """Validate and prepare the snapshot's evidence without changing its membership."""
    return prepare_evidence(
        EvidenceBatch(
            match_id=snapshot.match_id,
            batch_id="gallery-snapshot",
            producer="identity.gallery",
            identity_evidence=snapshot.evidence,
            assets=snapshot.assets,
        ),
        policy,
        feature_values=feature_values,
        risk_reviews=snapshot.risk_reviews,
    )


def eligible_template(
    template: Template,
    snapshot: GallerySnapshot,
    policy: MatchingPolicy,
    feature_values: Mapping[tuple[str, int], InlineFeature] | None = None,
    *,
    prepared_gallery: PreparedEvidence | None = None,
) -> tuple[PreparedSample | None, list[str]]:
    """Return the approved sample or explicit reasons it cannot be a reference.

    All references and current identity are checked, even for manually approved
    templates. Malformed features are errors from preparation, never exclusions.
    """
    reasons: list[str] = []
    if template.match_id != snapshot.match_id:
        raise ValueError("template and gallery must belong to the same match")
    if not template.active:
        reasons.append("inactive_template")
    if template.review_status != TemplateReviewStatus.APPROVED:
        reasons.append("template_not_approved")
    if template.feature_space_id != policy.feature_space_id:
        reasons.append("incompatible_feature_space")
    players = {player.player_id: player for player in snapshot.players}
    player = players.get(template.player_id)
    if player is None or player.alias_of is not None:
        reasons.append("missing_or_merged_player")
    samples = {sample.sample_id: sample for sample in snapshot.evidence.samples}
    sample = samples.get(template.sample_id)
    if sample is None:
        return None, [*reasons, "missing_template_sample"]
    prepared = prepared_gallery or prepare_gallery(snapshot, policy, feature_values)
    matches = [
        segment
        for segment in prepared.segments
        if any(item.sample.sample_id == sample.sample_id for item in segment.samples)
    ]
    if len(matches) != 1:
        excluded = [
            item.reason
            for segment in prepared.segments
            for item in segment.exclusions
            if item.sample_id == sample.sample_id
        ]
        return None, [*reasons, *(excluded or ["unresolved_template_segment"])]
    segment = matches[0]
    if not segment.segment.active:
        reasons.append("inactive_template_segment")
    if segment.risks:
        reasons.append("template_segment_risk")
    quality = sample.tracking_quality
    if quality is not None and quality.occluded is True:
        reasons.append("occluded_template")
    assignments = {assignment.segment_id: assignment for assignment in snapshot.assignments}
    assignment = assignments.get(segment.segment.segment_id)
    if (
        assignment is None
        or assignment.status != AssignmentStatus.CONFIRMED
        or assignment.player_id != template.player_id
    ):
        reasons.append("template_assignment_not_confirmed")
    if reasons:
        return None, sorted(set(reasons))
    return next(item for item in segment.samples if item.sample.sample_id == sample.sample_id), []
