"""Identity records and plans; reference existence is checked by workflows."""

import hashlib
import json
from enum import StrEnum
from typing import Literal, Self

from pydantic import ConfigDict, Field, StrictBool, model_validator

from app.contracts.common import (
    ContractModel,
    Identifier,
    JerseyNumber,
    NonNegativeFloat,
    NonNegativeInt,
    PlayerId,
    PositiveFloat,
    PositiveInt,
    Probability,
    Provenance,
    SegmentRange,
    Sha256,
)


class AssignmentStatus(StrEnum):
    UNASSIGNED = "unassigned"
    PROPOSED = "proposed"
    CONFIRMED = "confirmed"


class AssignmentMethod(StrEnum):
    MANUAL = "manual"
    AUTOMATIC = "automatic"
    SEED = "seed"
    IMPORTED = "imported"


class UnassignedReason(StrEnum):
    UNKNOWN = "unknown"
    OTHER_PERSON = "other_person"
    AMBIGUOUS = "ambiguous"
    REJECTED = "rejected"


class TemplateReviewStatus(StrEnum):
    UNREVIEWED = "unreviewed"
    APPROVED = "approved"
    REJECTED = "rejected"


class ResolutionDecision(StrEnum):
    MATCH_EXISTING = "match_existing"
    CREATE_NEW_PROPOSAL = "create_new_proposal"
    PENDING = "pending"
    REJECTED = "rejected"


class Segment(SegmentRange):
    """An identity interval, distinct from the upstream raw track it references."""

    segment_id: Identifier
    match_id: Identifier
    clip_id: Identifier
    track_id: Identifier
    active: StrictBool = True
    supersedes_segment_id: Identifier | None = None

    @model_validator(mode="after")
    def no_self_replacement(self) -> Self:
        if self.supersedes_segment_id == self.segment_id:
            raise ValueError("supersedes_segment_id must reference a different segment")
        return self


class Player(ContractModel):
    player_id: PlayerId
    match_id: Identifier
    display_name: Identifier | None = None
    team_id: Identifier | None = None
    jersey_number: JerseyNumber | None = None
    alias_of: PlayerId | None = None

    @model_validator(mode="after")
    def no_self_alias(self) -> Self:
        if self.alias_of == self.player_id:
            raise ValueError("alias_of must reference a different player")
        return self


class AssignmentEvidence(ContractModel):
    """A reason with an optional reference; it does not establish reference existence."""

    kind: Identifier
    reference_id: Identifier | None = None
    description: Identifier | None = None
    feature_space_id: Identifier | None = None
    distance: NonNegativeFloat | None = None

    @model_validator(mode="after")
    def distance_has_space(self) -> Self:
        if self.distance is not None and self.feature_space_id is None:
            raise ValueError("distance requires feature_space_id")
        return self


class Assignment(ContractModel):
    match_id: Identifier
    segment_id: Identifier
    player_id: PlayerId | None = None
    status: AssignmentStatus = AssignmentStatus.UNASSIGNED
    method: AssignmentMethod
    unassigned_reason: UnassignedReason | None = None
    evidence: list[AssignmentEvidence] = Field(default_factory=list)
    revision: NonNegativeInt

    @model_validator(mode="after")
    def status_matches_player(self) -> Self:
        if self.status == AssignmentStatus.UNASSIGNED and self.player_id is not None:
            raise ValueError("unassigned status requires player_id=null")
        if self.status != AssignmentStatus.UNASSIGNED and self.player_id is None:
            raise ValueError("proposed or confirmed status requires player_id")
        if self.status != AssignmentStatus.UNASSIGNED and self.unassigned_reason is not None:
            raise ValueError("unassigned_reason is only valid for unassigned status")
        return self


class Template(ContractModel):
    """Gallery membership references a sample in one comparable feature space."""

    template_id: Identifier
    match_id: Identifier
    player_id: PlayerId
    feature_space_id: Identifier
    sample_id: Identifier
    review_status: TemplateReviewStatus = TemplateReviewStatus.UNREVIEWED
    active: StrictBool = True
    revision: NonNegativeInt
    approval: "ReviewRecord | None" = None


class Candidate(ContractModel):
    match_id: Identifier
    segment_id: Identifier
    player_id: PlayerId
    feature_space_id: Identifier
    distance: NonNegativeFloat
    common_parts: PositiveInt
    gallery_revision: NonNegativeInt
    template_ids: list[Identifier] = Field(default_factory=list)
    score_kind: Literal["double_template", "single_template_preview"] = "double_template"
    query_sample_ids: list[Identifier] = Field(default_factory=list)


class Resolution(ContractModel):
    """A proposed decision; neither a candidate nor a new-ID proposal confirms identity."""

    match_id: Identifier
    segment_id: Identifier
    decision: ResolutionDecision
    player_id: PlayerId | None = None
    evidence: list[AssignmentEvidence] = Field(default_factory=list)
    assignment: Assignment | None = None
    reasons: list[Identifier] = Field(default_factory=list)
    requires_review: StrictBool = True
    automatic_confirmation_eligible: StrictBool = False
    feature_space_id: Identifier | None = None
    policy_version: Identifier | None = None
    gallery_revision: NonNegativeInt | None = None

    @model_validator(mode="after")
    def decision_matches_player(self) -> Self:
        if self.decision == ResolutionDecision.MATCH_EXISTING:
            if self.player_id is None:
                raise ValueError("match_existing requires player_id")
        elif self.player_id is not None:
            raise ValueError("only match_existing may provide player_id")
        if self.decision == ResolutionDecision.CREATE_NEW_PROPOSAL and not self.evidence:
            raise ValueError("create_new_proposal requires independent supporting evidence")
        if self.assignment is not None:
            if (self.assignment.match_id, self.assignment.segment_id) != (
                self.match_id,
                self.segment_id,
            ):
                raise ValueError("resolution assignment must have the same match and segment")
            if (
                self.decision == ResolutionDecision.MATCH_EXISTING
                and self.assignment.player_id != self.player_id
            ):
                raise ValueError("resolution and assignment player_id must agree")
        if self.automatic_confirmation_eligible and (
            self.decision != ResolutionDecision.MATCH_EXISTING
            or self.assignment is None
            or self.assignment.status != AssignmentStatus.CONFIRMED
            or self.assignment.method != AssignmentMethod.AUTOMATIC
            or self.requires_review
        ):
            raise ValueError("automatic confirmation requires an accepted automatic assignment")
        return self


class ReviewRecord(ContractModel):
    """Previously accepted human review context, not an executable review command."""

    actor: Identifier
    reason: Identifier
    revision: NonNegativeInt


class RiskKind(StrEnum):
    MIXED_IDENTITY = "mixed_identity"
    IDENTITY_SWITCH = "identity_switch"
    DUPLICATE_BOX = "duplicate_box"
    QUALITY_REVIEW = "quality_review"
    IDENTITY_CONFLICT = "identity_conflict"


class IdentityRisk(ContractModel):
    risk_id: Identifier
    kind: RiskKind
    match_id: Identifier
    segment_id: Identifier | None = None
    clip_id: Identifier | None = None
    track_id: Identifier | None = None
    sample_id: Identifier | None = None
    start_s: NonNegativeFloat | None = None
    end_s: NonNegativeFloat | None = None
    source: Provenance | None = None

    @model_validator(mode="after")
    def scoped_risk(self) -> Self:
        if not (self.segment_id or self.sample_id or (self.clip_id and self.track_id)):
            raise ValueError("risk requires segment, sample, or clip/track scope")
        if (self.start_s is None) != (self.end_s is None):
            raise ValueError("risk time bounds must be supplied together")
        if self.start_s is not None and self.end_s <= self.start_s:
            raise ValueError("risk end_s must be greater than start_s")
        return self


class RiskResolution(StrEnum):
    CLEARED = "cleared"
    CONFIRMED = "confirmed"


class RiskReview(ReviewRecord):
    risk_id: Identifier
    match_id: Identifier
    segment_id: Identifier
    resolution: RiskResolution
    clean_sample_ids: list[Identifier] = Field(default_factory=list)


class SampleExclusion(ContractModel):
    sample_id: Identifier
    reason: Identifier
    duplicate_of: Identifier | None = None


class Calibration(ContractModel):
    status: Literal["unvalidated", "validated"] = "unvalidated"
    dataset_version: Identifier
    policy_fingerprint: Sha256


class MatchingPolicy(ContractModel):
    """All matching parameters, including numerical validation tolerances.

    A validated calibration is an explicit external assertion bound to the exact
    policy fingerprint. Constructing it is not evidence that calibration occurred.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)

    feature_space_id: Identifier
    policy_version: Identifier = "identity-v1"
    distance_version: Literal["kpr_sqrt_visibility_l2_div2_v1"] = "kpr_sqrt_visibility_l2_div2_v1"
    aggregation_version: Literal["nearest_two_mean_segment_median_v1"] = (
        "nearest_two_mean_segment_median_v1"
    )
    template_admission_version: Literal["manual_approved_confirmed_v1"] = (
        "manual_approved_confirmed_v1"
    )
    min_common_parts: PositiveInt = 2
    max_templates_per_player: PositiveInt = 6
    auto_assign_enabled: StrictBool = False
    sample_bucket_s: PositiveFloat = 0.5
    min_query_buckets: PositiveInt = 3
    min_common_query_coverage: Probability = 0.8
    min_support_rate: Probability = 0.8
    l2_rtol: NonNegativeFloat = 1e-4
    l2_atol: NonNegativeFloat = 1e-5
    visibility_weight_epsilon: PositiveFloat = 1e-12
    tau: NonNegativeFloat | None = None
    delta: NonNegativeFloat | None = None
    calibration: Calibration | None = None

    @property
    def fingerprint(self) -> str:
        values = self.model_dump(mode="json", exclude={"calibration", "auto_assign_enabled"})
        encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def calibrated(self) -> bool:
        return bool(
            self.tau is not None
            and self.delta is not None
            and self.calibration is not None
            and self.calibration.status == "validated"
            and self.calibration.policy_fingerprint == self.fingerprint
        )

    @model_validator(mode="after")
    def valid_calibration(self) -> Self:
        if (self.tau is None) != (self.delta is None):
            raise ValueError("tau and delta must be configured together")
        if self.calibration is not None and self.calibration.status == "validated":
            if not self.calibrated:
                raise ValueError(
                    "validated calibration must match policy fingerprint and thresholds"
                )
        if self.auto_assign_enabled and not self.calibrated:
            raise ValueError("auto_assign_enabled requires an applicable validated calibration")
        return self


class DistanceResult(ContractModel):
    comparable: StrictBool
    distance: NonNegativeFloat | None = None
    common_parts: NonNegativeInt = 0
    reason: Identifier | None = None

    @model_validator(mode="after")
    def explicit_comparability(self) -> Self:
        if self.comparable != (self.distance is not None):
            raise ValueError("only comparable results have a finite distance")
        if not self.comparable and self.reason is None:
            raise ValueError("incomparable result requires a reason")
        return self


class TemplateComparison(ContractModel):
    template_id: Identifier
    sample_id: Identifier
    distance: NonNegativeFloat | None = None
    common_parts: NonNegativeInt = 0
    reason: Identifier | None = None


class SamplePlayerScore(ContractModel):
    sample_id: Identifier
    player_id: PlayerId
    distance: NonNegativeFloat | None = None
    preview_distance: NonNegativeFloat | None = None
    template_ids: list[Identifier] = Field(default_factory=list)
    comparisons: list[TemplateComparison] = Field(default_factory=list)
    common_parts: NonNegativeInt = 0
    reason: Identifier | None = None


class SampleDiagnostic(ContractModel):
    sample_id: Identifier
    bucket_index: NonNegativeInt
    region: NonNegativeInt = Field(le=2)
    representative: StrictBool
    scores: list[SamplePlayerScore] = Field(default_factory=list)
    supported_player_id: PlayerId | None = None
    reason: Identifier | None = None


class AssociationResult(ContractModel):
    match_id: Identifier
    segment_id: Identifier
    feature_space_id: Identifier
    policy_version: Identifier
    policy_fingerprint: Sha256
    gallery_revision: NonNegativeInt
    candidates: list[Candidate] = Field(default_factory=list)
    local_candidates: list[Candidate] = Field(default_factory=list)
    sample_diagnostics: list[SampleDiagnostic] = Field(default_factory=list)
    common_query_sample_ids: list[Identifier] = Field(default_factory=list)
    common_query_coverage: Probability = 0.0
    common_bucket_count: NonNegativeInt = 0
    covered_regions: list[NonNegativeInt] = Field(default_factory=list)
    first_player_id: PlayerId | None = None
    first_distance: NonNegativeFloat | None = None
    second_distance: NonNegativeFloat | None = None
    margin: NonNegativeFloat | None = None
    support_rate: Probability | None = None
    strong_disagreement_sample_ids: list[Identifier] = Field(default_factory=list)
    blocking_reasons: list[Identifier] = Field(default_factory=list)
    template_ids: list[Identifier] = Field(default_factory=list)
    gallery_conflict_template_ids: list[Identifier] = Field(default_factory=list)


class ConstraintKind(StrEnum):
    FORBID_PLAYER = "forbid_player"
    CANNOT_LINK = "cannot_link"
    BLOCK_AUTOMATIC = "block_automatic"


class ManualConstraint(ReviewRecord):
    constraint_id: Identifier
    match_id: Identifier
    segment_id: Identifier
    kind: ConstraintKind
    player_id: PlayerId | None = None
    related_segment_id: Identifier | None = None

    @model_validator(mode="after")
    def constraint_target(self) -> Self:
        if self.kind == ConstraintKind.FORBID_PLAYER and self.player_id is None:
            raise ValueError("forbid_player requires player_id")
        if self.kind == ConstraintKind.CANNOT_LINK:
            if self.related_segment_id is None or self.related_segment_id == self.segment_id:
                raise ValueError("cannot_link requires a distinct related_segment_id")
        return self


class NewIdentityRequest(ReviewRecord):
    """Accepted human intent to propose a new identity; no final player_id."""

    match_id: Identifier
    segment_id: Identifier
    support_sample_ids: list[Identifier] = Field(min_length=1)


class ManualTemplateReview(ReviewRecord):
    match_id: Identifier
    sample_id: Identifier
    player_id: PlayerId
    feature_space_id: Identifier
    review_status: TemplateReviewStatus
    template_id: Identifier | None = None


class GalleryAction(StrEnum):
    ACTIVATE = "activate"
    DEACTIVATE = "deactivate"
    AWAIT_REVIEW = "await_review"
    REPLACEMENT_REQUIRED = "replacement_required"


class GalleryChange(ContractModel):
    action: GalleryAction
    sample_id: Identifier
    player_id: PlayerId
    feature_space_id: Identifier
    template_id: Identifier | None = None
    reasons: list[Identifier] = Field(default_factory=list)
    review: ManualTemplateReview | None = None
    replacement_template_ids: list[Identifier] = Field(default_factory=list)


class GalleryPlan(ContractModel):
    match_id: Identifier
    gallery_revision: NonNegativeInt
    feature_space_id: Identifier
    policy_version: Identifier
    changes: list[GalleryChange] = Field(default_factory=list)


Template.model_rebuild()
