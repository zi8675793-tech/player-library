"""Interface data only; these models do not execute or authorize workflow operations."""

import math
from typing import Annotated, Generic, Literal, Self, TypeVar

from pydantic import Field, JsonValue, field_validator, model_validator

from app.contracts.common import (
    ContractModel,
    Identifier,
    JerseyNumber,
    NonNegativeFloat,
    NonNegativeInt,
    PlayerId,
    PositiveInt,
)
from app.contracts.events import EventOutcome
from app.contracts.evidence import EvidenceBatch
from app.contracts.identity import (
    AssignmentEvidence,
    ConstraintKind,
    Player,
    RiskResolution,
    Segment,
    TemplateReviewStatus,
    UnassignedReason,
)

DataT = TypeVar("DataT")


class SuccessResponse(ContractModel, Generic[DataT]):
    data: DataT
    revision: NonNegativeInt


class ErrorDetail(ContractModel):
    code: Identifier
    message: Identifier
    details: JsonValue = None

    @field_validator("details")
    @classmethod
    def finite_json(cls, value: JsonValue) -> JsonValue:
        def check(item: JsonValue) -> None:
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("details must contain only finite JSON numbers")
            if isinstance(item, dict):
                for child in item.values():
                    check(child)
            elif isinstance(item, list):
                for child in item:
                    check(child)

        check(value)
        return value


class ErrorResponse(ContractModel):
    error: ErrorDetail


class ImportRequest(ContractModel):
    batch: EvidenceBatch
    expected_revision: NonNegativeInt


class SeedIdentity(ContractModel):
    """Explicit human intent; request_id identifies this seed across delivery batches."""

    request_id: Identifier
    display_name: Identifier | None = None
    team_id: Identifier | None = None
    jersey_number: JerseyNumber | None = None
    segment_ids: Annotated[list[Identifier], Field(min_length=1)]
    clean_sample_ids: Annotated[list[Identifier], Field(min_length=1)]
    template_sample_ids: list[Identifier] = Field(default_factory=list)
    actor: Identifier
    reason: Identifier

    @model_validator(mode="after")
    def distinct_support(self) -> Self:
        for values in (self.segment_ids, self.clean_sample_ids, self.template_sample_ids):
            if len(values) != len(set(values)):
                raise ValueError("seed support identifiers must be unique")
        if not set(self.template_sample_ids) <= set(self.clean_sample_ids):
            raise ValueError("seed templates require explicit clean sample approval")
        return self


class InitializeRequest(ImportRequest):
    """Workflow checks the absence of existing initial identity anchors."""

    request_id: Identifier | None = None
    actor: Identifier = "operator"
    reason: Identifier = "initialize"
    seeds: list[SeedIdentity] = Field(default_factory=list)


class UpdateRequest(ImportRequest):
    """The same match may have any number of update batches."""


class EventImportRequest(ImportRequest):
    """Independent event submission; reference resolution remains a workflow concern."""


class MatchingPreviewRequest(ImportRequest):
    """A revision-scoped calculation with no requested persistence."""


class ReviewCommandBase(ContractModel):
    reason: Identifier


class ConfirmAssignmentCommand(ReviewCommandBase):
    kind: Literal["confirm_assignment"] = "confirm_assignment"
    segment_id: Identifier
    player_id: PlayerId
    evidence: list[AssignmentEvidence] = Field(default_factory=list)


class CreatePlayerCommand(ReviewCommandBase):
    kind: Literal["create_player"] = "create_player"
    player: Player
    segment_ids: list[Identifier] = Field(default_factory=list)
    evidence: list[AssignmentEvidence] = Field(default_factory=list)


class SplitSegmentCommand(ReviewCommandBase):
    kind: Literal["split_segment"] = "split_segment"
    segment_id: Identifier
    segments: Annotated[list[Segment], Field(min_length=2)]

    @model_validator(mode="after")
    def valid_replacements(self) -> Self:
        ids = [segment.segment_id for segment in self.segments]
        if len(ids) != len(set(ids)) or self.segment_id in ids:
            raise ValueError("split segments require distinct new segment_id values")
        first = self.segments[0]
        for segment in self.segments:
            if (segment.match_id, segment.clip_id, segment.track_id) != (
                first.match_id,
                first.clip_id,
                first.track_id,
            ):
                raise ValueError("split segments must reference the same match, clip and track")
            if segment.supersedes_segment_id != self.segment_id:
                raise ValueError("split segments must explicitly supersede segment_id")
        ordered = sorted(self.segments, key=lambda segment: segment.start_s)
        if any(left.end_s > right.start_s for left, right in zip(ordered, ordered[1:])):
            raise ValueError("replacement split segments must not overlap")
        framed = [segment for segment in ordered if segment.frame_start is not None]
        if any(
            left.frame_end_exclusive > right.frame_start for left, right in zip(framed, framed[1:])
        ):
            raise ValueError("split frame ranges must not overlap or reverse time order")
        return self


class MergePlayersCommand(ReviewCommandBase):
    kind: Literal["merge_players"] = "merge_players"
    player_id: PlayerId
    alias_player_ids: Annotated[list[PlayerId], Field(min_length=1)]

    @model_validator(mode="after")
    def distinct_players(self) -> Self:
        if self.player_id in self.alias_player_ids:
            raise ValueError("merge target cannot also be an alias player")
        if len(set(self.alias_player_ids)) != len(self.alias_player_ids):
            raise ValueError("alias_player_ids must be unique")
        return self


class SetAssignmentCommand(ReviewCommandBase):
    kind: Literal["set_assignment"] = "set_assignment"
    segment_id: Identifier
    player_id: PlayerId | None
    unassigned_reason: UnassignedReason | None = None
    evidence: list[AssignmentEvidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def unassigned_reason_requires_no_player(self) -> Self:
        if self.player_id is not None and self.unassigned_reason is not None:
            raise ValueError("unassigned_reason requires player_id=null")
        return self


class ReviewTemplateCommand(ReviewCommandBase):
    kind: Literal["review_template"] = "review_template"
    template_id: Identifier
    review_status: TemplateReviewStatus


class ReviewSampleCommand(ReviewCommandBase):
    """Review a sample before a template_id exists, independently of identity."""

    kind: Literal["review_sample"] = "review_sample"
    sample_id: Identifier
    player_id: PlayerId
    feature_space_id: Identifier
    review_status: TemplateReviewStatus


class ReviewRiskCommand(ReviewCommandBase):
    kind: Literal["review_risk"] = "review_risk"
    risk_id: Identifier
    segment_id: Identifier
    resolution: RiskResolution
    clean_sample_ids: list[Identifier] = Field(default_factory=list)


class SetConstraintCommand(ReviewCommandBase):
    kind: Literal["set_constraint"] = "set_constraint"
    constraint_id: Identifier
    segment_id: Identifier
    constraint_kind: ConstraintKind
    player_id: PlayerId | None = None
    related_segment_id: Identifier | None = None

    @model_validator(mode="after")
    def required_targets(self) -> Self:
        if self.constraint_kind == ConstraintKind.FORBID_PLAYER and self.player_id is None:
            raise ValueError("forbid_player requires player_id")
        if self.constraint_kind == ConstraintKind.CANNOT_LINK and (
            self.related_segment_id is None or self.related_segment_id == self.segment_id
        ):
            raise ValueError("cannot_link requires a distinct related_segment_id")
        return self


class RestoreRevisionCommand(ReviewCommandBase):
    kind: Literal["restore_revision"] = "restore_revision"
    target_revision: NonNegativeInt


ReviewCommand = Annotated[
    ConfirmAssignmentCommand
    | CreatePlayerCommand
    | SplitSegmentCommand
    | MergePlayersCommand
    | SetAssignmentCommand
    | ReviewTemplateCommand
    | ReviewSampleCommand
    | ReviewRiskCommand
    | SetConstraintCommand
    | RestoreRevisionCommand,
    Field(discriminator="kind"),
]


class ReviewRequest(ContractModel):
    match_id: Identifier
    expected_revision: NonNegativeInt
    actor: Identifier
    request_id: Identifier | None = None
    commands: Annotated[list[ReviewCommand], Field(min_length=1)]

    @model_validator(mode="after")
    def nested_match_scope(self) -> Self:
        for command in self.commands:
            if isinstance(command, CreatePlayerCommand):
                matches = [command.player.match_id]
            elif isinstance(command, SplitSegmentCommand):
                matches = [segment.match_id for segment in command.segments]
            else:
                continue
            if any(match_id != self.match_id for match_id in matches):
                raise ValueError("review command records must belong to request match_id")
        return self


class PlayerQuery(ContractModel):
    match_id: Identifier
    player_id: PlayerId | None = None
    team_id: Identifier | None = None
    offset: NonNegativeInt = 0
    limit: PositiveInt = 100


class EventQuery(ContractModel):
    match_id: Identifier
    player_id: PlayerId | None = None
    clip_id: Identifier | None = None
    event_type: Identifier | None = None
    outcome: EventOutcome | None = None
    role: Identifier | None = None
    start_s: NonNegativeFloat | None = None
    end_s: NonNegativeFloat | None = None
    offset: NonNegativeInt = 0
    limit: PositiveInt = 100

    @model_validator(mode="after")
    def valid_time_filter(self) -> Self:
        if self.start_s is not None and self.end_s is not None and self.end_s <= self.start_s:
            raise ValueError("event query end_s must be greater than start_s")
        return self


class ImportReport(ContractModel):
    match_id: Identifier
    batch_id: Identifier
    inserted_count: NonNegativeInt
    duplicate_count: NonNegativeInt
    review_count: NonNegativeInt
    warnings: list[Identifier] = Field(default_factory=list)


class AssetReference(ContractModel):
    """Public responses expose a controlled identifier instead of a local path."""

    asset_id: Identifier
    media_type: Identifier
