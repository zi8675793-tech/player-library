"""Original event evidence and separate, revision-dependent subject resolutions.

These models describe references without looking up their targets. In particular,
an event may arrive before its clips, tracks, or identity segments are registered.
"""

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from app.contracts.common import (
    ContractModel,
    FiniteFloat,
    Identifier,
    JerseyNumber,
    NonNegativeFloat,
    NonNegativeInt,
    PlayerId,
    Provenance,
)


class PointEventTime(ContractModel):
    """A single instant in seconds, without a fabricated interval endpoint."""

    kind: Literal["point"] = "point"
    time_s: NonNegativeFloat


class IntervalEventTime(ContractModel):
    """A nonempty, left-closed and right-open interval in seconds."""

    kind: Literal["interval"] = "interval"
    start_s: NonNegativeFloat
    end_s: NonNegativeFloat

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.end_s <= self.start_s:
            raise ValueError("end_s must be greater than start_s for an interval event")
        return self


EventTime = Annotated[PointEventTime | IntervalEventTime, Field(discriminator="kind")]


class TrackReference(ContractModel):
    """Support from a raw track, optionally localized within its own clip.

    If time is absent, the association workflow uses the event's time. This
    reference asserts neither that the track exists nor that it has one identity.
    """

    kind: Literal["track"] = "track"
    clip_id: Identifier
    track_id: Identifier
    time: EventTime | None = None


class SegmentReference(ContractModel):
    """Support from an identity segment that may not have arrived yet."""

    kind: Literal["segment"] = "segment"
    segment_id: Identifier
    time: EventTime | None = None


SubjectReference = Annotated[TrackReference | SegmentReference, Field(discriminator="kind")]


class Subject(ContractModel):
    """One independently resolved role; repeated roles and custom roles are valid."""

    subject_id: Identifier
    role: Identifier
    references: list[SubjectReference] = Field(default_factory=list)
    event_jersey_number: JerseyNumber | None = None
    event_team_id: Identifier | None = None


class EventOutcome(StrEnum):
    MADE = "made"
    MISSED = "missed"
    UNKNOWN = "unknown"


class ClockDirection(StrEnum):
    COUNT_UP = "count_up"
    COUNT_DOWN = "count_down"


class GameClock(ContractModel):
    """Competition time, independent of clip and source-video playback time."""

    period: Identifier
    clock_s: NonNegativeFloat
    direction: ClockDirection | None = None


class Event(ContractModel):
    """Immutable-source event facts, with no stored current player assignment.

    ``outcome=None`` means no result was supplied or the result is inapplicable;
    ``unknown`` can explicitly state an unknown applicable result. A shot does
    not imply ``made``. Scores and points are never inferred by this contract.
    """

    match_id: Identifier
    event_id: Identifier
    source_event_key: Identifier
    canonical_event_id: Identifier | None = None
    clip_id: Identifier
    event_type: Identifier
    time: EventTime
    outcome: EventOutcome | None = None
    points: NonNegativeInt | None = None
    source: Provenance
    raw_scores: dict[Identifier, FiniteFloat] = Field(default_factory=dict)
    subjects: list[Subject] = Field(default_factory=list)
    game_clock: GameClock | None = None

    @model_validator(mode="after")
    def validate_subject_ids(self) -> Self:
        subject_ids = [subject.subject_id for subject in self.subjects]
        if len(subject_ids) != len(set(subject_ids)):
            raise ValueError("subject_id must be unique within an event")
        return self


class SubjectMappingStatus(StrEnum):
    MAPPED = "mapped"
    UNMAPPED = "unmapped"


class ResolvedSubject(ContractModel):
    """A query-time result, separate from the subject's original references."""

    subject_id: Identifier
    role: Identifier
    status: SubjectMappingStatus
    player_id: PlayerId | None = None
    reason: Identifier | None = None

    @model_validator(mode="after")
    def validate_player_mapping(self) -> Self:
        if self.status == SubjectMappingStatus.MAPPED and self.player_id is None:
            raise ValueError("mapped subjects require player_id")
        if self.status == SubjectMappingStatus.UNMAPPED and self.player_id is not None:
            raise ValueError("unmapped subjects must not have player_id")
        return self


class SourceEventTime(ContractModel):
    """Derived source-video positioning, available only with an explicit mapping."""

    source_video_id: Identifier
    time: EventTime


class ResolvedEvent(ContractModel):
    """An event and the current resolution of every independent subject."""

    event: Event
    subjects: list[ResolvedSubject]
    validation_reasons: list[Identifier] = Field(default_factory=list)
    source_time: SourceEventTime | None = None

    @model_validator(mode="after")
    def validate_subject_correspondence(self) -> Self:
        expected = {subject.subject_id: subject.role for subject in self.event.subjects}
        resolved = {subject.subject_id: subject.role for subject in self.subjects}
        if len(resolved) != len(self.subjects) or resolved != expected:
            raise ValueError("resolved subjects must match every original subject_id and role")
        if self.source_time is not None and self.source_time.time.kind != self.event.time.kind:
            raise ValueError("source time must preserve the event's point or interval kind")
        return self
