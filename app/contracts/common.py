"""Shared scalar and provenance rules; never consult external state."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictStr, StringConstraints, model_validator

Identifier = Annotated[StrictStr, StringConstraints(min_length=1, pattern=r"\S")]
PlayerId = Annotated[StrictStr, StringConstraints(pattern=r"^pl_\S+$")]
JerseyNumber = Identifier
Sha256 = Annotated[StrictStr, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
NonNegativeFloat = Annotated[FiniteFloat, Field(ge=0)]
PositiveFloat = Annotated[FiniteFloat, Field(gt=0)]
Probability = Annotated[FiniteFloat, Field(ge=0, le=1)]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TimeRange(ContractModel):
    """A nonempty half-open range in clip seconds."""

    start_s: NonNegativeFloat
    end_s: NonNegativeFloat

    @model_validator(mode="after")
    def ordered_times(self) -> "TimeRange":
        if self.end_s <= self.start_s:
            raise ValueError("end_s must be greater than start_s for [start_s, end_s)")
        return self


class SegmentRange(TimeRange):
    frame_start: NonNegativeInt | None = None
    frame_end_exclusive: PositiveInt | None = None

    @model_validator(mode="after")
    def ordered_frames(self) -> "SegmentRange":
        if (self.frame_start is None) != (self.frame_end_exclusive is None):
            raise ValueError("frame_start and frame_end_exclusive must be provided together")
        if self.frame_start is not None and self.frame_end_exclusive <= self.frame_start:
            raise ValueError("frame_end_exclusive must be greater than frame_start")
        return self


class SourceKey(ContractModel):
    producer: Identifier
    record_type: Identifier
    source_record_id: Identifier
    evidence_version: Identifier


class SourceLocation(ContractModel):
    """A location within a declared source file, without opening that file."""

    file_id: Identifier
    line_number: PositiveInt | None = None
    record_key: Identifier | None = None


class Provenance(SourceKey):
    """An opaque source version and optional SHA-256 content fingerprint."""

    content_hash: Sha256 | None = None
    supersedes: SourceKey | None = None
    locations: list[SourceLocation] = Field(default_factory=list)

    @model_validator(mode="after")
    def distinct_version(self) -> "Provenance":
        current = SourceKey.model_validate(self.model_dump(include=set(SourceKey.model_fields)))
        if self.supersedes == current:
            raise ValueError("supersedes must identify a different source version")
        return self
