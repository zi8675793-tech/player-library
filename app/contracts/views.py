"""Typed current-state views returned by the Python workflow read API."""

from typing import Literal

from pydantic import Field, JsonValue

from app.contracts.common import ContractModel, Identifier, NonNegativeInt
from app.contracts.identity import Candidate, Player, Template


class PlayerView(ContractModel):
    player: Player
    confirmed_segment_ids: list[Identifier] = Field(default_factory=list)
    templates: list[Template] = Field(default_factory=list)
    made_count: NonNegativeInt = 0


class ReviewQueueItem(ContractModel):
    kind: Literal["identity", "risk", "template", "event"]
    segment_id: Identifier | None = None
    event_id: Identifier | None = None
    sample_id: Identifier | None = None
    feature_space_id: Identifier | None = None
    reasons: list[Identifier] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)
    details: dict[str, JsonValue] = Field(default_factory=dict)
