"""Typed current-state views returned by the Python workflow read API."""

from typing import Literal

from pydantic import Field, JsonValue, StrictBool

from app.contracts.common import (
    ContractModel,
    Identifier,
    JerseyNumber,
    NonNegativeFloat,
    NonNegativeInt,
)
from app.contracts.identity import Candidate, PersonRole, Player, PlayerId, Template


class ReviewActionSuggestion(ContractModel):
    """Structured action proposal for one-click human review in web UI / API."""

    action: Literal[
        "confirm_assignment",
        "create_player",
        "split_segment",
        "merge_players",
        "review_risk",
        "review_template",
        "set_assignment",
        "set_player_role",
        "dismiss_risk",
    ]
    label: str
    description: str | None = None
    target_player_id: PlayerId | None = None
    target_role: PersonRole | None = None


class ReviewQueueItem(ContractModel):
    """Actionable review item with complete spatial-temporal coordinates, risks, and one-click actions."""

    kind: Literal["identity", "risk", "template", "event"]
    segment_id: Identifier | None = None
    clip_id: Identifier | None = None
    track_id: Identifier | None = None
    start_s: NonNegativeFloat | None = None
    end_s: NonNegativeFloat | None = None
    event_id: Identifier | None = None
    sample_id: Identifier | None = None
    feature_space_id: Identifier | None = None
    reasons: list[Identifier] = Field(default_factory=list)
    risk_kinds: list[Identifier] = Field(default_factory=list)
    review_priority: Literal["critical", "high", "medium", "low"] = "medium"
    candidates: list[Candidate] = Field(default_factory=list)
    thumbnail_asset_ids: list[Identifier] = Field(default_factory=list)
    suggested_actions: list[ReviewActionSuggestion] = Field(default_factory=list)
    details: dict[str, JsonValue] = Field(default_factory=dict)


class PlayerView(ContractModel):
    """Profile view; role unknown is presented as anonymous profile, not formal player."""

    player: Player
    confirmed_segment_ids: list[Identifier] = Field(default_factory=list)
    templates: list[Template] = Field(default_factory=list)
    made_count: NonNegativeInt = 0
    is_formal_player: StrictBool = False
    profile_type: Identifier = "anonymous_identity_profile"


class PlayerStatsView(ContractModel):
    """Official formal player statistics view (excludes anonymous or non-player entities)."""

    match_id: Identifier
    player_id: PlayerId
    role: PersonRole
    display_name: Identifier | None = None
    jersey_number: JerseyNumber | None = None
    confirmed_segments_count: NonNegativeInt = 0
    approved_templates_count: NonNegativeInt = 0
    official_made_count: NonNegativeInt = 0
    is_formal_player: StrictBool = True
