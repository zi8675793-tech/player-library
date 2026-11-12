"""Revision-scoped Python workflow results, shared by examples and future transports."""

from pydantic import Field

from app.contracts.api import ImportReport
from app.contracts.identity import AssociationResult, GalleryPlan, Resolution


class WorkflowReport(ImportReport):
    changed: bool = False
    initialized: bool = False
    created_player_ids: list[str] = Field(default_factory=list)
    based_revision: int
    matching_revision: int
    associations: list[AssociationResult] = Field(default_factory=list)
    resolutions: list[Resolution] = Field(default_factory=list)
    gallery_plans: list[GalleryPlan] = Field(default_factory=list)
