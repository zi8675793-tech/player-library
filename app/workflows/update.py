"""Incremental evidence and independent event imports share transaction semantics."""

from app.contracts.api import EventImportRequest, UpdateRequest
from app.workflows.incremental import update_incremental
from app.workflows.ingest import ingest


def update_library(connection, request: UpdateRequest, **kwargs):
    if kwargs.get("incremental_discovery") or (
        kwargs.get("segmentation_report") is not None
        and kwargs.get("settings") is not None
        and getattr(kwargs["settings"].input, "adapter", None) == "machine"
    ):
        return update_incremental(connection, request, **kwargs)
    return ingest(connection, request, **kwargs)


def ingest_events(connection, request: EventImportRequest, **kwargs):
    return ingest(connection, request, events_only=True, **kwargs)


__all__ = ["ingest_events", "update_incremental", "update_library"]
