"""Build the first explicit human identity anchors through shared ingestion."""

from app.contracts.api import InitializeRequest
from app.workflows.ingest import ingest


def initialize_library(connection, request: InitializeRequest, **kwargs):
    return ingest(connection, request, initialize=True, **kwargs)
