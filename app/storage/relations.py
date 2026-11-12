"""Checks for complete formal references; raw evidence may arrive separately.

These are transaction-local primitives, not identity or review decisions. Call
inside the same write transaction as the corresponding formal record.
"""

import sqlite3

from app.contracts.evidence import FeatureFile, FileFeatureReference, Sample
from app.contracts.identity import (
    Assignment,
    IdentityRisk,
    ManualConstraint,
    ManualTemplateReview,
    RiskReview,
    Segment,
    Template,
)


class MissingReference(sqlite3.IntegrityError, ValueError):
    """A formal record refers to absent or incompatible same-match evidence."""


def _lookup(
    connection: sqlite3.Connection, table: str, match_id: str, column: str, value: str
) -> str:
    # table and column are fixed literals supplied only by this module.
    row = connection.execute(
        f"SELECT json FROM {table} WHERE match_id=? AND {column}=?", (match_id, value)
    ).fetchone()
    if row is None:
        raise MissingReference(f"{table} {value!r} not found in match {match_id!r}")
    return row[0]


def check_assignment_references(connection: sqlite3.Connection, assignment: Assignment) -> None:
    segment = Segment.model_validate_json(
        _lookup(connection, "segments", assignment.match_id, "segment_id", assignment.segment_id)
    )
    row = connection.execute(
        "SELECT 1 FROM tracks WHERE match_id=? AND clip_id=? AND track_id=?",
        (assignment.match_id, segment.clip_id, segment.track_id),
    ).fetchone()
    if row is None:
        raise MissingReference("assignment segment's track is missing in this match")
    if assignment.player_id is not None:
        _lookup(connection, "players", assignment.match_id, "player_id", assignment.player_id)


def _check_template_references(
    connection: sqlite3.Connection, template: Template | ManualTemplateReview
) -> None:
    match = template.match_id
    _lookup(connection, "players", match, "player_id", template.player_id)
    _lookup(connection, "feature_spaces", match, "feature_space_id", template.feature_space_id)
    sample = Sample.model_validate_json(
        _lookup(connection, "samples", match, "sample_id", template.sample_id)
    )
    if sample.segment_id is not None:
        segment = Segment.model_validate_json(
            _lookup(connection, "segments", match, "segment_id", sample.segment_id)
        )
        if (sample.clip_id, sample.track_id) != (segment.clip_id, segment.track_id) or not (
            segment.start_s <= sample.time_s < segment.end_s
        ):
            raise MissingReference("template sample is outside its referenced segment")
    features = [f for f in sample.features if f.feature_space_id == template.feature_space_id]
    if not features:
        raise MissingReference("template sample does not declare the referenced feature space")
    feature = features[0]
    if isinstance(feature, FileFeatureReference):
        row = connection.execute(
            "SELECT json FROM feature_files WHERE match_id=? AND feature_file_id=?",
            (match, feature.feature_file_id),
        ).fetchone()
        if row is None:
            raise MissingReference("template feature file is missing")
        file = FeatureFile.model_validate_json(row[0])
        if (
            file.feature_space_id != feature.feature_space_id
            or feature.row_index >= file.sample_count
        ):
            raise MissingReference("template feature file space or row is incompatible")


def check_template_references(connection: sqlite3.Connection, template: Template) -> None:
    _check_template_references(connection, template)


def check_template_review_references(
    connection: sqlite3.Connection, review: ManualTemplateReview
) -> None:
    """A template review addresses the same complete sample/space/player relation."""
    _check_template_references(connection, review)


def check_risk_review_references(connection: sqlite3.Connection, review: RiskReview) -> None:
    match = review.match_id
    risk = IdentityRisk.model_validate_json(
        _lookup(connection, "risks", match, "risk_id", review.risk_id)
    )
    segment = Segment.model_validate_json(
        _lookup(connection, "segments", match, "segment_id", review.segment_id)
    )
    if (
        (risk.segment_id is not None and risk.segment_id != review.segment_id)
        or (risk.clip_id is not None and risk.clip_id != segment.clip_id)
        or (risk.track_id is not None and risk.track_id != segment.track_id)
    ):
        raise MissingReference("risk review segment does not match the risk's explicit scope")
    if risk.sample_id is not None:
        _lookup(connection, "samples", match, "sample_id", risk.sample_id)
    for sample_id in review.clean_sample_ids:
        _lookup(connection, "samples", match, "sample_id", sample_id)


def check_constraint_references(
    connection: sqlite3.Connection, constraint: ManualConstraint
) -> None:
    match = constraint.match_id
    _lookup(connection, "segments", match, "segment_id", constraint.segment_id)
    if constraint.related_segment_id is not None:
        _lookup(connection, "segments", match, "segment_id", constraint.related_segment_id)
    if constraint.player_id is not None:
        _lookup(connection, "players", match, "player_id", constraint.player_id)
