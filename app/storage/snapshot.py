"""Consistent, match-scoped identity facts for the pure identity algorithms."""

import sqlite3
from dataclasses import dataclass, field

from pydantic import BaseModel

from app.contracts.evidence import (
    FeatureFile,
    FeatureSpace,
    GallerySnapshot,
    IdentityEvidence,
    InlineFeature,
    MediaAsset,
    Sample,
    Track,
    TrackObservation,
)
from app.contracts.identity import (
    Assignment,
    IdentityRisk,
    ManualConstraint,
    ManualTemplateReview,
    Player,
    RiskReview,
    Segment,
    Template,
)
from app.contracts.segmentation import SegmentProposal
from app.storage.features import load_feature_values, load_inline_features


@dataclass(frozen=True)
class IdentitySnapshot:
    """Facts read at one SQLite snapshot; no algorithm decision is made here."""

    gallery: GallerySnapshot
    feature_values: dict[tuple[str, int], InlineFeature]
    constraints: list[ManualConstraint]
    template_reviews: list[ManualTemplateReview]
    segment_proposals: dict[str, SegmentProposal] = field(default_factory=dict)
    proposal_evidence_fingerprints: dict[str, str | None] = field(default_factory=dict)


def _records(connection: sqlite3.Connection, sql: str, match_id: str, model: type[BaseModel]):
    return [model.model_validate_json(row[0]) for row in connection.execute(sql, (match_id,))]


def read_identity_snapshot(connection: sqlite3.Connection, match_id: str) -> IdentitySnapshot:
    """Read revision, gallery, features and human constraints from one read transaction.

    A caller's existing transaction remains under caller control. Otherwise this
    function opens and closes a read-only transaction, retaining no locks after
    return. Absent feature BLOBs remain absent from the returned mapping.
    """
    own_transaction = not connection.in_transaction
    if own_transaction:
        connection.execute("BEGIN")
    try:
        match = connection.execute(
            "SELECT revision FROM matches WHERE match_id=?", (match_id,)
        ).fetchone()
        if match is None:
            raise KeyError(f"unknown match {match_id!r}")
        revision = match[0]
        assets = _records(
            connection,
            "SELECT json FROM assets WHERE match_id=? ORDER BY asset_id",
            match_id,
            MediaAsset,
        )
        samples = _records(
            connection,
            "SELECT json FROM samples WHERE match_id=? ORDER BY sample_id",
            match_id,
            Sample,
        )
        inline = load_inline_features(connection, match_id)
        for sample in samples:
            for index, feature in enumerate(sample.features):
                if isinstance(feature, InlineFeature):
                    key = (sample.sample_id, feature.feature_space_id)
                    if key not in inline:
                        raise ValueError(
                            f"inline feature BLOB missing for sample {sample.sample_id!r}"
                        )
                    sample.features[index] = inline[key]
        evidence = IdentityEvidence(
            observations=_records(
                connection,
                "SELECT json FROM track_observations WHERE match_id=? "
                "ORDER BY clip_id, track_id, time_s, observation_id",
                match_id,
                TrackObservation,
            ),
            tracks=_records(
                connection,
                "SELECT json FROM tracks WHERE match_id=? ORDER BY clip_id, track_id",
                match_id,
                Track,
            ),
            segments=_records(
                connection,
                "SELECT json FROM segments WHERE match_id=? ORDER BY segment_id",
                match_id,
                Segment,
            ),
            samples=samples,
            feature_spaces=_records(
                connection,
                "SELECT json FROM feature_spaces WHERE match_id=? ORDER BY feature_space_id",
                match_id,
                FeatureSpace,
            ),
            feature_files=_records(
                connection,
                "SELECT json FROM feature_files WHERE match_id=? ORDER BY feature_file_id",
                match_id,
                FeatureFile,
            ),
            risks=_records(
                connection,
                "SELECT json FROM risks WHERE match_id=? ORDER BY risk_id",
                match_id,
                IdentityRisk,
            ),
        )
        gallery = GallerySnapshot(
            match_id=match_id,
            revision=revision,
            players=_records(
                connection,
                "SELECT json FROM players WHERE match_id=? AND alias_of IS NULL ORDER BY player_id",
                match_id,
                Player,
            ),
            assignments=_records(
                connection,
                "SELECT json FROM assignments WHERE match_id=? AND current=1 ORDER BY segment_id",
                match_id,
                Assignment,
            ),
            templates=_records(
                connection,
                "SELECT json FROM templates WHERE match_id=? AND current=1 ORDER BY template_id",
                match_id,
                Template,
            ),
            risk_reviews=_records(
                connection,
                """SELECT r.json FROM risk_reviews r WHERE r.match_id=?
                AND r.current=1
                ORDER BY r.risk_id, r.segment_id""",
                match_id,
                RiskReview,
            ),
            evidence=evidence,
            assets=assets,
        )
        constraints = _records(
            connection,
            """SELECT c.json FROM constraints c WHERE c.match_id=? AND c.current=1
                ORDER BY c.constraint_id""",
            match_id,
            ManualConstraint,
        )
        template_reviews = _records(
            connection,
            """SELECT r.json FROM template_reviews r WHERE r.match_id=?
               AND r.current=1
               ORDER BY r.sample_id, r.player_id, r.feature_space_id""",
            match_id,
            ManualTemplateReview,
        )
        proposals: dict[str, SegmentProposal] = {}
        proposal_fingerprints: dict[str, str | None] = {}
        table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='segmentation_proposals'"
        ).fetchone()
        if table_exists:
            for row in connection.execute(
                "SELECT segment_id, json, evidence_fingerprint FROM segmentation_proposals WHERE match_id=? ORDER BY segment_id",
                (match_id,),
            ):
                proposals[row[0]] = SegmentProposal.model_validate_json(row[1])
                proposal_fingerprints[row[0]] = row[2]
        return IdentitySnapshot(
            gallery,
            load_feature_values(connection, match_id),
            constraints,
            template_reviews,
            proposals,
            proposal_fingerprints,
        )
    finally:
        if own_transaction:
            connection.rollback()
