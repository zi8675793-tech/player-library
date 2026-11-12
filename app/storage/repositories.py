"""Match-scoped facts and history. None of these methods commits a transaction."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from app.contracts.common import Provenance, SourceKey, SourceLocation
from app.contracts.events import Event
from app.contracts.evidence import (
    Clip,
    EvidenceBatch,
    FeatureSpace,
    InlineFeature,
    Sample,
    SourceFile,
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
from app.storage.features import save_inline_feature
from app.storage.relations import (
    check_assignment_references,
    check_constraint_references,
    check_risk_review_references,
    check_template_references,
    check_template_review_references,
)


class StorageConflict(ValueError):
    """An immutable source, entity, or expected revision conflicts with stored facts."""


class RevisionConflict(StorageConflict):
    """The caller's snapshot is no longer current."""


@dataclass(frozen=True)
class RevisionAudit:
    match_id: str
    revision: int
    previous_revision: int
    actor: str
    reason: str
    changes: Any


@dataclass(frozen=True)
class SourceVersion:
    key: SourceKey
    entity_type: str
    entity_key: str
    fingerprint: str
    record: dict[str, Any]
    locations: list[SourceLocation]


def _json(value: BaseModel | dict | list) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _business(value: BaseModel) -> str:
    """Ignore provenance and its locators/hash; source version is separately indexed."""

    def remove_provenance(item: Any) -> Any:
        if isinstance(item, list):
            return [remove_provenance(value) for value in item]
        if isinstance(item, dict):
            return {
                key: remove_provenance(value)
                for key, value in item.items()
                if not (
                    key == "source"
                    and isinstance(value, dict)
                    and {"producer", "record_type", "source_record_id", "evidence_version"}
                    <= value.keys()
                )
            }
        return item

    return _json(remove_provenance(value.model_dump(mode="json")))


def _source_key(source: Provenance) -> tuple[str, str, str, str]:
    return (source.producer, source.record_type, source.source_record_id, source.evidence_version)


class Repository:
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection

    def _require_transaction(self) -> None:
        if not self.connection.in_transaction:
            raise RuntimeError("writes require an explicit transaction")

    def ensure_match(self, match_id: str) -> bool:
        self._require_transaction()
        cur = self.connection.execute(
            "INSERT INTO matches(match_id,revision) VALUES (?,0) ON CONFLICT DO NOTHING",
            (match_id,),
        )
        return cur.rowcount == 1

    def revision(self, match_id: str) -> int:
        row = self.connection.execute(
            "SELECT revision FROM matches WHERE match_id=?", (match_id,)
        ).fetchone()
        if row is None:
            raise KeyError(match_id)
        return row[0]

    def advance_revision(
        self, match_id: str, expected_revision: int, *, actor: str, reason: str, changes: Any
    ) -> int:
        """Atomically compare/update and audit; call inside the same write transaction.

        Do not call for a receipt-only or otherwise unchanged submission.
        """
        if not self.connection.in_transaction:
            raise RuntimeError("revision changes require an explicit transaction")
        if not actor.strip() or not reason.strip():
            raise ValueError("actor and reason are required")
        cur = self.connection.execute(
            "UPDATE matches SET revision=revision+1 WHERE match_id=? AND revision=?",
            (match_id, expected_revision),
        )
        if cur.rowcount != 1:
            raise RevisionConflict(
                f"stale or missing match {match_id!r} at revision {expected_revision}"
            )
        revision = expected_revision + 1
        self.connection.execute(
            "INSERT INTO revisions(match_id,revision,previous_revision,actor,reason,changes_json) "
            "VALUES (?,?,?,?,?,?)",
            (match_id, revision, expected_revision, actor, reason, _json(changes)),
        )
        from app.storage.revision_states import capture_state

        capture_state(self.connection, match_id, revision)
        return revision

    def _insert_entity(
        self,
        table: str,
        match_id: str,
        key: dict[str, Any],
        obj: BaseModel,
        extra: dict[str, Any] | None = None,
        *,
        replace_version: bool = False,
    ) -> bool:
        self._require_transaction()
        # All table/column names are internal constants, never user supplied SQL.
        where = " AND ".join(f"{column}=?" for column in ("match_id", *key))
        old = self.connection.execute(
            f"SELECT json FROM {table} WHERE {where}", (match_id, *key.values())
        ).fetchone()
        payload = _json(obj)
        if old:
            if old[0] == payload:
                return False
            if not replace_version:
                # Locations can change without changing the business fact.
                stored = type(obj).model_validate_json(old[0])
                if _business(stored) == _business(obj):
                    return False
                raise StorageConflict(f"conflicting {table} id {tuple(key.values())!r}")
            fields = {**(extra or {}), "json": payload}
            self.connection.execute(
                f"UPDATE {table} SET "
                + ",".join(f"{name}=?" for name in fields)
                + f" WHERE {where}",
                (*fields.values(), match_id, *key.values()),
            )
            return True
        from app.storage.revision_states import REVIEW_KEYS

        if table in REVIEW_KEYS:
            columns = REVIEW_KEYS[table]
            self.connection.execute(
                f"UPDATE {table} SET current=0 WHERE match_id=? AND "
                + " AND ".join(f"{column}=?" for column in columns),
                (match_id, *(key[column] for column in columns)),
            )
        fields = {"match_id": match_id, **key, **(extra or {}), "json": payload}
        self.connection.execute(
            f"INSERT INTO {table} ("
            + ",".join(fields)
            + ") VALUES ("
            + ",".join("?" for _ in fields)
            + ")",
            tuple(fields.values()),
        )
        return True

    def _source(
        self, match_id: str, obj: BaseModel, entity_type: str, entity_key: str
    ) -> tuple[bool, bool]:
        source = getattr(obj, "source", None)
        if source is None:
            return False, False
        fingerprint = hashlib.sha256(_business(obj).encode()).hexdigest()
        key = _source_key(source)
        row = self.connection.execute(
            "SELECT entity_type,entity_key,fingerprint FROM source_records WHERE match_id=? "
            "AND producer=? AND record_type=? AND source_record_id=? AND evidence_version=?",
            (match_id, *key),
        ).fetchone()
        if row:
            if tuple(row) != (entity_type, entity_key, fingerprint):
                raise StorageConflict(f"source version {key!r} has different business content")
            new = False
        else:
            previous = self.connection.execute(
                "SELECT evidence_version FROM source_records WHERE match_id=? AND producer=? "
                "AND record_type=? AND source_record_id=?",
                (match_id, *key[:3]),
            ).fetchall()
            if previous:
                if source.supersedes is None:
                    raise StorageConflict(
                        f"new source version {key!r} must explicitly supersede a version"
                    )
                old_key = (
                    source.supersedes.producer,
                    source.supersedes.record_type,
                    source.supersedes.source_record_id,
                    source.supersedes.evidence_version,
                )
                if old_key[:3] != key[:3]:
                    raise StorageConflict(
                        "supersedes must identify a version of the same source record"
                    )
                old = self.connection.execute(
                    "SELECT entity_type,entity_key FROM source_records WHERE match_id=? AND producer=? "
                    "AND record_type=? AND source_record_id=? AND evidence_version=?",
                    (match_id, *old_key),
                ).fetchone()
                if old is None or tuple(old) != (entity_type, entity_key):
                    raise StorageConflict("superseded source version or entity is missing")
            elif source.supersedes is not None:
                raise StorageConflict("superseded source version is missing")
            self.connection.execute(
                "INSERT INTO source_records(match_id,producer,record_type,source_record_id,evidence_version,"
                "entity_type,entity_key,fingerprint,json) VALUES (?,?,?,?,?,?,?,?,?)",
                (match_id, *key, entity_type, entity_key, fingerprint, _json(obj)),
            )
            new = True
        for location in source.locations:
            self.connection.execute(
                "INSERT INTO source_locations(match_id,producer,record_type,source_record_id,"
                "evidence_version,location_json) VALUES (?,?,?,?,?,?) ON CONFLICT DO NOTHING",
                (match_id, *key, _json(location)),
            )
        return new, bool(new and source.supersedes)

    def _evidence_entity(
        self,
        table: str,
        match_id: str,
        key: dict[str, Any],
        obj: BaseModel,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        entity_key = _json(list(key.values()))
        new_source, supersedes = self._source(match_id, obj, table, entity_key)
        if getattr(obj, "source", None) is not None and not new_source:
            # An old version can be resubmitted after a newer one is current.
            return False
        if supersedes:
            where = " AND ".join(f"{column}=?" for column in ("match_id", *key))
            row = self.connection.execute(
                f"SELECT json FROM {table} WHERE {where}", (match_id, *key.values())
            ).fetchone()
            if row is None:
                raise StorageConflict("superseded entity is missing")
            current = type(obj).model_validate_json(row[0])
            old_source = getattr(current, "source", None)
            superseded = obj.source.supersedes
            if old_source is None or _source_key(old_source) != (
                superseded.producer,
                superseded.record_type,
                superseded.source_record_id,
                superseded.evidence_version,
            ):
                raise StorageConflict("supersedes must identify the current entity version")
        changed = self._insert_entity(table, match_id, key, obj, extra, replace_version=supersedes)
        return new_source or changed

    def is_recorded_event(self, event: Event) -> bool:
        """Recognize identical immutable source evidence before applying late context."""
        row = self.connection.execute(
            "SELECT entity_type,entity_key,fingerprint FROM source_records WHERE match_id=? "
            "AND producer=? AND record_type=? AND source_record_id=? AND evidence_version=?",
            (event.match_id, *_source_key(event.source)),
        ).fetchone()
        return row is not None and tuple(row) == (
            "events",
            _json([event.event_id]),
            hashlib.sha256(_business(event).encode()).hexdigest(),
        )

    def store_batch(self, batch: EvidenceBatch) -> bool:
        """Store supplied facts; empty/absent branches never delete stored facts.

        Returns business change, excluding the separate handoff receipt and locations.
        An explicit transaction should encompass this call and revision/audit handling.
        """
        match = batch.match_id
        self.ensure_match(match)
        receipt = _json(batch)
        receipt_hash = hashlib.sha256(receipt.encode()).hexdigest()
        self.connection.execute(
            "INSERT INTO batches(match_id,batch_id,receipt_hash,json) VALUES (?,?,?,?) "
            "ON CONFLICT DO NOTHING",
            (match, batch.batch_id, receipt_hash, receipt),
        )
        changed = False

        def add(
            table: str, key: dict[str, Any], obj: BaseModel, extra: dict[str, Any] | None = None
        ) -> None:
            nonlocal changed
            changed = self._evidence_entity(table, match, key, obj, extra) or changed

        for source in batch.sources:
            # File hashes and locators describe a handoff, not a record's business content.
            payload = _json(source)
            self.connection.execute(
                "INSERT INTO source_files(match_id,file_id,json) VALUES (?,?,?) ON CONFLICT DO NOTHING",
                (match, source.file_id, payload),
            )
            self.connection.execute(
                "INSERT INTO source_file_observations(match_id,file_id,observation_hash,json) "
                "VALUES (?,?,?,?) ON CONFLICT DO NOTHING",
                (match, source.file_id, hashlib.sha256(payload.encode()).hexdigest(), payload),
            )
        for asset in batch.assets:
            add("assets", {"asset_id": asset.asset_id}, asset)
        for clip in batch.clips or []:
            add("clips", {"clip_id": clip.clip_id}, clip, {"asset_id": clip.asset_id})
        identity = batch.identity_evidence
        if identity:
            for space in identity.feature_spaces:
                add("feature_spaces", {"feature_space_id": space.feature_space_id}, space)
            for file in identity.feature_files:
                add(
                    "feature_files",
                    {"feature_file_id": file.feature_file_id},
                    file,
                    {"feature_space_id": file.feature_space_id},
                )
            for track in identity.tracks:
                add("tracks", {"clip_id": track.clip_id, "track_id": track.track_id}, track)
            for observation in identity.observations:
                add(
                    "track_observations",
                    {"observation_id": observation.observation_id},
                    observation,
                    {
                        "clip_id": observation.clip_id,
                        "track_id": observation.track_id,
                        "time_s": observation.time_s,
                        "frame_idx": observation.frame_idx,
                    },
                )
            for segment in identity.segments:
                reviewed = self.connection.execute(
                    "SELECT json FROM segment_review_history WHERE match_id=? AND segment_id=?",
                    (match, segment.segment_id),
                ).fetchall()
                if reviewed:
                    if any(
                        _business(Segment.model_validate_json(row[0])) == _business(segment)
                        for row in reviewed
                    ):
                        continue
                    raise StorageConflict("cannot overwrite a manually replaced identity segment")
                add(
                    "segments",
                    {"segment_id": segment.segment_id},
                    segment,
                    {
                        "clip_id": segment.clip_id,
                        "track_id": segment.track_id,
                        "start_s": segment.start_s,
                        "end_s": segment.end_s,
                        "active": int(segment.active),
                    },
                )
            for sample in identity.samples:
                old_row = self.connection.execute(
                    "SELECT json FROM samples WHERE match_id=? AND sample_id=?",
                    (match, sample.sample_id),
                ).fetchone()
                old_sample = Sample.model_validate_json(old_row[0]) if old_row else None
                existing_source = self.connection.execute(
                    "SELECT 1 FROM source_records WHERE match_id=? AND producer=? AND record_type=? "
                    "AND source_record_id=? AND evidence_version=?",
                    (match, *_source_key(sample.source)),
                ).fetchone()
                if (
                    old_sample
                    and sample.source.supersedes
                    and not existing_source
                    and (_source_key(old_sample.source) != _source_key(sample.source))
                ):
                    if self.connection.execute(
                        "SELECT 1 FROM templates WHERE match_id=? AND sample_id=? LIMIT 1",
                        (match, sample.sample_id),
                    ).fetchone():
                        raise StorageConflict(
                            "versioning a sample used by a template requires explicit review"
                        )
                add(
                    "samples",
                    {"sample_id": sample.sample_id},
                    sample,
                    {
                        "clip_id": sample.clip_id,
                        "track_id": sample.track_id,
                        "segment_id": sample.segment_id,
                    },
                )
                if (
                    old_sample
                    and sample.source.supersedes
                    and (
                        _source_key(old_sample.source)
                        == (
                            sample.source.supersedes.producer,
                            sample.source.supersedes.record_type,
                            sample.source.supersedes.source_record_id,
                            sample.source.supersedes.evidence_version,
                        )
                    )
                ):
                    self.connection.execute(
                        "INSERT INTO sample_feature_history "
                        "(match_id,producer,record_type,source_record_id,evidence_version,"
                        "sample_id,feature_space_id,dtype,shape_json,byte_order,values_blob,"
                        "visibility_shape_json,visibility_blob) "
                        "SELECT match_id,?,?,?,?,sample_id,feature_space_id,dtype,shape_json,"
                        "byte_order,values_blob,visibility_shape_json,visibility_blob "
                        "FROM sample_feature_values WHERE match_id=? AND sample_id=?",
                        (*_source_key(old_sample.source), match, sample.sample_id),
                    )
                    self.connection.execute(
                        "DELETE FROM sample_feature_values WHERE match_id=? AND sample_id=?",
                        (match, sample.sample_id),
                    )
            # Inline values need their declared space so their binary shape is unambiguous.
            for sample in identity.samples:
                current = self.connection.execute(
                    "SELECT json FROM samples WHERE match_id=? AND sample_id=?",
                    (match, sample.sample_id),
                ).fetchone()
                if _source_key(Sample.model_validate_json(current[0]).source) != _source_key(
                    sample.source
                ):
                    continue
                for feature in sample.features:
                    if isinstance(feature, InlineFeature):
                        row = self.connection.execute(
                            "SELECT json FROM feature_spaces WHERE match_id=? AND feature_space_id=?",
                            (match, feature.feature_space_id),
                        ).fetchone()
                        if row is None:
                            raise ValueError(
                                "inline feature requires its feature space to be stored"
                            )
                        changed = (
                            save_inline_feature(
                                self.connection,
                                match,
                                sample.sample_id,
                                FeatureSpace.model_validate_json(row[0]),
                                feature,
                            )
                            or changed
                        )
            risks = [
                *identity.risks,
                *(
                    risk
                    for sample in identity.samples
                    if sample.tracking_quality
                    for risk in sample.tracking_quality.risks
                ),
            ]
            for risk in risks:
                add("risks", {"risk_id": risk.risk_id}, risk)
        for event in batch.events or []:
            event_changed = self._evidence_entity(
                "events", match, {"event_id": event.event_id}, event, {"clip_id": event.clip_id}
            )
            changed = event_changed or changed
            if event_changed:
                self.connection.execute(
                    "DELETE FROM subject_refs WHERE match_id=? AND event_id=?",
                    (match, event.event_id),
                )
                self.connection.execute(
                    "DELETE FROM subjects WHERE match_id=? AND event_id=?", (match, event.event_id)
                )
                for subject in event.subjects:
                    self.connection.execute(
                        "INSERT INTO subjects(match_id,event_id,subject_id,role,json) VALUES (?,?,?,?,?)",
                        (match, event.event_id, subject.subject_id, subject.role, _json(subject)),
                    )
                    for index, ref in enumerate(subject.references):
                        self.connection.execute(
                            "INSERT INTO subject_refs(match_id,event_id,subject_id,ref_index,kind,clip_id,track_id,segment_id,json) "
                            "VALUES (?,?,?,?,?,?,?,?,?)",
                            (
                                match,
                                event.event_id,
                                subject.subject_id,
                                index,
                                ref.kind,
                                getattr(ref, "clip_id", None),
                                getattr(ref, "track_id", None),
                                getattr(ref, "segment_id", None),
                                _json(ref),
                            ),
                        )
        return changed

    def put_player(self, player: Player, *, update: bool = False) -> bool:
        return self._insert_entity(
            "players",
            player.match_id,
            {"player_id": player.player_id},
            player,
            {"alias_of": player.alias_of},
            replace_version=update,
        )

    def put_assignment(
        self, assignment: Assignment, *, committed_revision: int, replace: bool = False
    ) -> bool:
        """Record a decision under the commit revision, not its algorithm snapshot revision."""
        self._require_transaction()
        if committed_revision != self.revision(assignment.match_id) + 1:
            raise ValueError("committed_revision must be the next revision in the transaction")
        check_assignment_references(self.connection, assignment)
        result = assignment.model_copy(update={"revision": committed_revision})
        row = self.connection.execute(
            "SELECT json FROM assignments WHERE match_id=? AND segment_id=? AND current=1",
            (assignment.match_id, assignment.segment_id),
        ).fetchone()
        if row:
            old = Assignment.model_validate_json(row[0])
            if old.model_dump(exclude={"revision"}) == assignment.model_dump(exclude={"revision"}):
                return False
            if not replace:
                raise StorageConflict("changing an assignment requires explicit replacement")
            self.connection.execute(
                "UPDATE assignments SET current=0 WHERE match_id=? AND segment_id=? AND current=1",
                (assignment.match_id, assignment.segment_id),
            )
        self.connection.execute(
            "INSERT INTO assignments(match_id,segment_id,revision,player_id,current,json) VALUES (?,?,?,?,1,?)",
            (
                assignment.match_id,
                assignment.segment_id,
                committed_revision,
                assignment.player_id,
                _json(result),
            ),
        )
        return True

    def put_template(
        self, template: Template, *, committed_revision: int, replace: bool = False
    ) -> bool:
        self._require_transaction()
        if committed_revision != self.revision(template.match_id) + 1:
            raise ValueError("committed_revision must be the next revision in the transaction")
        check_template_references(self.connection, template)
        result = template.model_copy(update={"revision": committed_revision})
        row = self.connection.execute(
            "SELECT json FROM templates WHERE match_id=? AND template_id=? AND current=1",
            (template.match_id, template.template_id),
        ).fetchone()
        if row:
            old = Template.model_validate_json(row[0])
            if old.model_dump(exclude={"revision"}) == template.model_dump(exclude={"revision"}):
                return False
            if not replace:
                raise StorageConflict("changing a template requires explicit replacement")
            self.connection.execute(
                "UPDATE templates SET current=0 WHERE match_id=? AND template_id=? AND current=1",
                (template.match_id, template.template_id),
            )
            self.connection.execute(
                "DELETE FROM template_prototypes WHERE match_id=? AND template_id=?",
                (template.match_id, template.template_id),
            )
        self.connection.execute(
            "INSERT INTO templates(match_id,template_id,player_id,sample_id,feature_space_id,revision,current,json) VALUES (?,?,?,?,?,?,1,?)",
            (
                template.match_id,
                template.template_id,
                template.player_id,
                template.sample_id,
                template.feature_space_id,
                committed_revision,
                _json(result),
            ),
        )
        if template.active and template.prototype_id is not None:
            self.connection.execute(
                "INSERT OR REPLACE INTO template_prototypes(match_id,template_id,player_id,sample_id,prototype_id,time_bucket,source_component,reliable_parts_json,quality_score,json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    template.match_id,
                    template.template_id,
                    template.player_id,
                    template.sample_id,
                    template.prototype_id,
                    template.time_bucket,
                    template.source_component,
                    _json(template.reliable_parts),
                    template.quality_score,
                    _json(result),
                ),
            )
        return True

    def put_risk_review(self, review: RiskReview) -> bool:
        self._require_transaction()
        if review.revision != self.revision(review.match_id) + 1:
            raise ValueError("review revision must be the next commit revision")
        check_risk_review_references(self.connection, review)
        return self._insert_entity(
            "risk_reviews",
            review.match_id,
            {
                "risk_id": review.risk_id,
                "segment_id": review.segment_id,
                "revision": review.revision,
            },
            review,
        )

    def put_derived_risk(self, risk: IdentityRisk) -> bool:
        """Save a checked derivation; its source points to evidence, not a new source record."""
        return self._insert_entity("risks", risk.match_id, {"risk_id": risk.risk_id}, risk)

    def put_constraint(self, constraint: ManualConstraint) -> bool:
        self._require_transaction()
        if constraint.revision != self.revision(constraint.match_id) + 1:
            raise ValueError("constraint revision must be the next commit revision")
        check_constraint_references(self.connection, constraint)
        return self._insert_entity(
            "constraints",
            constraint.match_id,
            {"constraint_id": constraint.constraint_id, "revision": constraint.revision},
            constraint,
            {
                "segment_id": constraint.segment_id,
                "player_id": constraint.player_id,
                "related_segment_id": constraint.related_segment_id,
            },
        )

    def put_template_review(self, review: ManualTemplateReview) -> bool:
        self._require_transaction()
        if review.revision != self.revision(review.match_id) + 1:
            raise ValueError("review revision must be the next commit revision")
        check_template_review_references(self.connection, review)
        return self._insert_entity(
            "template_reviews",
            review.match_id,
            {
                "sample_id": review.sample_id,
                "player_id": review.player_id,
                "feature_space_id": review.feature_space_id,
                "revision": review.revision,
            },
            review,
        )

    def events(self, match_id: str) -> list[Event]:
        """Original event facts, including subjects and unresolved references."""
        return [
            Event.model_validate_json(row[0])
            for row in self.connection.execute(
                "SELECT json FROM events WHERE match_id=? ORDER BY event_id", (match_id,)
            )
        ]

    def clips(self, match_id: str) -> list[Clip]:
        return [
            Clip.model_validate_json(row[0])
            for row in self.connection.execute(
                "SELECT json FROM clips WHERE match_id=? ORDER BY clip_id", (match_id,)
            )
        ]

    def source_files(self, match_id: str) -> list[SourceFile]:
        return [
            SourceFile.model_validate_json(row[0])
            for row in self.connection.execute(
                "SELECT json FROM source_files WHERE match_id=? ORDER BY file_id", (match_id,)
            )
        ]

    def source_file_observations(self, match_id: str, file_id: str) -> list[SourceFile]:
        return [
            SourceFile.model_validate_json(row[0])
            for row in self.connection.execute(
                "SELECT json FROM source_file_observations WHERE match_id=? AND file_id=? "
                "ORDER BY observation_hash",
                (match_id, file_id),
            )
        ]

    def revisions(self, match_id: str) -> list[RevisionAudit]:
        rows = self.connection.execute(
            "SELECT revision,previous_revision,actor,reason,changes_json FROM revisions "
            "WHERE match_id=? ORDER BY revision",
            (match_id,),
        )
        return [
            RevisionAudit(match_id, rev, previous, actor, reason, json.loads(changes))
            for rev, previous, actor, reason, changes in rows
        ]

    def source_versions(
        self, match_id: str, producer: str, record_type: str, source_record_id: str
    ) -> list[SourceVersion]:
        rows = self.connection.execute(
            "SELECT evidence_version,entity_type,entity_key,fingerprint,json FROM source_records "
            "WHERE match_id=? AND producer=? AND record_type=? AND source_record_id=? "
            "ORDER BY evidence_version",
            (match_id, producer, record_type, source_record_id),
        ).fetchall()
        versions = []
        for version, entity_type, entity_key, fingerprint, payload in rows:
            key = SourceKey(
                producer=producer,
                record_type=record_type,
                source_record_id=source_record_id,
                evidence_version=version,
            )
            locations = [
                SourceLocation.model_validate_json(row[0])
                for row in self.connection.execute(
                    "SELECT location_json FROM source_locations WHERE match_id=? AND producer=? "
                    "AND record_type=? AND source_record_id=? AND evidence_version=? ORDER BY location_json",
                    (match_id, producer, record_type, source_record_id, version),
                )
            ]
            versions.append(
                SourceVersion(
                    key, entity_type, entity_key, fingerprint, json.loads(payload), locations
                )
            )
        return versions

    def assignment_history(self, match_id: str, segment_id: str) -> list[Assignment]:
        return [
            Assignment.model_validate_json(row[0])
            for row in self.connection.execute(
                "SELECT json FROM assignments WHERE match_id=? AND segment_id=? ORDER BY revision",
                (match_id, segment_id),
            )
        ]

    def template_history(self, match_id: str, template_id: str) -> list[Template]:
        return [
            Template.model_validate_json(row[0])
            for row in self.connection.execute(
                "SELECT json FROM templates WHERE match_id=? AND template_id=? ORDER BY revision",
                (match_id, template_id),
            )
        ]

    def receipts(self, match_id: str, batch_id: str) -> list[EvidenceBatch]:
        return [
            EvidenceBatch.model_validate_json(row[0])
            for row in self.connection.execute(
                "SELECT json FROM batches WHERE match_id=? AND batch_id=? ORDER BY receipt_hash",
                (match_id, batch_id),
            )
        ]

    def check_segment_overlaps(self, match_id: str) -> list[tuple[str, str]]:
        return [
            tuple(row)
            for row in self.connection.execute(
                "SELECT a.segment_id,b.segment_id FROM segments a JOIN segments b ON a.match_id=b.match_id "
                "AND a.clip_id=b.clip_id AND a.track_id=b.track_id AND a.segment_id<b.segment_id "
                "AND a.active=1 AND b.active=1 AND a.start_s<b.end_s AND b.start_s<a.end_s "
                "WHERE a.match_id=? ORDER BY a.segment_id,b.segment_id",
                (match_id,),
            )
        ]

    def check_alias_cycles(self, match_id: str) -> list[list[str]]:
        aliases = dict(
            self.connection.execute(
                "SELECT player_id,alias_of FROM players WHERE match_id=?", (match_id,)
            )
        )
        cycles: set[tuple[str, ...]] = set()
        for start in aliases:
            path: list[str] = []
            current = start
            while current is not None and current in aliases and current not in path:
                path.append(current)
                current = aliases[current]
            if current in path:
                cycle = path[path.index(current) :]
                i = cycle.index(min(cycle))
                cycles.add(tuple(cycle[i:] + cycle[:i]))
        return [list(cycle) for cycle in sorted(cycles)]

    def get_operation(
        self, match_id: str, kind: str, request_id: str, payload: dict
    ) -> dict | None:
        """A matching retry returns its original result, independent of a newer revision."""
        row = self.connection.execute(
            "SELECT payload_hash,result_json FROM workflow_operations WHERE match_id=? AND kind=? AND request_id=?",
            (match_id, kind, request_id),
        ).fetchone()
        fingerprint = hashlib.sha256(_json(payload).encode()).hexdigest()
        if row is None:
            return None
        if row[0] != fingerprint:
            raise StorageConflict("workflow request_id was already used with different content")
        return json.loads(row[1])

    def put_operation(
        self, match_id: str, kind: str, request_id: str, payload: dict, result: dict
    ) -> None:
        self._require_transaction()
        previous = self.get_operation(match_id, kind, request_id, payload)
        if previous is not None:
            if previous != result:
                raise StorageConflict("workflow operation result is immutable")
            return
        encoded = _json(payload)
        self.connection.execute(
            "INSERT INTO workflow_operations VALUES (?,?,?,?,?,?)",
            (
                match_id,
                kind,
                request_id,
                hashlib.sha256(encoded.encode()).hexdigest(),
                encoded,
                _json(result),
            ),
        )

    def record_review(self, match_id: str, revision: int, actor: str, commands: list[dict]) -> None:
        self._require_transaction()
        if revision != self.revision(match_id) + 1:
            raise ValueError("review revision must be the next commit revision")
        self.connection.execute(
            "INSERT INTO workflow_reviews VALUES (?,?,?,?)",
            (match_id, revision, actor, _json(commands)),
        )

    def review_records(self, match_id: str) -> list[dict]:
        return [
            {"revision": row[0], "actor": row[1], "commands": json.loads(row[2])}
            for row in self.connection.execute(
                "SELECT revision,actor,json FROM workflow_reviews WHERE match_id=? ORDER BY revision",
                (match_id,),
            )
        ]

    def put_clean_review(self, match_id: str, sample_id: str, review: dict) -> bool:
        self._require_transaction()
        if review["revision"] != self.revision(match_id) + 1:
            raise ValueError("clean review revision must be the next commit revision")
        row = self.connection.execute(
            "SELECT json FROM samples WHERE match_id=? AND sample_id=?", (match_id, sample_id)
        ).fetchone()
        if row is None:
            raise ValueError("clean review requires a stored sample")
        sample = Sample.model_validate_json(row[0])
        review = {
            **review,
            "source_key": sample.source.model_dump(include=set(SourceKey.model_fields)),
        }
        previous = self.clean_reviews(match_id).get(sample_id)
        if previous and {k: v for k, v in previous.items() if k != "revision"} == {
            k: v for k, v in review.items() if k != "revision"
        }:
            return False
        self.connection.execute(
            "UPDATE sample_clean_reviews SET current=0 WHERE match_id=? AND sample_id=?",
            (match_id, sample_id),
        )
        self.connection.execute(
            "INSERT INTO sample_clean_reviews(match_id,sample_id,revision,json) VALUES (?,?,?,?)",
            (match_id, sample_id, review["revision"], _json(review)),
        )
        return True

    def clean_reviews(self, match_id: str) -> dict[str, dict]:
        return {
            row[0]: json.loads(row[1])
            for row in self.connection.execute(
                "SELECT c.sample_id,c.json FROM sample_clean_reviews c WHERE c.match_id=? AND c.current=1",
                (match_id,),
            )
        }

    def replace_segment(self, segment: Segment, *, committed_revision: int) -> bool:
        """Keep the original segment in history; only review may retire its current range."""
        self._require_transaction()
        if committed_revision != self.revision(segment.match_id) + 1:
            raise ValueError("segment revision must be the next commit revision")
        old = self.connection.execute(
            "SELECT json FROM segments WHERE match_id=? AND segment_id=?",
            (segment.match_id, segment.segment_id),
        ).fetchone()
        if old and old[0] == _json(segment):
            return False
        if old:
            self.connection.execute(
                "INSERT INTO segment_review_history VALUES (?,?,?,?)",
                (segment.match_id, segment.segment_id, committed_revision, old[0]),
            )
        return self._insert_entity(
            "segments",
            segment.match_id,
            {"segment_id": segment.segment_id},
            segment,
            {
                "clip_id": segment.clip_id,
                "track_id": segment.track_id,
                "start_s": segment.start_s,
                "end_s": segment.end_s,
                "active": int(segment.active),
            },
            replace_version=bool(old),
        )

    def put_segmentation_proposal(
        self,
        match_id: str,
        proposal: SegmentProposal,
        evidence_fingerprint: str | None = None,
    ) -> bool:
        self._require_transaction()
        payload = _json(proposal)
        cur = self.connection.execute(
            """INSERT INTO segmentation_proposals(match_id, segment_id, parent_segment_id, status, evidence_fingerprint, json)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(match_id, segment_id) DO UPDATE SET
                   parent_segment_id = excluded.parent_segment_id,
                   status = excluded.status,
                   evidence_fingerprint = excluded.evidence_fingerprint,
                   json = excluded.json
            """,
            (
                match_id,
                proposal.segment.segment_id,
                proposal.parent_segment_id,
                proposal.status,
                evidence_fingerprint,
                payload,
            ),
        )
        return cur.rowcount > 0

    def segmentation_proposals(self, match_id: str) -> dict[str, SegmentProposal]:
        rows = self.connection.execute(
            "SELECT segment_id, json FROM segmentation_proposals WHERE match_id=?", (match_id,)
        ).fetchall()
        return {row[0]: SegmentProposal.model_validate_json(row[1]) for row in rows}
