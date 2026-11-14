"""Durable T1/T2 facts, immutable histories and legacy-compatible projections.

The caller owns BEGIN IMMEDIATE, revision checks and commit. No image paths are
opened here. Vectors are exact little-endian float32 BLOBs; JSON uses explicit
infinity tokens because SQLite rejects nonstandard Infinity JSON literals.
"""

import json
import math
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any

import numpy as np

from app.contracts.common import Provenance
from app.contracts.evidence import Clip, Track
from app.contracts.evidence import TrackObservation as LegacyObservation
from app.contracts.identity import Assignment, AssignmentEvidence, Player, Segment
from app.features.dual_stream import (
    AppearanceEmbedding,
    DualStreamFeature,
    JerseyNumberPrediction,
    TrackletSegment,
    TrackObservation,
)
from app.identity.constraints import SpatioTemporalConstraintEngine
from app.identity.matching import (
    DecisionOutcome,
    IncrementalDecision,
    IncrementalPlan,
    PlayerProfile,
    PlayerPrototype,
)
from app.storage.repositories import Repository, RevisionConflict, StorageConflict

OPERATION_KIND = "dual_stream_incremental"


@dataclass(frozen=True)
class DualStreamSnapshot:
    revision: int
    segments: tuple[TrackletSegment, ...]
    features: dict[str, DualStreamFeature]
    players: tuple[PlayerProfile, ...]
    decisions: dict[str, IncrementalDecision]
    assignments: dict[str, str]
    input_hashes: dict[str, str]


def dual_stream_json(value: Any) -> Any:
    """Return a standard-JSON-safe value, without mutating caller dataclasses."""
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: dual_stream_json(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return dual_stream_json(value.item())
    if isinstance(value, Mapping):
        return {str(key): dual_stream_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [dual_stream_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            raise ValueError("NaN cannot be persisted")
        return {"$float": "+inf" if value > 0 else "-inf"}
    return value


to_json = dual_stream_json


def _dumps(value: Any) -> str:
    return json.dumps(
        dual_stream_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _restore(value: Any) -> Any:
    if isinstance(value, dict):
        if value == {"$float": "+inf"}:
            return math.inf
        if value == {"$float": "-inf"}:
            return -math.inf
        return {key: _restore(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore(item) for item in value]
    return value


def legacy_player_id(player_id: str) -> str:
    """Project T2 UUIDs into the legacy PlayerId domain without changing T2 IDs."""
    return player_id if player_id.startswith("pl_") else "pl_" + player_id


def _require_transaction(connection: sqlite3.Connection) -> None:
    if not connection.in_transaction:
        raise RuntimeError("dual-stream writes require an explicit transaction")


def _feature_payload(feature: DualStreamFeature) -> tuple[str, bytes]:
    if not isinstance(feature, DualStreamFeature):
        raise TypeError("Expected DualStreamFeature")
    vector = feature.appearance.vector
    if (
        not isinstance(vector, np.ndarray)
        or vector.shape != (512,)
        or vector.dtype != np.float32
        or not np.isfinite(vector).all()
        or not np.isclose(np.linalg.norm(vector.astype(np.float64)), 1, rtol=1e-4, atol=1e-5)
    ):
        raise ValueError("appearance must be a normalized finite float32 (512,) vector")
    if not 0 <= feature.appearance.quality_score <= 1 or not feature.appearance.model_name.strip():
        raise ValueError("invalid appearance metadata")
    metadata = dual_stream_json(feature)
    del metadata["appearance"]["vector"]
    return _dumps(metadata), vector.astype("<f4", copy=False).tobytes(order="C")


def _load_feature(metadata: str, blob: bytes) -> DualStreamFeature:
    if len(blob) != 512 * 4:
        raise ValueError("invalid dual-stream vector byte count")
    data = json.loads(metadata)
    appearance = data["appearance"]
    vector = np.frombuffer(blob, dtype="<f4").astype(np.float32, copy=True)
    vector.setflags(write=False)
    jersey = data["jersey"]
    if jersey is not None:
        jersey["votes"] = {int(key): value for key, value in jersey["votes"].items()}
        jersey = JerseyNumberPrediction(**jersey)
    feature = DualStreamFeature(
        data["segment_id"],
        jersey,
        AppearanceEmbedding(vector=vector, **appearance),
    )
    _feature_payload(feature)
    return feature


def _load_segment(payload: str) -> TrackletSegment:
    data = json.loads(payload)
    observations = []
    for observation in data.pop("observations"):
        observation["bbox_xyxy"] = tuple(observation["bbox_xyxy"])
        observations.append(TrackObservation(**observation))
    return TrackletSegment(observations=tuple(observations), **data)


def _decision(payload: str) -> IncrementalDecision:
    data = _restore(json.loads(payload))
    data["outcome"] = DecisionOutcome(data["outcome"])
    return IncrementalDecision(**data)


def read_dual_stream_snapshot(
    connection: sqlite3.Connection,
    match_id: str,
    *,
    revision: int | None = None,
) -> DualStreamSnapshot:
    """Read one consistent snapshot; optionally reconstruct a historical revision."""
    own_transaction = not connection.in_transaction
    if own_transaction:
        connection.execute("BEGIN")
    try:
        row = connection.execute(
            "SELECT revision FROM matches WHERE match_id=?", (match_id,)
        ).fetchone()
        current = row[0] if row else 0
        if revision is None:
            revision = current
        if type(revision) is not int or not 0 <= revision <= current:
            raise ValueError("snapshot revision is outside the committed history")
        segments, features, hashes = [], {}, {}
        for row in connection.execute(
            "SELECT segment_id,input_hash,segment_json,feature_json,dtype,shape_json,byte_order,vector_blob "
            "FROM dual_stream_segments WHERE match_id=? AND created_revision<=? ORDER BY segment_id",
            (match_id, revision),
        ):
            sid, input_hash, segment_json, feature_json, dtype, shape, order, blob = row
            if (dtype, shape, order) != ("float32", "[512]", "little"):
                raise ValueError("unsupported dual-stream feature representation")
            segments.append(_load_segment(segment_json))
            features[sid] = _load_feature(feature_json, blob)
            hashes[sid] = input_hash
        prototypes = {}
        for row in connection.execute(
            "SELECT prototype_id,segment_id,json FROM prototypes "
            "WHERE match_id=? AND created_at_revision<=? ORDER BY prototype_id",
            (match_id, revision),
        ):
            data = json.loads(row[2])
            prototypes[row[0]] = PlayerPrototype(feature=features[row[1]], **data)
        players = []
        for row in connection.execute(
            "SELECT p.json FROM dual_stream_profiles p WHERE p.match_id=? AND p.revision="
            "(SELECT MAX(q.revision) FROM dual_stream_profiles q WHERE q.match_id=p.match_id "
            "AND q.player_id=p.player_id AND q.revision<=?) ORDER BY p.player_id",
            (match_id, revision),
        ):
            data = json.loads(row[0])
            ids = data.pop("prototype_ids")
            players.append(PlayerProfile(prototypes=tuple(prototypes[pid] for pid in ids), **data))
        decisions = {
            row[0]: _decision(row[1])
            for row in connection.execute(
                "SELECT d.segment_id,d.json FROM dual_stream_decisions d WHERE d.match_id=? "
                "AND d.revision=(SELECT MAX(q.revision) FROM dual_stream_decisions q "
                "WHERE q.match_id=d.match_id AND q.segment_id=d.segment_id AND q.revision<=?) "
                "ORDER BY d.segment_id",
                (match_id, revision),
            )
        }
        assignments = {}
        for sid, decision in decisions.items():
            target = decision.target_player_id
            if decision.outcome == DecisionOutcome.CREATED_NEW:
                target = decision.evidence["new_player_id"]
            if target is not None:
                assignments[sid] = target
        if revision == current:
            _validate_current_projections(
                connection, match_id, segments, players, decisions, assignments
            )
        return DualStreamSnapshot(
            revision,
            tuple(segments),
            features,
            tuple(players),
            decisions,
            assignments,
            hashes,
        )
    finally:
        if own_transaction:
            connection.rollback()


def _validate_current_projections(connection, match_id, segments, players, decisions, assignments):
    """Fail closed when review/restore has diverged from this engine's state."""
    managed_players = {legacy_player_id(player.player_id) for player in players}
    stored_players = {
        row[0]
        for row in connection.execute("SELECT player_id FROM players WHERE match_id=?", (match_id,))
    }
    stored_assignments = {
        row[0]
        for row in connection.execute(
            "SELECT segment_id FROM assignments WHERE match_id=? AND current=1", (match_id,)
        )
    }
    if stored_players != managed_players or stored_assignments != set(decisions):
        raise StorageConflict(
            "dual-stream workflow cannot ignore unmanaged legacy identities; explicit reconciliation required"
        )
    for table in ("constraints", "template_reviews", "risk_reviews", "sample_clean_reviews"):
        if connection.execute(
            f"SELECT 1 FROM {table} WHERE match_id=? AND current=1 LIMIT 1", (match_id,)
        ).fetchone():
            raise StorageConflict(
                "dual-stream workflow cannot consume or overwrite human review state; explicit reconciliation required"
            )
    for segment in segments:
        row = connection.execute(
            "SELECT json FROM segments WHERE match_id=? AND segment_id=?",
            (match_id, segment.segment_id),
        ).fetchone()
        expected = Segment(
            match_id=match_id,
            segment_id=segment.segment_id,
            clip_id=segment.video_id,
            track_id=str(segment.local_track_id),
            start_s=segment.start_time_s,
            end_s=math.nextafter(segment.end_time_s, math.inf),
            frame_start=segment.start_frame,
            frame_end_exclusive=segment.end_frame + 1,
        )
        if row is None or Segment.model_validate_json(row[0]) != expected:
            raise StorageConflict(
                "dual-stream segment projection was changed by review or restore; explicit reconciliation required"
            )
    for profile in players:
        row = connection.execute(
            "SELECT json FROM players WHERE match_id=? AND player_id=?",
            (match_id, legacy_player_id(profile.player_id)),
        ).fetchone()
        expected = Player(
            match_id=match_id,
            player_id=legacy_player_id(profile.player_id),
            display_name=profile.display_name,
            jersey_number=str(profile.confirmed_jersey_number)
            if profile.confirmed_jersey_number is not None
            else None,
        )
        if row is None or Player.model_validate_json(row[0]) != expected:
            raise StorageConflict(
                "dual-stream player projection was changed by review or restore; explicit reconciliation required"
            )
    for sid in decisions:
        row = connection.execute(
            "SELECT json,revision FROM assignments WHERE match_id=? AND segment_id=? AND current=1",
            (match_id, sid),
        ).fetchone()
        canonical_revision = connection.execute(
            "SELECT MAX(revision) FROM dual_stream_decisions WHERE match_id=? AND segment_id=?",
            (match_id, sid),
        ).fetchone()[0]
        target = assignments.get(sid)
        if row is None:
            raise StorageConflict("dual-stream assignment projection is missing")
        actual = Assignment.model_validate_json(row[0])
        if (
            actual.method != "automatic"
            or actual.status != ("proposed" if target is not None else "unassigned")
            or actual.player_id != (legacy_player_id(target) if target is not None else None)
            or row[1] != canonical_revision
        ):
            raise StorageConflict(
                "dual-stream assignment projection was changed by review or restore; explicit reconciliation required"
            )


def _project_segment(connection, match_id, segment):
    """Keep the existing readers valid without inventing observations or labels."""
    track_id = str(segment.local_track_id)
    # Legacy intervals are half-open, whereas T1 records the last observed time.
    end_s = math.nextafter(segment.end_time_s, math.inf)
    source = Provenance(
        producer="dual-stream-t3",
        record_type="track",
        source_record_id=f"{segment.video_id}:{track_id}",
        evidence_version="1",
    )
    track = Track(
        match_id=match_id,
        clip_id=segment.video_id,
        track_id=track_id,
        start_s=segment.start_time_s,
        end_s=end_s,
        frame_start=segment.start_frame,
        frame_end_exclusive=segment.end_frame + 1,
        observed_frame_indices=sorted(item.frame_idx for item in segment.observations),
        source=source,
    )
    old = connection.execute(
        "SELECT json FROM tracks WHERE match_id=? AND clip_id=? AND track_id=?",
        (match_id, segment.video_id, track_id),
    ).fetchone()
    if old:
        previous = Track.model_validate_json(old[0])
        if previous.source.producer != source.producer:
            raise StorageConflict("dual-stream input collides with an existing legacy track")
        track = track.model_copy(
            update={
                "start_s": min(previous.start_s, track.start_s),
                "end_s": max(previous.end_s, track.end_s),
                "frame_start": min(previous.frame_start, track.frame_start),
                "frame_end_exclusive": max(previous.frame_end_exclusive, track.frame_end_exclusive),
                "observed_frame_indices": sorted(
                    set(previous.observed_frame_indices or []) | set(track.observed_frame_indices)
                ),
            }
        )
    connection.execute(
        "INSERT INTO clips(match_id,clip_id,json) VALUES (?,?,?) ON CONFLICT DO NOTHING",
        (
            match_id,
            segment.video_id,
            Clip(match_id=match_id, clip_id=segment.video_id).model_dump_json(),
        ),
    )
    connection.execute(
        "INSERT INTO tracks(match_id,clip_id,track_id,json) VALUES (?,?,?,?) "
        "ON CONFLICT(match_id,clip_id,track_id) DO UPDATE SET json=excluded.json",
        (match_id, segment.video_id, track_id, track.model_dump_json()),
    )
    projection = Segment(
        match_id=match_id,
        segment_id=segment.segment_id,
        clip_id=segment.video_id,
        track_id=track_id,
        start_s=segment.start_time_s,
        end_s=end_s,
        frame_start=segment.start_frame,
        frame_end_exclusive=segment.end_frame + 1,
    )
    old = connection.execute(
        "SELECT json FROM segments WHERE match_id=? AND segment_id=?",
        (match_id, segment.segment_id),
    ).fetchone()
    if old and Segment.model_validate_json(old[0]) != projection:
        raise StorageConflict("dual-stream segment collides with existing identity evidence")
    connection.execute(
        "INSERT INTO segments(match_id,segment_id,clip_id,track_id,start_s,end_s,active,json) "
        "VALUES (?,?,?,?,?,?,1,?) ON CONFLICT DO NOTHING",
        (
            match_id,
            segment.segment_id,
            segment.video_id,
            track_id,
            projection.start_s,
            projection.end_s,
            projection.model_dump_json(),
        ),
    )
    for observation in segment.observations:
        oid = f"dual-stream:{segment.video_id}:{track_id}:{observation.frame_idx}"
        projected = LegacyObservation(
            observation_id=oid,
            match_id=match_id,
            clip_id=segment.video_id,
            track_id=track_id,
            time_s=observation.timestamp_s,
            bbox_xyxy=observation.bbox_xyxy,
            frame_idx=observation.frame_idx,
            detection_score=observation.confidence,
            source=source.model_copy(
                update={"record_type": "observation", "source_record_id": oid}
            ),
        )
        old = connection.execute(
            "SELECT json FROM track_observations WHERE match_id=? AND observation_id=?",
            (match_id, oid),
        ).fetchone()
        if old and LegacyObservation.model_validate_json(old[0]) != projected:
            raise StorageConflict("overlapping segments disagree about an observed frame")
        connection.execute(
            "INSERT INTO track_observations(match_id,observation_id,clip_id,track_id,time_s,frame_idx,json) "
            "VALUES (?,?,?,?,?,?,?) ON CONFLICT DO NOTHING",
            (
                match_id,
                oid,
                segment.video_id,
                track_id,
                observation.timestamp_s,
                observation.frame_idx,
                projected.model_dump_json(),
            ),
        )


def put_dual_stream_evidence(
    connection: sqlite3.Connection,
    match_id: str,
    segments: Sequence[TrackletSegment],
    features: Sequence[DualStreamFeature] | Mapping[str, DualStreamFeature],
    input_hashes: Mapping[str, str],
) -> bool:
    """Insert new immutable evidence; changed content under the same ID conflicts."""
    _require_transaction(connection)
    SpatioTemporalConstraintEngine(segments=segments)
    feature_map = (
        dict(features) if isinstance(features, Mapping) else {f.segment_id: f for f in features}
    )
    segment_ids = {segment.segment_id for segment in segments}
    if (
        len(segment_ids) != len(segments)
        or segment_ids != set(feature_map)
        or not segment_ids <= set(input_hashes)
    ):
        raise ValueError("segments, features and input hashes must have identical unique IDs")
    revision = Repository(connection).revision(match_id) + 1
    changed = False
    for segment in segments:
        sid = segment.segment_id
        feature = feature_map[sid]
        if (
            feature.segment_id != sid
            or not isinstance(input_hashes[sid], str)
            or not input_hashes[sid]
        ):
            raise ValueError("invalid feature binding or input hash")
        feature_json, blob = _feature_payload(feature)
        payload = (
            input_hashes[sid],
            _dumps(segment),
            feature_json,
            "float32",
            "[512]",
            "little",
            blob,
        )
        previous = connection.execute(
            "SELECT input_hash,segment_json,feature_json,dtype,shape_json,byte_order,vector_blob "
            "FROM dual_stream_segments WHERE match_id=? AND segment_id=?",
            (match_id, sid),
        ).fetchone()
        if previous:
            if tuple(previous) != payload:
                raise StorageConflict("dual-stream segment already contains different evidence")
            continue
        _project_segment(connection, match_id, segment)
        connection.execute(
            "INSERT INTO dual_stream_segments(match_id,segment_id,created_revision,input_hash,segment_json,"
            "feature_json,dtype,shape_json,byte_order,vector_blob) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (match_id, sid, revision, *payload),
        )
        changed = True
    return changed


def put_dual_stream_plan(
    connection: sqlite3.Connection,
    match_id: str,
    plan: IncrementalPlan,
    *,
    players: Sequence[PlayerProfile],
) -> None:
    """Persist decisions and the final gallery at R+1, preserving every older row."""
    _require_transaction(connection)
    repo = Repository(connection)
    if plan.from_revision != repo.revision(match_id) or plan.to_revision != plan.from_revision + 1:
        raise RevisionConflict("dual-stream plan is not based on the current revision")
    if not isinstance(plan, IncrementalPlan):
        raise TypeError("Expected IncrementalPlan")
    if any(not isinstance(profile, PlayerProfile) for profile in players):
        raise TypeError("Expected PlayerProfile")
    profile_map = {profile.player_id: profile for profile in players}
    if len(profile_map) != len(players):
        raise ValueError("duplicate player profile")
    previous_snapshot = read_dual_stream_snapshot(connection, match_id)
    if not {p.player_id for p in previous_snapshot.players} <= profile_map.keys():
        raise ValueError("final gallery must retain all historical player profiles")
    legacy_ids = [legacy_player_id(pid) for pid in profile_map]
    if len(set(legacy_ids)) != len(legacy_ids):
        raise StorageConflict("canonical player IDs collide in the legacy projection")
    seen_prototypes = set()
    for player in players:
        if (
            not isinstance(player.player_id, str)
            or not player.player_id.strip()
            or not isinstance(player.display_name, str)
            or not player.display_name.strip()
            or player.status not in {"active", "merged", "archived"}
            or type(player.created_at_revision) is not int
            or not 1 <= player.created_at_revision <= plan.to_revision
            or not isinstance(player.prototypes, tuple)
            or (
                player.confirmed_jersey_number is not None
                and (
                    type(player.confirmed_jersey_number) is not int
                    or not 0 <= player.confirmed_jersey_number <= 99
                )
            )
        ):
            raise ValueError("invalid player profile metadata")
        legacy_id = legacy_player_id(player.player_id)
        old = connection.execute(
            "SELECT 1 FROM players WHERE match_id=? AND player_id=?",
            (match_id, legacy_id),
        ).fetchone()
        canonical_old = next(
            (p for p in previous_snapshot.players if p.player_id == player.player_id), None
        )
        if old and canonical_old is None:
            raise StorageConflict("new dual-stream player collides with an existing player")
        if canonical_old is None and player.created_at_revision != plan.to_revision:
            raise ValueError("new player must be created at the commit revision")
        if canonical_old is not None and (
            player.created_at_revision != canonical_old.created_at_revision
            or player.display_name != canonical_old.display_name
            or player.confirmed_jersey_number != canonical_old.confirmed_jersey_number
        ):
            raise StorageConflict("existing player identity metadata is immutable")
        projection = Player(
            match_id=match_id,
            player_id=legacy_id,
            display_name=player.display_name,
            jersey_number=str(player.confirmed_jersey_number)
            if player.confirmed_jersey_number is not None
            else None,
        )
        if not old:
            repo.put_player(projection)
        prototype_ids = []
        for prototype in player.prototypes:
            if not isinstance(prototype, PlayerPrototype):
                raise TypeError("Expected PlayerPrototype")
            if prototype.prototype_id in seen_prototypes:
                raise ValueError("duplicate prototype ID in the final gallery")
            seen_prototypes.add(prototype.prototype_id)
            if (
                not isinstance(prototype.prototype_id, str)
                or not prototype.prototype_id.strip()
                or prototype.view_type
                not in {"frontal", "dorsal_with_number", "lateral", "generic"}
                or isinstance(prototype.weight, bool)
                or not math.isfinite(prototype.weight)
                or not 0 <= prototype.weight <= 1
                or type(prototype.created_at_revision) is not int
            ):
                raise ValueError("invalid prototype metadata")
            if prototype.player_id != player.player_id:
                raise ValueError("prototype owner differs from its profile")
            if not player.created_at_revision <= prototype.created_at_revision <= plan.to_revision:
                raise ValueError("prototype revision is outside its profile history")
            feature_row = connection.execute(
                "SELECT feature_json,vector_blob FROM dual_stream_segments WHERE match_id=? AND segment_id=?",
                (match_id, prototype.feature.segment_id),
            ).fetchone()
            if feature_row is None or tuple(feature_row) != _feature_payload(prototype.feature):
                raise StorageConflict(
                    "prototype feature is not the stored immutable segment feature"
                )
            payload = dual_stream_json(prototype)
            del payload["feature"]
            serialized = _dumps(payload)
            row = connection.execute(
                "SELECT player_id,segment_id,json FROM prototypes WHERE match_id=? AND prototype_id=?",
                (match_id, prototype.prototype_id),
            ).fetchone()
            expected = (player.player_id, prototype.feature.segment_id, serialized)
            if row and tuple(row) != expected:
                raise StorageConflict("prototype ID was reused with different evidence")
            if not row:
                if prototype.created_at_revision != plan.to_revision:
                    raise ValueError("new prototype must be created at the commit revision")
                connection.execute(
                    "INSERT INTO prototypes(match_id,prototype_id,player_id,legacy_player_id,segment_id,created_at_revision,json) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (
                        match_id,
                        prototype.prototype_id,
                        player.player_id,
                        legacy_id,
                        prototype.feature.segment_id,
                        prototype.created_at_revision,
                        serialized,
                    ),
                )
            prototype_ids.append(prototype.prototype_id)
        payload = dual_stream_json(player)
        del payload["prototypes"]
        payload["prototype_ids"] = prototype_ids
        serialized = _dumps(payload)
        row = connection.execute(
            "SELECT json FROM dual_stream_profiles WHERE match_id=? AND player_id=? ORDER BY revision DESC LIMIT 1",
            (match_id, player.player_id),
        ).fetchone()
        if row is None or row[0] != serialized:
            connection.execute(
                "INSERT INTO dual_stream_profiles(match_id,player_id,legacy_player_id,revision,json) VALUES (?,?,?,?,?)",
                (match_id, player.player_id, legacy_id, plan.to_revision, serialized),
            )
    if len({d.segment_id for d in plan.decisions}) != len(plan.decisions):
        raise ValueError("duplicate plan decision")
    fresh_ids = {
        row[0]
        for row in connection.execute(
            "SELECT segment_id FROM dual_stream_segments WHERE match_id=? AND created_revision=?",
            (match_id, plan.to_revision),
        )
    }
    if not fresh_ids <= {decision.segment_id for decision in plan.decisions}:
        raise ValueError("plan omits newly persisted segment decisions")
    for decision in plan.decisions:
        if not isinstance(decision, IncrementalDecision) or not isinstance(
            decision.outcome, DecisionOutcome
        ):
            raise TypeError("Expected IncrementalDecision with DecisionOutcome")
        if (
            math.isnan(decision.best_distance)
            or decision.best_distance < 0
            or math.isnan(decision.competitive_margin)
            or decision.competitive_margin < 0
            or not isinstance(decision.evidence, dict)
        ):
            raise ValueError("invalid incremental decision metadata")
        target = decision.target_player_id
        if decision.outcome == DecisionOutcome.CREATED_NEW:
            if target is not None:
                raise ValueError("created_new_identity must preserve target_player_id=None")
            target = decision.evidence["new_player_id"]
        if decision.outcome == DecisionOutcome.INSUFFICIENT and target is not None:
            raise ValueError("insufficient evidence cannot assign a player")
        if decision.outcome != DecisionOutcome.INSUFFICIENT and target not in profile_map:
            raise ValueError("decision refers to an absent player")
        existing = connection.execute(
            "SELECT json FROM assignments WHERE match_id=? AND segment_id=? AND current=1",
            (match_id, decision.segment_id),
        ).fetchone()
        if existing:
            assignment = Assignment.model_validate_json(existing[0])
            if assignment.method != "automatic" or assignment.status == "confirmed":
                raise StorageConflict(
                    "dual-stream planning cannot replace protected human assignments"
                )
        projection = Assignment(
            match_id=match_id,
            segment_id=decision.segment_id,
            player_id=legacy_player_id(target) if target is not None else None,
            status="proposed" if target is not None else "unassigned",
            method="automatic",
            unassigned_reason=None if target is not None else "ambiguous",
            revision=plan.from_revision,
            evidence=[
                AssignmentEvidence(
                    kind=decision.outcome.value, reference_id=f"dual-stream:{plan.to_revision}"
                )
            ],
        )
        repo.put_assignment(projection, committed_revision=plan.to_revision, replace=True)
        connection.execute(
            "INSERT INTO dual_stream_decisions(match_id,segment_id,revision,json) VALUES (?,?,?,?)",
            (match_id, decision.segment_id, plan.to_revision, _dumps(decision)),
        )


def find_dual_stream_operation(
    connection: sqlite3.Connection,
    match_id: str,
    *,
    request_id: str | None = None,
    input_fingerprint: str | None = None,
) -> dict[str, Any] | None:
    """Find an immutable request binding or a previously committed content hash."""
    if (request_id is None) == (input_fingerprint is None):
        raise ValueError("provide exactly one of request_id or input_fingerprint")
    column, value = (
        ("request_id", request_id)
        if request_id is not None
        else ("payload_hash", input_fingerprint)
    )
    row = connection.execute(
        "SELECT request_id,payload_hash,payload_json,result_json FROM workflow_operations "
        f"WHERE match_id=? AND kind=? AND {column}=? ORDER BY request_id LIMIT 1",
        (match_id, OPERATION_KIND, value),
    ).fetchone()
    if row is None:
        return None
    return {
        "request_id": row[0],
        "input_fingerprint": row[1],
        "payload": json.loads(row[2]),
        "result": json.loads(row[3]),
    }


def put_dual_stream_operation(
    connection: sqlite3.Connection,
    match_id: str,
    request_id: str,
    input_fingerprint: str,
    payload: Any,
    result: Any,
) -> None:
    """Bind a request and its original result atomically; never overwrite an audit."""
    _require_transaction(connection)
    previous = find_dual_stream_operation(connection, match_id, request_id=request_id)
    values = (input_fingerprint, _dumps(payload), _dumps(result))
    if previous:
        if input_fingerprint != previous["input_fingerprint"]:
            raise StorageConflict("dual-stream request ID is already bound to another operation")
        return
    connection.execute(
        "INSERT INTO workflow_operations(match_id,kind,request_id,payload_hash,payload_json,result_json) "
        "VALUES (?,?,?,?,?,?)",
        (match_id, OPERATION_KIND, request_id, *values),
    )
