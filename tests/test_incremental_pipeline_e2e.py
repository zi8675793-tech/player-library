"""T3 integration: real PNG ingestion, frozen T2 planning and atomic SQLite writes.

The image-content backends keep these storage tests deterministic and offline.
They still exercise DualStreamExtractor's real validation, image loading and
packaging; model accuracy belongs to the separately frozen T1/T4 benchmark.
"""

import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import app.workflows.incremental as workflow
from app.contracts.identity import Assignment, AssignmentMethod, AssignmentStatus, Player
from app.features.dual_stream import (
    AppearanceEmbedding,
    DualStreamExtractor,
    JerseyNumberPrediction,
    TrackletSegment,
    TrackObservation,
)
from app.identity.constraints import SpatioTemporalConstraintEngine
from app.identity.matching import DecisionOutcome, IncrementalMatcher
from app.storage.database import connect, transaction
from app.storage.dual_stream import legacy_player_id, read_dual_stream_snapshot
from app.storage.repositories import Repository, RevisionConflict, StorageConflict

MATCH = "t3-basketball"


class ImageAppearance:
    """Read a synthetic appearance angle from actual decoded PNG pixels."""

    def __init__(self):
        self.calls = 0

    def extract_track(self, images):
        self.calls += 1
        angle = math.radians(int(images[0][0, 0, 1]))
        vector = np.zeros(512, dtype=np.float32)
        vector[:2] = math.cos(angle), math.sin(angle)
        vector /= np.linalg.norm(vector)
        return AppearanceEmbedding(vector, float(images[0][0, 0, 2]) / 255, "t3-image-test-v1")


class ImageJersey:
    def extract_track(self, images):
        number = int(images[0][0, 0, 0]) or None
        return JerseyNumberPrediction(
            number,
            0.99 if number is not None else 0.0,
            number is not None,
            {number: 2.97} if number is not None else {},
            str(number) if number is not None else None,
        )


class RecordingMatcher(IncrementalMatcher):
    def __init__(self):
        super().__init__()
        self.calls = []

    def plan(self, features, players, **kwargs):
        self.calls.append(
            (
                tuple(feature.segment_id for feature in features),
                {proto.feature.segment_id for player in players for proto in player.prototypes},
                dict(kwargs.get("player_assignments") or {}),
            )
        )
        return super().plan(features, players, **kwargs)


@dataclass
class Pipeline:
    db_path: Path
    crop_root: Path
    connection: sqlite3.Connection
    extractor: DualStreamExtractor
    appearance: ImageAppearance
    sequence: int = 0

    def segment(self, video, *, track=1, angle=10, number=27, x=0):
        observations = []
        for frame in (0, 6, 12):
            relative = f"{video}/samples/track-{track}-{frame}.png"
            path = self.crop_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (24, 48), (number or 0, angle, 230)).save(path)
            observations.append(
                TrackObservation(
                    str(video),
                    track,
                    frame,
                    frame / 60,
                    (float(x), 0.0, float(x + 24), 48.0),
                    0.95,
                    relative,
                )
            )
        return TrackletSegment(
            f"{video}_{track}_0", str(video), track, 0, 12, 0.0, 0.2, 3, tuple(observations)
        )

    def submit(self, segments, *, expected=None, request_id=None, batch_id=None, matcher=None):
        self.sequence += 1
        if expected is None:
            expected = read_dual_stream_snapshot(self.connection, MATCH).revision
        return workflow.update_dual_stream(
            self.connection,
            match_id=MATCH,
            batch_id=batch_id or f"batch-{self.sequence}",
            request_id=request_id or f"request-{self.sequence}",
            expected_revision=expected,
            segments=segments,
            extractor=self.extractor,
            matcher=matcher,
        )

    def snapshot(self):
        return read_dual_stream_snapshot(self.connection, MATCH)


@pytest.fixture
def pipeline(tmp_path):
    db_path = tmp_path / "t3.sqlite3"
    appearance = ImageAppearance()
    connection = connect(db_path)
    context = Pipeline(
        db_path,
        tmp_path / "track",
        connection,
        DualStreamExtractor(
            tmp_path / "track", appearance_extractor=appearance, jersey_extractor=ImageJersey()
        ),
        appearance,
    )
    try:
        yield context
    finally:
        connection.close()


def _count(connection, table):
    return connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def test_sequential_batches_resolve_pending_and_expand_prototypes(pipeline):
    player_id = None
    expected_sizes = (0, 2, 3, 4, 5)
    for revision, (video, prototype_count) in enumerate(
        zip((20, 6, 43, 50, 68), expected_sizes, strict=True), start=1
    ):
        segment = pipeline.segment(video, number=None if revision == 1 else 27)
        result = pipeline.submit([segment], expected=revision - 1)
        snapshot = pipeline.snapshot()
        assert result.data["changed"] is True
        assert result.revision == snapshot.revision == revision
        assert len(snapshot.segments) == len(snapshot.features) == revision
        if revision == 1:
            assert snapshot.players == ()
            assert snapshot.assignments == {}
            assert snapshot.decisions[segment.segment_id].outcome == DecisionOutcome.INSUFFICIENT
        else:
            assert len(snapshot.players) == 1
            player = snapshot.players[0]
            player_id = player_id or player.player_id
            assert player.player_id == player_id
            assert player.confirmed_jersey_number == 27
            assert len(player.prototypes) == prototype_count
            assert set(snapshot.assignments.values()) == {player_id}
            assert len(snapshot.assignments) == revision
            assert all(proto.created_at_revision <= revision for proto in player.prototypes)
        assert pipeline.connection.execute("PRAGMA foreign_key_check").fetchall() == []

    assert _count(pipeline.connection, "revisions") == 5
    assert _count(pipeline.connection, "workflow_operations") == 5
    history = pipeline.connection.execute(
        "SELECT revision,player_id,current FROM assignments "
        "WHERE match_id=? AND segment_id=? ORDER BY revision",
        (MATCH, "20_1_0"),
    ).fetchall()
    assert history[0][0] == 1 and history[0][1] is None and history[0][2] == 0
    assert history[-1][1] == legacy_player_id(player_id) and history[-1][2] == 1
    for row in pipeline.connection.execute("SELECT json FROM players"):
        assert Player.model_validate_json(row[0]).player_id == legacy_player_id(player_id)
    for row in pipeline.connection.execute("SELECT json FROM assignments WHERE current=1"):
        assignment = Assignment.model_validate_json(row[0])
        assert assignment.status == AssignmentStatus.PROPOSED
        assert assignment.method == AssignmentMethod.AUTOMATIC


def test_same_request_and_same_content_replay_skip_models(pipeline):
    segments = [pipeline.segment(20), pipeline.segment(6)]
    original = pipeline.submit(segments, request_id="stable", expected=0)
    calls = pipeline.appearance.calls
    audit_count = _count(pipeline.connection, "workflow_operations")
    replay = pipeline.submit(segments, request_id="stable", expected=0)
    reordered = pipeline.submit(
        list(reversed(segments)), request_id="new-handoff", batch_id="renamed-batch", expected=0
    )
    assert original.data["changed"] is True
    assert replay.data["changed"] is reordered.data["changed"] is False
    assert replay.revision == reordered.revision == 1
    assert original.data["input_fingerprint"] == reordered.data["input_fingerprint"]
    assert pipeline.appearance.calls == calls
    assert _count(pipeline.connection, "revisions") == 1
    assert _count(pipeline.connection, "workflow_operations") >= audit_count


def test_old_batch_replay_reports_current_revision_without_reverting(pipeline):
    first = pipeline.segment(20)
    pipeline.submit([first], request_id="first", expected=0)
    pipeline.submit([pipeline.segment(6)], expected=1)
    player_id = pipeline.snapshot().players[0].player_id
    replay = pipeline.submit([first], request_id="first", expected=0)
    assert replay.data["changed"] is False
    assert replay.revision == pipeline.snapshot().revision == 2
    assert pipeline.snapshot().assignments[first.segment_id] == player_id


def test_mixed_overlap_batch_extracts_and_stores_only_fresh_evidence(pipeline):
    first = pipeline.segment(20)
    pipeline.submit([first], expected=0)
    before = pipeline.snapshot().features[first.segment_id].appearance.vector.copy()
    second = pipeline.segment(6)
    result = pipeline.submit([first, second], expected=1, request_id="overlap")
    assert result.data["changed"] is True
    assert result.revision == 2
    assert pipeline.appearance.calls == 2
    assert len(pipeline.snapshot().segments) == 2
    assert len(pipeline.snapshot().players) == 1
    np.testing.assert_array_equal(
        pipeline.snapshot().features[first.segment_id].appearance.vector, before
    )
    assert _count(pipeline.connection, "dual_stream_segments") == 2
    replay = pipeline.submit([second, first], expected=1, request_id="overlap")
    assert replay.data["changed"] is False
    assert replay.revision == 2


def test_request_id_collision_is_rejected_before_stale_revision(pipeline):
    pipeline.submit([pipeline.segment(20)], request_id="collision", expected=0)
    with pytest.raises(StorageConflict):
        pipeline.submit([pipeline.segment(6)], request_id="collision", expected=0)
    assert pipeline.snapshot().revision == 1
    assert len(pipeline.snapshot().segments) == 1


def test_new_evidence_with_stale_revision_cannot_write(pipeline):
    pipeline.submit([pipeline.segment(20)], expected=0)
    with pytest.raises(RevisionConflict):
        pipeline.submit([pipeline.segment(6)], expected=0)
    assert pipeline.snapshot().revision == 1
    assert _count(pipeline.connection, "dual_stream_segments") == 1
    assert _count(pipeline.connection, "workflow_operations") == 1


def test_changed_crop_bytes_are_conflicting_evidence(pipeline):
    segment = pipeline.segment(20)
    pipeline.submit([segment], expected=0)
    image_path = pipeline.crop_root / segment.observations[0].crop_path
    Image.new("RGB", (24, 48), (10, 80, 230)).save(image_path)
    with pytest.raises(StorageConflict):
        pipeline.submit([segment], expected=1)
    assert pipeline.snapshot().revision == 1
    assert _count(pipeline.connection, "dual_stream_segments") == 1


def test_operation_receipt_and_machine_evidence_are_immutable(pipeline):
    pipeline.submit([pipeline.segment(20)], expected=0)
    for statement in (
        "UPDATE workflow_operations SET result_json='{}'",
        "DELETE FROM workflow_operations",
        "UPDATE dual_stream_segments SET input_hash='changed'",
        "DELETE FROM dual_stream_segments",
    ):
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            with transaction(pipeline.connection):
                pipeline.connection.execute(statement)
    assert pipeline.snapshot().revision == 1
    assert _count(pipeline.connection, "workflow_operations") == 1
    assert _count(pipeline.connection, "dual_stream_segments") == 1


def test_crop_mutation_during_extraction_rolls_back_and_can_retry(pipeline, monkeypatch):
    segment = pipeline.segment(20)
    image_path = pipeline.crop_root / segment.observations[0].crop_path
    original = pipeline.appearance.extract_track

    def mutate_after_decode(images):
        feature = original(images)
        Image.new("RGB", (24, 48), (10, 80, 230)).save(image_path)
        return feature

    monkeypatch.setattr(pipeline.appearance, "extract_track", mutate_after_decode)
    with pytest.raises(StorageConflict, match="changed during"):
        pipeline.submit([segment], request_id="race", expected=0)
    assert _count(pipeline.connection, "matches") == 0
    assert _count(pipeline.connection, "dual_stream_segments") == 0
    assert _count(pipeline.connection, "workflow_operations") == 0
    monkeypatch.undo()
    segment = pipeline.segment(20)
    retry = pipeline.submit([segment], request_id="race", expected=0)
    assert retry.data["changed"] is True
    assert retry.revision == 1


@pytest.mark.parametrize("fault", ["operation", "revision"])
def test_interruption_rolls_back_every_table_and_exact_input_replays(pipeline, monkeypatch, fault):
    segments = [pipeline.segment(20), pipeline.segment(6)]
    if fault == "operation":
        original = workflow.put_dual_stream_operation

        def fail(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("simulated interruption after audit write")

        monkeypatch.setattr(workflow, "put_dual_stream_operation", fail)
    else:
        original = Repository.advance_revision

        def fail(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("simulated interruption after revision write")

        monkeypatch.setattr(Repository, "advance_revision", fail)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        pipeline.submit(segments, request_id="interrupted", expected=0)
    assert pipeline.connection.in_transaction is False
    for table in (
        "matches",
        "players",
        "assignments",
        "prototypes",
        "dual_stream_segments",
        "dual_stream_profiles",
        "dual_stream_decisions",
        "workflow_operations",
        "revisions",
    ):
        assert _count(pipeline.connection, table) == 0, table
    monkeypatch.undo()
    replay = pipeline.submit(segments, request_id="interrupted", expected=0)
    assert replay.data["changed"] is True
    assert replay.revision == 1
    assert len(pipeline.snapshot().players) == 1


def test_reopen_restores_float32_features_and_canonical_profile_ids(pipeline):
    segments = [pipeline.segment(20, angle=7), pipeline.segment(6, angle=13)]
    pipeline.submit(segments, expected=0)
    before = pipeline.snapshot()
    other = connect(pipeline.db_path)
    try:
        after = read_dual_stream_snapshot(other, MATCH)
        assert after.revision == before.revision
        assert after.assignments == before.assignments
        assert after.players[0].player_id == before.players[0].player_id
        assert after.segments == before.segments
        for sid, feature in before.features.items():
            restored = after.features[sid]
            assert restored.appearance.vector.dtype == np.float32
            np.testing.assert_array_equal(restored.appearance.vector, feature.appearance.vector)
            assert restored.jersey == feature.jersey
        assert [p.prototype_id for p in after.players[0].prototypes] == [
            p.prototype_id for p in before.players[0].prototypes
        ]
    finally:
        other.close()


def test_historical_snapshots_keep_pending_resolution_and_prototype_versions(pipeline):
    first = pipeline.segment(20, number=None)
    pipeline.submit([first], expected=0)
    pipeline.submit([pipeline.segment(6)], expected=1)
    pipeline.submit([pipeline.segment(43)], expected=2)
    snapshots = [
        read_dual_stream_snapshot(pipeline.connection, MATCH, revision=revision)
        for revision in range(4)
    ]
    assert [snapshot.revision for snapshot in snapshots] == [0, 1, 2, 3]
    assert [len(snapshot.segments) for snapshot in snapshots] == [0, 1, 2, 3]
    assert [len(snapshot.input_hashes) for snapshot in snapshots] == [0, 1, 2, 3]
    assert snapshots[0].players == snapshots[1].players == ()
    assert snapshots[1].decisions[first.segment_id].outcome == DecisionOutcome.INSUFFICIENT
    assert snapshots[1].assignments == {}
    assert len(snapshots[2].players[0].prototypes) == 2
    assert len(snapshots[3].players[0].prototypes) == 3
    assert snapshots[2].players[0].player_id == snapshots[3].players[0].player_id
    assert snapshots[2].assignments[first.segment_id] == snapshots[3].assignments[first.segment_id]
    with pytest.raises(ValueError, match="revision"):
        read_dual_stream_snapshot(pipeline.connection, MATCH, revision=4)


def test_new_queries_keep_fixed_revision_decisions_until_a_later_batch(pipeline):
    pipeline.submit(
        [pipeline.segment(20, angle=0, number=None), pipeline.segment(6, angle=0, number=None)],
        expected=0,
    )
    b_segments = [
        pipeline.segment(43, angle=70, number=None),
        pipeline.segment(50, angle=80, number=None),
    ]
    ambiguous = pipeline.segment(68, angle=115, number=None)
    matcher = RecordingMatcher()
    pipeline.submit([*b_segments, ambiguous], expected=1, matcher=matcher)
    snapshot = pipeline.snapshot()
    assert len(snapshot.players) == 2
    assert ambiguous.segment_id not in snapshot.assignments
    assert snapshot.decisions[ambiguous.segment_id].outcome == DecisionOutcome.INSUFFICIENT
    assert snapshot.decisions[ambiguous.segment_id].evidence["snapshot_revision"] == 1
    assert sum(ambiguous.segment_id in call[0] for call in matcher.calls) == 1
    b_player = snapshot.assignments[b_segments[0].segment_id]
    pipeline.submit([pipeline.segment(70, angle=80, number=None)], expected=2)
    assert pipeline.snapshot().assignments[ambiguous.segment_id] == b_player


def test_manual_confirmation_divergence_cannot_be_automatically_overwritten(pipeline):
    first = pipeline.segment(20)
    pipeline.submit([first, pipeline.segment(6)], expected=0)
    canonical = pipeline.snapshot().assignments[first.segment_id]
    manual = Assignment(
        match_id=MATCH,
        segment_id=first.segment_id,
        player_id=legacy_player_id(canonical),
        status=AssignmentStatus.CONFIRMED,
        method=AssignmentMethod.MANUAL,
        revision=1,
    )
    repo = Repository(pipeline.connection)
    with transaction(pipeline.connection):
        repo.put_assignment(manual, committed_revision=2, replace=True)
        repo.advance_revision(MATCH, 1, actor="reviewer", reason="confirmation", changes={})
    with pytest.raises(StorageConflict, match="review or restore"):
        pipeline.submit([pipeline.segment(43)], expected=2)
    assert repo.revision(MATCH) == 2
    row = pipeline.connection.execute(
        "SELECT json FROM assignments WHERE match_id=? AND segment_id=? AND current=1",
        (MATCH, first.segment_id),
    ).fetchone()
    persisted = Assignment.model_validate_json(row[0])
    assert persisted.method == AssignmentMethod.MANUAL
    assert persisted.status == AssignmentStatus.CONFIRMED
    assert persisted.revision == 2
    assert _count(pipeline.connection, "dual_stream_segments") == 2
    assert _count(pipeline.connection, "workflow_operations") == 1


def test_leave_one_out_reassigns_history_and_retracts_old_owner_prototype(pipeline):
    a_segments = [
        pipeline.segment(20, angle=0, number=None),
        pipeline.segment(6, angle=10, number=None),
    ]
    pipeline.submit(a_segments, expected=0)
    a_player = pipeline.snapshot().assignments[a_segments[0].segment_id]
    b_segments = [
        pipeline.segment(43, angle=80, number=None),
        pipeline.segment(50, angle=90, number=None),
    ]
    pipeline.submit(b_segments, expected=1)
    b_player = pipeline.snapshot().assignments[b_segments[0].segment_id]
    assert a_player != b_player
    query = pipeline.segment(68, angle=40, number=None)
    pipeline.submit([query], expected=2)
    assert pipeline.snapshot().assignments[query.segment_id] == a_player
    old_prototype_id = next(
        proto.prototype_id
        for player in pipeline.snapshot().players
        for proto in player.prototypes
        if proto.feature.segment_id == query.segment_id
    )

    matcher = RecordingMatcher()
    result = pipeline.submit(
        [pipeline.segment(70, angle=65, number=None)], expected=3, matcher=matcher
    )
    snapshot = pipeline.snapshot()
    assert result.revision == snapshot.revision == 4
    assert snapshot.assignments[query.segment_id] == b_player
    profiles = {player.player_id: player for player in snapshot.players}
    assert query.segment_id not in {p.feature.segment_id for p in profiles[a_player].prototypes}
    assert query.segment_id in {p.feature.segment_id for p in profiles[b_player].prototypes}
    query_calls = [call for call in matcher.calls if query.segment_id in call[0]]
    assert query_calls
    assert all(
        query.segment_id not in call[1] and query.segment_id not in call[2] for call in query_calls
    )
    history = pipeline.connection.execute(
        "SELECT revision,player_id,current FROM assignments "
        "WHERE match_id=? AND segment_id=? ORDER BY revision",
        (MATCH, query.segment_id),
    ).fetchall()
    assert history[0][0] == 3 and history[0][1] == legacy_player_id(a_player)
    assert history[0][2] == 0
    assert history[-1][0] == 4 and history[-1][1] == legacy_player_id(b_player)
    assert history[-1][2] == 1
    assert snapshot.decisions[query.segment_id].evidence["reassessment"] == "leave_one_out"
    historical = read_dual_stream_snapshot(pipeline.connection, MATCH, revision=3)
    assert historical.assignments[query.segment_id] == a_player
    assert old_prototype_id in {
        proto.prototype_id for player in historical.players for proto in player.prototypes
    }
    assert (
        pipeline.connection.execute(
            "SELECT count(*) FROM prototypes WHERE match_id=? AND prototype_id=?",
            (MATCH, old_prototype_id),
        ).fetchone()[0]
        == 1
    )


def test_cannot_link_uses_all_history_across_batches(pipeline):
    first = [pipeline.segment(20), pipeline.segment(6)]
    pipeline.submit(first, expected=0)
    player_id = pipeline.snapshot().players[0].player_id
    conflicting = pipeline.segment(20, track=2, x=200, number=None)
    pipeline.submit([conflicting], expected=1)
    snapshot = pipeline.snapshot()
    assert conflicting.segment_id not in snapshot.assignments
    assert snapshot.decisions[conflicting.segment_id].outcome == DecisionOutcome.INSUFFICIENT
    engine = SpatioTemporalConstraintEngine(
        segments=snapshot.segments, player_assignments=snapshot.assignments
    )
    assert engine.is_segment_forbidden_for_player(conflicting.segment_id, player_id)
    pipeline.submit([pipeline.segment(6, track=2, x=200, number=None)], expected=2)
    snapshot = pipeline.snapshot()
    assert len(snapshot.players) == 2
    assert snapshot.assignments[conflicting.segment_id] != player_id
    engine = SpatioTemporalConstraintEngine(segments=snapshot.segments)
    assigned = list(snapshot.assignments.items())
    for index, (left, left_player) in enumerate(assigned):
        for right, right_player in assigned[index + 1 :]:
            if left_player == right_player:
                assert engine.pair_relation(left, right) == "clear"


def test_jersey_conflict_never_pollutes_same_appearance_profile(pipeline):
    pipeline.submit([pipeline.segment(20), pipeline.segment(6)], expected=0)
    original_player = pipeline.snapshot().players[0].player_id
    other = pipeline.segment(43, number=10)
    pipeline.submit([other], expected=1)
    assert pipeline.snapshot().assignments[other.segment_id] != original_player
    pipeline.submit([pipeline.segment(50, number=10)], expected=2)
    snapshot = pipeline.snapshot()
    assert snapshot.assignments[other.segment_id] != original_player
    assert {player.confirmed_jersey_number for player in snapshot.players} == {10, 27}
    for player in snapshot.players:
        assert {p.feature.jersey.number for p in player.prototypes} == {
            player.confirmed_jersey_number
        }


def test_single_video_numbered_batch_creates_player_at_cold_start(pipeline):
    segment = pipeline.segment(20, number=27)
    result = pipeline.submit([segment], expected=0)
    snapshot = pipeline.snapshot()
    assert result.data["changed"] is True
    assert result.revision == 1
    assert len(snapshot.players) == 1
    player = snapshot.players[0]
    assert player.confirmed_jersey_number == 27
    assert snapshot.decisions[segment.segment_id].outcome == DecisionOutcome.CREATED_NEW
    assert (
        snapshot.decisions[segment.segment_id].evidence["reason"] == "confirmed_jersey_number_seed"
    )
    assert snapshot.assignments[segment.segment_id] == player.player_id


def test_jersey_safety_checks_assigned_evidence_when_prototype_capacity_is_full(pipeline):
    limited = IncrementalMatcher(max_prototypes=2)
    pipeline.submit(
        [pipeline.segment(20, number=None), pipeline.segment(6, number=None)],
        expected=0,
        matcher=limited,
    )
    player_id = pipeline.snapshot().players[0].player_id
    known = pipeline.segment(43, number=27)
    pipeline.submit([known], expected=1, matcher=limited)
    snapshot = pipeline.snapshot()
    assert snapshot.assignments[known.segment_id] == player_id
    assert len(snapshot.players[0].prototypes) == 2
    assert all(proto.feature.jersey.number is None for proto in snapshot.players[0].prototypes)

    # The 27 lives only on the assigned segment, not on a prototype. Planning
    # must still treat it as physical evidence and defer the 10 instead of
    # letting it reach the commit validator.
    conflicting = pipeline.segment(50, number=10)
    result = pipeline.submit([conflicting], expected=2, matcher=limited)
    snapshot = pipeline.snapshot()
    assert result.revision == snapshot.revision == 3
    assert snapshot.assignments[known.segment_id] == player_id
    assert snapshot.assignments.get(conflicting.segment_id) != player_id
    decision = snapshot.decisions[conflicting.segment_id]
    # The 10 is physically separated, so it may seed its own numbered profile;
    # it must never join the profile carrying 27 evidence.
    assert decision.outcome in {DecisionOutcome.INSUFFICIENT, DecisionOutcome.CREATED_NEW}
    assert decision.evidence.get("reason") != "matched_existing"
    assert _count(pipeline.connection, "dual_stream_segments") == 4
    assert _count(pipeline.connection, "workflow_operations") == 3


def test_begin_immediate_precedes_snapshot_and_competing_writer_cannot_commit(pipeline):
    statements = []
    pipeline.connection.set_trace_callback(statements.append)
    pipeline.submit([pipeline.segment(20)], expected=0)
    pipeline.connection.set_trace_callback(None)
    assert statements[0].strip().upper() == "BEGIN IMMEDIATE"
    competing = connect(pipeline.db_path, timeout=0)
    try:
        with transaction(pipeline.connection):
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                workflow.update_dual_stream(
                    competing,
                    match_id=MATCH,
                    batch_id="other",
                    request_id="other",
                    expected_revision=1,
                    segments=[pipeline.segment(6)],
                    extractor=pipeline.extractor,
                )
        assert pipeline.snapshot().revision == 1
        assert _count(pipeline.connection, "dual_stream_segments") == 1
    finally:
        competing.close()
