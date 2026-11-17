"""Cross-batch jersey-number evidence must use one consistent definition.

Planning used `confirmed_jersey_number + selected prototypes`, while the commit
validator scanned every assigned segment. A profile whose assigned segment
carried number 5 but whose prototype slots were full could therefore accept a
23 candidate during planning and fail the commit check. These regression tests
pin the unified "observed numbers" definition:

* a number seen on any assigned segment blocks an incompatible candidate;
* digit-compatible numbers (5 / 15) are not a conflict;
* a confirmed 27 profile never absorbs a 5 candidate.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import app.workflows.incremental as workflow
from app.features.dual_stream import (
    AppearanceEmbedding,
    DualStreamExtractor,
    DualStreamFeature,
    JerseyNumberPrediction,
    TrackletSegment,
    TrackObservation,
)
from app.identity.matching import (
    DecisionOutcome,
    IncrementalDecision,
    IncrementalMatcher,
    PlayerProfile,
    PlayerPrototype,
    _has_conflicting_numbers,
    observed_numbers,
)
from app.storage.database import connect
from app.storage.dual_stream import read_dual_stream_snapshot
from app.storage.repositories import StorageConflict

MATCH = "number-evidence-basketball"


class ImageAppearance:
    """Angle taken from the PNG's green channel; quality from the blue channel."""

    def extract_track(self, images):
        angle = math.radians(int(images[0][0, 0, 1]))
        vector = np.zeros(512, dtype=np.float32)
        vector[:2] = math.cos(angle), math.sin(angle)
        vector /= np.linalg.norm(vector)
        return AppearanceEmbedding(vector, float(images[0][0, 0, 2]) / 255, "number-evidence-v1")


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


@dataclass
class Pipeline:
    root: Path
    connection: object
    extractor: DualStreamExtractor
    sequence: int = 0

    def segment(self, video, *, track=1, number=None, angle=10, x=0):
        observations = []
        for frame in (0, 6, 12):
            relative = f"{video}/samples/track-{track}-{frame}.png"
            path = self.root / relative
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

    def submit(self, segments, *, expected, matcher):
        self.sequence += 1
        return workflow.update_dual_stream(
            self.connection,
            match_id=MATCH,
            batch_id=f"batch-{self.sequence}",
            request_id=f"request-{self.sequence}",
            expected_revision=expected,
            segments=segments,
            extractor=self.extractor,
            matcher=matcher,
        )

    def snapshot(self):
        return read_dual_stream_snapshot(self.connection, MATCH)


@pytest.fixture
def pipeline(tmp_path):
    connection = connect(tmp_path / "numbers.sqlite3")
    context = Pipeline(
        tmp_path / "track",
        connection,
        DualStreamExtractor(
            tmp_path / "track",
            appearance_extractor=ImageAppearance(),
            jersey_extractor=ImageJersey(),
        ),
    )
    try:
        yield context
    finally:
        connection.close()


def conflicting_profiles(snapshot) -> list[tuple[PlayerProfile, set[int]]]:
    """Return every committed profile whose observed numbers contain a conflict."""
    broken = []
    for player in snapshot.players:
        numbers = observed_numbers(player, snapshot.assignments, snapshot.features)
        if _has_conflicting_numbers(numbers):
            broken.append((player, numbers))
    return broken


def anonymous_profile(pipeline, matcher):
    pipeline.submit(
        [
            pipeline.segment(20, number=None, angle=10),
            pipeline.segment(6, number=None, angle=10),
        ],
        expected=0,
        matcher=matcher,
    )
    snapshot = pipeline.snapshot()
    assert len(snapshot.players) == 1
    player = snapshot.players[0]
    assert player.confirmed_jersey_number is None
    return player


def test_assigned_number_blocks_conflicting_candidate_across_batches(pipeline):
    """5号只在已归属段里（原型已满），23号不得被吸进同一档案。"""
    limited = IncrementalMatcher(max_prototypes=2)
    profile = anonymous_profile(pipeline, limited)

    five = pipeline.segment(43, number=5, angle=10)
    pipeline.submit([five], expected=1, matcher=limited)
    snapshot = pipeline.snapshot()
    assert snapshot.assignments[five.segment_id] == profile.player_id
    # Prototype capacity is full, so the 5 never becomes a prototype: exactly
    # the gap between planning-time and commit-time number knowledge.
    assert all(proto.feature.jersey.number is None for proto in snapshot.players[0].prototypes)

    twenty_three = pipeline.segment(63, number=23, angle=10)
    result = pipeline.submit([twenty_three], expected=2, matcher=limited)

    assert result.revision == 3
    snapshot = pipeline.snapshot()
    assert snapshot.revision == 3
    assert snapshot.assignments[five.segment_id] == profile.player_id
    assert snapshot.assignments.get(twenty_three.segment_id) != profile.player_id
    assert conflicting_profiles(snapshot) == []


def test_digit_compatible_fifteen_is_not_a_cross_batch_conflict(pipeline):
    """5号与15号在同一档案内属于既有数字兼容语义，不得误判为冲突。"""
    limited = IncrementalMatcher(max_prototypes=2)
    profile = anonymous_profile(pipeline, limited)

    five = pipeline.segment(43, number=5, angle=10)
    pipeline.submit([five], expected=1, matcher=limited)
    fifteen = pipeline.segment(50, number=15, angle=10)
    result = pipeline.submit([fifteen], expected=2, matcher=limited)

    assert result.revision == 3
    snapshot = pipeline.snapshot()
    assert snapshot.assignments[five.segment_id] == profile.player_id
    assert snapshot.assignments[fifteen.segment_id] == profile.player_id
    assert conflicting_profiles(snapshot) == []


def test_confirmed_27_profile_never_absorbs_a_five(pipeline):
    """命名兼容：已有27号确认档案时，5号候选不得被吸入。"""
    matcher = IncrementalMatcher()
    twenty_seven = pipeline.segment(20, number=27, angle=10)
    pipeline.submit([twenty_seven], expected=0, matcher=matcher)
    snapshot = pipeline.snapshot()
    assert len(snapshot.players) == 1
    profile = snapshot.players[0]
    assert profile.confirmed_jersey_number == 27

    five = pipeline.segment(6, number=5, angle=10)
    result = pipeline.submit([five], expected=1, matcher=matcher)

    assert result.revision == 2
    snapshot = pipeline.snapshot()
    assert snapshot.assignments.get(five.segment_id) != profile.player_id
    assert conflicting_profiles(snapshot) == []
    assert observed_numbers(profile, snapshot.assignments, snapshot.features) == {27}


def _synthetic_segment(video, track, frames=3):
    observations = tuple(
        TrackObservation(video, track, frame, frame / 60, (0.0, 0.0, 10.0, 20.0), 0.95, None)
        for frame in range(frames)
    )
    return TrackletSegment(
        f"{video}_{track}_0",
        video,
        track,
        0,
        frames - 1,
        0.0,
        (frames - 1) / 60,
        frames,
        observations,
    )


def _synthetic_feature(segment_id, number):
    vector = np.zeros(512, dtype=np.float32)
    vector[0] = 1.0
    jersey = (
        None
        if number is None
        else JerseyNumberPrediction(number, 0.99, True, {number: 2.97}, str(number))
    )
    return DualStreamFeature(segment_id, jersey, AppearanceEmbedding(vector, 0.9, "synthetic"))


def _synthetic_decision(segment_id, player_id):
    return IncrementalDecision(
        segment_id,
        DecisionOutcome.MATCHED_EXISTING,
        player_id,
        0.1,
        0.5,
        {"reason": "matched_existing"},
    )


def _repair(players, assignments, segments, features, decisions, *, revision=2, matcher=None):
    return workflow._validate_dual_stream_commit(
        players,
        assignments,
        segments,
        features,
        decisions=decisions,
        revision=revision,
        matcher=matcher or IncrementalMatcher(max_prototypes=6),
    )


def test_commit_repair_keeps_strongest_anchor_and_rebuilds_prototypes():
    """直接构造不一致终态：修复而不是中断，弱证据段撤回待定。"""
    strong = _synthetic_segment("20", 1, frames=9)
    weak = _synthetic_segment("6", 2, frames=3)
    features = {
        strong.segment_id: _synthetic_feature(strong.segment_id, 5),
        weak.segment_id: _synthetic_feature(weak.segment_id, 23),
    }
    player = PlayerProfile("p", "未命名球员", None, (), "active", 1)
    decisions = {
        strong.segment_id: _synthetic_decision(strong.segment_id, "p"),
        weak.segment_id: _synthetic_decision(weak.segment_id, "p"),
    }
    players, assignments, repaired_decisions, demoted = _repair(
        (player,),
        {strong.segment_id: "p", weak.segment_id: "p"},
        (strong, weak),
        features,
        decisions,
    )

    assert demoted == [weak.segment_id]
    assert assignments == {strong.segment_id: "p"}
    decision = repaired_decisions[weak.segment_id]
    assert decision.outcome == DecisionOutcome.INSUFFICIENT
    assert decision.target_player_id is None
    assert decision.evidence["reason"] == "jersey_number_conflict_repair"
    assert len(players) == 1
    assert {p.feature.segment_id for p in players[0].prototypes} == {strong.segment_id}
    assert not _has_conflicting_numbers(observed_numbers(players[0], assignments, features))


def test_commit_repair_prefers_confirmed_number_over_longer_conflicting_segment():
    confirmed = _synthetic_segment("20", 1, frames=2)
    conflicting = _synthetic_segment("6", 2, frames=40)
    features = {
        confirmed.segment_id: _synthetic_feature(confirmed.segment_id, 27),
        conflicting.segment_id: _synthetic_feature(conflicting.segment_id, 10),
    }
    prototype = PlayerPrototype(
        "proto-27",
        "p",
        "generic",
        features[confirmed.segment_id],
        0.9,
        2,
    )
    player = PlayerProfile("p", "27号球员", 27, (prototype,), "active", 1)
    decisions = {
        confirmed.segment_id: _synthetic_decision(confirmed.segment_id, "p"),
        conflicting.segment_id: _synthetic_decision(conflicting.segment_id, "p"),
    }
    players, assignments, repaired_decisions, demoted = _repair(
        (player,),
        {confirmed.segment_id: "p", conflicting.segment_id: "p"},
        (confirmed, conflicting),
        features,
        decisions,
    )

    assert demoted == [conflicting.segment_id]
    assert set(assignments) == {confirmed.segment_id}
    assert repaired_decisions[conflicting.segment_id].outcome == DecisionOutcome.INSUFFICIENT
    assert {p.feature.segment_id for p in players[0].prototypes} == {confirmed.segment_id}


def test_commit_repair_keeps_digit_compatible_numbers():
    five = _synthetic_segment("20", 1, frames=4)
    fifteen = _synthetic_segment("6", 2, frames=4)
    features = {
        five.segment_id: _synthetic_feature(five.segment_id, 5),
        fifteen.segment_id: _synthetic_feature(fifteen.segment_id, 15),
    }
    player = PlayerProfile("p", "未命名球员", None, (), "active", 1)
    decisions = {
        five.segment_id: _synthetic_decision(five.segment_id, "p"),
        fifteen.segment_id: _synthetic_decision(fifteen.segment_id, "p"),
    }
    players, assignments, _decisions, demoted = _repair(
        (player,),
        {five.segment_id: "p", fifteen.segment_id: "p"},
        (five, fifteen),
        features,
        decisions,
    )

    assert demoted == []
    assert assignments == {five.segment_id: "p", fifteen.segment_id: "p"}
    assert not _has_conflicting_numbers(observed_numbers(players[0], assignments, features))


def test_commit_repair_still_rejects_genuine_internal_inconsistency():
    segment = _synthetic_segment("20", 1)
    features = {segment.segment_id: _synthetic_feature(segment.segment_id, 5)}
    player = PlayerProfile("p", "未命名球员", None, (), "active", 1)
    decisions = {segment.segment_id: _synthetic_decision(segment.segment_id, "p")}

    with pytest.raises(StorageConflict, match="unknown dual-stream player"):
        _repair((player,), {segment.segment_id: "ghost"}, (segment,), features, decisions)

    orphan = PlayerPrototype(
        "proto-orphan",
        "p",
        "generic",
        _synthetic_feature("6_2_0", 5),
        0.9,
        2,
    )
    orphaned = PlayerProfile("p", "未命名球员", None, (orphan,), "active", 1)
    with pytest.raises(StorageConflict, match="no matching current assignment"):
        _repair(
            (orphaned,),
            {segment.segment_id: "p"},
            (segment,),
            features,
            decisions,
        )
