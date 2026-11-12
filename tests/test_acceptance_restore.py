"""Business invariants for restoring complete, auditable identity state."""

import pytest
from test_review_advanced_attacks import _setup_match_with_two_players_and_events

from app.contracts.api import (
    CreatePlayerCommand,
    MergePlayersCommand,
    PlayerQuery,
    RestoreRevisionCommand,
    ReviewRequest,
    SplitSegmentCommand,
)
from app.contracts.identity import MatchingPolicy, Player, Segment
from app.storage.database import connect
from app.storage.repositories import Repository
from app.storage.snapshot import read_identity_snapshot
from app.workflows._consistency import validate_templates
from app.workflows.read import get_players
from app.workflows.review import apply_review


def review(db, commands):
    return apply_review(
        db,
        ReviewRequest(
            match_id="m",
            expected_revision=Repository(db).revision("m"),
            actor="human",
            commands=commands,
        ),
    )


def profiles(db):
    return {
        p.player.player_id: (
            p.confirmed_segment_ids,
            p.made_count,
            [t.template_id for t in p.templates],
        )
        for p in get_players(db, PlayerQuery(match_id="m")).data
    }


def split(db):
    review(
        db,
        [
            SplitSegmentCommand(
                segment_id="s1",
                reason="inspect split",
                segments=[
                    Segment(
                        match_id="m",
                        clip_id="c",
                        track_id="t",
                        segment_id=name,
                        start_s=start,
                        end_s=end,
                        supersedes_segment_id="s1",
                    )
                    for name, start, end in [("s1a", 0, 2.5), ("s1b", 2.5, 5)]
                ],
            )
        ],
    )


def test_restore_split_reinstates_segments_goals_and_eligible_templates(tmp_path):
    db = connect(tmp_path / "split.sqlite")
    _setup_match_with_two_players_and_events(db)
    before = profiles(db)
    split(db)
    review(db, [RestoreRevisionCommand(target_revision=2, reason="undo split")])
    assert profiles(db) == before
    state = read_identity_snapshot(db, "m")
    assert {s.segment_id for s in state.gallery.evidence.segments if s.active} == {"s1", "s2"}
    validate_templates(
        state.gallery, state.feature_values, lambda space: MatchingPolicy(feature_space_id=space)
    )
    assert len(Repository(db).review_records("m")) == 3


def test_restore_failure_rolls_back_every_fact_and_audit(tmp_path, monkeypatch):
    import app.workflows._restore as restore

    db = connect(tmp_path / "rollback.sqlite")
    _setup_match_with_two_players_and_events(db)
    split(db)
    before = list(db.iterdump())

    def fail(*args, **kwargs):
        raise ValueError("forced template validation failure")

    monkeypatch.setattr(restore, "validate_templates", fail)
    with pytest.raises(ValueError, match="forced template"):
        review(db, [RestoreRevisionCommand(target_revision=2, reason="undo split")])
    assert list(db.iterdump()) == before


def test_restore_to_restore_keeps_separate_players_templates_and_new_history(tmp_path):
    db = connect(tmp_path / "chain.sqlite")
    _setup_match_with_two_players_and_events(db)
    before = profiles(db)
    review(
        db,
        [MergePlayersCommand(player_id="pl_a", alias_player_ids=["pl_b"], reason="inspect merge")],
    )
    assert profiles(db)["pl_a"][1] == 2
    review(db, [RestoreRevisionCommand(target_revision=2, reason="undo merge")])
    assert profiles(db) == before
    review(
        db,
        [
            CreatePlayerCommand(
                player=Player(match_id="m", player_id="pl_c"), reason="independent later work"
            )
        ],
    )
    review(db, [RestoreRevisionCommand(target_revision=4, reason="restore a restore")])
    assert {p: profiles(db)[p] for p in before} == before
    state = read_identity_snapshot(db, "m")
    validate_templates(
        state.gallery, state.feature_values, lambda space: MatchingPolicy(feature_space_id=space)
    )
    history = Repository(db).assignment_history("m", "s2")
    assert [(a.revision, a.player_id) for a in history] == [
        (2, "pl_b"),
        (3, "pl_a"),
        (4, "pl_b"),
        (6, "pl_b"),
    ]
    assert [
        r[0]
        for r in db.execute(
            "SELECT revision FROM template_reviews WHERE sample_id='smp2' ORDER BY revision"
        )
    ] == [2, 4, 6]
    assert len(Repository(db).review_records("m")) == 5


def test_restore_constraint_selection_is_versioned_without_deleting_audit(tmp_path):
    from app.contracts.api import SetConstraintCommand

    db = connect(tmp_path / "constraints.sqlite")
    _setup_match_with_two_players_and_events(db)
    review(
        db,
        [
            SetConstraintCommand(
                constraint_id="forbid",
                segment_id="s1",
                constraint_kind="forbid_player",
                player_id="pl_b",
                reason="distinct identity",
            )
        ],
    )
    assert len(read_identity_snapshot(db, "m").constraints) == 1
    review(db, [RestoreRevisionCommand(target_revision=2, reason="before constraint")])
    assert read_identity_snapshot(db, "m").constraints == []
    assert db.execute("SELECT COUNT(*) FROM constraints").fetchone()[0] == 1
    review(db, [RestoreRevisionCommand(target_revision=3, reason="reinstate constraint")])
    constraints = read_identity_snapshot(db, "m").constraints
    assert len(constraints) == 1
    assert constraints[0].revision == 5
    assert db.execute("SELECT COUNT(*) FROM constraints").fetchone()[0] == 2


def test_restore_revalidates_templates_under_selected_policy_and_rolls_back(tmp_path):
    db = connect(tmp_path / "policy.sqlite")
    _setup_match_with_two_players_and_events(db)
    split(db)
    before = list(db.iterdump())
    with pytest.raises(ValueError):
        apply_review(
            db,
            ReviewRequest(
                match_id="m",
                expected_revision=3,
                actor="human",
                commands=[
                    RestoreRevisionCommand(target_revision=2, reason="restore with stricter policy")
                ],
            ),
            policies=[MatchingPolicy(feature_space_id="fs", min_common_parts=4)],
        )
    assert list(db.iterdump()) == before


def test_repeated_restore_reinstates_and_retires_replacement_edges(tmp_path):
    db = connect(tmp_path / "split-chain.sqlite")
    _setup_match_with_two_players_and_events(db)
    before = profiles(db)
    split(db)
    review(db, [RestoreRevisionCommand(target_revision=2, reason="undo split")])
    assert profiles(db) == before
    review(db, [RestoreRevisionCommand(target_revision=3, reason="reinstate split")])
    state = read_identity_snapshot(db, "m")
    segments = {s.segment_id: s for s in state.gallery.evidence.segments}
    assert not segments["s1"].active
    assert segments["s1a"].active and segments["s1a"].supersedes_segment_id == "s1"
    assert segments["s1b"].active and segments["s1b"].supersedes_segment_id == "s1"
    assert profiles(db)["pl_a"][1] == 0
    review(db, [RestoreRevisionCommand(target_revision=4, reason="return to restored original")])
    assert profiles(db) == before
    validate_templates(
        read_identity_snapshot(db, "m").gallery,
        state.feature_values,
        lambda space: MatchingPolicy(feature_space_id=space),
    )
