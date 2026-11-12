"""Human maintenance preserves evidence, invalidates galleries and commits atomically."""

import pytest
from test_identity_prepare_distance import batch, sample

from app.contracts.api import (
    ConfirmAssignmentCommand,
    CreatePlayerCommand,
    EventQuery,
    MergePlayersCommand,
    PlayerQuery,
    RestoreRevisionCommand,
    ReviewRequest,
    ReviewSampleCommand,
    SetAssignmentCommand,
    SetConstraintCommand,
    SplitSegmentCommand,
)
from app.contracts.events import (
    Event,
    PointEventTime,
    SegmentReference,
    Subject,
)
from app.contracts.evidence import Clip, Track, TrackingQuality
from app.contracts.identity import Player, Segment
from app.storage.database import connect, transaction
from app.storage.repositories import Repository, RevisionConflict, StorageConflict
from app.storage.snapshot import read_identity_snapshot
from app.workflows._review_ops import current_evidence
from app.workflows.read import get_events, get_players
from app.workflows.review import UnsupportedReviewCommand, apply_review


def setup(conn):
    evidence = batch([sample("one", 0.5), sample("two", 1.5)])
    evidence.identity_evidence.tracks = [
        Track(
            match_id="m",
            clip_id="c",
            track_id="t",
            start_s=0,
            end_s=3,
            source=evidence.identity_evidence.samples[0].source.model_copy(
                update={"record_type": "track", "source_record_id": "t"}
            ),
        )
    ]
    repo = Repository(conn)
    with transaction(conn):
        repo.ensure_match("m")
        repo.store_batch(evidence)
        repo.advance_revision("m", 0, actor="source", reason="evidence", changes={})
    return evidence


def review(conn, revision, commands, request_id=None):
    return apply_review(
        conn,
        ReviewRequest(
            match_id="m",
            expected_revision=revision,
            actor="human",
            request_id=request_id,
            commands=commands,
        ),
    )


def initialize(conn):
    setup(conn)
    return review(
        conn,
        1,
        [
            CreatePlayerCommand(
                player=Player(match_id="m", player_id="pl_a"),
                segment_ids=["s"],
                reason="explicit identity",
            ),
            ReviewSampleCommand(
                sample_id="one",
                player_id="pl_a",
                feature_space_id="fs",
                review_status="approved",
                reason="clean appearance",
            ),
        ],
    )


def test_review_reassignment_revokes_old_approval_and_retry_is_idempotent(tmp_path):
    conn = connect(tmp_path / "library.sqlite")
    result = initialize(conn)
    assert result.revision == 2
    state = read_identity_snapshot(conn, "m")
    assert state.gallery.templates[0].active
    assert state.gallery.templates[0].approval.actor == "human"
    revision = review(
        conn,
        2,
        [
            CreatePlayerCommand(
                player=Player(match_id="m", player_id="pl_b"), reason="another player"
            )
        ],
    ).revision
    request = ReviewRequest(
        match_id="m",
        expected_revision=revision,
        actor="human",
        request_id="move",
        commands=[
            SetAssignmentCommand(segment_id="s", player_id="pl_b", reason="correct identity")
        ],
    )
    result = apply_review(conn, request)
    assert apply_review(conn, request) == result
    snapshot = read_identity_snapshot(conn, "m").gallery
    assert snapshot.assignments[0].player_id == "pl_b"
    assert not any(template.active for template in snapshot.templates)
    assert len(Repository(conn).revisions("m")) == result.revision
    with pytest.raises(StorageConflict, match="different content"):
        apply_review(conn, request.model_copy(update={"actor": "changed"}))
    conn.close()


def test_split_preserves_raw_samples_and_allows_only_range_checked_projection(tmp_path):
    conn = connect(tmp_path / "library.sqlite")
    initialize(conn)
    replacements = [
        Segment(
            match_id="m",
            segment_id="left",
            clip_id="c",
            track_id="t",
            start_s=0,
            end_s=1,
            supersedes_segment_id="s",
        ),
        Segment(
            match_id="m",
            segment_id="right",
            clip_id="c",
            track_id="t",
            start_s=1,
            end_s=3,
            supersedes_segment_id="s",
        ),
    ]
    result = review(
        conn,
        2,
        [
            SplitSegmentCommand(segment_id="s", segments=replacements, reason="identity switches"),
            ConfirmAssignmentCommand(segment_id="left", player_id="pl_a", reason="reviewed left"),
        ],
    )
    state = read_identity_snapshot(conn, "m")
    assert not any(template.active for template in state.gallery.templates)
    assert {item.segment_id for item in state.gallery.evidence.samples} == {"s"}
    mapped = current_evidence(state.gallery.evidence)
    assert {sample.sample_id: sample.segment_id for sample in mapped.samples} == {
        "one": "left",
        "two": "right",
    }
    assert {sample.sample_id: sample.segment_id for sample in state.gallery.evidence.samples} == {
        "one": "s",
        "two": "s",
    }
    assert Repository(conn).check_segment_overlaps("m") == []
    # Re-delivery of the original evidence cannot revive a manually retired segment.
    evidence = batch([sample("one", 0.5), sample("two", 1.5)])
    with transaction(conn):
        assert not Repository(conn).store_batch(evidence)
    assert Repository(conn).revision("m") == result.revision
    conn.close()


def test_review_rollback_stale_and_unsupported(tmp_path):
    conn = connect(tmp_path / "library.sqlite")
    initialize(conn)
    with pytest.raises(ValueError, match="forbid_player"):
        review(
            conn,
            2,
            [
                CreatePlayerCommand(
                    player=Player(match_id="m", player_id="pl_new"), reason="temporary"
                ),
                SetConstraintCommand(
                    constraint_id="forbidden",
                    segment_id="s",
                    constraint_kind="forbid_player",
                    player_id="pl_a",
                    reason="incompatible",
                ),
            ],
        )
    snapshot = read_identity_snapshot(conn, "m").gallery
    assert [player.player_id for player in snapshot.players] == ["pl_a"]
    assert snapshot.revision == 2
    with pytest.raises(RevisionConflict):
        review(conn, 1, [SetAssignmentCommand(segment_id="s", player_id=None, reason="stale")])
    with pytest.raises(UnsupportedReviewCommand, match="creating aliases"):
        review(
            conn,
            2,
            [
                CreatePlayerCommand(
                    player=Player(match_id="m", player_id="pl_alias", alias_of="pl_a"),
                    reason="unsupported alias creation",
                )
            ],
        )
    with pytest.raises(ValueError, match="alias player"):
        review(
            conn,
            2,
            [MergePlayersCommand(player_id="pl_a", alias_player_ids=["pl_b"], reason="missing")],
        )
    assert Repository(conn).revision("m") == 2
    conn.close()


def test_invalid_split_rolls_back_and_does_not_move_all_samples(tmp_path):
    conn = connect(tmp_path / "library.sqlite")
    initialize(conn)
    replacements = [
        Segment(
            match_id="m",
            segment_id="x",
            clip_id="c",
            track_id="t",
            start_s=0,
            end_s=1,
            supersedes_segment_id="s",
        ),
        Segment(
            match_id="m",
            segment_id="y",
            clip_id="c",
            track_id="t",
            start_s=1,
            end_s=4,
            supersedes_segment_id="s",
        ),
    ]
    with pytest.raises(ValueError, match="original segment range"):
        review(
            conn,
            2,
            [SplitSegmentCommand(segment_id="s", segments=replacements, reason="bad range")],
        )
    snapshot = read_identity_snapshot(conn, "m").gallery
    assert snapshot.evidence.segments[0].active
    assert snapshot.templates[0].active
    assert snapshot.revision == 2
    conn.close()


def test_historical_approval_does_not_block_later_unrelated_review(tmp_path):
    conn = connect(tmp_path / "library.sqlite")
    initialize(conn)
    review(conn, 2, [SetAssignmentCommand(segment_id="s", player_id=None, reason="unconfirmed")])
    review(
        conn,
        3,
        [
            CreatePlayerCommand(
                player=Player(match_id="m", player_id="pl_b"), reason="different person"
            )
        ],
    )
    review(
        conn,
        4,
        [ConfirmAssignmentCommand(segment_id="s", player_id="pl_a", reason="confirmed again")],
    )
    assert not read_identity_snapshot(conn, "m").gallery.templates[0].active
    review(
        conn,
        5,
        [
            ReviewSampleCommand(
                sample_id="one",
                player_id="pl_a",
                feature_space_id="fs",
                review_status="approved",
                reason="reviewed after correction",
            )
        ],
    )
    gallery = read_identity_snapshot(conn, "m").gallery
    assert len(gallery.templates) == 1
    assert gallery.templates[0].active
    assert gallery.templates[0].approval.revision == 6
    conn.close()


def test_same_human_fact_new_request_id_does_not_advance_revision(tmp_path):
    conn = connect(tmp_path / "library.sqlite")
    setup(conn)
    commands = [
        CreatePlayerCommand(
            player=Player(match_id="m", player_id="pl_a"),
            segment_ids=["s"],
            reason="human identity",
        )
    ]
    review(conn, 1, commands, request_id="first")
    result = review(conn, 2, commands, request_id="second")
    assert result.revision == 2
    assert Repository(conn).revision("m") == 2
    assert len(Repository(conn).review_records("m")) == 1
    conn.close()


def test_opposing_template_commands_for_same_target_roll_back(tmp_path):
    conn = connect(tmp_path / "library.sqlite")
    initialize(conn)
    with pytest.raises(ValueError, match="duplicate template review target"):
        review(
            conn,
            2,
            [
                ReviewSampleCommand(
                    sample_id="one",
                    player_id="pl_a",
                    feature_space_id="fs",
                    review_status="rejected",
                    reason="reject",
                ),
                ReviewSampleCommand(
                    sample_id="one",
                    player_id="pl_a",
                    feature_space_id="fs",
                    review_status="approved",
                    reason="approve",
                ),
            ],
        )
    assert Repository(conn).revision("m") == 2
    assert read_identity_snapshot(conn, "m").gallery.templates[0].active
    conn.close()


def test_risk_review_remains_independent_of_identity_and_template_approval(tmp_path):
    from app.contracts.api import ReviewRiskCommand
    from app.contracts.evidence import EvidenceBatch, IdentityEvidence
    from app.contracts.identity import IdentityRisk

    conn = connect(tmp_path / "library.sqlite")
    initialize(conn)
    repo = Repository(conn)
    with transaction(conn):
        repo.store_batch(
            EvidenceBatch(
                match_id="m",
                batch_id="risk",
                producer="source",
                identity_evidence=IdentityEvidence(
                    risks=[
                        IdentityRisk(
                            match_id="m", risk_id="mixed", kind="mixed_identity", segment_id="s"
                        )
                    ]
                ),
            )
        )
        repo.advance_revision("m", 2, actor="source", reason="risk", changes={})
    review(
        conn,
        3,
        [
            ReviewRiskCommand(
                risk_id="mixed",
                segment_id="s",
                resolution="confirmed",
                reason="mixed identity remains",
            )
        ],
    )
    state = read_identity_snapshot(conn, "m")
    assert state.gallery.assignments[0].status == "confirmed"
    assert not state.gallery.templates[0].active
    review(
        conn,
        4,
        [
            ReviewRiskCommand(
                risk_id="mixed",
                segment_id="s",
                resolution="cleared",
                clean_sample_ids=["one"],
                reason="scope inspected",
            )
        ],
    )
    state = read_identity_snapshot(conn, "m")
    assert state.gallery.risk_reviews[0].resolution == "cleared"
    assert not state.gallery.templates[0].active
    assert repo.clean_reviews("m") == {}
    conn.close()


def test_restore_revision_normal_flow_from_initial(tmp_path):
    conn = connect(tmp_path / "lib.sqlite")
    # R=1: Setup evidence batch
    setup(conn)
    snapshot_r1 = read_identity_snapshot(conn, "m")
    assert snapshot_r1.gallery.revision == 1
    assert snapshot_r1.gallery.assignments == []
    assert snapshot_r1.gallery.templates == []

    # R=2: Review creating player pl_a, assignment and approved template
    res_init = review(
        conn,
        1,
        [
            CreatePlayerCommand(
                player=Player(match_id="m", player_id="pl_a"),
                segment_ids=["s"],
                reason="explicit identity",
            ),
            ReviewSampleCommand(
                sample_id="one",
                player_id="pl_a",
                feature_space_id="fs",
                review_status="approved",
                reason="clean appearance",
            ),
        ],
    )
    assert res_init.revision == 2
    snapshot_r2 = read_identity_snapshot(conn, "m")
    assert snapshot_r2.gallery.revision == 2
    assert len(snapshot_r2.gallery.assignments) == 1
    assert snapshot_r2.gallery.assignments[0].player_id == "pl_a"
    assert len(snapshot_r2.gallery.templates) == 1
    assert snapshot_r2.gallery.templates[0].active is True

    # Execute restore to R=1 -> results in R=3
    res_restore = review(
        conn,
        2,
        [RestoreRevisionCommand(target_revision=1, reason="restore to initial r1")],
    )
    assert res_restore.revision == 3
    snapshot_r3 = read_identity_snapshot(conn, "m")
    assert snapshot_r3.gallery.revision == 3
    # Assert player assignments and templates at R=3 match R=1 completely
    assert snapshot_r3.gallery.assignments == snapshot_r1.gallery.assignments == []
    assert snapshot_r3.gallery.templates == snapshot_r1.gallery.templates == []
    conn.close()


def test_restore_revision_to_previous_custom_state_and_alias(tmp_path):
    conn = connect(tmp_path / "lib.sqlite")
    initialize(conn)  # R=2: pl_a assigned to segment s, template approved
    snapshot_r2 = read_identity_snapshot(conn, "m")
    assert len(snapshot_r2.gallery.assignments) == 1
    assert snapshot_r2.gallery.assignments[0].player_id == "pl_a"
    assert snapshot_r2.gallery.templates[0].active is True

    # R=3: Reassign to pl_b (which deactivates pl_a's template)
    res_r3 = review(
        conn,
        2,
        [
            CreatePlayerCommand(
                player=Player(match_id="m", player_id="pl_b"),
                segment_ids=["s"],
                reason="reassign to b",
            )
        ],
    )
    assert res_r3.revision == 3
    snapshot_r3 = read_identity_snapshot(conn, "m")
    assert snapshot_r3.gallery.assignments[0].player_id == "pl_b"
    assert not any(t.active for t in snapshot_r3.gallery.templates)

    # Execute restore to R=2 -> results in R=4
    res_r4 = review(
        conn,
        3,
        [RestoreRevisionCommand(target_revision=2, reason="restore to r2")],
    )
    assert res_r4.revision == 4
    snapshot_r4 = read_identity_snapshot(conn, "m")
    assert snapshot_r4.gallery.revision == 4
    assert [a.model_dump(exclude={"revision"}) for a in snapshot_r4.gallery.assignments] == [
        a.model_dump(exclude={"revision"}) for a in snapshot_r2.gallery.assignments
    ]
    assert all(a.revision == 4 for a in snapshot_r4.gallery.assignments)
    assert [t.model_dump(exclude={"revision"}) for t in snapshot_r4.gallery.templates] == [
        t.model_dump(exclude={"revision"}) for t in snapshot_r2.gallery.templates
    ]
    assert all(t.revision == 4 for t in snapshot_r4.gallery.templates)
    assert snapshot_r4.gallery.assignments[0].player_id == "pl_a"
    assert snapshot_r4.gallery.templates[0].active is True
    assert snapshot_r4.gallery.templates[0].review_status == "approved"

    # Further test: Merge pl_b into pl_a at R=5, then restore to R=4 -> R=6
    res_r5 = review(
        conn,
        4,
        [
            MergePlayersCommand(
                player_id="pl_a",
                alias_player_ids=["pl_b"],
                reason="merge b into a",
            )
        ],
    )
    assert res_r5.revision == 5
    snapshot_r5 = read_identity_snapshot(conn, "m")
    # pl_b is alias so filtered from active gallery players
    assert [p.player_id for p in snapshot_r5.gallery.players] == ["pl_a"]

    # Restore to R=4 -> results in R=6
    res_r6 = review(
        conn,
        5,
        [RestoreRevisionCommand(target_revision=4, reason="undo merge")],
    )
    assert res_r6.revision == 6
    snapshot_r6 = read_identity_snapshot(conn, "m")
    assert {p.player_id for p in snapshot_r6.gallery.players} == {"pl_a", "pl_b"}
    # Verify pl_b is no longer alias
    raw_players = {
        row[0]: (row[1], Player.model_validate_json(row[2]))
        for row in conn.execute("SELECT player_id, alias_of, json FROM players WHERE match_id='m'")
    }
    assert raw_players["pl_b"][0] is None
    assert raw_players["pl_b"][1].alias_of is None

    conn.close()


def test_restore_revision_illegal_cases(tmp_path):
    conn = connect(tmp_path / "lib.sqlite")
    initialize(conn)  # Current revision is 2

    # Illegal 1: target_revision == current_revision
    with pytest.raises(ValueError, match="strictly less than current revision"):
        review(conn, 2, [RestoreRevisionCommand(target_revision=2, reason="same revision")])

    # Illegal 2: target_revision > current_revision (future version)
    with pytest.raises(ValueError, match="strictly less than current revision"):
        review(conn, 2, [RestoreRevisionCommand(target_revision=3, reason="future revision")])

    # Illegal 3: target_revision does not exist (e.g. 0 or 999)
    with pytest.raises(ValueError, match="does not exist in revisions"):
        review(conn, 2, [RestoreRevisionCommand(target_revision=0, reason="version zero")])
    with pytest.raises(ValueError, match="strictly less than current revision"):
        review(conn, 2, [RestoreRevisionCommand(target_revision=999, reason="huge revision")])

    # Illegal 4: stale expected_revision
    with pytest.raises(RevisionConflict):
        review(conn, 1, [RestoreRevisionCommand(target_revision=1, reason="stale expected")])

    # Illegal 5: combined with another command
    with pytest.raises(ValueError, match="cannot be combined"):
        review(
            conn,
            2,
            [
                RestoreRevisionCommand(target_revision=1, reason="r1"),
                ConfirmAssignmentCommand(segment_id="s", player_id="pl_a", reason="confirm"),
            ],
        )

    # Revision remains unchanged after all rejected attempts
    assert Repository(conn).revision("m") == 2
    conn.close()


def test_restore_revision_idempotency_and_audit(tmp_path):
    conn = connect(tmp_path / "lib.sqlite")
    initialize(conn)  # R=2

    # R=3: make a change
    review(
        conn,
        2,
        [
            CreatePlayerCommand(
                player=Player(match_id="m", player_id="pl_b"),
                segment_ids=["s"],
                reason="move to b",
            )
        ],
    )
    assert Repository(conn).revision("m") == 3

    # R=4: restore to R=2 with explicit request_id
    restore_request = ReviewRequest(
        match_id="m",
        expected_revision=3,
        actor="auditor",
        request_id="restore_req_audit_001",
        commands=[RestoreRevisionCommand(target_revision=2, reason="audit rollback")],
    )
    res1 = apply_review(conn, restore_request)
    assert res1.revision == 4

    # Idempotency: re-executing identical request returns cached result without advancing revision
    res2 = apply_review(conn, restore_request)
    assert res2 == res1
    assert Repository(conn).revision("m") == 4

    # Monotonic revision chain in revisions table
    repo = Repository(conn)
    revisions = repo.revisions("m")
    assert [r.revision for r in revisions] == [1, 2, 3, 4]
    for i, r in enumerate(revisions):
        assert r.revision == i + 1
        assert r.previous_revision == i

    restore_audit = revisions[3]
    assert restore_audit.revision == 4
    assert restore_audit.previous_revision == 3  # parent_revision before restore
    assert "restore_revision:2" in restore_audit.reason
    assert restore_audit.changes["target_revision"] == 2
    assert restore_audit.changes["restored"] is True

    # Audit in workflow_reviews table
    review_history = repo.review_records("m")
    assert len(review_history) == 3  # rev 2, 3, 4
    last_review = review_history[-1]
    assert last_review["revision"] == 4
    assert last_review["actor"] == "auditor"
    assert last_review["commands"][0]["kind"] == "restore_revision"
    assert last_review["commands"][0]["target_revision"] == 2

    # Intermediate history is preserved in assignments table
    assignments_history = repo.assignment_history("m", "s")
    assert len(assignments_history) == 3
    assert assignments_history[0].player_id == "pl_a"
    assert assignments_history[1].player_id == "pl_b"
    assert assignments_history[2].player_id == "pl_a"
    assert assignments_history[2].revision == 4

    conn.close()


def test_merge_players_normal_flow_migrates_templates_and_accumulates_events(tmp_path):
    conn = connect(tmp_path / "merge.sqlite")
    repo = Repository(conn)

    clip = Clip(match_id="m", clip_id="c", duration_s=10.0, frame_timing="cfr", fps=10.0)
    track = Track(
        match_id="m",
        clip_id="c",
        track_id="t",
        start_s=0.0,
        end_s=10.0,
        source=sample("one", 1.0).source.model_copy(
            update={"record_type": "track", "source_record_id": "t"}
        ),
        observed_frame_indices=list(range(100)),
    )
    s1 = Segment(match_id="m", clip_id="c", track_id="t", segment_id="s1", start_s=0.0, end_s=5.0)
    s2 = Segment(match_id="m", clip_id="c", track_id="t", segment_id="s2", start_s=5.0, end_s=10.0)
    smp1 = sample(
        "smp1",
        2.0,
        quality=TrackingQuality(usability="usable", clean_review="approved", score=0.8),
    ).model_copy(update={"segment_id": "s1"})
    smp2 = sample(
        "smp2",
        7.0,
        quality=TrackingQuality(usability="usable", clean_review="approved", score=0.9),
    ).model_copy(update={"segment_id": "s2"})

    ev1 = Event(
        match_id="m",
        event_id="e1",
        clip_id="c",
        source_event_key="k1",
        event_type="shot",
        outcome="made",
        time=PointEventTime(time_s=2.0),
        source=smp1.source.model_copy(update={"record_type": "event", "source_record_id": "e1"}),
        subjects=[
            Subject(
                subject_id="sub1", role="shooter", references=[SegmentReference(segment_id="s1")]
            )
        ],
    )
    ev2 = Event(
        match_id="m",
        event_id="e2",
        clip_id="c",
        source_event_key="k2",
        event_type="shot",
        outcome="made",
        time=PointEventTime(time_s=7.0),
        source=smp2.source.model_copy(update={"record_type": "event", "source_record_id": "e2"}),
        subjects=[
            Subject(
                subject_id="sub2", role="shooter", references=[SegmentReference(segment_id="s2")]
            )
        ],
    )

    initial_batch = batch([smp1, smp2], end_s=10.0)
    initial_batch.clips = [clip]
    initial_batch.identity_evidence.tracks = [track]
    initial_batch.identity_evidence.segments = [s1, s2]
    initial_batch.events = [ev1, ev2]

    with transaction(conn):
        repo.ensure_match("m")
        repo.store_batch(initial_batch)
        repo.advance_revision("m", 0, actor="source", reason="evidence", changes={})

    # R=2: Create pl_a and pl_b with confirmed segments and approved templates
    review(
        conn,
        1,
        [
            CreatePlayerCommand(
                player=Player(match_id="m", player_id="pl_a"),
                segment_ids=["s1"],
                reason="player a",
            ),
            CreatePlayerCommand(
                player=Player(match_id="m", player_id="pl_b"),
                segment_ids=["s2"],
                reason="player b",
            ),
            ReviewSampleCommand(
                sample_id="smp1",
                player_id="pl_a",
                feature_space_id="fs",
                review_status="approved",
                reason="tpl a",
            ),
            ReviewSampleCommand(
                sample_id="smp2",
                player_id="pl_b",
                feature_space_id="fs",
                review_status="approved",
                reason="tpl b",
            ),
        ],
    )

    # Before merge: verify separate players, events, templates
    players_before = get_players(conn, PlayerQuery(match_id="m")).data
    assert len(players_before) == 2
    by_id_before = {p.player.player_id: p for p in players_before}
    assert by_id_before["pl_a"].made_count == 1
    assert len(by_id_before["pl_a"].templates) == 1
    assert by_id_before["pl_a"].confirmed_segment_ids == ["s1"]
    assert by_id_before["pl_b"].made_count == 1
    assert len(by_id_before["pl_b"].templates) == 1
    assert by_id_before["pl_b"].confirmed_segment_ids == ["s2"]

    events_a_before = get_events(conn, EventQuery(match_id="m", player_id="pl_a")).data
    assert len(events_a_before) == 1 and events_a_before[0].event.event_id == "e1"
    events_b_before = get_events(conn, EventQuery(match_id="m", player_id="pl_b")).data
    assert len(events_b_before) == 1 and events_b_before[0].event.event_id == "e2"

    # Execute Merge: pl_b merged into pl_a (R=2 -> R=3)
    res_merge = review(
        conn,
        2,
        [MergePlayersCommand(player_id="pl_a", alias_player_ids=["pl_b"], reason="same person")],
    )
    assert res_merge.revision == 3

    # Downstream read consistency:
    # 1. get_players only returns primary players (alias_of IS NULL)
    players_after = get_players(conn, PlayerQuery(match_id="m")).data
    assert len(players_after) == 1
    pl_a_view = players_after[0]
    assert pl_a_view.player.player_id == "pl_a"
    assert pl_a_view.player.alias_of is None

    # Querying merged player specifically returns empty
    assert get_players(conn, PlayerQuery(match_id="m", player_id="pl_b")).data == []

    # 2. Made count accumulated: 1 + 1 = 2
    assert pl_a_view.made_count == 2

    # 3. Confirmed segments migrated
    assert pl_a_view.confirmed_segment_ids == ["s1", "s2"]

    # 4. Gallery templates migrated to pl_a
    assert len(pl_a_view.templates) == 2
    assert {t.sample_id for t in pl_a_view.templates} == {"smp1", "smp2"}
    assert all(t.player_id == "pl_a" for t in pl_a_view.templates)
    assert all(t.active for t in pl_a_view.templates)

    # 5. get_events consistency: both events now belong to pl_a
    events_a_after = get_events(conn, EventQuery(match_id="m", player_id="pl_a")).data
    assert len(events_a_after) == 2
    assert {e.event.event_id for e in events_a_after} == {"e1", "e2"}
    events_b_after = get_events(conn, EventQuery(match_id="m", player_id="pl_b")).data
    assert len(events_b_after) == 0

    # 6. Database entity persistence & audit checks
    alias_row = conn.execute(
        "SELECT player_id, alias_of FROM players WHERE match_id='m' AND player_id='pl_b'"
    ).fetchone()
    assert alias_row["alias_of"] == "pl_a"

    history_s2 = repo.assignment_history("m", "s2")
    assert len(history_s2) == 2
    assert history_s2[0].player_id == "pl_b" and history_s2[0].revision == 2
    assert history_s2[1].player_id == "pl_a" and history_s2[1].revision == 3

    conn.close()


def test_merge_players_alias_cycle_detection_rolls_back(tmp_path):
    conn = connect(tmp_path / "cycle.sqlite")
    initialize(conn)  # R=2: pl_a exists

    # R=3: Create pl_b and pl_c
    review(
        conn,
        2,
        [
            CreatePlayerCommand(player=Player(match_id="m", player_id="pl_b"), reason="b"),
            CreatePlayerCommand(player=Player(match_id="m", player_id="pl_c"), reason="c"),
        ],
    )

    # R=4: Merge pl_b into pl_a -> pl_b.alias_of = "pl_a"
    review(
        conn,
        3,
        [MergePlayersCommand(player_id="pl_a", alias_player_ids=["pl_b"], reason="b into a")],
    )
    repo = Repository(conn)
    assert repo.revision("m") == 4

    # Case 1: 2-player cycle: merge pl_a into pl_b (pl_b is already alias of pl_a)
    with pytest.raises(ValueError, match="player alias cycle"):
        review(
            conn,
            4,
            [MergePlayersCommand(player_id="pl_b", alias_player_ids=["pl_a"], reason="a into b")],
        )

    # Verify transaction rollback
    assert repo.revision("m") == 4
    row_a = conn.execute(
        "SELECT alias_of FROM players WHERE match_id='m' AND player_id='pl_a'"
    ).fetchone()
    assert row_a[0] is None
    row_b = conn.execute(
        "SELECT alias_of FROM players WHERE match_id='m' AND player_id='pl_b'"
    ).fetchone()
    assert row_b[0] == "pl_a"

    # Case 2: Multi-hop 3-player cycle: pl_c -> pl_b -> pl_a
    # First: merge pl_c into pl_b (R=4 -> R=5)
    review(
        conn,
        4,
        [MergePlayersCommand(player_id="pl_b", alias_player_ids=["pl_c"], reason="c into b")],
    )
    assert repo.revision("m") == 5

    # Now attempt to merge pl_a into pl_c (forming pl_a -> pl_c -> pl_b -> pl_a cycle)
    with pytest.raises(ValueError, match="player alias cycle"):
        review(
            conn,
            5,
            [
                MergePlayersCommand(
                    player_id="pl_c", alias_player_ids=["pl_a"], reason="a into c cycle"
                )
            ],
        )

    # Verify rollback
    assert repo.revision("m") == 5
    row_a_after = conn.execute(
        "SELECT alias_of FROM players WHERE match_id='m' AND player_id='pl_a'"
    ).fetchone()
    assert row_a_after[0] is None

    conn.close()


def test_merge_players_gallery_capacity_truncates_excess(tmp_path):
    conn = connect(tmp_path / "capacity.sqlite")
    repo = Repository(conn)

    # Create 8 samples across 2 segments sa and sb with varying tracking scores
    # pl_a will own sa and get 4 samples: sa_1..sa_4 (scores 0.6, 0.7, 0.8, 0.9)
    # pl_b will own sb and get 4 samples: sb_1..sb_4 (scores 0.65, 0.75, 0.85, 0.95)
    # Combined = 8 templates > max_templates_per_player (6)
    sa = Segment(match_id="m", clip_id="c", track_id="t", segment_id="sa", start_s=0.0, end_s=2.5)
    sb = Segment(match_id="m", clip_id="c", track_id="t", segment_id="sb", start_s=2.5, end_s=5.0)
    samples_a = [
        sample(
            f"sa_{i}",
            0.1 * i,
            quality=TrackingQuality(
                usability="usable", clean_review="approved", score=0.5 + 0.1 * i
            ),
        ).model_copy(update={"segment_id": "sa"})
        for i in range(1, 5)
    ]
    samples_b = [
        sample(
            f"sb_{i}",
            2.5 + 0.1 * i,
            quality=TrackingQuality(
                usability="usable", clean_review="approved", score=0.55 + 0.1 * i
            ),
        ).model_copy(update={"segment_id": "sb"})
        for i in range(1, 5)
    ]
    all_samples = samples_a + samples_b
    evidence = batch(all_samples, end_s=5.0)
    evidence.identity_evidence.tracks = [
        Track(
            match_id="m",
            clip_id="c",
            track_id="t",
            start_s=0.0,
            end_s=5.0,
            source=evidence.identity_evidence.samples[0].source.model_copy(
                update={"record_type": "track", "source_record_id": "t"}
            ),
        )
    ]
    evidence.identity_evidence.segments = [sa, sb]

    with transaction(conn):
        repo.ensure_match("m")
        repo.store_batch(evidence)
        repo.advance_revision("m", 0, actor="source", reason="evidence", changes={})

    # R=2: Create pl_a (owns sa) and pl_b (owns sb) and approve all 8 samples
    commands = [
        CreatePlayerCommand(
            player=Player(match_id="m", player_id="pl_a"), segment_ids=["sa"], reason="a"
        ),
        CreatePlayerCommand(
            player=Player(match_id="m", player_id="pl_b"), segment_ids=["sb"], reason="b"
        ),
    ]
    for s in samples_a:
        commands.append(
            ReviewSampleCommand(
                sample_id=s.sample_id,
                player_id="pl_a",
                feature_space_id="fs",
                review_status="approved",
                reason="clean",
            )
        )
    for s in samples_b:
        commands.append(
            ReviewSampleCommand(
                sample_id=s.sample_id,
                player_id="pl_b",
                feature_space_id="fs",
                review_status="approved",
                reason="clean",
            )
        )
    review(conn, 1, commands)

    state_before = read_identity_snapshot(conn, "m")
    active_templates_before = [t for t in state_before.gallery.templates if t.active]
    assert len(active_templates_before) == 8
    assert len([t for t in active_templates_before if t.player_id == "pl_a"]) == 4
    assert len([t for t in active_templates_before if t.player_id == "pl_b"]) == 4

    # Merge pl_b into pl_a: total 8 templates for pl_a in feature space fs > max_templates_per_player (6)
    res_merge = review(
        conn,
        2,
        [MergePlayersCommand(player_id="pl_a", alias_player_ids=["pl_b"], reason="capacity test")],
    )
    assert res_merge.revision == 3

    # Verify downstream: get_players returns pl_a with exactly 6 active approved templates
    players = get_players(conn, PlayerQuery(match_id="m")).data
    assert len(players) == 1
    pl_a_view = players[0]
    assert len(pl_a_view.templates) == 6
    assert all(t.active for t in pl_a_view.templates)
    assert all(t.player_id == "pl_a" for t in pl_a_view.templates)

    # In database: exactly 8 templates exist with current=1, 6 active and 2 inactive
    rows = conn.execute(
        "SELECT template_id, player_id, json FROM templates WHERE match_id='m' AND current=1"
    ).fetchall()
    assert len(rows) == 8
    assert all(r["player_id"] == "pl_a" for r in rows)
    import json

    active_flags = [json.loads(r["json"])["active"] for r in rows]
    assert active_flags.count(True) == 6
    assert active_flags.count(False) == 2

    # Verify the 2 deactivated templates are the lowest scoring ones:
    # sa_1 (score 0.6) and sb_1 (score 0.65)
    inactive_sample_ids = {
        json.loads(r["json"])["sample_id"] for r in rows if not json.loads(r["json"])["active"]
    }
    assert inactive_sample_ids == {"sa_1", "sb_1"}

    conn.close()
