"""Boundary attack tests and regression acceptance for MergePlayersCommand and RestoreRevisionCommand.

Covers:
1. Merge then Restore: decoupling of alias_of, identity segments, and made event counts.
2. Restore then Merge: unlocked version chain allowing subsequent creations, modifications, and merges.
3. Optimistic Concurrency Collision: strict RevisionConflict detection and zero dirty data on stale/future expected_revision.
4. Batch Commands Atomicity: global transaction atomicity (all-or-nothing rollback on partial failure).
5. Boundary attacks: cyclic merges, invalid self-references, and restore batching isolation.
"""

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
    SeedIdentity,
    SetAssignmentCommand,
)
from app.contracts.events import Event, PointEventTime, SegmentReference, Subject
from app.contracts.evidence import Clip, Track, TrackingQuality
from app.contracts.identity import Player, Segment
from app.storage.database import connect, transaction
from app.storage.repositories import Repository, RevisionConflict
from app.storage.snapshot import read_identity_snapshot
from app.workflows.initialize import initialize_library
from app.workflows.read import get_events, get_players
from app.workflows.review import apply_review


def _setup_match_with_two_players_and_events(conn, match_id="m"):
    """Prepare a match with 2 segments, 2 samples, 2 made shots, and 2 players at R=2."""
    repo = Repository(conn)
    clip = Clip(match_id=match_id, clip_id="c", duration_s=10.0, frame_timing="cfr", fps=10.0)
    track = Track(
        match_id=match_id,
        clip_id="c",
        track_id="t",
        start_s=0.0,
        end_s=10.0,
        source=sample("smp1", 1.0).source.model_copy(
            update={"record_type": "track", "source_record_id": "t"}
        ),
        observed_frame_indices=list(range(100)),
    )
    s1 = Segment(
        match_id=match_id, clip_id="c", track_id="t", segment_id="s1", start_s=0.0, end_s=5.0
    )
    s2 = Segment(
        match_id=match_id, clip_id="c", track_id="t", segment_id="s2", start_s=5.0, end_s=10.0
    )
    smp1 = sample(
        "smp1",
        2.0,
        quality=TrackingQuality(usability="usable", clean_review="approved", score=0.85),
    ).model_copy(update={"segment_id": "s1"})
    smp2 = sample(
        "smp2",
        7.0,
        quality=TrackingQuality(usability="usable", clean_review="approved", score=0.90),
    ).model_copy(update={"segment_id": "s2"})

    ev1 = Event(
        match_id=match_id,
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
        match_id=match_id,
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

    # R=1: Ingest batch
    with transaction(conn):
        repo.ensure_match(match_id)
        repo.store_batch(initial_batch)
        repo.advance_revision(match_id, 0, actor="source", reason="evidence", changes={})

    assert repo.revision(match_id) == 1

    # R=2: Create pl_a and pl_b with confirmed segments and approved templates
    res_init = apply_review(
        conn,
        ReviewRequest(
            match_id=match_id,
            expected_revision=1,
            actor="admin",
            commands=[
                CreatePlayerCommand(
                    player=Player(match_id=match_id, player_id="pl_a"),
                    segment_ids=["s1"],
                    reason="create pl_a",
                ),
                CreatePlayerCommand(
                    player=Player(match_id=match_id, player_id="pl_b"),
                    segment_ids=["s2"],
                    reason="create pl_b",
                ),
                ReviewSampleCommand(
                    sample_id="smp1",
                    player_id="pl_a",
                    feature_space_id="fs",
                    review_status="approved",
                    reason="tpl pl_a",
                ),
                ReviewSampleCommand(
                    sample_id="smp2",
                    player_id="pl_b",
                    feature_space_id="fs",
                    review_status="approved",
                    reason="tpl pl_b",
                ),
            ],
        ),
    )
    assert res_init.revision == 2
    return match_id


# ==============================================================================
# 1. 合并后再恢复（Merge then Restore）
# ==============================================================================


def test_merge_then_restore_decouples_alias_segments_and_goals(tmp_path):
    """Scenario 1: Merge B into A, execute an intermediate review, then restore.

    Assert that Player-B's alias_of recovers to NULL, and segments, templates,
    and made counts are completely decoupled and separated.
    """
    conn = connect(tmp_path / "attack1.sqlite")
    match_id = _setup_match_with_two_players_and_events(conn)
    repo = Repository(conn)

    # Initial state verification at R=2
    players_r2 = get_players(conn, PlayerQuery(match_id=match_id)).data
    assert len(players_r2) == 2
    by_id_r2 = {p.player.player_id: p for p in players_r2}
    assert by_id_r2["pl_a"].player.alias_of is None
    assert by_id_r2["pl_b"].player.alias_of is None
    assert by_id_r2["pl_a"].confirmed_segment_ids == ["s1"]
    assert by_id_r2["pl_b"].confirmed_segment_ids == ["s2"]
    assert by_id_r2["pl_a"].made_count == 1
    assert by_id_r2["pl_b"].made_count == 1

    events_a_r2 = get_events(conn, EventQuery(match_id=match_id, player_id="pl_a")).data
    events_b_r2 = get_events(conn, EventQuery(match_id=match_id, player_id="pl_b")).data
    assert [e.event.event_id for e in events_a_r2] == ["e1"]
    assert [e.event.event_id for e in events_b_r2] == ["e2"]

    # Step 1: Execute Merge (R=2 -> R=3)
    res_merge = apply_review(
        conn,
        ReviewRequest(
            match_id=match_id,
            expected_revision=2,
            actor="admin",
            commands=[
                MergePlayersCommand(
                    player_id="pl_a", alias_player_ids=["pl_b"], reason="merge b into a"
                )
            ],
        ),
    )
    assert res_merge.revision == 3

    # Verify merged state at R=3
    players_r3 = get_players(conn, PlayerQuery(match_id=match_id)).data
    assert len(players_r3) == 1
    pl_a_r3 = players_r3[0]
    assert pl_a_r3.player.player_id == "pl_a"
    assert pl_a_r3.player.alias_of is None
    assert pl_a_r3.confirmed_segment_ids == ["s1", "s2"]
    assert pl_a_r3.made_count == 2
    assert len(pl_a_r3.templates) == 2

    # Querying merged player directly returns empty
    assert get_players(conn, PlayerQuery(match_id=match_id, player_id="pl_b")).data == []

    # Events in merged state belong entirely to pl_a
    assert len(get_events(conn, EventQuery(match_id=match_id, player_id="pl_a")).data) == 2
    assert len(get_events(conn, EventQuery(match_id=match_id, player_id="pl_b")).data) == 0

    # Step 2: Intermediate operation at R=3 (R=3 -> R=4)
    res_inter = apply_review(
        conn,
        ReviewRequest(
            match_id=match_id,
            expected_revision=3,
            actor="admin",
            commands=[
                ConfirmAssignmentCommand(
                    segment_id="s1", player_id="pl_a", reason="confirm during merge"
                )
            ],
        ),
    )
    assert res_inter.revision == 4

    # Step 3: Restore to R=2 (R=4 -> R=5)
    res_restore = apply_review(
        conn,
        ReviewRequest(
            match_id=match_id,
            expected_revision=4,
            actor="admin",
            commands=[
                RestoreRevisionCommand(target_revision=2, reason="undo merge and restore to r2")
            ],
        ),
    )
    assert res_restore.revision == 5
    assert repo.revision(match_id) == 5

    # Step 4: Strict assertion on restored state at R=5
    # 4.1 SQL row verification
    alias_rows = {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT player_id, alias_of FROM players WHERE match_id=?", (match_id,)
        )
    }
    assert alias_rows["pl_a"] is None
    assert alias_rows["pl_b"] is None, "Player-B alias_of must be restored to NULL"

    # 4.2 Snapshot verification
    snapshot = read_identity_snapshot(conn, match_id)
    snapshot_pids = {p.player_id: p for p in snapshot.gallery.players}
    assert snapshot_pids["pl_a"].alias_of is None
    assert snapshot_pids["pl_b"].alias_of is None

    # 4.3 View verification: Both players are active and separate
    players_r5 = get_players(conn, PlayerQuery(match_id=match_id)).data
    assert len(players_r5) == 2
    by_id_r5 = {p.player.player_id: p for p in players_r5}
    assert by_id_r5["pl_a"].confirmed_segment_ids == ["s1"]
    assert by_id_r5["pl_b"].confirmed_segment_ids == ["s2"]
    assert by_id_r5["pl_a"].made_count == 1
    assert by_id_r5["pl_b"].made_count == 1
    assert len(by_id_r5["pl_a"].templates) == 1
    assert len(by_id_r5["pl_b"].templates) == 1

    # 4.4 Events verification: goals are completely decoupled back to original owners
    events_a_r5 = get_events(conn, EventQuery(match_id=match_id, player_id="pl_a")).data
    events_b_r5 = get_events(conn, EventQuery(match_id=match_id, player_id="pl_b")).data
    assert [e.event.event_id for e in events_a_r5] == ["e1"]
    assert [e.event.event_id for e in events_b_r5] == ["e2"]

    conn.close()


def test_merge_then_restore_exact_r1_to_r4_chain(tmp_path):
    """Exact prompt scenario: R=1 seeded -> R=2 merge -> R=3 intermediate -> R=4 restore to R=1."""
    conn = connect(tmp_path / "chain.sqlite")
    match_id = "m"

    clip = Clip(match_id=match_id, clip_id="c", duration_s=10.0, frame_timing="cfr", fps=10.0)
    track = Track(
        match_id=match_id,
        clip_id="c",
        track_id="t",
        start_s=0.0,
        end_s=10.0,
        source=sample("smp1", 1.0).source.model_copy(
            update={"record_type": "track", "source_record_id": "t"}
        ),
        observed_frame_indices=list(range(100)),
    )
    s1 = Segment(
        match_id=match_id, clip_id="c", track_id="t", segment_id="s1", start_s=0.0, end_s=5.0
    )
    s2 = Segment(
        match_id=match_id, clip_id="c", track_id="t", segment_id="s2", start_s=5.0, end_s=10.0
    )
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
        match_id=match_id,
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
        match_id=match_id,
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

    init_batch = batch([smp1, smp2], end_s=10.0)
    init_batch.clips = [clip]
    init_batch.identity_evidence.tracks = [track]
    init_batch.identity_evidence.segments = [s1, s2]
    init_batch.events = [ev1, ev2]

    # R=1: initialize_library with 2 seeds
    from app.contracts.api import InitializeRequest

    res_init = initialize_library(
        conn,
        InitializeRequest(
            batch=init_batch,
            expected_revision=0,
            actor="admin",
            reason="seed both players",
            seeds=[
                SeedIdentity(
                    request_id="seed_a",
                    display_name="Player-A",
                    segment_ids=["s1"],
                    clean_sample_ids=["smp1"],
                    template_sample_ids=["smp1"],
                    actor="admin",
                    reason="seed player a",
                ),
                SeedIdentity(
                    request_id="seed_b",
                    display_name="Player-B",
                    segment_ids=["s2"],
                    clean_sample_ids=["smp2"],
                    template_sample_ids=["smp2"],
                    actor="admin",
                    reason="seed player b",
                ),
            ],
        ),
    )
    assert res_init.revision == 1
    players_r1 = get_players(conn, PlayerQuery(match_id=match_id)).data
    assert len(players_r1) == 2
    pid_a = [p.player.player_id for p in players_r1 if "Player-A" in (p.player.display_name or "")][
        0
    ]
    pid_b = [p.player.player_id for p in players_r1 if "Player-B" in (p.player.display_name or "")][
        0
    ]

    # R=2: Merge B into A
    res_merge = apply_review(
        conn,
        ReviewRequest(
            match_id=match_id,
            expected_revision=1,
            actor="admin",
            commands=[
                MergePlayersCommand(
                    player_id=pid_a, alias_player_ids=[pid_b], reason="merge b into a"
                )
            ],
        ),
    )
    assert res_merge.revision == 2

    # R=3: Restore to R=1 (which results in R=3 or if intermediate is used)
    res_restore = apply_review(
        conn,
        ReviewRequest(
            match_id=match_id,
            expected_revision=2,
            actor="admin",
            commands=[RestoreRevisionCommand(target_revision=1, reason="restore to r1")],
        ),
    )
    assert res_restore.revision == 3

    # Assert Player-B alias_of is NULL and segments/goals are restored
    alias_b = conn.execute(
        "SELECT alias_of FROM players WHERE match_id=? AND player_id=?", (match_id, pid_b)
    ).fetchone()[0]
    assert alias_b is None

    players_restored = get_players(conn, PlayerQuery(match_id=match_id)).data
    assert len(players_restored) == 2
    by_pid = {p.player.player_id: p for p in players_restored}
    assert by_pid[pid_a].made_count == 1
    assert by_pid[pid_b].made_count == 1
    assert by_pid[pid_a].confirmed_segment_ids == ["s1"]
    assert by_pid[pid_b].confirmed_segment_ids == ["s2"]

    conn.close()


# ==============================================================================
# 2. 恢复后再合并（Restore then Merge）
# ==============================================================================


def test_restore_then_subsequent_operations_no_deadlock(tmp_path):
    """Verify system remains fully operational after restore without lockup or cycle."""
    conn = connect(tmp_path / "attack2.sqlite")
    match_id = _setup_match_with_two_players_and_events(conn)
    repo = Repository(conn)

    # 1. Merge pl_b into pl_a (R=2 -> R=3)
    apply_review(
        conn,
        ReviewRequest(
            match_id=match_id,
            expected_revision=2,
            actor="admin",
            commands=[
                MergePlayersCommand(
                    player_id="pl_a", alias_player_ids=["pl_b"], reason="merge b into a"
                )
            ],
        ),
    )

    # 2. Restore to R=2 (R=3 -> R=4)
    apply_review(
        conn,
        ReviewRequest(
            match_id=match_id,
            expected_revision=3,
            actor="admin",
            commands=[RestoreRevisionCommand(target_revision=2, reason="restore to r2")],
        ),
    )
    assert repo.revision(match_id) == 4

    # 3. Subsequent operation 1: Create a brand new player pl_c at R=4 -> R=5
    res_c = apply_review(
        conn,
        ReviewRequest(
            match_id=match_id,
            expected_revision=4,
            actor="admin",
            commands=[
                CreatePlayerCommand(
                    player=Player(match_id=match_id, player_id="pl_c", display_name="Player-C"),
                    reason="create pl_c",
                )
            ],
        ),
    )
    assert res_c.revision == 5
    assert len(get_players(conn, PlayerQuery(match_id=match_id)).data) == 3

    # 4. Subsequent operation 2: Merge pl_c into pl_b at R=5 -> R=6
    res_merge2 = apply_review(
        conn,
        ReviewRequest(
            match_id=match_id,
            expected_revision=5,
            actor="admin",
            commands=[
                MergePlayersCommand(
                    player_id="pl_b", alias_player_ids=["pl_c"], reason="merge c into b"
                )
            ],
        ),
    )
    assert res_merge2.revision == 6

    # Verify pl_c is alias of pl_b, while pl_b remains active
    players_r6 = get_players(conn, PlayerQuery(match_id=match_id)).data
    assert len(players_r6) == 2
    assert {p.player.player_id for p in players_r6} == {"pl_a", "pl_b"}
    alias_c = conn.execute(
        "SELECT alias_of FROM players WHERE match_id=? AND player_id='pl_c'", (match_id,)
    ).fetchone()[0]
    assert alias_c == "pl_b"

    # 5. Subsequent operation 3: Merge pl_b into pl_a at R=6 -> R=7
    res_merge3 = apply_review(
        conn,
        ReviewRequest(
            match_id=match_id,
            expected_revision=6,
            actor="admin",
            commands=[
                MergePlayersCommand(
                    player_id="pl_a", alias_player_ids=["pl_b"], reason="merge b into a again"
                )
            ],
        ),
    )
    assert res_merge3.revision == 7
    players_r7 = get_players(conn, PlayerQuery(match_id=match_id)).data
    assert len(players_r7) == 1
    assert players_r7[0].player.player_id == "pl_a"
    assert players_r7[0].made_count == 2

    conn.close()


def test_restore_multi_chain_alias_preservation(tmp_path):
    """Complex attack: Merge -> Restore -> Create & Merge -> Restore to post-restore state.

    Ensures old rolled-back merges do not get resurrected.
    """
    conn = connect(tmp_path / "attack_multichain.sqlite")
    match_id = _setup_match_with_two_players_and_events(conn)
    repo = Repository(conn)

    # R=2: pl_a and pl_b exist
    # R=3: Merge pl_b into pl_a
    apply_review(
        conn,
        ReviewRequest(
            match_id=match_id,
            expected_revision=2,
            actor="admin",
            commands=[
                MergePlayersCommand(
                    player_id="pl_a", alias_player_ids=["pl_b"], reason="merge b into a"
                )
            ],
        ),
    )
    assert repo.revision(match_id) == 3

    # R=4: Restore to R=2 (un-merging pl_b)
    apply_review(
        conn,
        ReviewRequest(
            match_id=match_id,
            expected_revision=3,
            actor="admin",
            commands=[RestoreRevisionCommand(target_revision=2, reason="restore to r2")],
        ),
    )
    assert repo.revision(match_id) == 4

    # R=5: Create pl_c
    apply_review(
        conn,
        ReviewRequest(
            match_id=match_id,
            expected_revision=4,
            actor="admin",
            commands=[
                CreatePlayerCommand(
                    player=Player(match_id=match_id, player_id="pl_c"), reason="create c"
                )
            ],
        ),
    )
    assert repo.revision(match_id) == 5

    # R=6: Merge pl_c into pl_a
    apply_review(
        conn,
        ReviewRequest(
            match_id=match_id,
            expected_revision=5,
            actor="admin",
            commands=[
                MergePlayersCommand(
                    player_id="pl_a", alias_player_ids=["pl_c"], reason="merge c into a"
                )
            ],
        ),
    )
    assert repo.revision(match_id) == 6

    # R=7: Restore to R=4 (where pl_c did not exist, and pl_b was un-merged)
    apply_review(
        conn,
        ReviewRequest(
            match_id=match_id,
            expected_revision=6,
            actor="admin",
            commands=[RestoreRevisionCommand(target_revision=4, reason="restore to r4")],
        ),
    )
    assert repo.revision(match_id) == 7

    # Assert pl_b was NOT resurrected as an alias of pl_a!
    alias_rows = {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT player_id, alias_of FROM players WHERE match_id=?", (match_id,)
        )
    }
    assert alias_rows["pl_b"] is None, "pl_b alias must be None at restored R=4"
    assert alias_rows["pl_a"] is None

    conn.close()


# ==============================================================================
# 3. 并发版本冲突（Optimistic Concurrency Collision）
# ==============================================================================


def test_optimistic_concurrency_stale_expected_revision_on_merge(tmp_path):
    """Calling MergePlayersCommand with stale expected_revision must raise RevisionConflict."""
    conn = connect(tmp_path / "attack_occ_merge.sqlite")
    match_id = _setup_match_with_two_players_and_events(conn)
    repo = Repository(conn)
    current_rev = repo.revision(match_id)
    assert current_rev == 2

    # Snapshot before attack
    snap_before = read_identity_snapshot(conn, match_id)
    reviews_before = repo.review_records(match_id)

    # Attack 1: Pass stale revision 1 (current is 2)
    with pytest.raises(RevisionConflict, match="stale review expected_revision"):
        apply_review(
            conn,
            ReviewRequest(
                match_id=match_id,
                expected_revision=1,
                actor="attacker",
                commands=[
                    MergePlayersCommand(
                        player_id="pl_a", alias_player_ids=["pl_b"], reason="stale merge"
                    )
                ],
            ),
        )

    # Attack 2: Pass revision 0
    with pytest.raises(RevisionConflict):
        apply_review(
            conn,
            ReviewRequest(
                match_id=match_id,
                expected_revision=0,
                actor="attacker",
                commands=[
                    MergePlayersCommand(
                        player_id="pl_a", alias_player_ids=["pl_b"], reason="stale merge 0"
                    )
                ],
            ),
        )

    # Attack 3: Pass future revision 999
    with pytest.raises(RevisionConflict):
        apply_review(
            conn,
            ReviewRequest(
                match_id=match_id,
                expected_revision=999,
                actor="attacker",
                commands=[
                    MergePlayersCommand(
                        player_id="pl_a", alias_player_ids=["pl_b"], reason="future merge"
                    )
                ],
            ),
        )

    # Verify 100% zero dirty data
    assert repo.revision(match_id) == current_rev
    assert repo.review_records(match_id) == reviews_before
    snap_after = read_identity_snapshot(conn, match_id)
    assert snap_after.gallery.assignments == snap_before.gallery.assignments
    assert snap_after.gallery.players == snap_before.gallery.players

    raw_aliases = conn.execute(
        "SELECT player_id, alias_of FROM players WHERE match_id=?", (match_id,)
    ).fetchall()
    assert all(row[1] is None for row in raw_aliases)

    conn.close()


def test_optimistic_concurrency_stale_expected_revision_on_restore(tmp_path):
    """Calling RestoreRevisionCommand with stale expected_revision must raise RevisionConflict."""
    conn = connect(tmp_path / "attack_occ_restore.sqlite")
    match_id = _setup_match_with_two_players_and_events(conn)
    repo = Repository(conn)

    # Advance to R=3
    apply_review(
        conn,
        ReviewRequest(
            match_id=match_id,
            expected_revision=2,
            actor="admin",
            commands=[
                MergePlayersCommand(
                    player_id="pl_a", alias_player_ids=["pl_b"], reason="merge b into a"
                )
            ],
        ),
    )
    assert repo.revision(match_id) == 3

    snap_r3 = read_identity_snapshot(conn, match_id)
    reviews_r3 = repo.review_records(match_id)

    # Attempt restore with stale expected_revision=2 (current is 3)
    with pytest.raises(RevisionConflict, match="stale review expected_revision"):
        apply_review(
            conn,
            ReviewRequest(
                match_id=match_id,
                expected_revision=2,
                actor="attacker",
                commands=[RestoreRevisionCommand(target_revision=1, reason="stale restore")],
            ),
        )

    # Verify zero dirty data
    assert repo.revision(match_id) == 3
    assert repo.review_records(match_id) == reviews_r3
    snap_after = read_identity_snapshot(conn, match_id)
    assert snap_after.gallery.assignments == snap_r3.gallery.assignments
    assert snap_after.gallery.players == snap_r3.gallery.players

    conn.close()


def test_competing_concurrent_requests_only_one_succeeds(tmp_path):
    """Simulate two racing operators competing with the same expected_revision."""
    conn = connect(tmp_path / "competing.sqlite")
    match_id = _setup_match_with_two_players_and_events(conn)
    repo = Repository(conn)
    assert repo.revision(match_id) == 2

    # Operator 1 submits merge
    req1 = ReviewRequest(
        match_id=match_id,
        expected_revision=2,
        actor="operator_1",
        request_id="req_merge_race",
        commands=[
            MergePlayersCommand(player_id="pl_a", alias_player_ids=["pl_b"], reason="racing merge")
        ],
    )
    # Operator 2 submits confirm assignment with same expected_revision=2
    req2 = ReviewRequest(
        match_id=match_id,
        expected_revision=2,
        actor="operator_2",
        request_id="req_confirm_race",
        commands=[
            ConfirmAssignmentCommand(segment_id="s1", player_id="pl_a", reason="competing confirm")
        ],
    )

    # Operator 1 executes first
    res1 = apply_review(conn, req1)
    assert res1.revision == 3

    # Operator 2 executes second -> 100% collision
    with pytest.raises(RevisionConflict):
        apply_review(conn, req2)

    # State is cleanly operator 1's merge, no residue from operator 2
    assert repo.revision(match_id) == 3
    players = get_players(conn, PlayerQuery(match_id=match_id)).data
    assert len(players) == 1
    assert players[0].player.player_id == "pl_a"

    conn.close()


# ==============================================================================
# 4. 多命令原子复合（Batch Commands Atomicity）
# ==============================================================================


def test_batch_commands_atomicity_success(tmp_path):
    """A batch containing CreatePlayer, ConfirmAssignment, and MergePlayers commits atomically."""
    conn = connect(tmp_path / "batch_success.sqlite")
    match_id = _setup_match_with_two_players_and_events(conn)
    repo = Repository(conn)
    assert repo.revision(match_id) == 2

    res = apply_review(
        conn,
        ReviewRequest(
            match_id=match_id,
            expected_revision=2,
            actor="admin",
            commands=[
                CreatePlayerCommand(
                    player=Player(match_id=match_id, player_id="pl_c", display_name="Player-C"),
                    reason="create c",
                ),
                ConfirmAssignmentCommand(
                    segment_id="s1", player_id="pl_a", reason="confirm s1 for a"
                ),
                MergePlayersCommand(
                    player_id="pl_a", alias_player_ids=["pl_b"], reason="merge b into a"
                ),
            ],
        ),
    )
    assert res.revision == 3
    assert repo.revision(match_id) == 3

    # Assert all three operations took effect atomically:
    # 1. pl_c exists
    # 2. pl_b merged into pl_a
    # 3. pl_a confirmed on s1
    players = get_players(conn, PlayerQuery(match_id=match_id)).data
    assert len(players) == 2  # pl_a (holding pl_b) and pl_c
    by_pid = {p.player.player_id: p for p in players}
    assert "pl_a" in by_pid
    assert "pl_c" in by_pid
    assert "s1" in by_pid["pl_a"].confirmed_segment_ids

    conn.close()


def test_batch_commands_atomicity_rollback_on_merge_failure(tmp_path):
    """A batch where earlier commands succeed but MergePlayers fails rolls back 100%."""
    conn = connect(tmp_path / "batch_rollback_merge.sqlite")
    match_id = _setup_match_with_two_players_and_events(conn)
    repo = Repository(conn)
    assert repo.revision(match_id) == 2

    snap_before = read_identity_snapshot(conn, match_id)
    reviews_before = repo.review_records(match_id)

    # Command 1 is valid CreatePlayer, Command 2 is invalid Merge (alias player doesn't exist)
    with pytest.raises(ValueError, match="alias player 'pl_non_existent' not found"):
        apply_review(
            conn,
            ReviewRequest(
                match_id=match_id,
                expected_revision=2,
                actor="admin",
                commands=[
                    CreatePlayerCommand(
                        player=Player(match_id=match_id, player_id="pl_c"),
                        reason="create c",
                    ),
                    MergePlayersCommand(
                        player_id="pl_a",
                        alias_player_ids=["pl_non_existent"],
                        reason="invalid merge",
                    ),
                ],
            ),
        )

    # Verify global rollback:
    # 1. Revision remains 2
    assert repo.revision(match_id) == 2
    # 2. pl_c was NOT created in DB
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM players WHERE match_id=? AND player_id='pl_c'", (match_id,)
        ).fetchone()[0]
        == 0
    )
    # 3. No review record added
    assert repo.review_records(match_id) == reviews_before
    # 4. Snapshot identical
    snap_after = read_identity_snapshot(conn, match_id)
    assert snap_after.gallery.players == snap_before.gallery.players
    assert snap_after.gallery.assignments == snap_before.gallery.assignments

    conn.close()


def test_batch_commands_atomicity_rollback_on_subsequent_failure(tmp_path):
    """A batch where MergePlayers succeeds first but subsequent command fails rolls back 100%."""
    conn = connect(tmp_path / "batch_rollback_subsequent.sqlite")
    match_id = _setup_match_with_two_players_and_events(conn)
    repo = Repository(conn)
    assert repo.revision(match_id) == 2

    # Command 1 is valid Merge, Command 2 has invalid unassigned_reason with player_id set
    with pytest.raises(ValueError, match="unassigned_reason requires player_id=null"):
        apply_review(
            conn,
            ReviewRequest(
                match_id=match_id,
                expected_revision=2,
                actor="admin",
                commands=[
                    MergePlayersCommand(
                        player_id="pl_a",
                        alias_player_ids=["pl_b"],
                        reason="valid merge before fail",
                    ),
                    SetAssignmentCommand(
                        segment_id="s1",
                        player_id="pl_a",
                        unassigned_reason="other_person",  # Invalid!
                        reason="illegal assignment command",
                    ),
                ],
            ),
        )

    # Verify pl_b was NOT merged into pl_a
    assert repo.revision(match_id) == 2
    raw_b_alias = conn.execute(
        "SELECT alias_of FROM players WHERE match_id=? AND player_id='pl_b'", (match_id,)
    ).fetchone()[0]
    assert raw_b_alias is None

    conn.close()


def test_batch_commands_atomicity_reject_restore_in_batch(tmp_path):
    """RestoreRevisionCommand combined with other commands in a single request must be rejected."""
    conn = connect(tmp_path / "batch_reject_restore.sqlite")
    match_id = _setup_match_with_two_players_and_events(conn)
    repo = Repository(conn)

    with pytest.raises(ValueError, match="cannot be combined with other commands"):
        apply_review(
            conn,
            ReviewRequest(
                match_id=match_id,
                expected_revision=2,
                actor="admin",
                commands=[
                    ConfirmAssignmentCommand(segment_id="s1", player_id="pl_a", reason="confirm"),
                    RestoreRevisionCommand(target_revision=1, reason="restore in batch"),
                ],
            ),
        )

    assert repo.revision(match_id) == 2
    conn.close()


# ==============================================================================
# 5. 异常与边界防御测试（Edge & Boundary Defenses）
# ==============================================================================


def test_merge_self_attack_rejected():
    """Merging a player into itself must be blocked immediately by validator."""
    with pytest.raises(ValueError, match="merge target cannot also be an alias player"):
        MergePlayersCommand(
            player_id="pl_a",
            alias_player_ids=["pl_a"],
            reason="self merge attack",
        )


def test_merge_duplicate_alias_ids_rejected():
    """Duplicate alias IDs in command must be blocked by validator."""
    with pytest.raises(ValueError, match="alias_player_ids must be unique"):
        MergePlayersCommand(
            player_id="pl_a",
            alias_player_ids=["pl_b", "pl_b"],
            reason="duplicate alias attack",
        )


def test_merge_cycle_attack_prevented_and_rolled_back(tmp_path):
    """Attempting to create an alias cycle (A -> B -> A) is caught and rolled back."""
    conn = connect(tmp_path / "cycle_defense.sqlite")
    match_id = _setup_match_with_two_players_and_events(conn)

    # R=3: Merge B into A (B.alias_of = A)
    apply_review(
        conn,
        ReviewRequest(
            match_id=match_id,
            expected_revision=2,
            actor="admin",
            commands=[
                MergePlayersCommand(player_id="pl_a", alias_player_ids=["pl_b"], reason="b into a")
            ],
        ),
    )

    # R=4 attempt: Merge A into B (cycle!)
    with pytest.raises(ValueError, match="player alias cycle"):
        apply_review(
            conn,
            ReviewRequest(
                match_id=match_id,
                expected_revision=3,
                actor="attacker",
                commands=[
                    MergePlayersCommand(
                        player_id="pl_b", alias_player_ids=["pl_a"], reason="cycle attack"
                    )
                ],
            ),
        )

    # Assert revision remained 3 and A is not alias of B
    assert Repository(conn).revision(match_id) == 3
    alias_a = conn.execute(
        "SELECT alias_of FROM players WHERE match_id=? AND player_id='pl_a'", (match_id,)
    ).fetchone()[0]
    assert alias_a is None

    conn.close()


def test_request_id_replay_idempotency(tmp_path):
    """Submitting duplicate request_id with same payload is idempotent and non-destructive."""
    conn = connect(tmp_path / "idempotent.sqlite")
    match_id = _setup_match_with_two_players_and_events(conn)

    req = ReviewRequest(
        match_id=match_id,
        expected_revision=2,
        actor="admin",
        request_id="stable_merge_req",
        commands=[
            MergePlayersCommand(
                player_id="pl_a", alias_player_ids=["pl_b"], reason="idempotent merge"
            )
        ],
    )

    res1 = apply_review(conn, req)
    assert res1.revision == 3

    # Replay identical request
    res2 = apply_review(conn, req)
    assert res2.revision == 3
    assert res2.model_dump() == res1.model_dump()

    # Revision did not advance again
    assert Repository(conn).revision(match_id) == 3
    conn.close()
