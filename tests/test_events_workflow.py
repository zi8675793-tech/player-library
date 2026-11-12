"""Role/time/replacement rules and revision-consistent current event views."""

import pytest

from app.contracts.api import EventQuery, PlayerQuery
from app.contracts.common import Provenance
from app.contracts.events import (
    Event,
    IntervalEventTime,
    PointEventTime,
    SegmentReference,
    Subject,
    TrackReference,
)
from app.contracts.evidence import Clip, EvidenceBatch, IdentityEvidence, Track
from app.contracts.identity import Assignment, Player, Segment
from app.events import associate_event, validate_events
from app.storage.database import connect, transaction
from app.storage.repositories import Repository
from app.workflows.read import get_events, get_players, get_review_queue


def source(key, record_type="event"):
    return Provenance(
        producer="synthetic", record_type=record_type, source_record_id=key, evidence_version="1"
    )


def context():
    clip = Clip(match_id="m", clip_id="c", duration_s=10, frame_timing="cfr", fps=10)
    track = Track(
        match_id="m",
        clip_id="c",
        track_id="t",
        start_s=0,
        end_s=10,
        source=source("t", "track"),
        observed_frame_indices=list(range(100)),
    )
    left = Segment(match_id="m", clip_id="c", track_id="t", segment_id="left", start_s=0, end_s=5)
    right = Segment(
        match_id="m", clip_id="c", track_id="t", segment_id="right", start_s=5, end_s=10
    )
    assignments = [
        Assignment(
            match_id="m",
            segment_id=segment.segment_id,
            player_id=player,
            method="manual",
            status="confirmed",
            revision=1,
        )
        for segment, player in [(left, "pl_a"), (right, "pl_b")]
    ]
    return {
        "clips": [clip],
        "tracks": [track],
        "segments": [left, right],
        "assignments": assignments,
    }


def event(key="e", *, time=None, references=None, role="shooter", **kwargs):
    return Event(
        match_id="m",
        event_id=key,
        source_event_key=key,
        clip_id="c",
        event_type="shot",
        time=time or PointEventTime(time_s=2),
        source=source(key),
        subjects=[
            Subject(
                subject_id="subject",
                role=role,
                references=references
                if references is not None
                else [TrackReference(clip_id="c", track_id="t")],
            )
        ],
        **kwargs,
    )


def mapped(event_value, ctx=None):
    return associate_event(event_value, **(ctx or context())).subjects[0]


def test_point_half_open_boundary_and_explicit_reference_interval():
    assert mapped(event(time=PointEventTime(time_s=5))).player_id == "pl_b"
    assert set(mapped(event(time=PointEventTime(time_s=10))).reason.split(";")) == {
        "event_outside_clip",
        "support_outside_clip",
    }
    # The original event is a point, but its explicit support spans a switch.
    subject = mapped(
        event(
            references=[
                TrackReference(
                    clip_id="c", track_id="t", time=IntervalEventTime(start_s=4, end_s=6)
                )
            ]
        )
    )
    assert subject.status == "unmapped"
    assert subject.reason == "identity_changes_within_support"
    assert (
        mapped(
            event(
                references=[
                    TrackReference(
                        clip_id="c", track_id="t", time=IntervalEventTime(start_s=4, end_s=5)
                    )
                ]
            )
        ).player_id
        == "pl_a"
    )


def test_independent_roles_and_multiple_support_agreement():
    e = event(
        references=[SegmentReference(segment_id="left"), TrackReference(clip_id="c", track_id="t")]
    )
    e.subjects.append(
        Subject(
            subject_id="assist",
            role="assister",
            references=[SegmentReference(segment_id="right", time=PointEventTime(time_s=6))],
        )
    )
    result = associate_event(e, **context())
    assert [s.player_id for s in result.subjects] == ["pl_a", "pl_b"]
    e.subjects[0].references.append(
        SegmentReference(segment_id="right", time=PointEventTime(time_s=6))
    )
    assert mapped(e).reason == "subject_supports_disagree"


def test_interval_gap_unconfirmed_and_overlap_never_choose_nearest():
    ctx = context()
    ctx["segments"][1] = ctx["segments"][1].model_copy(update={"start_s": 6})
    assert mapped(event(time=IntervalEventTime(start_s=4, end_s=7)), ctx).reason == "identity_gap"
    ctx = context()
    ctx["assignments"][0] = ctx["assignments"][0].model_copy(update={"status": "proposed"})
    assert mapped(event(), ctx).reason == "identity_not_confirmed"
    ctx = context()
    ctx["segments"][1] = ctx["segments"][1].model_copy(update={"start_s": 1})
    assert mapped(event(), ctx).reason == "identity_conflict"
    assert (
        mapped(event(time=IntervalEventTime(start_s=2, end_s=3)), ctx).reason == "identity_conflict"
    )


def test_explicit_replacements_only_and_current_ranges():
    ctx = context()
    original = Segment(
        match_id="m", clip_id="c", track_id="t", segment_id="old", start_s=0, end_s=10, active=False
    )
    ctx["segments"] = [
        original,
        *(s.model_copy(update={"supersedes_segment_id": "old"}) for s in ctx["segments"]),
    ]
    assert mapped(event(references=[SegmentReference(segment_id="old")]), ctx).player_id == "pl_a"
    assert (
        mapped(
            event(references=[SegmentReference(segment_id="old", time=PointEventTime(time_s=5))]),
            ctx,
        ).player_id
        == "pl_b"
    )
    assert (
        mapped(
            event(
                references=[
                    SegmentReference(segment_id="old", time=IntervalEventTime(start_s=4, end_s=6))
                ]
            ),
            ctx,
        ).status
        == "unmapped"
    )
    # An unrelated active segment on the same track is not an implicit replacement.
    ctx["segments"] = [original, *context()["segments"]]
    assert (
        mapped(event(references=[SegmentReference(segment_id="old")]), ctx).reason
        == "segment_inactive_without_replacement"
    )
    ctx["segments"][1] = ctx["segments"][1].model_copy(
        update={"supersedes_segment_id": "old", "end_s": 11}
    )
    assert (
        mapped(event(references=[SegmentReference(segment_id="old")]), ctx).reason
        == "replacement_outside_original_range"
    )


def test_actual_observations_missing_or_unknown():
    ctx = context()
    ctx["tracks"][0].observed_frame_indices.remove(20)
    assert mapped(event(), ctx).reason == "missing_observation_frames"
    assert mapped(event(time=IntervalEventTime(start_s=1.9, end_s=2)), ctx).player_id == "pl_a"
    assert (
        mapped(event(time=IntervalEventTime(start_s=1.9, end_s=2.1)), ctx).reason
        == "missing_observation_frames"
    )
    ctx["tracks"][0].observed_frame_indices = None
    assert mapped(event(), ctx).reason == "observation_information_missing"
    ctx["tracks"][0].observed_frame_indices = []
    assert mapped(event(), ctx).reason == "no_actual_observations"
    ctx = context()
    ctx["clips"][0].frame_timing = "vfr"
    assert mapped(event(), ctx).reason == "observation_time_mapping_missing"


def test_cross_clip_needs_mapping_or_reference_local_time():
    ctx = context()
    ctx["clips"].append(Clip(match_id="m", clip_id="other", duration_s=10))
    e = event().model_copy(update={"clip_id": "other"})
    assert mapped(e, ctx).reason == "cross_clip_time_mapping_missing"
    e.subjects[0].references[0].time = PointEventTime(time_s=6)
    assert mapped(e, ctx).player_id == "pl_b"
    e.subjects[0].references[0].time = None
    # model_copy does not revalidate nested assignments; reconstruct proper contracts.
    from app.contracts.evidence import SourceMapping

    ctx["clips"][0].source_mapping = SourceMapping(source_video_id="v", source_start_s=10)
    ctx["clips"][1].source_mapping = SourceMapping(source_video_id="v", source_start_s=14)
    assert mapped(e, ctx).player_id == "pl_b"  # other 2s -> source 16s -> c 6s
    result = associate_event(e, **ctx)
    assert result.source_time.time.time_s == 16
    assert associate_event(event(), **context()).source_time is None


def test_validate_known_ranges_preserves_late_targets_and_outcomes():
    ctx = context()
    kwargs = {key: value for key, value in ctx.items() if key != "assignments"}
    ordinary = event()
    assert validate_events([ordinary], match_id="m", **kwargs)[0].outcome is None
    late = event(references=[SegmentReference(segment_id="not-yet-received")])
    assert validate_events([late], match_id="m", **kwargs) == [late]
    with pytest.raises(ValueError, match="outside target"):
        validate_events(
            [
                event(
                    references=[SegmentReference(segment_id="left", time=PointEventTime(time_s=5))]
                )
            ],
            match_id="m",
            **kwargs,
        )
    with pytest.raises(ValueError, match="one match"):
        validate_events([ordinary], match_id="other", **kwargs)


def test_reads_current_identity_filters_canonical_stats_and_reopen(tmp_path):
    path = tmp_path / "events.sqlite3"
    ctx = context()
    events = [
        event("made", outcome="made", canonical_event_id="basket"),
        event("replay", outcome="made", canonical_event_id="basket"),
        event("ordinary"),
        event("assist", outcome="made", role="assister"),
        event("conflict-a", outcome="made", canonical_event_id="conflict"),
        event(
            "conflict-b",
            time=PointEventTime(time_s=6),
            outcome="made",
            canonical_event_id="conflict",
        ),
        event("late", references=[SegmentReference(segment_id="future")]),
    ]
    conn = connect(path)
    repo = Repository(conn)
    with transaction(conn):
        repo.store_batch(
            EvidenceBatch(
                match_id="m",
                batch_id="b",
                producer="synthetic",
                clips=ctx["clips"],
                identity_evidence=IdentityEvidence(tracks=ctx["tracks"], segments=ctx["segments"]),
                events=events,
            )
        )
        for player_id in ["pl_a", "pl_b"]:
            repo.put_player(Player(match_id="m", player_id=player_id))
        for assignment in ctx["assignments"]:
            repo.put_assignment(assignment, committed_revision=1)
        repo.advance_revision("m", 0, actor="tester", reason="seed", changes={})
    conn.close()
    conn = connect(path)
    result = get_players(conn, PlayerQuery(match_id="m"))
    assert result.revision == 1
    assert {v.player.player_id: v.made_count for v in result.data} == {"pl_a": 1, "pl_b": 0}
    assert (
        len(
            get_events(
                conn, EventQuery(match_id="m", player_id="pl_a", role="shooter", outcome="made")
            ).data
        )
        == 3
    )
    assert get_events(conn, EventQuery(match_id="m", start_s=2, end_s=3)).revision == 1
    assert not get_events(conn, EventQuery(match_id="m", start_s=3, end_s=5)).data
    queue = get_review_queue(conn, "m")
    assert any(
        item.event_id == "late" and item.reasons == ["segment_not_received"] for item in queue.data
    )
    assert any(item.reasons == ["canonical_scorers_conflict"] for item in queue.data)
    assert not conn.in_transaction
    assert "player_id" not in Event.model_fields
    conn.close()


def test_frame_scope_and_context_ambiguity_are_checked():
    ctx = context()
    ctx["segments"][0] = ctx["segments"][0].model_copy(
        update={"frame_start": 30, "frame_end_exclusive": 50}
    )
    assert mapped(event(), ctx).reason == "support_outside_segment_frame_range"
    ctx = context()
    ctx["assignments"].append(ctx["assignments"][0])
    with pytest.raises(ValueError, match="duplicate records"):
        mapped(event(), ctx)


def test_queue_missing_features_refreshes_and_revocation_marks_old_diagnostics(tmp_path):
    from test_identity_prepare_distance import batch, sample

    from app.contracts.api import (
        CreatePlayerCommand,
        InitializeRequest,
        ReviewRequest,
        SetAssignmentCommand,
        UpdateRequest,
    )
    from app.workflows.initialize import initialize_library
    from app.workflows.review import apply_review
    from app.workflows.update import update_library

    conn = connect(tmp_path / "queue.sqlite3")
    supplied = batch([sample("late-feature", 1)])
    supplied.identity_evidence.tracks = [context()["tracks"][0]]
    # There is enough identity scope, but initially no feature declaration or value.
    initial = EvidenceBatch(
        match_id="m",
        batch_id="scope",
        producer="synthetic",
        identity_evidence=IdentityEvidence(
            tracks=supplied.identity_evidence.tracks, segments=supplied.identity_evidence.segments
        ),
    )
    initialized = initialize_library(conn, InitializeRequest(batch=initial, expected_revision=0))
    assert not initialized.data.initialized
    queue = get_review_queue(conn, "m").data
    assert len(queue) == 1 and "missing_feature_space" in queue[0].reasons
    supplied.identity_evidence.segments = []
    supplied.identity_evidence.tracks = []
    updated = update_library(conn, UpdateRequest(batch=supplied, expected_revision=1))
    queue = get_review_queue(conn, "m").data
    assert len(queue) == 1 and queue[0].feature_space_id == "fs"
    assert "missing_feature_space" not in queue[0].reasons
    assert not queue[0].candidates
    confirmed = apply_review(
        conn,
        ReviewRequest(
            match_id="m",
            expected_revision=updated.revision,
            actor="human",
            commands=[
                CreatePlayerCommand(
                    player=Player(match_id="m", player_id="pl_a"),
                    segment_ids=["s"],
                    reason="observed identity",
                )
            ],
        ),
    )
    assert not [item for item in get_review_queue(conn, "m").data if item.kind == "identity"]
    apply_review(
        conn,
        ReviewRequest(
            match_id="m",
            expected_revision=confirmed.revision,
            actor="human",
            commands=[
                SetAssignmentCommand(
                    segment_id="s",
                    player_id=None,
                    unassigned_reason="rejected",
                    reason="wrong identity",
                )
            ],
        ),
    )
    queue = [item for item in get_review_queue(conn, "m").data if item.kind == "identity"]
    assert len(queue) == 1
    assert "matching_diagnostics_outdated" in queue[0].reasons
    assert "human_rejected" in queue[0].reasons
    assert queue[0].details["current_assignment"]["player_id"] is None
    assert not queue[0].candidates
    conn.close()


def test_historical_event_incompatible_late_support_stays_unmapped(tmp_path):
    from app.contracts.api import EventImportRequest, UpdateRequest
    from app.workflows.update import ingest_events, update_library

    conn = connect(tmp_path / "late-support.sqlite3")
    late_event = event(
        references=[SegmentReference(segment_id="left", time=PointEventTime(time_s=8))]
    )
    stored = ingest_events(
        conn,
        EventImportRequest(
            expected_revision=0,
            batch=EvidenceBatch(
                match_id="m", batch_id="early-event", producer="synthetic", events=[late_event]
            ),
        ),
    )
    ctx = context()
    update_library(
        conn,
        UpdateRequest(
            expected_revision=stored.revision,
            batch=EvidenceBatch(
                match_id="m",
                batch_id="late-scope",
                producer="synthetic",
                clips=ctx["clips"],
                identity_evidence=IdentityEvidence(tracks=ctx["tracks"], segments=ctx["segments"]),
            ),
        ),
    )
    result = get_events(conn, EventQuery(match_id="m"))
    assert len(result.data) == 1
    assert result.data[0].subjects[0].reason == "support_outside_referenced_segment"
    assert result.data[0].event == late_event
    conn.close()
