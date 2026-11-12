"""Late context must not turn invalid events into statistics or break retries."""

import pytest

from app.contracts.api import (
    CreatePlayerCommand,
    EventImportRequest,
    EventQuery,
    PlayerQuery,
    ReviewRequest,
    UpdateRequest,
)
from app.contracts.common import Provenance
from app.contracts.events import Event, IntervalEventTime, PointEventTime, Subject, TrackReference
from app.contracts.evidence import Clip, EvidenceBatch, IdentityEvidence, Track
from app.contracts.identity import Player, Segment
from app.storage.database import connect
from app.storage.repositories import Repository
from app.workflows.read import get_events, get_players, get_review_queue
from app.workflows.review import apply_review
from app.workflows.update import ingest_events, update_library


def source(key):
    return Provenance(
        producer="regression", record_type="record", source_record_id=key, evidence_version="1"
    )


def event(time, support=None):
    return Event(
        match_id="m",
        event_id="e",
        source_event_key="e",
        clip_id="c",
        event_type="shot",
        time=time,
        outcome="made",
        source=source("e"),
        subjects=[
            Subject(
                subject_id="shooter",
                role="shooter",
                references=[
                    TrackReference(
                        clip_id="c", track_id="t", time=support or PointEventTime(time_s=2)
                    )
                ],
            )
        ],
    )


def send_event(db, item, batch_id="events"):
    return ingest_events(
        db,
        EventImportRequest(
            expected_revision=Repository(db).revision("m")
            if db.execute("SELECT 1 FROM matches WHERE match_id='m'").fetchone()
            else 0,
            batch=EvidenceBatch(
                match_id="m", batch_id=batch_id, producer="regression", events=[item]
            ),
        ),
    )


@pytest.mark.parametrize(
    "time",
    [PointEventTime(time_s=100), PointEventTime(time_s=10), IntervalEventTime(start_s=9, end_s=11)],
)
def test_late_clip_quarantines_out_of_bounds_event_and_preserves_identity(tmp_path, time):
    db = connect(tmp_path / "late.sqlite")
    item = event(time)
    send_event(db, item)
    update_library(
        db,
        UpdateRequest(
            expected_revision=1,
            batch=EvidenceBatch(
                match_id="m",
                batch_id="late",
                producer="regression",
                clips=[Clip(match_id="m", clip_id="c", duration_s=10, frame_timing="cfr", fps=10)],
                identity_evidence=IdentityEvidence(
                    tracks=[
                        Track(
                            match_id="m",
                            clip_id="c",
                            track_id="t",
                            start_s=0,
                            end_s=10,
                            observed_frame_indices=list(range(100)),
                            source=source("t"),
                        )
                    ],
                    segments=[
                        Segment(
                            match_id="m",
                            clip_id="c",
                            track_id="t",
                            segment_id="s",
                            start_s=0,
                            end_s=10,
                        )
                    ],
                ),
            ),
        ),
    )
    apply_review(
        db,
        ReviewRequest(
            match_id="m",
            expected_revision=2,
            actor="human",
            commands=[
                CreatePlayerCommand(
                    player=Player(match_id="m", player_id="pl_a"),
                    segment_ids=["s"],
                    reason="verified",
                )
            ],
        ),
    )
    resolved = get_events(db, EventQuery(match_id="m")).data[0]
    assert resolved.event == item
    assert resolved.subjects[0].status == "unmapped"
    assert "event_outside_clip" in resolved.subjects[0].reason
    player = get_players(db, PlayerQuery(match_id="m")).data[0]
    assert player.confirmed_segment_ids == ["s"]
    assert player.made_count == 0
    assert any(
        q.kind == "event" and q.event_id == "e" and "event_outside_clip" in q.reasons
        for q in get_review_queue(db, "m").data
    )


def late_track(db):
    item = event(PointEventTime(time_s=1.5), IntervalEventTime(start_s=0, end_s=5))
    send_event(db, item)
    update_library(
        db,
        UpdateRequest(
            expected_revision=1,
            batch=EvidenceBatch(
                match_id="m",
                batch_id="track",
                producer="regression",
                identity_evidence=IdentityEvidence(
                    tracks=[
                        Track(
                            match_id="m",
                            clip_id="c",
                            track_id="t",
                            start_s=1,
                            end_s=2,
                            source=source("t"),
                        )
                    ]
                ),
            ),
        ),
    )
    return item


def test_late_track_keeps_historical_retry_idempotent(tmp_path):
    db = connect(tmp_path / "retry.sqlite")
    item = late_track(db)
    response = send_event(db, item, "different-batch")
    assert response.revision == 2
    assert len(Repository(db).events("m")) == 1
    assert get_events(db, EventQuery(match_id="m")).data[0].subjects[0].status == "unmapped"


@pytest.mark.parametrize("mode", ["new-event", "new-version", "same-version-mutation"])
def test_late_track_still_rejects_new_invalid_or_rewritten_evidence(tmp_path, mode):
    from app.contracts.common import SourceKey

    db = connect(tmp_path / "new.sqlite")
    item = late_track(db).model_copy(deep=True)
    if mode == "new-event":
        item.event_id = "new"
        item.source_event_key = "new"
        item.source = source("new")
    elif mode == "new-version":
        item.source = item.source.model_copy(
            update={
                "evidence_version": "2",
                "supersedes": SourceKey(
                    **item.source.model_dump(
                        include={"producer", "record_type", "source_record_id", "evidence_version"}
                    )
                ),
            }
        )
    else:
        from app.contracts.events import EventOutcome

        item.outcome = EventOutcome.MISSED
    before = list(db.iterdump())
    with pytest.raises(ValueError):
        send_event(db, item, "changed")
    assert list(db.iterdump()) == before


def test_event_bounds_remain_reviewable_without_subjects():
    from app.events.associate import associate_event

    item = event(PointEventTime(time_s=100)).model_copy(update={"subjects": []})
    resolved = associate_event(
        item, assignments=[], segments=[], clips=[Clip(match_id="m", clip_id="c", duration_s=10)]
    )
    assert resolved.validation_reasons == ["event_outside_clip"]
    assert resolved.subjects == []


def test_valid_interval_can_end_exactly_at_clip_end():
    from app.events.associate import associate_event

    item = event(IntervalEventTime(start_s=9, end_s=10))
    resolved = associate_event(
        item, assignments=[], segments=[], clips=[Clip(match_id="m", clip_id="c", duration_s=10)]
    )
    assert resolved.validation_reasons == []
