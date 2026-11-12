"""Cross-batch provenance, late references and caller-owned transactions."""

import pytest

from app.contracts.common import Provenance, SourceLocation
from app.contracts.events import Event, PointEventTime, Subject, TrackReference
from app.contracts.evidence import Clip, EvidenceBatch, IdentityEvidence, Track
from app.storage.database import connect, transaction
from app.storage.repositories import Repository, RevisionConflict, StorageConflict


def source(version="1", *, supersedes=None, line=1):
    return Provenance(
        producer="producer",
        record_type="event",
        source_record_id="source-event",
        evidence_version=version,
        supersedes=supersedes,
        locations=[SourceLocation(file_id="file", line_number=line)],
    )


def batch(batch_id, *, event=None, identity=None, clips=None, match="match"):
    return EvidenceBatch(
        match_id=match,
        batch_id=batch_id,
        producer="producer",
        events=[event] if event else [],
        identity_evidence=identity,
        clips=clips,
    )


def event(src=None, *, event_type="shot", match="match"):
    return Event(
        match_id=match,
        event_id="event",
        source_event_key="source-event",
        clip_id="clip",
        event_type=event_type,
        time=PointEventTime(time_s=1),
        source=src or source(),
        subjects=[
            Subject(
                subject_id="subject",
                role="shooter",
                references=[TrackReference(clip_id="clip", track_id="track")],
            )
        ],
    )


def test_event_arrives_first_duplicates_conflicts_and_new_version(tmp_path):
    db = connect(tmp_path / "db.sqlite")
    repo = Repository(db)
    with transaction(db):
        assert repo.store_batch(batch("one", event=event()))
        repo.advance_revision("match", 0, actor="tester", reason="ingest", changes=["event"])
    with transaction(db):
        assert not repo.store_batch(batch("two", event=event(source(line=42))))
        assert not repo.store_batch(batch("empty", clips=[]))
    assert repo.revision("match") == 1
    assert db.execute("SELECT count(*) FROM events").fetchone()[0] == 1
    assert db.execute("SELECT count(*) FROM subject_refs").fetchone()[0] == 1
    with pytest.raises(StorageConflict), transaction(db):
        repo.store_batch(batch("conflict", event=event(event_type="pass")))
    old_key = source().model_dump(
        include={"producer", "record_type", "source_record_id", "evidence_version"}
    )
    with transaction(db):
        assert repo.store_batch(
            batch("version-two", event=event(source("2", supersedes=old_key), event_type="pass"))
        )
        repo.advance_revision("match", 1, actor="tester", reason="correction", changes=["event"])
    assert db.execute("SELECT count(*) FROM source_records").fetchone()[0] == 2
    with transaction(db):
        assert not repo.store_batch(batch("old-again", event=event()))
    assert (
        Event.model_validate_json(db.execute("SELECT json FROM events").fetchone()[0]).event_type
        == "pass"
    )
    with pytest.raises(StorageConflict), transaction(db):
        repo.store_batch(batch("bad-version", event=event(source("3"), event_type="rebound")))
    assert db.execute("SELECT count(*) FROM source_records").fetchone()[0] == 2
    assert repo.revision("match") == 2
    track_source = Provenance(
        producer="producer", record_type="track", source_record_id="track", evidence_version="1"
    )
    track = Track(
        match_id="match", clip_id="clip", track_id="track", start_s=0, end_s=2, source=track_source
    )
    with transaction(db):
        assert repo.store_batch(
            batch(
                "late",
                clips=[Clip(match_id="match", clip_id="clip")],
                identity=IdentityEvidence(tracks=[track]),
            )
        )
        repo.advance_revision("match", 2, actor="tester", reason="late track", changes=["track"])
    assert db.execute("SELECT count(*) FROM subject_refs").fetchone()[0] == 1
    db.close()


def test_failed_write_and_stale_revision_rollback_all_changes(tmp_path):
    path = tmp_path / "db.sqlite"
    a, b = connect(path), connect(path)
    first, second = Repository(a), Repository(b)
    with transaction(a):
        first.ensure_match("match")
    with pytest.raises(RuntimeError, match="fail"), transaction(a):
        assert first.store_batch(batch("rolled-back", event=event()))
        first.advance_revision("match", 0, actor="tester", reason="test", changes=["event"])
        raise RuntimeError("fail")
    assert first.revision("match") == 0
    assert a.execute("SELECT count(*) FROM events").fetchone()[0] == 0
    assert a.execute("SELECT count(*) FROM revisions").fetchone()[0] == 0
    with transaction(a):
        assert first.store_batch(batch("one", event=event()))
        first.advance_revision("match", 0, actor="tester", reason="first", changes=["event"])
    with pytest.raises(RevisionConflict), transaction(b):
        second.store_batch(batch("stale", clips=[Clip(match_id="match", clip_id="late")]))
        second.advance_revision("match", 0, actor="tester", reason="stale", changes=["clip"])
    assert second.revision("match") == 1
    assert b.execute("SELECT count(*) FROM clips WHERE clip_id='late'").fetchone()[0] == 0
    assert b.execute("SELECT count(*) FROM revisions").fetchone()[0] == 1
    a.close()
    b.close()


def test_match_scope_and_formal_relationships(tmp_path):
    import sqlite3

    from app.contracts.identity import Assignment, Player, Segment

    db = connect(tmp_path / "db.sqlite")
    repo = Repository(db)
    with transaction(db):
        for match in ("one", "two"):
            item = event(match=match)
            assert repo.store_batch(batch("same", event=item, match=match))
        repo.put_player(Player(match_id="one", player_id="pl_same"))
        repo.put_player(Player(match_id="two", player_id="pl_same"))
        repo.put_player(Player(match_id="one", player_id="pl_only_one"))
        segment = Segment(
            match_id="two",
            segment_id="segment",
            clip_id="clip",
            track_id="track",
            start_s=0,
            end_s=2,
        )
        repo.store_batch(
            batch("segment", match="two", identity=IdentityEvidence(segments=[segment]))
        )
    assert db.execute("SELECT count(*) FROM events WHERE event_id='event'").fetchone()[0] == 2
    assignment = Assignment(
        match_id="two",
        segment_id="segment",
        player_id="pl_only_one",
        status="confirmed",
        method="manual",
        revision=0,
    )
    with pytest.raises(sqlite3.IntegrityError), transaction(db):
        repo.put_assignment(assignment, committed_revision=1)
    assert db.execute("SELECT count(*) FROM assignments").fetchone()[0] == 0
    db.close()


def test_explicit_checks_detect_overlap_and_alias_cycle(tmp_path):
    from app.contracts.identity import Player, Segment

    db = connect(tmp_path / "db.sqlite")
    repo = Repository(db)
    segments = [
        Segment(
            match_id="match", segment_id="a", clip_id="clip", track_id="track", start_s=0, end_s=2
        ),
        Segment(
            match_id="match", segment_id="b", clip_id="clip", track_id="track", start_s=1, end_s=3
        ),
        Segment(
            match_id="match",
            segment_id="other-clip",
            clip_id="other",
            track_id="track",
            start_s=1,
            end_s=3,
        ),
        Segment(
            match_id="match",
            segment_id="adjacent",
            clip_id="clip",
            track_id="track",
            start_s=3,
            end_s=4,
        ),
    ]
    with transaction(db):
        repo.store_batch(batch("segments", identity=IdentityEvidence(segments=segments)))
        repo.put_player(Player(match_id="match", player_id="pl_a"))
        repo.put_player(Player(match_id="match", player_id="pl_b", alias_of="pl_a"))
        repo.put_player(Player(match_id="match", player_id="pl_a", alias_of="pl_b"), update=True)
        assert repo.check_segment_overlaps("match") == [("a", "b")]
        assert repo.check_alias_cycles("match") == [["pl_a", "pl_b"]]
    db.close()


def test_same_batch_reordered_is_receipt_only(tmp_path):
    db = connect(tmp_path / "db.sqlite")
    repo = Repository(db)
    first = event()
    second = first.model_copy(
        update={
            "event_id": "event-two",
            "source_event_key": "second",
            "source": Provenance(
                producer="producer",
                record_type="event",
                source_record_id="second",
                evidence_version="1",
            ),
        }
    )
    ordered = EvidenceBatch(
        match_id="match", batch_id="same", producer="producer", events=[first, second]
    )
    reordered = ordered.model_copy(update={"events": [second, first]})
    with transaction(db):
        assert repo.store_batch(ordered)
        repo.advance_revision("match", 0, actor="tester", reason="ingest", changes=["events"])
    with transaction(db):
        assert not repo.store_batch(reordered)
    assert repo.revision("match") == 1
    assert db.execute("SELECT count(*) FROM events").fetchone()[0] == 2
    db.close()
