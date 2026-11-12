"""Machine frame evidence survives shared ingestion, source versions and upgrades."""

import pytest

from app.contracts.api import UpdateRequest
from app.contracts.common import Provenance, SourceKey
from app.contracts.evidence import EvidenceBatch, IdentityEvidence, Track, TrackObservation
from app.storage.database import connect
from app.storage.repositories import Repository, StorageConflict
from app.storage.snapshot import read_identity_snapshot
from app.workflows.update import update_library


def observation(index=0):
    return TrackObservation(
        observation_id=f"obs-{index}",
        match_id="m",
        clip_id="c",
        track_id="t",
        time_s=float(index),
        frame_idx=index * 25,
        bbox_xyxy=(1.0 + index, 2.0, 11.0 + index, 22.0),
        detection_score=0.85,
        source=Provenance(
            producer="tracker",
            record_type="observation",
            source_record_id=f"c:t:{index}",
            evidence_version="v1",
        ),
    )


def batch(observations, batch_id="one", *, tracks=True):
    return EvidenceBatch(
        match_id="m",
        batch_id=batch_id,
        producer="tracker",
        identity_evidence=IdentityEvidence(
            observations=observations,
            tracks=[
                Track(
                    match_id="m",
                    clip_id="c",
                    track_id="t",
                    start_s=0.0,
                    end_s=3.0,
                    source=Provenance(
                        producer="tracker",
                        record_type="track",
                        source_record_id="c:t",
                        evidence_version="v1",
                    ),
                )
            ]
            if tracks
            else [],
        ),
    )


def ingest(db, value):
    return update_library(
        db,
        UpdateRequest(
            batch=value,
            expected_revision=Repository(db).revision("m")
            if db.execute("SELECT 1 FROM matches WHERE match_id='m'").fetchone()
            else 0,
        ),
    )


def test_observations_round_trip_and_reordered_handoff_are_idempotent(tmp_path):
    path = tmp_path / "new.sqlite3"
    db = connect(path)
    original = [observation(0), observation(1), observation(2)]
    first = ingest(db, batch(original))
    assert first.revision == 1
    assert read_identity_snapshot(db, "m").gallery.evidence.observations == original
    second = ingest(db, batch(list(reversed(original)), "next"))
    assert second.revision == 1 and not second.data.changed
    assert db.execute("SELECT count(*) FROM track_observations").fetchone()[0] == 3
    assert db.execute("PRAGMA foreign_key_check").fetchall() == []
    db.close()
    reopened = connect(path)
    assert read_identity_snapshot(reopened, "m").gallery.evidence.observations == original
    reopened.close()


def test_observation_source_conflict_rolls_back_whole_handoff(tmp_path):
    db = connect(tmp_path / "new.sqlite3")
    first = observation()
    ingest(db, batch([first]))
    before = list(db.iterdump())
    changed = first.model_copy(update={"detection_score": 0.3})
    with pytest.raises(StorageConflict, match="different business content"):
        ingest(db, batch([observation(1), changed], "bad"))
    assert list(db.iterdump()) == before
    db.close()


def test_observation_explicit_source_version_is_preserved_and_old_replay_cannot_revert(tmp_path):
    db = connect(tmp_path / "new.sqlite3")
    first = observation()
    ingest(db, batch([first]))
    updated = first.model_copy(
        update={
            "bbox_xyxy": (2.0, 2.0, 12.0, 22.0),
            "source": first.source.model_copy(
                update={
                    "evidence_version": "v2",
                    "supersedes": SourceKey(
                        **first.source.model_dump(include=set(SourceKey.model_fields))
                    ),
                }
            ),
        }
    )
    assert ingest(db, batch([updated], "corrected")).revision == 2
    assert read_identity_snapshot(db, "m").gallery.evidence.observations == [updated]
    assert not ingest(db, batch([first], "old-replay")).data.changed
    assert read_identity_snapshot(db, "m").gallery.evidence.observations == [updated]
    assert (
        db.execute(
            "SELECT count(*) FROM source_records WHERE entity_type='track_observations'"
        ).fetchone()[0]
        == 2
    )
    db.close()


def test_observation_can_precede_track_and_late_incompatible_track_rolls_back(tmp_path):
    db = connect(tmp_path / "new.sqlite3")
    ingest(db, batch([observation(2)], tracks=False))
    bad = batch([], "late")
    bad.identity_evidence.tracks[0].end_s = 1.0
    before = list(db.iterdump())
    with pytest.raises(ValueError, match="outside"):
        ingest(db, bad)
    assert list(db.iterdump()) == before
    assert read_identity_snapshot(db, "m").gallery.evidence.observations == [observation(2)]
    db.close()


def test_v5_upgrade_keeps_existing_identity_and_adds_empty_observation_store(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    db = connect(path)
    db.execute("INSERT INTO matches VALUES ('m',0)")
    db.execute(
        "INSERT INTO players(match_id,player_id,json) VALUES ('m','pl_stable',?)",
        ('{"match_id":"m","player_id":"pl_stable"}',),
    )
    db.execute("DROP TABLE identity_discovery_reports")
    db.execute("DROP TABLE track_observations")
    db.execute("DELETE FROM schema_migrations WHERE version>=6")
    db.close()
    db = connect(path)
    state = read_identity_snapshot(db, "m")
    assert state.gallery.players[0].player_id == "pl_stable"
    assert not state.gallery.evidence.observations
    assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    db.close()
