"""SQLite schema upgrades preserve data and fail atomically."""

import shutil
import sqlite3
from pathlib import Path

import pytest

from app.storage.database import connect, migrate


def test_reopen_and_repeat_migrations_preserve_data(tmp_path):
    path = tmp_path / "nested" / "library.sqlite3"
    db = connect(path)
    assert db.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    db.execute("CREATE TABLE retained (player_id TEXT PRIMARY KEY)")
    db.execute("INSERT INTO retained VALUES (?)", ("pl_stable",))
    db.execute("INSERT INTO matches(match_id, revision) VALUES ('match', 0)")
    db.execute(
        "INSERT INTO players(match_id, player_id, json) VALUES (?, ?, ?)",
        ("match", "pl_stable", '{"match_id":"match","player_id":"pl_stable"}'),
    )
    db.commit()
    versions = db.execute("SELECT * FROM schema_migrations ORDER BY version").fetchall()
    migrate(db)
    assert db.execute("SELECT * FROM schema_migrations ORDER BY version").fetchall() == versions
    db.close()

    reopened = connect(path)
    try:
        assert reopened.execute("SELECT player_id FROM retained").fetchone()[0] == "pl_stable"
        assert reopened.execute("SELECT player_id FROM players").fetchone()[0] == "pl_stable"
        assert reopened.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert (
            reopened.execute("SELECT * FROM schema_migrations ORDER BY version").fetchall()
            == versions
        )
    finally:
        reopened.close()


def test_failed_migration_rolls_back_ddl_and_preserves_existing_data(tmp_path):
    path = tmp_path / "library.sqlite3"
    db = connect(path)
    db.execute("CREATE TABLE retained (value TEXT)")
    db.execute("INSERT INTO retained VALUES ('existing')")
    db.commit()
    previous = db.execute("SELECT * FROM schema_migrations ORDER BY version").fetchall()
    migration_dir = tmp_path / "migrations"
    migration_dir.mkdir()
    originals = Path(__file__).parents[1] / "app" / "storage" / "migrations"
    for sql in originals.glob("*.sql"):
        shutil.copy(sql, migration_dir / sql.name)
    next_version = max(int(path.name.split("_", 1)[0]) for path in originals.glob("*.sql")) + 1
    (migration_dir / f"{next_version:03d}_broken.sql").write_text(
        "CREATE TABLE partial_migration (value TEXT);\n"
        "INSERT INTO table_that_does_not_exist VALUES (1);\n",
        encoding="utf-8",
    )
    with pytest.raises(sqlite3.Error):
        migrate(db, migrations_dir=migration_dir)
    assert db.execute("SELECT value FROM retained").fetchone()[0] == "existing"
    assert not db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='partial_migration'"
    ).fetchone()
    assert db.execute("SELECT * FROM schema_migrations ORDER BY version").fetchall() == previous
    db.close()


def legacy_database(path):
    """Build populated v3 data, removing only the new schema in this test copy."""
    from test_review_advanced_attacks import _setup_match_with_two_players_and_events

    db = connect(path)
    _setup_match_with_two_players_and_events(db)
    db.execute("DROP TABLE revision_states")
    db.execute("DROP TABLE identity_discovery_reports")
    for table in ("risk_reviews", "template_reviews", "constraints", "sample_clean_reviews"):
        db.execute(f"ALTER TABLE {table} DROP COLUMN current")
    db.execute("DROP TABLE matching_diagnostics")
    db.execute("DROP TABLE IF EXISTS incremental_operations")
    db.execute("DELETE FROM schema_migrations WHERE version>=4")
    db.close()


def test_v3_upgrade_preserves_state_and_can_restore_legacy_revision(tmp_path):
    from app.contracts.api import RestoreRevisionCommand, ReviewRequest
    from app.storage.repositories import Repository
    from app.workflows.review import apply_review

    path = tmp_path / "legacy.sqlite3"
    legacy_database(path)
    db = connect(path)
    assert Repository(db).revision("m") == 2
    assert db.execute("SELECT COUNT(*) FROM players").fetchone()[0] == 2
    assert db.execute("SELECT COUNT(*) FROM revision_states").fetchone()[0] == 1
    response = apply_review(
        db,
        ReviewRequest(
            match_id="m",
            expected_revision=2,
            actor="human",
            commands=[
                RestoreRevisionCommand(
                    target_revision=1, reason="restore pre-upgrade evidence state"
                )
            ],
        ),
    )
    assert response.revision == 3
    assert not db.execute("SELECT 1 FROM assignments WHERE current=1").fetchall()
    assert not db.execute("SELECT 1 FROM templates WHERE current=1").fetchall()
    assert db.execute("SELECT COUNT(*) FROM template_reviews").fetchone()[0] == 2
    assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert db.execute("PRAGMA foreign_key_check").fetchall() == []
    db.close()


@pytest.mark.parametrize("superseded", [False, True])
def test_legacy_clean_review_binding_only_migrates_unambiguous_source(tmp_path, superseded):
    from app.contracts.api import InitializeRequest, SeedIdentity, UpdateRequest
    from app.contracts.common import SourceKey
    from app.contracts.evidence import EvidenceBatch, IdentityEvidence
    from app.storage.repositories import Repository
    from app.storage.snapshot import read_identity_snapshot
    from app.workflows._review_ops import current_evidence
    from app.workflows.initialize import initialize_library
    from app.workflows.update import update_library
    from examples.workflow_v1 import synthetic_batch

    path = tmp_path / "legacy-clean.sqlite"
    db = connect(path)
    batch = synthetic_batch("a", with_event=False)
    initialize_library(
        db,
        InitializeRequest(
            batch=batch,
            expected_revision=0,
            actor="human",
            reason="verified",
            seeds=[
                SeedIdentity(
                    request_id="a",
                    segment_ids=["a"],
                    clean_sample_ids=["a-0"],
                    actor="human",
                    reason="old clean approval",
                )
            ],
        ),
    )
    if superseded:
        sample = batch.identity_evidence.samples[0].model_copy(deep=True)
        sample.source = sample.source.model_copy(
            update={
                "evidence_version": "2",
                "supersedes": SourceKey(
                    **sample.source.model_dump(include=set(SourceKey.model_fields))
                ),
            }
        )
        sample.features[0].values = [0.0, 1.0] * 3
        sample.tracking_quality = None
        update_library(
            db,
            UpdateRequest(
                expected_revision=1,
                batch=EvidenceBatch(
                    match_id=batch.match_id,
                    batch_id="new",
                    producer="test",
                    identity_evidence=IdentityEvidence(samples=[sample]),
                ),
            ),
        )
    db.execute("UPDATE sample_clean_reviews SET json=json_remove(json,'$.source_key')")
    db.execute("DROP TABLE revision_states")
    db.execute("DROP TABLE identity_discovery_reports")
    for table in ("risk_reviews", "template_reviews", "constraints", "sample_clean_reviews"):
        db.execute(f"ALTER TABLE {table} DROP COLUMN current")
    db.execute("DROP TABLE matching_diagnostics")
    db.execute("DROP TABLE IF EXISTS incremental_operations")
    db.execute("DELETE FROM schema_migrations WHERE version>=4")
    db.close()
    db = connect(path)
    evidence = current_evidence(
        read_identity_snapshot(db, batch.match_id).gallery.evidence,
        clean_reviews=Repository(db).clean_reviews(batch.match_id),
    )
    sample = next(s for s in evidence.samples if s.sample_id == "a-0")
    assert sample.tracking_quality.clean_review == ("unreviewed" if superseded else "approved")
    assert db.execute("SELECT COUNT(*) FROM sample_clean_reviews").fetchone()[0] == 1
    assert Repository(db).revision(batch.match_id) == (2 if superseded else 1)
    db.close()


def test_legacy_deleted_review_history_is_not_guessed_during_restore(tmp_path):
    from app.contracts.api import RestoreRevisionCommand, ReviewRequest
    from app.workflows.review import apply_review

    path = tmp_path / "lost-history.sqlite"
    legacy_database(path)
    db = sqlite3.connect(path)
    db.execute("UPDATE matches SET revision=3 WHERE match_id='m'")
    db.execute("INSERT INTO revisions VALUES ('m',3,2,'human','restore old implementation','{}')")
    db.execute(
        "INSERT INTO workflow_reviews VALUES ('m',3,'human',?)",
        ('[{"kind":"restore_revision","target_revision":1,"reason":"undo"}]',),
    )
    db.execute("UPDATE assignments SET current=0")
    db.execute("UPDATE templates SET current=0")
    db.execute("DELETE FROM template_reviews")
    db.commit()
    db.close()
    db = connect(path)
    before = list(db.iterdump())
    with pytest.raises(ValueError, match="no reliable state manifest"):
        apply_review(
            db,
            ReviewRequest(
                match_id="m",
                expected_revision=3,
                actor="human",
                commands=[
                    RestoreRevisionCommand(
                        target_revision=2, reason="cannot reconstruct lost approvals"
                    )
                ],
            ),
        )
    assert list(db.iterdump()) == before
    db.close()
