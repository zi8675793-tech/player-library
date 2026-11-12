"""Cold-start workflow acceptance uses generated observations, never identity answers."""

import json

import pytest
from test_machine_input import machine_settings
from test_track_consistency import make_batch

from app.contracts.api import (
    RestoreRevisionCommand,
    ReviewRequest,
    ReviewTemplateCommand,
    SetAssignmentCommand,
    UpdateRequest,
)
from app.contracts.evidence import Clip, EvidenceBatch, IdentityEvidence
from app.contracts.identity import Player
from app.identity.consistency import segment_tracks
from app.presentation import baseline, cli
from app.storage.database import connect, transaction
from app.storage.repositories import Repository, RevisionConflict, StorageConflict
from app.workflows.discover import discover_library
from app.workflows.read import get_discovery_reports, get_identity_state
from app.workflows.review import apply_review
from app.workflows.update import update_library


def cross_clip_batch():
    pieces = []
    for clip in ("alpha", "beta"):
        text = make_batch(angles=(0.0, 0.02, 0.01, -0.01, 0.0, 0.01)).model_dump_json()
        for old, new in (
            ('"clip"', json.dumps(clip)),
            ("parent-0", f"parent-{clip}"),
            ("s-0-", f"s-{clip}-"),
            ("o-0-", f"o-{clip}-"),
            ('"t-0"', json.dumps(f"t-{clip}")),
        ):
            text = text.replace(old, new)
        pieces.append(EvidenceBatch.model_validate_json(text))
    evidence = IdentityEvidence(
        tracks=[t for p in pieces for t in p.identity_evidence.tracks],
        observations=[o for p in pieces for o in p.identity_evidence.observations],
        segments=[s for p in pieces for s in p.identity_evidence.segments],
        samples=[s for p in pieces for s in p.identity_evidence.samples],
        feature_spaces=pieces[0].identity_evidence.feature_spaces,
    )
    return EvidenceBatch(
        match_id="m",
        batch_id="synthetic",
        producer="synthetic",
        clips=[c for p in pieces for c in p.clips],
        identity_evidence=evidence,
    )


def prepared_database(tmp_path):
    settings = machine_settings(tmp_path / "unused-machine-root", tmp_path)
    result = segment_tracks(cross_clip_batch())
    db = connect(":memory:")
    imported = update_library(
        db,
        UpdateRequest(batch=result.batch, expected_revision=0),
        feature_values={},
        settings=settings,
    )
    return db, settings, result, imported.revision


def test_discovery_persists_anonymous_predictions_automatic_support_and_replay(tmp_path):
    db, settings, result, revision = prepared_database(tmp_path)
    accepted = discover_library(
        db,
        result.batch,
        segmentation_report=result.report,
        feature_values={},
        settings=settings,
        expected_revision=revision,
    )
    assert accepted.revision == revision + 1
    state = get_identity_state(db, "m")
    assert len(state.data.players) == 1
    assert len(state.data.assignments) == 2
    assert {a.status for a in state.data.assignments} == {"proposed"}
    assert {a.method for a in state.data.assignments} == {"automatic"}
    assert state.data.players[0].jersey_number is None
    assert len(state.data.templates) >= 2
    assert all(t.admission == "automatic_experimental" for t in state.data.templates)
    assert all(t.review_status == "unreviewed" and t.approval is None for t in state.data.templates)
    reports = get_discovery_reports(db, "m")
    assert reports.data[0]["report"]["policy"]["calibration"] == "unvalidated"
    assert len(reports.data[0]["report"]["groups"]) == 1
    assert Repository(db).review_records("m") == []
    replay = discover_library(
        db,
        result.batch.model_copy(update={"batch_id": "other-batch"}),
        segmentation_report=result.report,
        feature_values={},
        settings=settings,
        expected_revision=accepted.revision,
    )
    assert replay.revision == accepted.revision
    assert replay.data["changed"] is False
    assert get_identity_state(db, "m") == state
    assert len(Repository(db).revisions("m")) == 2
    db.close()


def test_discovery_refuses_existing_players_and_rolls_back(tmp_path):
    db, settings, result, revision = prepared_database(tmp_path)
    with transaction(db):
        repo = Repository(db)
        repo.put_player(Player(match_id="m", player_id="pl_existing"))
        repo.advance_revision("m", revision, actor="human", reason="fixture", changes={})
    before = get_identity_state(db, "m")
    with pytest.raises(StorageConflict, match="empty identity library"):
        discover_library(
            db,
            result.batch,
            segmentation_report=result.report,
            feature_values={},
            settings=settings,
            expected_revision=revision + 1,
        )
    assert get_identity_state(db, "m") == before
    assert get_discovery_reports(db, "m").data == []
    db.close()


def test_discovery_checks_revision_before_writing_predictions(tmp_path):
    db, settings, result, revision = prepared_database(tmp_path)
    with pytest.raises(RevisionConflict):
        discover_library(
            db,
            result.batch,
            segmentation_report=result.report,
            feature_values={},
            settings=settings,
            expected_revision=revision - 1,
        )
    assert get_identity_state(db, "m").data.players == []
    assert get_discovery_reports(db, "m").data == []
    db.close()


def test_discovery_commit_failure_rolls_back_players_assignments_support_and_revision(
    tmp_path, monkeypatch
):
    db, settings, result, revision = prepared_database(tmp_path)
    before = get_identity_state(db, "m")

    def fail_template(*args, **kwargs):
        raise RuntimeError("injected template write failure")

    monkeypatch.setattr(Repository, "put_template", fail_template)
    with pytest.raises(RuntimeError, match="injected"):
        discover_library(
            db,
            result.batch,
            segmentation_report=result.report,
            feature_values={},
            settings=settings,
            expected_revision=revision,
        )
    assert get_identity_state(db, "m") == before
    assert get_discovery_reports(db, "m").data == []
    assert len(Repository(db).revisions("m")) == 1
    db.close()


def test_discovery_cli_exports_nonempty_experimental_groups_templates_unknown_and_hashes(
    tmp_path, monkeypatch
):
    from test_machine_input import digest

    settings = machine_settings(tmp_path / "unused-machine-root", tmp_path)
    monkeypatch.setattr(baseline, "PROJECT_VAR", settings.paths.runtime_root)
    monkeypatch.setattr(
        baseline,
        "load_machine_input",
        lambda settings, batch_id: (
            cross_clip_batch().model_copy(update={"batch_id": batch_id}),
            {},
        ),
    )
    assert (
        cli.main(
            [
                "baseline",
                "--config",
                str(tmp_path / "machine.toml"),
                "--run-id",
                "discovered",
                "--discover-identities",
            ]
        )
        == 0
    )
    output = settings.paths.runtime_root / "discovered"

    def read(name):
        return json.loads((output / name).read_text())

    prediction = read("predictions.json")
    discovery = read("discovery.json")
    assert len(prediction["identity_groups"]) == 1
    assert len(prediction["identity_groups"][0]["member_segment_ids"]) == 2
    assert prediction["unknown_segments"] == []
    assert prediction["calibration"] == "unvalidated"
    assert prediction["identity_status"] == "experimental"
    assert len(prediction["templates"]) >= 2
    assert discovery["report"]["groups"][0]["templates"][0]["witness_sample_ids"]
    assert all(t["approval"] is None for t in prediction["templates"])
    assert read("summary.json")["discovery_idempotency"]["changed"] is False
    assert read("config.json")["identity_discovery_enabled"] is True
    assert read("config.json")["segment_tracks_enabled"] is True
    hashes = read("artifact-hashes.json")
    assert {"discovery.json", "segmentation.json", "player-library.sqlite3"} <= hashes.keys()
    assert all(digest(output / name) == checksum for name, checksum in hashes.items())


def test_discovered_state_survives_ordinary_ingest_and_human_template_rejection_restore(tmp_path):
    db, settings, result, revision = prepared_database(tmp_path)
    accepted = discover_library(
        db,
        result.batch,
        segmentation_report=result.report,
        feature_values={},
        settings=settings,
        expected_revision=revision,
    )
    before = get_identity_state(db, "m")
    replay = update_library(
        db,
        UpdateRequest(batch=result.batch, expected_revision=accepted.revision),
        settings=settings,
        feature_values={},
    )
    assert replay.data.changed is False
    assert get_identity_state(db, "m") == before
    late = update_library(
        db,
        UpdateRequest(
            batch=EvidenceBatch(
                match_id="m",
                batch_id="late",
                producer="synthetic",
                clips=[Clip(match_id="m", clip_id="late")],
                identity_evidence=IdentityEvidence(),
            ),
            expected_revision=replay.revision,
        ),
        settings=settings,
        feature_values={},
    )
    after = get_identity_state(db, "m")
    assert after.data.players == before.data.players
    assert [a.model_dump(exclude={"revision"}) for a in after.data.assignments] == [
        a.model_dump(exclude={"revision"}) for a in before.data.assignments
    ]
    assert after.data.templates == before.data.templates
    template = after.data.templates[0]
    reviewed = apply_review(
        db,
        ReviewRequest(
            match_id="m",
            expected_revision=late.revision,
            actor="human",
            commands=[
                ReviewTemplateCommand(
                    template_id=template.template_id,
                    review_status="rejected",
                    reason="explicit sample review",
                )
            ],
        ),
        settings=settings,
    )
    current = get_identity_state(db, "m")
    rejected = next(t for t in current.data.templates if t.template_id == template.template_id)
    assert rejected.review_status == "rejected"
    assert not rejected.active
    assert rejected.admission == "automatic_experimental"
    assert rejected.approval is None
    assert current.data.assignments == after.data.assignments
    apply_review(
        db,
        ReviewRequest(
            match_id="m",
            expected_revision=reviewed.revision,
            actor="human",
            commands=[
                RestoreRevisionCommand(target_revision=accepted.revision, reason="undo rejection")
            ],
        ),
        settings=settings,
    )
    restored = get_identity_state(db, "m")
    assert all(t.active and t.review_status == "unreviewed" for t in restored.data.templates)
    assert len(get_discovery_reports(db, "m").data) == 1
    db.close()


def test_human_confirmation_preserves_auto_support_without_fabricating_template_approval(tmp_path):
    db, settings, result, revision = prepared_database(tmp_path)
    accepted = discover_library(
        db,
        result.batch,
        segmentation_report=result.report,
        feature_values={},
        settings=settings,
        expected_revision=revision,
    )
    before = get_identity_state(db, "m")
    assignment = before.data.assignments[0]
    apply_review(
        db,
        ReviewRequest(
            match_id="m",
            expected_revision=accepted.revision,
            actor="human",
            commands=[
                SetAssignmentCommand(
                    segment_id=assignment.segment_id,
                    player_id=assignment.player_id,
                    reason="independent identity review",
                )
            ],
        ),
        settings=settings,
    )
    after = get_identity_state(db, "m")
    confirmed = next(a for a in after.data.assignments if a.segment_id == assignment.segment_id)
    assert confirmed.status == "confirmed" and confirmed.method == "manual"
    assert all(t.review_status == "unreviewed" and t.approval is None for t in after.data.templates)
    assert Repository(db).review_records("m")[-1]["actor"] == "human"
    db.close()


def test_human_removal_of_witness_retires_dependent_auto_templates(tmp_path):
    db, settings, result, revision = prepared_database(tmp_path)
    accepted = discover_library(
        db,
        result.batch,
        segmentation_report=result.report,
        feature_values={},
        settings=settings,
        expected_revision=revision,
    )
    before = get_identity_state(db, "m")
    assignment = before.data.assignments[0]
    apply_review(
        db,
        ReviewRequest(
            match_id="m",
            expected_revision=accepted.revision,
            actor="human",
            commands=[
                SetAssignmentCommand(
                    segment_id=assignment.segment_id,
                    player_id=None,
                    reason="contradictory identity evidence",
                )
            ],
        ),
        settings=settings,
    )
    after = get_identity_state(db, "m")
    assert all(not t.active for t in after.data.templates)
    assert len(get_discovery_reports(db, "m").data) == 1
    assert any(
        a.segment_id == assignment.segment_id and a.player_id is None and a.method == "manual"
        for a in after.data.assignments
    )
    db.close()
