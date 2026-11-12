"""Cross-module acceptance against temporary SQLite, using synthetic source facts."""

import sqlite3

import pytest

from app.contracts.api import (
    ConfirmAssignmentCommand,
    CreatePlayerCommand,
    EventImportRequest,
    EventQuery,
    InitializeRequest,
    MatchingPreviewRequest,
    PlayerQuery,
    ReviewRequest,
    ReviewSampleCommand,
    SeedIdentity,
    SetAssignmentCommand,
    SplitSegmentCommand,
    UpdateRequest,
)
from app.contracts.events import EventOutcome, IntervalEventTime, SegmentReference
from app.contracts.evidence import EvidenceBatch, FeatureFile, FileFeatureReference, SourceFile
from app.contracts.identity import Player, Segment
from app.storage.database import connect
from app.storage.repositories import Repository, RevisionConflict, StorageConflict
from app.storage.snapshot import read_identity_snapshot
from app.workflows.ingest import preview_matching
from app.workflows.initialize import initialize_library
from app.workflows.read import get_events, get_players, get_review_queue
from app.workflows.review import apply_review
from app.workflows.update import ingest_events, update_library
from examples.workflow_v1 import synthetic_batch


@pytest.fixture
def db(tmp_path):
    connection = connect(tmp_path / "library.sqlite3")
    yield connection
    connection.close()


def seed_request(batch=None, expected_revision=0, request_id="init", seeds=True):
    batch = batch or synthetic_batch("anchor")
    name = batch.identity_evidence.segments[0].segment_id
    return InitializeRequest(
        batch=batch,
        expected_revision=expected_revision,
        request_id=request_id,
        actor="reviewer",
        reason="explicit identity anchors",
        seeds=[
            SeedIdentity(
                request_id=f"seed-{name}",
                display_name=name,
                segment_ids=[name],
                clean_sample_ids=[f"{name}-0", f"{name}-2"],
                template_sample_ids=[f"{name}-0", f"{name}-2"],
                actor="reviewer",
                reason="human selected identity and clean templates",
            )
        ]
        if seeds
        else [],
    )


def revision(db):
    return Repository(db).revision("workflow-demo")


def review(db, commands, request_id=None):
    return apply_review(
        db,
        ReviewRequest(
            match_id="workflow-demo",
            expected_revision=revision(db),
            actor="reviewer",
            request_id=request_id,
            commands=commands,
        ),
    )


def test_seed_reopen_and_business_retries(db, tmp_path):
    request = seed_request()
    initialized = initialize_library(db, request)
    player = initialized.data.created_player_ids[0]
    assert initialized.revision == 1 and initialized.data.initialized
    assert len(get_players(db, PlayerQuery(match_id="workflow-demo")).data[0].templates) == 2
    assert (
        initialize_library(db, request) == initialized
    )  # Exact retry accepts original expected_revision.
    renamed = request.model_copy(deep=True, update={"request_id": None, "expected_revision": 1})
    renamed.batch.batch_id = "another-delivery"
    repeated = initialize_library(db, renamed)
    assert repeated.revision == 1 and not repeated.data.changed
    db.close()
    reopened = connect(tmp_path / "library.sqlite3")
    try:
        profiles = get_players(reopened, PlayerQuery(match_id="workflow-demo"))
        assert profiles.data[0].player.player_id == player
        assert profiles.data[0].made_count == 1
        assert (
            len(get_events(reopened, EventQuery(match_id="workflow-demo", player_id=player)).data)
            == 1
        )
    finally:
        reopened.close()


def test_people_only_event_only_keep_ids_and_do_not_run_reid(db, monkeypatch):
    player = initialize_library(db, seed_request()).data.created_player_ids[0]
    people = synthetic_batch("next", with_event=False)
    updated = update_library(db, UpdateRequest(batch=people, expected_revision=1))
    assert updated.revision == 2 and updated.data.created_player_ids == []
    assert {
        candidate.player_id
        for a in updated.data.associations
        for candidate in [*a.candidates, *a.local_candidates]
    } == {player}
    assert all(r.assignment is None for r in updated.data.resolutions if r.segment_id == "next")
    assert len(get_players(db, PlayerQuery(match_id="workflow-demo")).data) == 1
    review(db, [ConfirmAssignmentCommand(segment_id="next", player_id=player, reason="inspected")])
    only_events = synthetic_batch("next").model_copy(
        update={"identity_evidence": None, "clips": None}
    )
    monkeypatch.setattr(
        "app.workflows.ingest.associate_segments",
        lambda *a, **k: pytest.fail("event-only ran ReID"),
    )
    result = ingest_events(db, EventImportRequest(batch=only_events, expected_revision=3))
    assert result.revision == 4
    assert len(get_events(db, EventQuery(match_id="workflow-demo", player_id=player)).data) == 2
    only_events.batch_id = "repeated"
    assert (
        ingest_events(db, EventImportRequest(batch=only_events, expected_revision=4)).revision == 4
    )


def test_without_seed_no_identity_and_later_initialize_is_allowed(db):
    batch = synthetic_batch("anchor")
    result = initialize_library(db, seed_request(batch, seeds=False))
    assert result.revision == 1 and not result.data.initialized
    assert get_players(db, PlayerQuery(match_id="workflow-demo")).data == []
    assert get_review_queue(db, "workflow-demo").data
    second = initialize_library(db, seed_request(expected_revision=1, request_id="with-seeds"))
    assert second.revision == 2 and second.data.initialized
    with pytest.raises(StorageConflict):
        initialize_library(
            db,
            seed_request(synthetic_batch("another"), expected_revision=2, request_id="other-init"),
        )
    assert revision(db) == 2


def test_event_arrives_before_identity_same_event_resolves(db):
    batch = synthetic_batch("anchor")
    early = batch.model_copy(update={"identity_evidence": None, "clips": None})
    result = ingest_events(db, EventImportRequest(batch=early, expected_revision=0))
    assert result.revision == 1
    assert (
        get_events(db, EventQuery(match_id="workflow-demo")).data[0].subjects[0].status
        == "unmapped"
    )
    initialize_library(db, seed_request(expected_revision=1))
    assert (
        get_events(db, EventQuery(match_id="workflow-demo")).data[0].subjects[0].status == "mapped"
    )
    assert len(Repository(db).events("workflow-demo")) == 1


def test_feature_only_supplement_reprocesses_saved_pending(db):
    initialize_library(db, seed_request())
    batch = synthetic_batch("late", with_event=False)
    space_id = batch.identity_evidence.feature_spaces[0].feature_space_id
    values = {}
    for index, sample in enumerate(batch.identity_evidence.samples):
        values[("late-values", index)] = sample.features[0]
        sample.features = [
            FileFeatureReference(
                feature_space_id=space_id, feature_file_id="late-values", row_index=index
            )
        ]
    batch.identity_evidence.feature_files = [
        FeatureFile(
            feature_file_id="late-values",
            feature_space_id=space_id,
            sample_count=3,
            values_file=SourceFile(file_id="values", path="values.npz"),
            index_file=SourceFile(file_id="index", path="index.jsonl"),
            visibility_file=SourceFile(file_id="vis", path="values.npz"),
        )
    ]
    first = update_library(db, UpdateRequest(batch=batch, expected_revision=1))
    diagnostic = next(a for a in first.data.associations if a.segment_id == "late")
    assert not diagnostic.local_candidates
    supplement = EvidenceBatch(
        match_id="workflow-demo", batch_id="features-only", producer="example"
    )
    second = update_library(
        db, UpdateRequest(batch=supplement, expected_revision=2), feature_values=values
    )
    assert second.revision == 3
    assert any(a.local_candidates for a in second.data.associations if a.segment_id == "late")
    assert len(read_identity_snapshot(db, "workflow-demo").feature_values) == 3
    assert (
        update_library(
            db, UpdateRequest(batch=supplement, expected_revision=3), feature_values=values
        ).revision
        == 3
    )


def test_late_segment_processes_old_samples_without_resubmission(db):
    initialize_library(db, seed_request())
    batch = synthetic_batch("late", with_event=False)
    segment = batch.identity_evidence.segments.pop()
    first = update_library(db, UpdateRequest(batch=batch, expected_revision=1))
    assert all(a.segment_id != "late" for a in first.data.associations)
    from app.contracts.evidence import IdentityEvidence

    late = EvidenceBatch(
        match_id="workflow-demo",
        batch_id="segment-only",
        producer="example",
        identity_evidence=IdentityEvidence(segments=[segment]),
    )
    second = update_library(db, UpdateRequest(batch=late, expected_revision=2))
    assert any(a.local_candidates for a in second.data.associations if a.segment_id == "late")


def test_reassignment_revokes_templates_and_moves_same_event(db):
    first = initialize_library(db, seed_request()).data.created_player_ids[0]
    review(
        db,
        [
            CreatePlayerCommand(
                player=Player(match_id="workflow-demo", player_id="pl_other"),
                reason="explicit new person",
            ),
            SetAssignmentCommand(
                segment_id="anchor", player_id="pl_other", reason="correct wrong identity"
            ),
        ],
    )
    state = read_identity_snapshot(db, "workflow-demo")
    assert not any(template.active for template in state.gallery.templates)
    assert not get_events(db, EventQuery(match_id="workflow-demo", player_id=first)).data
    assert len(get_events(db, EventQuery(match_id="workflow-demo", player_id="pl_other")).data) == 1
    # New owner's approval is independent, and an unrelated update cannot replay stale old approval.
    review(
        db,
        [
            ReviewSampleCommand(
                sample_id="anchor-0",
                player_id="pl_other",
                feature_space_id="demo-kpr",
                review_status="approved",
                reason="approve after corrected owner",
            )
        ],
    )
    updated = update_library(
        db,
        UpdateRequest(
            batch=synthetic_batch("more", with_event=False), expected_revision=revision(db)
        ),
    )
    assert updated.data.created_player_ids == []
    assert {
        t.player_id
        for t in read_identity_snapshot(db, "workflow-demo").gallery.templates
        if t.active
    } == {"pl_other"}
    assert len(Repository(db).events("workflow-demo")) == 1


def test_split_replacement_uses_ranges_and_original_samples_unchanged(db):
    player = initialize_library(db, seed_request()).data.created_player_ids[0]
    segments = [
        Segment(
            match_id="workflow-demo",
            segment_id="left",
            clip_id="anchor",
            track_id="track",
            start_s=0.0,
            end_s=1.5,
            frame_start=0,
            frame_end_exclusive=15,
            supersedes_segment_id="anchor",
        ),
        Segment(
            match_id="workflow-demo",
            segment_id="right",
            clip_id="anchor",
            track_id="track",
            start_s=1.5,
            end_s=3.0,
            frame_start=15,
            frame_end_exclusive=30,
            supersedes_segment_id="anchor",
        ),
    ]
    review(
        db,
        [
            SplitSegmentCommand(segment_id="anchor", segments=segments, reason="identity switches"),
            ConfirmAssignmentCommand(segment_id="left", player_id=player, reason="left is known"),
            CreatePlayerCommand(
                player=Player(match_id="workflow-demo", player_id="pl_right"),
                segment_ids=["right"],
                reason="right new person",
            ),
        ],
    )
    state = read_identity_snapshot(db, "workflow-demo")
    assert all(sample.segment_id == "anchor" for sample in state.gallery.evidence.samples)
    assert not any(t.active for t in state.gallery.templates)
    assert (
        get_events(db, EventQuery(match_id="workflow-demo")).data[0].subjects[0].player_id == player
    )
    event = synthetic_batch("anchor").events[0].model_copy(deep=True)
    event.event_id = event.source_event_key = "interval-support"
    event.source = event.source.model_copy(update={"source_record_id": "interval-support"})
    event.subjects[0].references = [
        SegmentReference(segment_id="anchor", time=IntervalEventTime(start_s=0.0, end_s=3.0))
    ]
    ingest_events(
        db,
        EventImportRequest(
            batch=EvidenceBatch(
                match_id="workflow-demo", batch_id="interval", producer="example", events=[event]
            ),
            expected_revision=2,
        ),
    )
    result = next(
        e
        for e in get_events(db, EventQuery(match_id="workflow-demo")).data
        if e.event.event_id == "interval-support"
    )
    assert result.subjects[0].status == "unmapped"
    # Late update reruns all original samples only through checked replacement projection.
    update_library(
        db, UpdateRequest(batch=synthetic_batch("other", with_event=False), expected_revision=3)
    )
    assert all(
        s.segment_id == "anchor"
        for s in read_identity_snapshot(db, "workflow-demo").gallery.evidence.samples
        if s.clip_id == "anchor"
    )


def test_fixed_gallery_and_human_decisions_survive_update(db):
    player = initialize_library(db, seed_request()).data.created_player_ids[0]
    state = read_identity_snapshot(db, "workflow-demo")
    ids = {t.template_id for t in state.gallery.templates if t.active}
    batch = synthetic_batch("next", with_event=False)
    report = update_library(db, UpdateRequest(batch=batch, expected_revision=1)).data
    assert all(set(a.template_ids) <= ids for a in report.associations)
    assert {a.gallery_revision for a in report.associations} == {1}
    review(
        db,
        [
            SetAssignmentCommand(
                segment_id="next",
                player_id=None,
                unassigned_reason="rejected",
                reason="other person",
            )
        ],
    )
    update_library(
        db, UpdateRequest(batch=synthetic_batch("another", with_event=False), expected_revision=3)
    )
    assignment = next(
        a
        for a in read_identity_snapshot(db, "workflow-demo").gallery.assignments
        if a.segment_id == "next"
    )
    assert assignment.method == "manual" and assignment.unassigned_reason == "rejected"
    assert (
        len(
            get_players(db, PlayerQuery(match_id="workflow-demo", player_id=player))
            .data[0]
            .templates
        )
        == 2
    )


def test_failure_rollback_stale_revision_and_preview_no_writes(db, monkeypatch):
    initialize_library(db, seed_request())
    batch = synthetic_batch("next", with_event=False)
    before = db.total_changes
    request = MatchingPreviewRequest(batch=batch, expected_revision=1)
    preview = preview_matching(db, request)
    assert preview.data.associations
    assert db.total_changes == before and revision(db) == 1
    assert not Repository(db).receipts("workflow-demo", batch.batch_id)
    with pytest.raises(RevisionConflict):
        update_library(db, UpdateRequest(batch=batch, expected_revision=0))
    original = Repository.put_template

    def fail(*args, **kwargs):
        raise sqlite3.IntegrityError("injected midway")

    monkeypatch.setattr(Repository, "put_template", fail)
    with pytest.raises(sqlite3.IntegrityError, match="injected midway"):
        update_library(db, UpdateRequest(batch=batch, expected_revision=1))
    monkeypatch.setattr(Repository, "put_template", original)
    assert revision(db) == 1
    assert not Repository(db).receipts("workflow-demo", batch.batch_id)
    assert "next" not in {
        s.segment_id for s in read_identity_snapshot(db, "workflow-demo").gallery.evidence.segments
    }


def test_matches_isolated_and_conflicting_source_rejected(db):
    player = initialize_library(db, seed_request()).data.created_player_ids[0]
    other = synthetic_batch("anchor", match_id="other-match")
    result = initialize_library(db, seed_request(other))
    assert result.data.created_player_ids[0] != player
    assert len(get_players(db, PlayerQuery(match_id="workflow-demo")).data) == 1
    bad = synthetic_batch("anchor")
    bad.events[0].outcome = EventOutcome.MISSED
    with pytest.raises(StorageConflict):
        update_library(db, UpdateRequest(batch=bad, expected_revision=1))
    assert revision(db) == 1
