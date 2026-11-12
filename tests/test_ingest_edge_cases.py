"""Edge cases for initialization retries and persisted pending evidence."""

import pytest
from test_workflow_v1 import revision, seed_request

from app.contracts.api import (
    InitializeRequest,
    ReviewRequest,
    SeedIdentity,
    SplitSegmentCommand,
    UpdateRequest,
)
from app.contracts.evidence import EvidenceBatch, FeatureFile, FileFeatureReference, SourceFile
from app.contracts.identity import Segment
from app.storage.database import connect
from app.storage.repositories import StorageConflict
from app.storage.snapshot import read_identity_snapshot
from app.workflows.initialize import initialize_library
from app.workflows.review import apply_review
from app.workflows.update import update_library
from examples.workflow_v1 import synthetic_batch


@pytest.fixture
def db(tmp_path):
    connection = connect(tmp_path / "library.sqlite3")
    yield connection
    connection.close()


def file_features(batch):
    values = {}
    evidence = batch.identity_evidence
    for index, sample in enumerate(evidence.samples):
        values[("features", index)] = sample.features[0]
        sample.features = [
            FileFeatureReference(
                feature_space_id="demo-kpr", feature_file_id="features", row_index=index
            )
        ]
    evidence.feature_files = [
        FeatureFile(
            feature_file_id="features",
            feature_space_id="demo-kpr",
            sample_count=len(evidence.samples),
            values_file=SourceFile(file_id="values", path="features.npz"),
            visibility_file=SourceFile(file_id="values", path="features.npz"),
            index_file=SourceFile(file_id="index", path="index.jsonl"),
        )
    ]
    return values


def test_initialize_retry_cannot_hide_conflicting_parsed_feature_payload(db):
    batch = synthetic_batch("anchor")
    values = file_features(batch)
    request = seed_request(batch)
    first = initialize_library(db, request, feature_values=values)
    changed = {key: value.model_copy(deep=True) for key, value in values.items()}
    changed[("features", 0)].values = [0.0, 1.0] * 3
    with pytest.raises(StorageConflict, match="request|feature|content"):
        initialize_library(db, request, feature_values=changed)
    assert revision(db) == first.revision
    assert read_identity_snapshot(db, "workflow-demo").feature_values == values


def test_initialize_can_use_checked_descendant_sample_after_preidentity_split(db):
    batch = synthetic_batch("anchor", with_event=False)
    update_library(db, UpdateRequest(batch=batch, expected_revision=0))
    replacements = [
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
    apply_review(
        db,
        ReviewRequest(
            match_id="workflow-demo",
            expected_revision=1,
            actor="reviewer",
            commands=[
                SplitSegmentCommand(
                    segment_id="anchor", segments=replacements, reason="separate known ranges"
                )
            ],
        ),
    )
    request = InitializeRequest(
        batch=EvidenceBatch(match_id="workflow-demo", batch_id="seed-only", producer="test"),
        expected_revision=2,
        request_id="seed-after-split",
        actor="reviewer",
        reason="establish first anchor after split",
        seeds=[
            SeedIdentity(
                request_id="left-person",
                segment_ids=["left"],
                clean_sample_ids=["anchor-0"],
                template_sample_ids=["anchor-0"],
                actor="reviewer",
                reason="range checked clean anchor",
            )
        ],
    )
    result = initialize_library(db, request)
    assert result.data.initialized and len(result.data.created_player_ids) == 1
    snapshot = read_identity_snapshot(db, "workflow-demo")
    assert any(template.active for template in snapshot.gallery.templates)
    assert all(sample.segment_id == "anchor" for sample in snapshot.gallery.evidence.samples)
