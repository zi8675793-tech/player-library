"""A clean approval authorizes one immutable source version only."""

from app.contracts.api import InitializeRequest, PlayerQuery, SeedIdentity, UpdateRequest
from app.contracts.common import SourceKey
from app.contracts.evidence import EvidenceBatch, IdentityEvidence
from app.storage.database import connect, transaction
from app.storage.repositories import Repository
from app.storage.snapshot import read_identity_snapshot
from app.workflows._review_ops import current_evidence
from app.workflows.initialize import initialize_library
from app.workflows.read import get_players
from app.workflows.update import update_library
from examples.workflow_v1 import synthetic_batch


def projected(db):
    return current_evidence(
        read_identity_snapshot(db, "workflow-demo").gallery.evidence,
        clean_reviews=Repository(db).clean_reviews("workflow-demo"),
    ).samples[0]


def test_superseded_seed_sample_requires_new_clean_approval_without_losing_identity(tmp_path):
    db = connect(tmp_path / "clean.sqlite")
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
                    reason="v1 clean",
                )
            ],
        ),
    )
    assert projected(db).tracking_quality.clean_review == "approved"
    original_player = get_players(db, PlayerQuery(match_id="workflow-demo")).data[0]
    sample = batch.identity_evidence.samples[0].model_copy(deep=True)
    old = SourceKey.model_validate(sample.source.model_dump(include=set(SourceKey.model_fields)))
    sample.source = sample.source.model_copy(update={"evidence_version": "2", "supersedes": old})
    sample.features[0].values = [0.0, 1.0] * 3
    sample.tracking_quality = None
    update_library(
        db,
        UpdateRequest(
            expected_revision=1,
            batch=EvidenceBatch(
                match_id="workflow-demo",
                batch_id="v2",
                producer="regression",
                identity_evidence=IdentityEvidence(samples=[sample]),
            ),
        ),
    )
    assert projected(db).source.evidence_version == "2"
    assert projected(db).tracking_quality.clean_review == "unreviewed"
    player = get_players(db, PlayerQuery(match_id="workflow-demo")).data[0]
    assert player.player.player_id == original_player.player.player_id
    assert player.confirmed_segment_ids == original_player.confirmed_segment_ids
    with transaction(db):
        repo = Repository(db)
        repo.put_clean_review(
            "workflow-demo",
            "a-0",
            {
                "actor": "human",
                "reason": "v2 independently inspected",
                "review_status": "approved",
                "revision": 3,
            },
        )
        repo.advance_revision("workflow-demo", 2, actor="human", reason="clean v2", changes={})
    assert projected(db).tracking_quality.clean_review == "approved"
    assert db.execute("SELECT COUNT(*) FROM sample_clean_reviews").fetchone()[0] == 2
