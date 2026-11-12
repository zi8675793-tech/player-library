"""Business and regression tests for Stage 6:
- Role boundary decoupling (player, referee, staff, unknown_role)
- Anonymous profiles vs formal player statistics
- Review Queue contract enrichment (video 3 duplicate box one-click review)
- Prediction vs independent scoring isolation firewall
- Stage 6 9-metrics benchmark reporting
"""

import pytest

from app.contracts.api import (
    ImportRequest,
    PlayerQuery,
    ReviewRequest,
    ReviewRiskCommand,
    SetPlayerRoleCommand,
)
from app.contracts.common import Provenance
from app.contracts.evidence import (
    Clip,
    EvidenceBatch,
    IdentityEvidence,
    Sample,
    Track,
    TrackObservation,
)
from app.contracts.identity import (
    IdentityRisk,
    PersonRole,
    Player,
    Segment,
)
from app.evaluation.guard import (
    GroundTruthLeakageError,
    PrematureScoringError,
    assert_frozen_before_scoring,
    assert_no_ground_truth_in_evidence,
)
from app.evaluation.metrics import Stage6BenchmarkMetrics
from app.storage.database import connect
from app.storage.repositories import Repository
from app.workflows.ingest import ingest
from app.workflows.read import get_formal_player_stats, get_players, get_review_queue
from app.workflows.review import apply_review


def _prov(key: str) -> Provenance:
    return Provenance(
        producer="synthetic_stage6",
        record_type="test",
        source_record_id=key,
        evidence_version="1",
    )


def test_person_role_defaults_and_decoupling():
    """Player default role is unknown_role (anonymous identity profile)."""
    p = Player(player_id="pl_anon", match_id="m1")
    assert p.role == PersonRole.UNKNOWN_ROLE
    assert p.role == "unknown_role"

    # Can be explicitly created as referee or staff
    p_ref = Player(player_id="pl_ref", match_id="m1", role=PersonRole.REFEREE)
    assert p_ref.role == PersonRole.REFEREE

    p_player = Player(
        player_id="pl_star", match_id="m1", role=PersonRole.PLAYER, jersey_number="23"
    )
    assert p_player.role == PersonRole.PLAYER


def test_role_revision_without_rebuilding_identity(tmp_path):
    """Setting player role does not require rebuilding player or altering assignments."""
    db_path = tmp_path / "test_role.sqlite3"
    conn = connect(db_path)

    repo = Repository(conn)
    from app.storage.database import transaction

    with transaction(conn):
        repo.ensure_match("m_role")
        # Insert anonymous profile
        p = Player(player_id="pl_1", match_id="m_role", role=PersonRole.UNKNOWN_ROLE)
        repo.put_player(p)
        # Advance rev 0 -> 1
        repo.advance_revision("m_role", 0, actor="test", reason="init", changes={})

    # Read profiles before role confirmation
    views_r1 = get_players(conn, PlayerQuery(match_id="m_role")).data
    assert len(views_r1) == 1
    assert views_r1[0].player.role == PersonRole.UNKNOWN_ROLE
    assert not views_r1[0].is_formal_player
    assert views_r1[0].profile_type == "anonymous_identity_profile"

    # Formal stats must be empty for unknown role
    stats_r1 = get_formal_player_stats(conn, "m_role").data
    assert len(stats_r1) == 0

    # Apply SetPlayerRoleCommand: elevate to formal player
    req = ReviewRequest(
        match_id="m_role",
        expected_revision=1,
        actor="referee_judge",
        commands=[
            SetPlayerRoleCommand(
                reason="confirmed_jersey_number",
                player_id="pl_1",
                role=PersonRole.PLAYER,
            )
        ],
    )
    res = apply_review(conn, req)
    assert res.revision == 2

    # Check updated views
    views_r2 = get_players(conn, PlayerQuery(match_id="m_role")).data
    assert len(views_r2) == 1
    assert views_r2[0].player.role == PersonRole.PLAYER
    assert views_r2[0].is_formal_player
    assert views_r2[0].profile_type == "formal_player"

    # Filter with formal_players_only=True
    formal_views = get_players(conn, PlayerQuery(match_id="m_role", formal_players_only=True)).data
    assert len(formal_views) == 1

    # Now formal stats includes pl_1
    stats_r2 = get_formal_player_stats(conn, "m_role").data
    assert len(stats_r2) == 1
    assert stats_r2[0].player_id == "pl_1"
    assert stats_r2[0].role == PersonRole.PLAYER
    assert stats_r2[0].is_formal_player


def test_review_queue_enrichment_and_duplicate_box(tmp_path):
    """Video 3 duplicate box track generates critical review item with one-click actions."""
    db_path = tmp_path / "test_rq.sqlite3"
    conn = connect(db_path)

    # Ingest synthetic batch mirroring clip 3: track 6 has duplicate_box
    seg = Segment(
        segment_id="seg_c3_t6",
        match_id="m_video3",
        clip_id="3",
        track_id="6",
        start_s=0.0,
        end_s=12.48,
    )
    track6 = Track(
        match_id="m_video3",
        clip_id="3",
        track_id="6",
        start_s=0.0,
        end_s=12.48,
        source=_prov("trk6"),
    )
    track1 = Track(
        match_id="m_video3",
        clip_id="3",
        track_id="1",
        start_s=0.0,
        end_s=12.86,
        source=_prov("trk1"),
    )
    obs = TrackObservation(
        observation_id="obs_c3_t6_0",
        match_id="m_video3",
        clip_id="3",
        track_id="6",
        time_s=0.0,
        frame_idx=0,
        bbox_xyxy=(100.0, 100.0, 200.0, 300.0),
        source=_prov("obs6"),
    )
    sample = Sample(
        sample_id="s_c3_t6_0",
        match_id="m_video3",
        clip_id="3",
        track_id="6",
        time_s=0.0,
        frame_idx=0,
        segment_id="seg_c3_t6",
        image_asset_id="asset_img_t6",
        source=_prov("smp6"),
    )
    dup_risk = IdentityRisk(
        risk_id="risk_dup_t6",
        kind="duplicate_box",
        match_id="m_video3",
        segment_id="seg_c3_t6",
        clip_id="3",
        track_id="6",
        start_s=0.0,
        end_s=12.48,
        source=_prov("rsk6"),
    )

    batch = EvidenceBatch(
        match_id="m_video3",
        batch_id="b1",
        producer="synthetic",
        clips=[Clip(match_id="m_video3", clip_id="3", duration_s=15.0)],
        identity_evidence=IdentityEvidence(
            tracks=[track1, track6],
            segments=[seg],
            observations=[obs],
            samples=[sample],
            risks=[dup_risk],
        ),
    )

    ingest_res = ingest(conn, ImportRequest(batch=batch, expected_revision=0))
    assert ingest_res.revision == 1

    # Read review queue
    rq = get_review_queue(conn, "m_video3").data
    assert len(rq) >= 1

    # Find the risk review item for duplicate_box
    risk_item = next(item for item in rq if item.kind == "risk" and "duplicate_box" in item.reasons)
    assert risk_item.clip_id == "3"
    assert risk_item.track_id == "6"
    assert risk_item.start_s == 0.0
    assert risk_item.end_s == 12.48
    assert risk_item.review_priority == "critical"
    assert "asset_img_t6" in risk_item.thumbnail_asset_ids
    assert "1" in risk_item.details["overlapping_tracks"]

    # Verify suggested actions contract
    action_kinds = [a.action for a in risk_item.suggested_actions]
    assert "review_risk" in action_kinds
    assert "split_segment" in action_kinds
    assert "set_assignment" in action_kinds

    # Test one-click execution: reviewer executes ReviewRiskCommand to dismiss duplicate_box
    review_res = apply_review(
        conn,
        ReviewRequest(
            match_id="m_video3",
            expected_revision=1,
            actor="expert_reviewer",
            commands=[
                ReviewRiskCommand(
                    reason="verified_not_duplicate",
                    risk_id="risk_dup_t6",
                    segment_id="seg_c3_t6",
                    resolution="cleared",
                )
            ],
        ),
    )
    assert review_res.revision == 2

    # Verify risk is no longer active in review queue
    rq_after = get_review_queue(conn, "m_video3").data
    active_risks = [
        item for item in rq_after if item.kind == "risk" and item.segment_id == "seg_c3_t6"
    ]
    assert len(active_risks) == 0


def test_inference_isolation_guard():
    """Ground truth isolation guard catches leakage and prevents premature scoring."""
    # 1. Evidence source with annotations.csv is blocked
    with pytest.raises(GroundTruthLeakageError, match="Forbidden ground truth source"):
        assert_no_ground_truth_in_evidence(
            ["/workspace/incoming/track/1/tracks.csv", "../annotations.csv"]
        )

    # 2. Evidence source with target directory is blocked
    with pytest.raises(GroundTruthLeakageError, match="Forbidden ground truth source"):
        assert_no_ground_truth_in_evidence(["test/target/feature_contract.json"])

    # 3. Valid machine sources pass cleanly
    assert_no_ground_truth_in_evidence(
        ["track/3/tracks.csv", "identity/3/samples.jsonl", "video/3.mp4"]
    )

    # 4. Premature scoring check on unfrozen directory raises PrematureScoringError
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp_dir:
        with pytest.raises(PrematureScoringError, match="artifact-hashes.json missing"):
            assert_frozen_before_scoring(Path(tmp_dir))


def test_stage6_benchmark_metrics_contract():
    """Stage6BenchmarkMetrics contract supports all 9 defined metrics."""
    metrics = Stage6BenchmarkMetrics(
        correct_coverage_samples=1.0,
        correct_coverage_observations=1.0,
        correct_coverage_duration=1.0,
        merge_error_rate=0.0,
        merge_error_count=0,
        split_error_rate=0.0,
        split_error_count=0,
        assigned_purity=1.0,
        cross_clip_pairwise_precision=1.0,
        cross_clip_pairwise_recall=0.0556,
        cross_clip_pairwise_f1=0.1053,
        cross_clip_pairwise_tp=148,
        cross_clip_pairwise_fp=0,
        cross_clip_pairwise_fn=2512,
        deferred_rate=0.743,
        deferred_count=137,
        total_candidate_segments=158,
        incremental_duplicate_rate=0.0,
        incremental_duplicate_count=0,
        order_stability_ari=0.9377,
        order_stability_agreement=0.9995,
        non_player_pollution_rate=0.0,
        non_player_pollution_status="zero_pollution",
    )
    dumped = metrics.model_dump()
    assert dumped["merge_error_count"] == 0
    assert dumped["cross_clip_pairwise_precision"] == 1.0
    assert dumped["order_stability_ari"] == 0.9377
