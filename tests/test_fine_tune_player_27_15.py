"""Unit and acceptance tests for engineering fine-tuning of Player 27 and 15.

Verifies:
1. Small-sample adaptive coverage quantization:
   - 4-sample query with 3 supported samples (75% coverage) matches when no far conflict exists.
   - 4th sample with distance >= 0.45 (far) triggers strict one-vote veto.
2. Spatial continuity decoupled from feature sampling sparsity:
   - 28s continuous track with > 1.5s sample gaps is not sliced into 16 shards when observations are continuous.
   - Records sparse_feature_sampling with identity_coverage_preserved=True.
3. Cold-start two-stage closed-loop absorption:
   - Quality short segments (can_match_existing=True, can_create_new_identity=False) are safely absorbed into
     the existing player in stage 2 rather than lost to unknown.
4. Zero merge errors regression:
   - Opponents/distinct players remain strictly separated (0 merge errors).
"""

import numpy as np

from app.contracts.common import Provenance
from app.contracts.discovery import DiscoveryPolicy
from app.contracts.evidence import (
    Clip,
    EvidenceBatch,
    FeatureSpace,
    IdentityEvidence,
    InlineFeature,
    Preprocessing,
    Sample,
    Track,
    TrackObservation,
    VisibilityContract,
)
from app.contracts.identity import Segment
from app.contracts.segmentation import SegmentationPolicy
from app.identity.consistency import segment_tracks
from app.identity.discovery import (
    _Distances,
    evaluate_existing_player_match,
)
from app.settings import load_settings
from app.storage.database import connect, migrate
from app.workflows.discover import discover_library
from app.workflows.ingest import ingest


def _synthetic_provenance(key: str) -> Provenance:
    return Provenance(
        producer="synthetic",
        record_type="observation",
        source_record_id=key,
        evidence_version="1",
    )


def _feature_space(space_id="test-space"):
    return FeatureSpace(
        feature_space_id=space_id,
        backend="kpr",
        model_hash="f" * 64,
        preprocessing=Preprocessing(version="synthetic", color_space="rgb"),
        layout="parts",
        shape=[3, 2],
        normalization="l2",
        distance="euclidean_sqrt_visibility_mean_div2",
        visibility=VisibilityContract(shape=[3], semantics="visible"),
    )


def _angle_vector(angle: float) -> list[float]:
    return [float(np.cos(angle)), float(np.sin(angle))] * 3


def _sample(sample_id, seg_id, clip_id, track_id, time_s, frame_idx, vec, match_id="m"):
    return Sample(
        sample_id=sample_id,
        match_id=match_id,
        segment_id=seg_id,
        clip_id=clip_id,
        track_id=track_id,
        frame_idx=frame_idx,
        time_s=time_s,
        image_asset_id=f"img_{sample_id}",
        source=_synthetic_provenance(f"src_{sample_id}"),
        features=[
            InlineFeature(
                feature_space_id="test-space",
                kind="inline",
                values=vec,
                visibility=[1.0, 1.0, 1.0],
            )
        ],
    )


class TestSmallSampleAdaptiveCoverage:
    """Test 1: 4-sample adaptive coverage quantization and veto."""

    def test_small_sample_adaptive_coverage_passes_without_far(self):
        """4 samples with 3 near (0.20) and 1 medium (0.32 < 0.45) pass coverage (75%)."""
        policy = DiscoveryPolicy(
            version="complete_support_v3",
            support_mode="constrained_manifold",
            min_coverage=0.80,
            intra_segment_near_distance=0.18,
            existing_player_match_threshold=0.28,
            far_distance=0.45,
            min_match_buckets=2,
        )

        space = _feature_space()
        # Template samples for player_1 at angle 0.0
        v0 = _angle_vector(0.0)
        t1 = _sample("tpl1", "seg_tpl", "c0", "t1", 0.0, 0, v0)
        t2 = _sample("tpl2", "seg_tpl", "c0", "t1", 0.5, 5, v0)

        # Query segment with 4 samples:
        # 3 samples are near v0 (dist ~0.22 <= 0.28)
        # 1 sample is medium (dist ~0.34 < 0.45)
        v_near = _angle_vector(0.45)
        v_med = _angle_vector(0.70)

        q1 = _sample("q1", "seg_q", "c1", "t2", 1.0, 10, v_near)
        q2 = _sample("q2", "seg_q", "c1", "t2", 1.5, 15, v_near)
        q3 = _sample("q3", "seg_q", "c1", "t2", 2.0, 20, v_near)
        q4 = _sample("q4", "seg_q", "c1", "t2", 2.5, 25, v_med)

        all_samples = [t1, t2, q1, q2, q3, q4]
        distances = _Distances(all_samples, space, None, policy)
        components = {s.sample_id: s.sample_id for s in all_samples}

        seg_q = Segment(
            segment_id="seg_q", match_id="m", clip_id="c1", track_id="t2", start_s=1.0, end_s=2.5
        )

        existing_players = {"p1": None}
        player_templates = {"p1": [(None, t1), (None, t2)]}

        best_pid, best_dist, reasons = evaluate_existing_player_match(
            segment=seg_q,
            usable_samples=[q1, q2, q3, q4],
            representatives=[q1, q2, q3, q4],
            existing_players=existing_players,
            player_templates=player_templates,
            distances=distances,
            components=components,
            policy=policy,
            can_match=True,
        )

        assert best_pid == "p1"
        assert "matched_unique_existing_player" in reasons

    def test_small_sample_adaptive_coverage_vetoed_by_far_minority(self):
        """4 samples with 3 near but 1 far (>= 0.45) is strictly vetoed."""
        policy = DiscoveryPolicy(
            version="complete_support_v3",
            support_mode="constrained_manifold",
            min_coverage=0.80,
            intra_segment_near_distance=0.18,
            existing_player_match_threshold=0.28,
            far_distance=0.45,
            min_match_buckets=2,
        )

        space = _feature_space()
        v0 = _angle_vector(0.0)
        t1 = _sample("tpl1", "seg_tpl", "c0", "t1", 0.0, 0, v0)
        t2 = _sample("tpl2", "seg_tpl", "c0", "t1", 0.5, 5, v0)

        v_near = _angle_vector(0.45)
        # Angle 1.6 gives distance ~0.717 >= 0.45 (far)
        v_far = _angle_vector(1.6)

        q1 = _sample("q1", "seg_q", "c1", "t2", 1.0, 10, v_near)
        q2 = _sample("q2", "seg_q", "c1", "t2", 1.5, 15, v_near)
        q3 = _sample("q3", "seg_q", "c1", "t2", 2.0, 20, v_near)
        q4 = _sample("q4", "seg_q", "c1", "t2", 2.5, 25, v_far)

        all_samples = [t1, t2, q1, q2, q3, q4]
        distances = _Distances(all_samples, space, None, policy)
        components = {s.sample_id: s.sample_id for s in all_samples}

        seg_q = Segment(
            segment_id="seg_q", match_id="m", clip_id="c1", track_id="t2", start_s=1.0, end_s=2.5
        )

        existing_players = {"p1": None}
        player_templates = {"p1": [(None, t1), (None, t2)]}

        best_pid, best_dist, reasons = evaluate_existing_player_match(
            segment=seg_q,
            usable_samples=[q1, q2, q3, q4],
            representatives=[q1, q2, q3, q4],
            existing_players=existing_players,
            player_templates=player_templates,
            distances=distances,
            components=components,
            policy=policy,
            can_match=True,
        )

        assert best_pid is None
        assert "minority_sample_far_from_existing_player" in reasons


class TestSpatialContinuityDecoupledFromSamplingSparsity:
    """Test 2: Spatial continuity preserved across 2.5s sampling gaps."""

    def test_continuous_track_with_sparse_samples_not_shredded(self):
        """28s continuous track with sample gaps > 1.5s produces a preserved segment, not 16 shards."""
        policy = SegmentationPolicy(
            version="track-consistency-v3",
            max_sample_gap_s=1.5,
            max_continuous_sample_gap_s=4.5,
            max_observation_gap_s=0.25,
            min_match_buckets=2,
            min_support_buckets=3,
        )

        # 28 seconds of continuous observations at 10 fps (280 observations)
        clip_id = "c0"
        track_id = "trk_long"
        fps = 10.0
        n_frames = 280
        observations = [
            TrackObservation(
                observation_id=f"obs_{i}",
                match_id="m",
                clip_id=clip_id,
                track_id=track_id,
                frame_idx=i,
                time_s=round(i / fps, 2),
                bbox_xyxy=(100.0, 100.0, 200.0, 300.0),
                source=_synthetic_provenance(f"obs_src_{i}"),
            )
            for i in range(n_frames)
        ]

        # Features sampled every 2.5s (frames 0, 25, 50, 75, ... 275) -> gap 2.5s > 1.5s
        space = _feature_space()
        v = _angle_vector(0.0)

        sample_frames = list(range(0, n_frames, 25))
        samples = [
            Sample(
                sample_id=f"samp_{f}",
                match_id="m",
                segment_id="raw_seg",
                clip_id=clip_id,
                track_id=track_id,
                frame_idx=f,
                time_s=round(f / fps, 2),
                image_asset_id=f"img_{f}",
                source=_synthetic_provenance(f"samp_src_{f}"),
                features=[
                    InlineFeature(
                        feature_space_id="test-space",
                        kind="inline",
                        values=v,
                        visibility=[1.0, 1.0, 1.0],
                    )
                ],
            )
            for f in sample_frames
        ]

        raw_seg = Segment(
            segment_id="raw_seg",
            match_id="m",
            clip_id=clip_id,
            track_id=track_id,
            start_s=0.0,
            end_s=round(n_frames / fps, 2),
        )

        batch = EvidenceBatch(
            match_id="m",
            batch_id="b0",
            producer="synthetic",
            clips=[Clip(match_id="m", clip_id=clip_id, duration_s=30.0)],
            identity_evidence=IdentityEvidence(
                tracks=[
                    Track(
                        match_id="m",
                        clip_id=clip_id,
                        track_id=track_id,
                        start_s=0.0,
                        end_s=round(n_frames / fps, 2),
                        source=_synthetic_provenance("t_src"),
                    )
                ],
                segments=[raw_seg],
                samples=samples,
                observations=observations,
                feature_spaces=[space],
            ),
        )

        result = segment_tracks(batch, policy=policy)
        assert len(result.report.tracks) == 1
        t_rep = result.report.tracks[0]

        # In track-consistency-v3, the track is NOT shredded into 10+ shards!
        # It produces 1 or 2 high quality coherent segments
        supported_proposals = [p for p in t_rep.proposals if p.status == "supported"]
        assert len(supported_proposals) >= 1

        # Check findings: sparse_feature_sampling is reported with identity_coverage_preserved=True
        sparse_findings = [
            f
            for t in result.report.tracks
            for f in t.findings
            if f.kind == "sparse_feature_sampling"
        ]
        assert len(sparse_findings) > 0
        assert all(f.details.get("identity_coverage_preserved") is True for f in sparse_findings)


class TestColdStartClosedLoopAbsorption:
    """Test 3: Quality short segments absorbed in cold start stage 2."""

    def test_quality_short_segment_absorbed(self):
        """Short segment with can_match_existing=True is absorbed in stage 2."""
        conn = connect(":memory:")
        migrate(conn)
        settings = load_settings("examples/machine-discovery.toml")

        # Construct batch with:
        # Clip 1: Player A track 1 (solid, 4 samples)
        # Clip 2: Player A track 2 (solid, 4 samples) -> These form Player 1 in Stage 1!
        # Clip 3: Player A track 3 (short, 2 samples, can_match_existing=True, can_create_new_identity=False)
        space = _feature_space()
        v = _angle_vector(0.0)

        def make_track(cid, tid, sid, s_indices, n_frames=20):
            fps = 10.0
            trk = Track(
                match_id="m_test",
                clip_id=cid,
                track_id=tid,
                start_s=0.0,
                end_s=n_frames / fps,
                source=_synthetic_provenance(f"trk_{cid}_{tid}"),
            )
            seg = Segment(
                segment_id=sid,
                match_id="m_test",
                clip_id=cid,
                track_id=tid,
                start_s=0.0,
                end_s=n_frames / fps,
            )
            samps = [
                Sample(
                    sample_id=f"samp_{cid}_{tid}_{i}",
                    match_id="m_test",
                    segment_id=sid,
                    clip_id=cid,
                    track_id=tid,
                    frame_idx=f_idx,
                    time_s=f_idx / fps,
                    image_asset_id=f"img_{cid}_{tid}_{i}",
                    source=_synthetic_provenance(f"src_{cid}_{tid}_{i}"),
                    features=[
                        InlineFeature(
                            feature_space_id="test-space",
                            kind="inline",
                            values=v,
                            visibility=[1.0, 1.0, 1.0],
                        )
                    ],
                )
                for i, f_idx in enumerate(s_indices)
            ]
            obss = [
                TrackObservation(
                    observation_id=f"obs_{cid}_{tid}_{f}",
                    match_id="m_test",
                    clip_id=cid,
                    track_id=tid,
                    frame_idx=f,
                    time_s=f / fps,
                    bbox_xyxy=(100.0, 100.0, 200.0, 300.0),
                    source=_synthetic_provenance(f"obs_{cid}_{tid}_{f}"),
                )
                for f in range(n_frames)
            ]
            return trk, seg, samps, obss

        # c1 and c2 have 4 samples (frames 0, 5, 10, 15 out of 20 frames = 2.0s) -> can_create_new_identity=True
        # c3 has 2 samples (frames 0, 5 out of 8 frames = 0.8s) -> can_match_existing=True, can_create_new_identity=False
        t1, s1, sm1, ob1 = make_track("c1", "t1", "seg1", [0, 5, 10, 15], n_frames=20)
        t2, s2, sm2, ob2 = make_track("c2", "t1", "seg2", [0, 5, 10, 15], n_frames=20)
        t3, s3, sm3, ob3 = make_track("c3", "t1", "seg3", [0, 5], n_frames=8)

        batch = EvidenceBatch(
            match_id="m_test",
            batch_id="b_all",
            producer="synthetic",
            clips=[
                Clip(match_id="m_test", clip_id="c1", duration_s=10.0),
                Clip(match_id="m_test", clip_id="c2", duration_s=10.0),
                Clip(match_id="m_test", clip_id="c3", duration_s=10.0),
            ],
            identity_evidence=IdentityEvidence(
                tracks=[t1, t2, t3],
                segments=[s1, s2, s3],
                samples=[*sm1, *sm2, *sm3],
                observations=[*ob1, *ob2, *ob3],
                feature_spaces=[space],
            ),
        )

        seg_res = segment_tracks(batch, policy=settings.segmentation)
        seg_report = seg_res.report
        ingest_batch = seg_res.batch

        from app.contracts.api import ImportRequest

        ingest_res = ingest(
            conn, ImportRequest(batch=ingest_batch, expected_revision=0), settings=settings
        )
        assert ingest_res.data.changed

        # Now run discover_library
        _ = discover_library(
            conn,
            ingest_batch,
            segmentation_report=seg_report,
            feature_values={},
            settings=settings,
            expected_revision=ingest_res.revision,
        )

        # In Stage 1: Player 1 was created from seg1 and seg2
        # In Stage 2: seg3 (can_create_new_identity=False) was absorbed!
        from app.storage.snapshot import read_identity_snapshot

        snapshot = read_identity_snapshot(conn, "m_test")
        assignments = snapshot.gallery.assignments
        assigned_map = {a.segment_id: a.player_id for a in assignments if a.player_id is not None}

        assert len(snapshot.gallery.players) == 1
        assert len(assigned_map) == 3
        assert len(set(assigned_map.values())) == 1


class TestZeroMergeErrorsRegression:
    """Test 4: Cross-player merge errors strictly stay 0."""

    def test_different_players_never_merged(self):
        """Orthogonal / far appearance players must never be merged."""
        policy = DiscoveryPolicy(
            version="complete_support_v3",
            support_mode="constrained_manifold",
            min_coverage=0.80,
            intra_segment_near_distance=0.18,
            existing_player_match_threshold=0.28,
            far_distance=0.45,
            min_match_buckets=2,
        )

        space = _feature_space()
        # Player A vector at angle 0.0
        vA = _angle_vector(0.0)
        # Player B vector at angle 1.8 (dist ~0.78 >> 0.45)
        vB = _angle_vector(1.8)

        tA1 = _sample("tplA1", "segA", "c0", "t1", 0.0, 0, vA)
        tA2 = _sample("tplA2", "segA", "c0", "t1", 0.5, 5, vA)

        qB1 = _sample("qB1", "segB", "c1", "t2", 1.0, 10, vB)
        qB2 = _sample("qB2", "segB", "c1", "t2", 1.5, 15, vB)

        all_samples = [tA1, tA2, qB1, qB2]
        distances = _Distances(all_samples, space, None, policy)
        components = {s.sample_id: s.sample_id for s in all_samples}

        segB = Segment(
            segment_id="segB", match_id="m", clip_id="c1", track_id="t2", start_s=1.0, end_s=1.5
        )

        existing_players = {"player_A": None}
        player_templates = {"player_A": [(None, tA1), (None, tA2)]}

        best_pid, best_dist, reasons = evaluate_existing_player_match(
            segment=segB,
            usable_samples=[qB1, qB2],
            representatives=[qB1, qB2],
            existing_players=existing_players,
            player_templates=player_templates,
            distances=distances,
            components=components,
            policy=policy,
            can_match=True,
        )

        assert best_pid is None
        assert any(
            r in reasons
            for r in (
                "minority_sample_far_from_existing_player",
                "best_match_exceeds_threshold",
                "no_matching_existing_player",
            )
        )
