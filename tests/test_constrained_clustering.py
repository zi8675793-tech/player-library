"""Unit tests for constrained hierarchical clustering with physical hard constraints.

Verifies:
1. Viewpoint transfer (continuous manifold chaining):
   - A is similar to B (0.20), B is similar to C (0.21), but A and C are farther (0.32).
   - Under average linkage, transitive merge succeeds without suicidal bridge execution.
2. Hard spatio-temporal mutual exclusion:
   - Tracks overlapping in time within the same clip (co-occurring on screen) can NEVER
     be merged, even with identical features (distance 0.01).
3. Transitive barrier defense:
   - Tracks conflicting via hard constraints cannot be chained together through a third track.
4. Output partition:
   - Multi-clip clusters (len(clip_ids) >= 2) marked as is_core_player=True, cluster_type="core".
   - Single-clip intermittent clusters marked as is_core_player=False, cluster_type="single_clip".
   - Unmerged single tracks marked as cluster_type="unassigned".
5. Linkage criteria comparison and edge cases.
"""

import numpy as np
import pytest

from app.identity.clustering import (
    DiscoveredCluster,
    compute_cluster_distance,
    constrained_hierarchical_clustering,
)
from app.identity.constraints import SpatioTemporalConstraintEngine
from app.identity.representation import TrackRepresentation, compute_track_distance


def _make_track(
    clip_id: str,
    track_id: int,
    start_s: float,
    end_s: float,
    feature_vec: np.ndarray,
    parts: int = 6,
    vis_val: float = 0.9,
) -> TrackRepresentation:
    feature_mat = np.tile(feature_vec, (parts, 1)).astype(np.float32)
    vis_arr = np.ones((parts,), dtype=np.float32) * vis_val
    return TrackRepresentation(
        clip_id=clip_id,
        track_id=track_id,
        start_s=start_s,
        end_s=end_s,
        feature=feature_mat,
        visibility=vis_arr,
        sample_ids=[f"s_{clip_id}_{track_id}"],
    )


def _create_viewpoint_features() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create three 128D unit vectors with pairwise distances exactly 0.20, 0.21, 0.32."""
    d_ab, d_bc, d_ac = 0.20, 0.21, 0.32
    th_ab = 2.0 * float(np.arcsin(d_ab))
    th_bc = 2.0 * float(np.arcsin(d_bc))
    th_ac = 2.0 * float(np.arcsin(d_ac))

    cos_ab, sin_ab = float(np.cos(th_ab)), float(np.sin(th_ab))
    cos_bc, sin_bc = float(np.cos(th_bc)), float(np.sin(th_bc))
    cos_ac = float(np.cos(th_ac))

    cos_phi = (cos_ac - cos_ab * cos_bc) / (sin_ab * sin_bc)
    sin_phi = float(np.sqrt(np.clip(1.0 - cos_phi**2, 0.0, 1.0)))

    f_b = np.zeros(128, dtype=np.float32)
    f_b[0] = 1.0

    f_a = np.zeros(128, dtype=np.float32)
    f_a[0] = cos_ab
    f_a[1] = sin_ab

    f_c = np.zeros(128, dtype=np.float32)
    f_c[0] = cos_bc
    f_c[1] = sin_bc * cos_phi
    f_c[2] = sin_bc * sin_phi

    return f_a, f_b, f_c


class TestViewpointTransfer:
    """Acceptance test 1: Viewpoint transfer manifold chaining replaces suicidal bridge execution."""

    def test_viewpoint_transfer_merges_across_clips(self):
        f_a, f_b, f_c = _create_viewpoint_features()

        track_a = _make_track("clip_1", 1, 0.0, 5.0, f_a)
        track_b = _make_track("clip_2", 2, 0.0, 5.0, f_b)
        track_c = _make_track("clip_3", 3, 0.0, 5.0, f_c)

        # Verify underlying feature distances
        d_ab = compute_track_distance(track_a, track_b)
        d_bc = compute_track_distance(track_b, track_c)
        d_ac = compute_track_distance(track_a, track_c)

        assert d_ab is not None and np.isclose(d_ab, 0.20, atol=1e-4)
        assert d_bc is not None and np.isclose(d_bc, 0.21, atol=1e-4)
        assert d_ac is not None and np.isclose(d_ac, 0.32, atol=1e-4)

        tracks = [track_a, track_b, track_c]
        engine = SpatioTemporalConstraintEngine(tracks=tracks)

        # No spatio-temporal conflict across different clips
        assert engine.is_forbidden(0, 1) is False
        assert engine.is_forbidden(1, 2) is False
        assert engine.is_forbidden(0, 2) is False

        # Run constrained hierarchical clustering with average linkage
        # A-B (0.20) merges first; then {A, B} and C has average distance (0.21 + 0.32)/2 = 0.265 <= 0.28
        clusters = constrained_hierarchical_clustering(
            tracks=tracks,
            constraints=engine,
            threshold=0.28,
            linkage="average",
        )

        assert len(clusters) == 1
        core_cluster = clusters[0]
        assert isinstance(core_cluster, DiscoveredCluster)
        assert set(core_cluster.track_indices) == {0, 1, 2}
        assert core_cluster.clip_ids == {"clip_1", "clip_2", "clip_3"}
        assert core_cluster.is_core_player is True
        assert core_cluster.cluster_type == "core"

        # Internal average distance: (0.20 + 0.21 + 0.32) / 3 ≈ 0.2433
        expected_internal_avg = (0.20 + 0.21 + 0.32) / 3.0
        assert np.isclose(core_cluster.average_internal_distance, expected_internal_avg, atol=1e-3)

    def test_viewpoint_transfer_with_precomputed_distance_matrix(self):
        # Explicit test with distance matrix [0.0, 0.20, 0.32], [0.20, 0.0, 0.21], [0.32, 0.21, 0.0]
        dist_mat = np.array(
            [
                [0.0, 0.20, 0.32],
                [0.20, 0.0, 0.21],
                [0.32, 0.21, 0.0],
            ],
            dtype=np.float64,
        )

        f_dummy = np.zeros(128, dtype=np.float32)
        f_dummy[0] = 1.0
        tracks = [
            _make_track("clip_1", 1, 0.0, 5.0, f_dummy),
            _make_track("clip_2", 2, 0.0, 5.0, f_dummy),
            _make_track("clip_3", 3, 0.0, 5.0, f_dummy),
        ]
        engine = SpatioTemporalConstraintEngine(tracks=tracks)

        clusters = constrained_hierarchical_clustering(
            tracks=tracks,
            constraints=engine,
            threshold=0.28,
            linkage="average",
            distance_matrix=dist_mat,
        )

        assert len(clusters) == 1
        assert clusters[0].track_indices == [0, 1, 2]
        assert clusters[0].is_core_player is True
        assert np.isclose(clusters[0].average_internal_distance, 0.73 / 3.0, atol=1e-4)

    def test_single_linkage_viewpoint_transfer(self):
        f_a, f_b, f_c = _create_viewpoint_features()
        tracks = [
            _make_track("clip_1", 1, 0.0, 5.0, f_a),
            _make_track("clip_2", 2, 0.0, 5.0, f_b),
            _make_track("clip_3", 3, 0.0, 5.0, f_c),
        ]
        engine = SpatioTemporalConstraintEngine(tracks=tracks)

        # Single linkage: min(0.21, 0.32) = 0.21 <= 0.28 -> merges
        clusters = constrained_hierarchical_clustering(
            tracks=tracks,
            constraints=engine,
            threshold=0.28,
            linkage="single",
        )
        assert len(clusters) == 1
        assert clusters[0].track_indices == [0, 1, 2]
        assert clusters[0].is_core_player is True

    def test_complete_linkage_strictness(self):
        f_a, f_b, f_c = _create_viewpoint_features()
        tracks = [
            _make_track("clip_1", 1, 0.0, 5.0, f_a),
            _make_track("clip_2", 2, 0.0, 5.0, f_b),
            _make_track("clip_3", 3, 0.0, 5.0, f_c),
        ]
        engine = SpatioTemporalConstraintEngine(tracks=tracks)

        # Under complete linkage with threshold=0.28, {A, B} merges (0.20 <= 0.28),
        # but C cannot merge because max(0.21, 0.32) = 0.32 > 0.28
        clusters_strict = constrained_hierarchical_clustering(
            tracks=tracks,
            constraints=engine,
            threshold=0.28,
            linkage="complete",
        )
        assert len(clusters_strict) == 2
        assert clusters_strict[0].track_indices == [0, 1]
        assert clusters_strict[1].track_indices == [2]

        # If threshold is relaxed to 0.35, complete linkage merges all 3
        clusters_relaxed = constrained_hierarchical_clustering(
            tracks=tracks,
            constraints=engine,
            threshold=0.35,
            linkage="complete",
        )
        assert len(clusters_relaxed) == 1
        assert clusters_relaxed[0].track_indices == [0, 1, 2]


class TestHardPhysicalConstraint:
    """Acceptance test 2: Same-screen co-occurrence hard constraint is strictly enforced."""

    def test_identical_features_same_screen_never_merge(self):
        # Two tracks with nearly identical features (distance 0.01)
        f_vec = np.zeros(128, dtype=np.float32)
        f_vec[0] = 1.0

        # Construct slightly rotated vector for distance ~0.01
        theta = 2.0 * float(np.arcsin(0.01))
        f_vec_close = np.zeros(128, dtype=np.float32)
        f_vec_close[0] = float(np.cos(theta))
        f_vec_close[1] = float(np.sin(theta))

        # Both tracks are in the SAME clip (clip_1) and overlap in time [2.0, 5.0]
        track_0 = _make_track("clip_1", 10, start_s=0.0, end_s=5.0, feature_vec=f_vec)
        track_1 = _make_track("clip_1", 20, start_s=2.0, end_s=7.0, feature_vec=f_vec_close)

        d = compute_track_distance(track_0, track_1)
        assert d is not None and np.isclose(d, 0.01, atol=1e-4)

        tracks = [track_0, track_1]
        engine = SpatioTemporalConstraintEngine(tracks=tracks)

        # Overlap without reliable spatial witnesses stays unresolved, not truth.
        assert engine.is_forbidden(0, 1) is False
        assert engine.is_unresolved(0, 1) is True
        assert engine.can_merge_clusters({0}, {1}) is False

        # Clustering must NEVER merge them, even with threshold=0.50
        clusters = constrained_hierarchical_clustering(
            tracks=tracks,
            constraints=engine,
            threshold=0.50,
            linkage="average",
        )

        assert len(clusters) == 2
        assert clusters[0].track_indices == [0]
        assert clusters[1].track_indices == [1]
        assert clusters[0].is_core_player is False
        assert clusters[1].is_core_player is False
        assert clusters[0].cluster_type == "unassigned"
        assert clusters[1].cluster_type == "unassigned"

    def test_indirect_chain_blocked_by_hard_constraint(self):
        """Track A and Track C are in the same clip and overlap (conflict).

        Track B is in a different clip and similar to both A and C.
        B can merge with A, but C must NOT be merged into {A, B}.
        """
        f_a, f_b, f_c = _create_viewpoint_features()

        # A and C both in clip_1, overlapping in time [2.0, 4.0]
        track_a = _make_track("clip_1", 1, start_s=0.0, end_s=4.0, feature_vec=f_a)
        track_b = _make_track("clip_2", 2, start_s=0.0, end_s=5.0, feature_vec=f_b)
        track_c = _make_track("clip_1", 3, start_s=2.0, end_s=6.0, feature_vec=f_c)

        tracks = [track_a, track_b, track_c]
        engine = SpatioTemporalConstraintEngine(tracks=tracks)

        assert engine.is_forbidden(0, 2) is False
        assert engine.is_unresolved(0, 2) is True
        # B does not conflict with A or C individually
        assert engine.is_forbidden(0, 1) is False
        assert engine.is_forbidden(1, 2) is False

        # Merging with threshold=0.28
        # Step 1: A and B merge (d=0.20 <= 0.28) -> cluster {0, 1}
        # Step 2: Candidates are {0, 1} and {2}. Because 0 and 2 are forbidden,
        # can_merge_clusters({0, 1}, {2}) is False!
        clusters = constrained_hierarchical_clustering(
            tracks=tracks,
            constraints=engine,
            threshold=0.28,
            linkage="average",
        )

        assert len(clusters) == 2
        # Cluster {0, 1} spans clip_1 and clip_2 -> is_core_player = True
        assert clusters[0].track_indices == [0, 1]
        assert clusters[0].is_core_player is True
        assert clusters[0].cluster_type == "core"

        # Cluster {2} is isolated -> unassigned
        assert clusters[1].track_indices == [2]
        assert clusters[1].is_core_player is False
        assert clusters[1].cluster_type == "unassigned"


class TestClusterPartitioning:
    """Verifies output classification into core players, single-clip players, and unassigned."""

    def test_single_clip_intermittent_player(self):
        # Two tracks in the same clip but at disjoint time intervals (no conflict)
        f_vec = np.zeros(128, dtype=np.float32)
        f_vec[0] = 1.0

        track_0 = _make_track("clip_1", 1, start_s=0.0, end_s=3.0, feature_vec=f_vec)
        track_1 = _make_track("clip_1", 2, start_s=5.0, end_s=8.0, feature_vec=f_vec)

        tracks = [track_0, track_1]
        engine = SpatioTemporalConstraintEngine(tracks=tracks)

        assert engine.is_forbidden(0, 1) is False

        clusters = constrained_hierarchical_clustering(
            tracks=tracks,
            constraints=engine,
            threshold=0.28,
            linkage="average",
        )

        assert len(clusters) == 1
        c = clusters[0]
        assert c.track_indices == [0, 1]
        assert c.clip_ids == {"clip_1"}
        assert c.is_core_player is False
        assert c.is_single_clip is True
        assert c.cluster_type == "single_clip"
        assert c.category == "single_clip"
        assert c.status == "single_clip"
        assert c.average_internal_distance == 0.0

    def test_unassigned_isolated_fragment(self):
        f_a = np.zeros(128, dtype=np.float32)
        f_a[0] = 1.0
        f_b = np.zeros(128, dtype=np.float32)
        f_b[1] = 1.0  # Orthogonal, distance ~0.707 > 0.28

        track_0 = _make_track("clip_1", 1, start_s=0.0, end_s=5.0, feature_vec=f_a)
        track_1 = _make_track("clip_2", 2, start_s=0.0, end_s=5.0, feature_vec=f_b)

        tracks = [track_0, track_1]
        engine = SpatioTemporalConstraintEngine(tracks=tracks)

        clusters = constrained_hierarchical_clustering(
            tracks=tracks,
            constraints=engine,
            threshold=0.28,
            linkage="average",
        )

        assert len(clusters) == 2
        for c in clusters:
            assert c.is_core_player is False
            assert c.is_unassigned is True
            assert c.cluster_type == "unassigned"
            assert c.average_internal_distance == 0.0


class TestEdgeCases:
    """Verifies edge cases and input validation."""

    def test_empty_tracks(self):
        engine = SpatioTemporalConstraintEngine(tracks=[])
        assert constrained_hierarchical_clustering([], engine) == []

    def test_single_track(self):
        f = np.zeros(128, dtype=np.float32)
        f[0] = 1.0
        track = _make_track("clip_1", 1, 0.0, 5.0, f)
        engine = SpatioTemporalConstraintEngine(tracks=[track])

        clusters = constrained_hierarchical_clustering([track], engine)
        assert len(clusters) == 1
        assert clusters[0].track_indices == [0]
        assert clusters[0].is_core_player is False
        assert clusters[0].is_unassigned is True
        assert clusters[0].average_internal_distance == 0.0

    def test_invalid_linkage_raises_value_error(self):
        f = np.zeros(128, dtype=np.float32)
        f[0] = 1.0
        tracks = [
            _make_track("clip_1", 1, 0.0, 5.0, f),
            _make_track("clip_2", 2, 0.0, 5.0, f),
        ]
        engine = SpatioTemporalConstraintEngine(tracks=tracks)

        with pytest.raises(ValueError, match="Unsupported linkage criterion"):
            constrained_hierarchical_clustering(tracks, engine, linkage="ward")

    def test_compute_cluster_distance_linkage_variants(self):
        dist_mat = np.array(
            [
                [0.0, 0.1, 0.3],
                [0.1, 0.0, 0.5],
                [0.3, 0.5, 0.0],
            ],
            dtype=np.float64,
        )

        # Cluster A = {0, 1}, Cluster B = {2}
        # Pairwise distances: D(0, 2) = 0.3, D(1, 2) = 0.5
        c_a = [0, 1]
        c_b = [2]

        assert np.isclose(
            compute_cluster_distance(c_a, c_b, dist_mat, linkage="complete"),
            0.5,
        )
        assert np.isclose(
            compute_cluster_distance(c_a, c_b, dist_mat, linkage="average"),
            0.4,
        )
        assert np.isclose(
            compute_cluster_distance(c_a, c_b, dist_mat, linkage="single"),
            0.3,
        )
