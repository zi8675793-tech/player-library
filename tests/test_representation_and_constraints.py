"""Unit tests for trajectory representation pooling and spatio-temporal constraints.

Verifies:
1. Multi-part weighted pooling and L2 normalization correctness.
2. Low-visibility frame filtering and visibility averaging.
3. Multi-part weighted distance calculation and comparability thresholds.
4. Spatio-temporal hard mutual exclusion (Must-Not-Link):
   - Same clip overlapping time tracks are forbidden (is_forbidden = True).
   - Different clips or non-overlapping time tracks are not forbidden (is_forbidden = False).
5. Cluster merge validation (can_merge_clusters).
6. Integration with TrackObservation evidence.
"""

import numpy as np
import pytest

from app.contracts.common import Provenance
from app.contracts.evidence import FeatureSpace, InlineFeature, Sample, TrackObservation
from app.identity.constraints import ForbiddenPairs, SpatioTemporalConstraintEngine
from app.identity.representation import (
    TrackRepresentation,
    compute_track_distance,
    pool_track_features,
)


def _make_provenance(record_id: str = "rec_1") -> Provenance:
    return Provenance(
        producer="test",
        record_type="observation",
        source_record_id=record_id,
        evidence_version="v1",
    )


def _make_feature_space(
    parts: int = 6, dimension: int = 128, fs_id: str = "kpr_test"
) -> FeatureSpace:
    return FeatureSpace(
        feature_space_id=fs_id,
        backend="kpr",
        model_hash="0" * 64,
        preprocessing={"version": "v1", "color_space": "RGB"},
        layout="parts",
        shape=[parts, dimension],
        normalization="l2",
        distance="euclidean_sqrt_visibility_mean_div2",
        visibility={"shape": [parts], "semantics": "visibility"},
    )


def _make_unit_vector(dimension: int, index: int = 0) -> np.ndarray:
    vec = np.zeros(dimension, dtype=np.float32)
    vec[index % dimension] = 1.0
    return vec


def _make_track_rep(
    clip_id: str = "clip_1",
    track_id: int = 1,
    start_s: float = 0.0,
    end_s: float = 5.0,
    parts: int = 6,
    dimension: int = 128,
    visibility: list[float] | None = None,
    features: list[np.ndarray] | None = None,
) -> TrackRepresentation:
    if visibility is None:
        vis_arr = np.ones((parts,), dtype=np.float32) * 0.9
    else:
        vis_arr = np.asarray(visibility, dtype=np.float32)

    if features is None:
        feat_arr = np.zeros((parts, dimension), dtype=np.float32)
        for p in range(parts):
            feat_arr[p] = _make_unit_vector(dimension, p)
    else:
        feat_arr = np.asarray(features, dtype=np.float32)

    return TrackRepresentation(
        clip_id=clip_id,
        track_id=track_id,
        start_s=start_s,
        end_s=end_s,
        feature=feat_arr,
        visibility=vis_arr,
        sample_ids=[f"s_{clip_id}_{track_id}_1"],
    )


class TestTrackRepresentation:
    def test_dataclass_attributes_and_equality(self):
        feat = np.ones((6, 128), dtype=np.float32)
        vis = np.ones(6, dtype=np.float32) * 0.8
        t1 = TrackRepresentation("c1", 1, 0.0, 5.0, feat, vis, ["s1"])
        t2 = TrackRepresentation("c1", 1, 0.0, 5.0, feat.copy(), vis.copy(), ["s1"])
        t3 = TrackRepresentation("c1", 2, 0.0, 5.0, feat, vis, ["s2"])

        assert t1 == t2
        assert t1 != t3
        assert hash(t1) == hash(t2)
        assert len({t1, t2}) == 1
        with pytest.raises(AttributeError):
            t1.start_s = 2.0  # frozen dataclass


class TestPooling:
    def test_pool_single_sample_l2_normalized(self):
        sample = {
            "clip_id": "clip_1",
            "track_id": 10,
            "sample_id": "s1",
            "time_s": 1.5,
            "feature": np.array(
                [
                    _make_unit_vector(128, 0) * 3.0,
                    _make_unit_vector(128, 1) * 5.0,
                    _make_unit_vector(128, 2),
                    _make_unit_vector(128, 3),
                    _make_unit_vector(128, 4),
                    _make_unit_vector(128, 5),
                ],
                dtype=np.float32,
            ),
            "visibility": np.array([0.9, 0.85, 0.7, 0.6, 0.5, 0.4], dtype=np.float32),
        }

        reps = pool_track_features([sample], min_vis=0.2)
        assert ("clip_1", 10) in reps
        rep = reps[("clip_1", 10)]

        assert rep.clip_id == "clip_1"
        assert rep.track_id == 10
        assert rep.start_s == 1.5
        assert rep.end_s == 1.5
        assert rep.sample_ids == ["s1"]
        assert rep.feature.shape == (6, 128)
        assert rep.visibility.shape == (6,)

        for p in range(6):
            norm = np.linalg.norm(rep.feature[p])
            assert np.isclose(norm, 1.0, atol=1e-5)
            assert np.isclose(rep.visibility[p], sample["visibility"][p])

    def test_pool_multiple_samples_weighted_l2(self):
        f1_p0 = _make_unit_vector(128, 0)
        f2_p0 = _make_unit_vector(128, 1)

        sample1 = {
            "clip_id": "c1",
            "track_id": 1,
            "sample_id": "s1",
            "time_s": 1.0,
            "feature": np.array([f1_p0] * 6, dtype=np.float32),
            "visibility": np.array([0.8] * 6, dtype=np.float32),
        }
        sample2 = {
            "clip_id": "c1",
            "track_id": 1,
            "sample_id": "s2",
            "time_s": 3.0,
            "feature": np.array([f2_p0] * 6, dtype=np.float32),
            "visibility": np.array([0.6] * 6, dtype=np.float32),
        }

        reps = pool_track_features([sample1, sample2], min_vis=0.2)
        rep = reps[("c1", 1)]

        assert rep.start_s == 1.0
        assert rep.end_s == 3.0
        assert rep.sample_ids == ["s1", "s2"]

        expected_weighted_sum = 0.8 * f1_p0 + 0.6 * f2_p0
        expected_norm = np.linalg.norm(expected_weighted_sum)
        expected_f0 = expected_weighted_sum / expected_norm

        for p in range(6):
            np.testing.assert_allclose(rep.feature[p], expected_f0, atol=1e-5)
            assert np.isclose(np.linalg.norm(rep.feature[p]), 1.0, atol=1e-5)
            assert np.isclose(rep.visibility[p], 0.7, atol=1e-5)

    def test_pool_filters_low_visibility_frames(self):
        f1 = _make_unit_vector(128, 0)
        f2 = _make_unit_vector(128, 1)

        sample1 = {
            "clip_id": "c1",
            "track_id": 1,
            "sample_id": "s1",
            "time_s": 0.0,
            "feature": np.array([f1] * 6, dtype=np.float32),
            "visibility": np.array([0.9, 0.1, 0.9, 0.05, 0.9, 0.9], dtype=np.float32),
        }
        sample2 = {
            "clip_id": "c1",
            "track_id": 1,
            "sample_id": "s2",
            "time_s": 2.0,
            "feature": np.array([f2] * 6, dtype=np.float32),
            "visibility": np.array([0.15, 0.8, 0.9, 0.1, 0.9, 0.9], dtype=np.float32),
        }

        reps = pool_track_features([sample1, sample2], min_vis=0.2)
        rep = reps[("c1", 1)]

        # Part 0: sample1 (v=0.9 >= 0.2), sample2 (v=0.15 < 0.2) filtered out -> only sample1
        np.testing.assert_allclose(rep.feature[0], f1, atol=1e-5)
        assert np.isclose(rep.visibility[0], 0.9)

        # Part 1: sample1 (v=0.1 < 0.2) filtered out, sample2 (v=0.8 >= 0.2) kept -> only sample2
        np.testing.assert_allclose(rep.feature[1], f2, atol=1e-5)
        assert np.isclose(rep.visibility[1], 0.8)

        # Part 3: both sample1 (0.05) and sample2 (0.1) are below min_vis 0.2 -> 0 feature, 0.0 vis
        np.testing.assert_allclose(rep.feature[3], np.zeros(128), atol=1e-5)
        assert rep.visibility[3] == 0.0

    def test_pool_with_contracts_sample_and_feature_space(self):
        space = _make_feature_space(parts=6, dimension=128)
        f_vec = [1.0, *([0.0] * 127)] * 6

        sample = Sample(
            match_id="m1",
            sample_id="sam_101",
            clip_id="clip_A",
            track_id="42",
            time_s=12.5,
            source=_make_provenance("sam_101"),
            features=[
                InlineFeature(
                    feature_space_id=space.feature_space_id,
                    values=f_vec,
                    visibility=[0.85] * 6,
                )
            ],
        )

        reps = pool_track_features([sample], space=space, min_vis=0.2)
        assert ("clip_A", 42) in reps
        rep = reps[("clip_A", 42)]
        assert rep.clip_id == "clip_A"
        assert rep.track_id == 42
        assert rep.start_s == 12.5
        assert rep.end_s == 12.5
        assert rep.sample_ids == ["sam_101"]
        assert rep.feature.shape == (6, 128)
        assert np.allclose(rep.visibility, 0.85)

    def test_pool_empty_samples(self):
        reps = pool_track_features([])
        assert reps == {}


class TestTrackDistance:
    def test_identical_tracks_distance_zero(self):
        t1 = _make_track_rep(clip_id="c1", track_id=1, visibility=[0.8] * 6)
        t2 = _make_track_rep(clip_id="c2", track_id=2, visibility=[0.9] * 6)
        dist = compute_track_distance(t1, t2, min_common_parts=2, min_vis=0.2)
        assert dist is not None
        assert np.isclose(dist, 0.0, atol=1e-6)

    def test_orthogonal_and_opposite_distance(self):
        f1 = [_make_unit_vector(128, 0)] * 6
        f2 = [_make_unit_vector(128, 1)] * 6  # Orthogonal to f1, L2 diff = sqrt(2)
        v1 = [1.0] * 6
        v2 = [1.0] * 6

        t1 = _make_track_rep(features=f1, visibility=v1)
        t2 = _make_track_rep(features=f2, visibility=v2)

        dist = compute_track_distance(t1, t2, min_common_parts=2)
        assert dist is not None
        # d = (6 * 1.0 * sqrt(2)) / (2 * 6 * 1.0) = sqrt(2) / 2
        expected = float(np.sqrt(2) / 2.0)
        assert np.isclose(dist, expected, atol=1e-5)

        # Opposite vectors: diff norm = 2.0, d = 2.0 / 2.0 = 1.0
        f3 = [-_make_unit_vector(128, 0)] * 6
        t3 = _make_track_rep(features=f3, visibility=v1)
        dist_opp = compute_track_distance(t1, t3, min_common_parts=2)
        assert dist_opp is not None
        assert np.isclose(dist_opp, 1.0, atol=1e-5)

    def test_insufficient_common_parts_returns_none(self):
        # t1 visible only at part 0
        v1 = [0.8, 0.1, 0.1, 0.1, 0.1, 0.1]
        # t2 visible at part 0 and part 1
        v2 = [0.8, 0.8, 0.1, 0.1, 0.1, 0.1]

        t1 = _make_track_rep(visibility=v1)
        t2 = _make_track_rep(visibility=v2)

        # Overlapping parts >= min_vis(0.2) is only {0} (1 part), while min_common_parts=2
        assert compute_track_distance(t1, t2, min_common_parts=2, min_vis=0.2) is None
        # With min_common_parts=1, it succeeds
        assert compute_track_distance(t1, t2, min_common_parts=1, min_vis=0.2) is not None

    def test_partially_overlapping_parts_weighted_properly(self):
        # Part 0 and Part 1 are mutually visible >= 0.2
        v1 = [0.9, 0.4, 0.1, 0.0, 0.0, 0.0]
        v2 = [0.4, 0.9, 0.0, 0.0, 0.0, 0.0]
        f1 = [_make_unit_vector(128, 0), _make_unit_vector(128, 0)] + [
            _make_unit_vector(128, 0)
        ] * 4
        f2 = [_make_unit_vector(128, 1), _make_unit_vector(128, 0)] + [
            _make_unit_vector(128, 1)
        ] * 4

        t1 = _make_track_rep(features=f1, visibility=v1)
        t2 = _make_track_rep(features=f2, visibility=v2)

        # Part 0 diff norm = sqrt(2), weight = sqrt(0.9 * 0.4) = 0.6
        # Part 1 diff norm = 0.0, weight = sqrt(0.4 * 0.9) = 0.6
        # Sum weights = 1.2
        # d = (0.6 * sqrt(2) + 0.6 * 0.0) / (2 * 1.2) = 0.6 * sqrt(2) / 2.4 = sqrt(2) / 4
        dist = compute_track_distance(t1, t2, min_common_parts=2, min_vis=0.2)
        assert dist is not None
        expected = float(np.sqrt(2) / 4.0)
        assert np.isclose(dist, expected, atol=1e-5)


class TestSpatioTemporalConstraintEngine:
    @staticmethod
    def _separated_observations(track_a="0", track_b="1"):
        observations = []
        for i in range(3):
            for track, x in ((track_a, 0.0), (track_b, 100.0)):
                observations.append(
                    TrackObservation(
                        observation_id=f"separated-{track}-{i}",
                        match_id="m1",
                        clip_id="clip_1",
                        track_id=track,
                        time_s=i + 0.1,
                        frame_idx=i,
                        bbox_xyxy=(x, 0.0, x + 20.0, 50.0),
                        source=_make_provenance(f"separated-{track}-{i}"),
                    )
                )
        return observations

    def test_same_clip_overlapping_time_is_forbidden(self):
        # Track 0: [0.0, 5.0] in clip_1
        # Track 1: [3.0, 8.0] in clip_1  -> max(0, 3)=3 < min(5, 8)=5 (overlap 3..5)
        # Track 2: [1.0, 4.0] in clip_1  -> enclosed within Track 0
        t0 = _make_track_rep(clip_id="clip_1", track_id=0, start_s=0.0, end_s=5.0)
        t1 = _make_track_rep(clip_id="clip_1", track_id=1, start_s=3.0, end_s=8.0)
        t2 = _make_track_rep(clip_id="clip_1", track_id=2, start_s=1.0, end_s=4.0)

        engine = SpatioTemporalConstraintEngine(tracks=[t0, t1, t2])

        assert all(
            engine.is_unresolved(i, j) for i, j in ((0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1))
        )
        assert not engine.forbidden_pairs
        assert engine.can_merge_clusters({0}, {1}) is False

        engine = SpatioTemporalConstraintEngine(
            tracks=[t0, t1, t2], observations=self._separated_observations()
        )
        assert engine.is_forbidden(0, 1) is True
        assert engine.is_forbidden(1, 0) is True
        assert engine.can_merge_clusters({0}, {1}) is False

    def test_different_clip_not_forbidden(self):
        # Same exact time interval [2.0, 6.0], but different clips
        t0 = _make_track_rep(clip_id="clip_A", track_id=1, start_s=2.0, end_s=6.0)
        t1 = _make_track_rep(clip_id="clip_B", track_id=2, start_s=2.0, end_s=6.0)

        engine = SpatioTemporalConstraintEngine(tracks=[t0, t1])

        assert engine.is_forbidden(0, 1) is False
        assert engine.is_forbidden(1, 0) is False
        assert (0, 1) not in engine.forbidden_pairs
        assert len(engine.forbidden_pairs) == 0

    def test_same_clip_disjoint_and_touching_time_not_forbidden(self):
        # Track 0: [0.0, 3.0]
        # Track 1: [3.0, 6.0] -> touching boundary: max(0, 3) < min(3, 6) -> 3 < 3 is False
        # Track 2: [7.0, 10.0] -> completely disjoint: max(0, 7) < min(3, 10) -> 7 < 3 is False
        t0 = _make_track_rep(clip_id="clip_1", track_id=0, start_s=0.0, end_s=3.0)
        t1 = _make_track_rep(clip_id="clip_1", track_id=1, start_s=3.0, end_s=6.0)
        t2 = _make_track_rep(clip_id="clip_1", track_id=2, start_s=7.0, end_s=10.0)

        engine = SpatioTemporalConstraintEngine(tracks=[t0, t1, t2])

        assert engine.is_forbidden(0, 1) is False
        assert engine.is_forbidden(1, 0) is False
        assert engine.is_forbidden(0, 2) is False
        assert engine.is_forbidden(1, 2) is False
        assert len(engine.forbidden_pairs) == 0

    def test_self_is_not_forbidden(self):
        t0 = _make_track_rep(clip_id="clip_1", track_id=0, start_s=0.0, end_s=5.0)
        engine = SpatioTemporalConstraintEngine(tracks=[t0])
        assert engine.is_forbidden(0, 0) is False
        assert (0, 0) not in engine.forbidden_pairs

    def test_can_merge_clusters(self):
        # t0: clip_1 [0.0, 5.0]
        # t1: clip_1 [4.0, 8.0] (conflicts with t0)
        # t2: clip_2 [0.0, 5.0] (in clip_2, no conflict with t0 or t1)
        # t3: clip_1 [10.0, 15.0] (in clip_1, disjoint with t0 and t1)
        t0 = _make_track_rep(clip_id="clip_1", track_id=0, start_s=0.0, end_s=5.0)
        t1 = _make_track_rep(clip_id="clip_1", track_id=1, start_s=4.0, end_s=8.0)
        t2 = _make_track_rep(clip_id="clip_2", track_id=2, start_s=0.0, end_s=5.0)
        t3 = _make_track_rep(clip_id="clip_1", track_id=3, start_s=10.0, end_s=15.0)

        engine = SpatioTemporalConstraintEngine(tracks=[t0, t1, t2, t3])

        # Cluster {0} and {1} cannot be merged because 0 and 1 are forbidden
        assert engine.can_merge_clusters({0}, {1}) is False
        assert engine.can_merge_clusters({0, 2}, {1, 3}) is False

        # Cluster {0} and {2} can be merged (different clips)
        assert engine.can_merge_clusters({0}, {2}) is True
        # Cluster {0, 2} and {3} can be merged (t0 and t3 are non-overlapping in clip_1; t2 in clip_2)
        assert engine.can_merge_clusters({0, 2}, {3}) is True

    def test_observations_expand_track_bounds(self):
        # Sample times are non-overlapping: t0 [0.0, 1.0], t1 [5.0, 6.0]
        t0 = _make_track_rep(clip_id="clip_1", track_id=1, start_s=0.0, end_s=1.0)
        t1 = _make_track_rep(clip_id="clip_1", track_id=2, start_s=5.0, end_s=6.0)

        # Observations expand t0 up to 4.0s and t1 down to 3.0s -> overlap at 3.0..4.0s!
        obs0 = TrackObservation(
            observation_id="o1",
            match_id="m1",
            clip_id="clip_1",
            track_id="1",
            time_s=4.0,
            bbox_xyxy=(10.0, 10.0, 50.0, 50.0),
            source=_make_provenance("obs1"),
        )
        obs1 = TrackObservation(
            observation_id="o2",
            match_id="m1",
            clip_id="clip_1",
            track_id="2",
            time_s=3.0,
            bbox_xyxy=(60.0, 60.0, 90.0, 90.0),
            source=_make_provenance("obs2"),
        )

        engine = SpatioTemporalConstraintEngine(tracks=[t0, t1], observations=[obs0, obs1])
        assert engine.is_unresolved(0, 1) is True
        assert (0, 1) not in engine.forbidden_pairs
        assert engine.can_merge_clusters({0}, {1}) is False


class TestForbiddenPairs:
    def test_custom_set_behavior(self):
        fp = ForbiddenPairs()
        fp.add((2, 5))
        assert (2, 5) in fp
        assert (5, 2) in fp
        assert (1, 5) not in fp
        assert len(fp) == 1
        assert list(fp) == [(2, 5)]

        # Adding reversed pair does not duplicate
        fp.add((5, 2))
        assert len(fp) == 1

        # Self-pair ignored
        fp.add((3, 3))
        assert len(fp) == 1
        assert (3, 3) not in fp
