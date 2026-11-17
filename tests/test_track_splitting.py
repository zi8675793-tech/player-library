"""Unit tests for unsupervised change-point detection and trajectory splitting."""

from unittest.mock import MagicMock

import numpy as np

from app.features.appearance import AppearanceEmbedding, AppearanceExtractor
from app.features.split import (
    ChangePoint,
    _extract_torso_hsv_metrics,
    detect_track_change_points,
    split_track_observations,
)


def _make_dummy_crop(color_bgr: tuple[int, int, int], size: int = 100) -> np.ndarray:
    crop = np.zeros((size, size, 3), dtype=np.uint8)
    crop[:] = color_bgr
    # Add texture so laplacian variance is > 30 if needed
    noise = np.random.randint(-20, 20, (size, size, 3))
    crop = np.clip(crop.astype(int) + noise, 0, 255).astype(np.uint8)
    return crop


def test_torso_hsv_metrics():
    green = _make_dummy_crop((20, 100, 20))
    sat, hist = _extract_torso_hsv_metrics(green)
    assert 0 <= sat <= 255
    assert len(hist) == 18 * 8
    assert np.isclose(hist.sum(), 1.0)


def test_no_change_point_on_short_track():
    crops = [
        (10, _make_dummy_crop((20, 100, 20)), 0.9),
        (50, _make_dummy_crop((20, 100, 20)), 0.9),
        (100, _make_dummy_crop((20, 100, 20)), 0.9),
    ]
    cps = detect_track_change_points(crops, min_track_duration=300)
    assert cps == []


def test_change_point_detection_on_distinct_regimes():
    # Mock appearance extractor to return two distinct orthogonal embeddings
    mock_extractor = MagicMock(spec=AppearanceExtractor)

    v1 = np.zeros(512, dtype=np.float32)
    v1[0] = 1.0  # Regime 1 embedding
    v2 = np.zeros(512, dtype=np.float32)
    v2[1] = 1.0  # Regime 2 embedding (cosine dist = 1.0)

    emb1 = AppearanceEmbedding(vector=v1, quality_score=0.9, model_name="test")
    emb2 = AppearanceEmbedding(vector=v2, quality_score=0.9, model_name="test")

    mock_extractor.extract.side_effect = [emb1] * 5 + [emb2] * 5

    dark_green = (80, 95, 65)  # Realistic pine green (sat ~85)
    bright_yellow = (20, 210, 240)  # Realistic yellow (sat ~230)

    crops = [
        (0, _make_dummy_crop(dark_green), 0.9),
        (50, _make_dummy_crop(dark_green), 0.9),
        (100, _make_dummy_crop(dark_green), 0.9),
        (150, _make_dummy_crop(dark_green), 0.9),
        (200, _make_dummy_crop(dark_green), 0.9),
        # Transition between F200 and F350
        (350, _make_dummy_crop(bright_yellow), 0.9),
        (400, _make_dummy_crop(bright_yellow), 0.9),
        (450, _make_dummy_crop(bright_yellow), 0.9),
        (500, _make_dummy_crop(bright_yellow), 0.9),
        (550, _make_dummy_crop(bright_yellow), 0.9),
    ]

    cps = detect_track_change_points(
        crops,
        min_track_duration=300,
        min_subsegment_duration=100,
        appearance_extractor=mock_extractor,
    )

    assert len(cps) == 1
    cp = cps[0]
    assert cp.left_frame == 200
    assert cp.right_frame == 350
    assert cp.transition_start == 200
    assert cp.transition_end == 350
    assert cp.d_sat >= 35.0
    assert cp.d_step >= 0.25


def test_split_track_observations():
    rows = [
        {"frame_idx": f, "timestamp_s": f / 30.0, "bbox_xyxy": [10, 10, 50, 50], "det_score": 0.9}
        for f in range(0, 600, 10)
    ]
    cp = ChangePoint(
        left_frame=200,
        right_frame=350,
        transition_start=200,
        transition_end=350,
        d_step=0.4,
        d_sat=100.0,
        chi2=0.5,
        d_cross=0.4,
        margin=0.2,
        reason="test",
    )

    subs = split_track_observations(rows, [cp])
    assert len(subs) == 3

    sub1, transition, sub2 = subs
    assert sub1[0]["frame_idx"] == 0
    assert sub1[-1]["frame_idx"] == 200

    assert sub2[0]["frame_idx"] == 350
    assert sub2[-1]["frame_idx"] == 590

    # Transition-buffer observations are preserved as their own pending segment.
    assert [r["frame_idx"] for r in transition] == list(range(210, 350, 10))
    flat = sorted(r["frame_idx"] for sub in subs for r in sub)
    assert flat == list(range(0, 600, 10))
