"""Same-team (same jersey color) ID-switch splitting: red-first tests.

Covers the four acceptance scenarios:
1. Same-color sequence with a clean 27 -> 10 per-crop OCR vote flip and no
   saturation jump must produce exactly one change point, and both sides must
   remain complete after splitting.
2. Same-color, same-number, appearance-continuous tracks must produce zero
   change points (false-alarm guard).
3. Transition-buffer observations must never be dropped: every input row must
   appear in exactly one returned subsegment (a pending transition segment).
4. The existing cross-team saturation/color regression scenario still splits.

Synthetic crops encode machine evidence statelessly in two marker pixels so
fakes are deterministic and order-independent:
  - rgb[0, 0, 0] carries the jersey number the fake OCR should read (255=none),
  - rgb[0, 1, 0] carries the appearance regime (0 or 1).
"""

import numpy as np

from app.features.appearance import AppearanceExtractor
from app.features.jersey_number import JerseyNumberExtractor, JerseyNumberPrediction
from app.features.split import detect_track_change_points, split_track_observations


class _MarkerOCR(JerseyNumberExtractor):
    """Stateless fake single-image OCR reading the marker pixel."""

    def __init__(self):
        super().__init__(reader=object())

    def extract(self, image):
        number = int(image[0, 0, 0])
        if number == 255:
            return JerseyNumberPrediction(None, 0.0, True, {})
        return JerseyNumberPrediction(number, 0.95, True, {number: 0.95}, str(number))


def _marker_encoder(rgb):
    v = np.zeros(512, dtype=np.float32)
    v[int(rgb[0, 1, 0])] = 1.0
    return v


def _appearance():
    return AppearanceExtractor(encoder=_marker_encoder, model_name="test-marker-encoder")


def _crop(number: int, regime: int = 0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    crop = np.full((100, 60, 3), (80, 95, 65), dtype=np.int16)
    crop += rng.integers(-15, 15, crop.shape, dtype=np.int16)
    crop = np.clip(crop, 0, 255).astype(np.uint8)
    crop[0, 0, :] = number
    crop[0, 1, :] = (regime, regime, regime)
    return crop


def _dense_rows(last: int = 779):
    return [
        {"frame_idx": f, "timestamp_s": f / 60.0, "bbox_xyxy": [10, 10, 50, 90], "det_score": 0.9}
        for f in range(0, last + 1)
    ]


def test_same_color_ocr_number_switch_is_detected_and_both_sides_complete():
    # 20 samples over frames 0..760; identical green torso, identical appearance;
    # first half votes 27, second half votes 10. Only OCR can see the switch.
    crops = [
        (f * 40, _crop(27 if k < 10 else 10, regime=0, seed=k), 0.9)
        for k, f in enumerate(range(20))
    ]
    cps = detect_track_change_points(
        crops,
        min_track_duration=300,
        min_subsegment_duration=120,
        appearance_extractor=_appearance(),
        jersey_extractor=_MarkerOCR(),
    )
    assert len(cps) == 1
    cp = cps[0]
    assert cp.left_number == 27
    assert cp.right_number == 10
    assert "ocr" in cp.reason
    # The flip is between sample frames 360 (27) and 400 (10).
    assert cp.left_frame == 360
    assert cp.right_frame == 400

    rows = _dense_rows()
    subs = split_track_observations(rows, cps)
    flat = [r["frame_idx"] for sub in subs for r in sub]
    assert sorted(flat) == list(range(0, 780))
    left = next(sub for sub in subs if sub[0]["frame_idx"] == 0)
    right = next(sub for sub in subs if sub[-1]["frame_idx"] == 779)
    assert left[-1]["frame_idx"] == cp.left_frame
    assert right[0]["frame_idx"] == cp.right_frame
    # Both stable regions stay whole inside their own subsegment.
    assert {r["frame_idx"] for r in left} >= set(range(0, 361))
    assert {r["frame_idx"] for r in right} >= set(range(400, 780))


def test_same_number_same_color_continuous_track_has_no_split():
    crops = [(f * 40, _crop(27, regime=0, seed=k), 0.9) for k, f in enumerate(range(20))]
    cps = detect_track_change_points(
        crops,
        min_track_duration=300,
        min_subsegment_duration=120,
        appearance_extractor=_appearance(),
        jersey_extractor=_MarkerOCR(),
    )
    assert cps == []


def test_digit_compatible_votes_do_not_split():
    # 27 vs 7 are digit-compatible under the frozen number semantics.
    crops = [
        (f * 40, _crop(27 if k < 10 else 7, regime=0, seed=k), 0.9) for k, f in enumerate(range(20))
    ]
    cps = detect_track_change_points(
        crops,
        min_track_duration=300,
        min_subsegment_duration=120,
        appearance_extractor=_appearance(),
        jersey_extractor=_MarkerOCR(),
    )
    assert cps == []


def test_no_ocr_votes_continuous_appearance_has_no_split():
    crops = [(f * 40, _crop(255, regime=0, seed=k), 0.9) for k, f in enumerate(range(20))]
    cps = detect_track_change_points(
        crops,
        min_track_duration=300,
        min_subsegment_duration=120,
        appearance_extractor=_appearance(),
        jersey_extractor=_MarkerOCR(),
    )
    assert cps == []


def test_same_color_appearance_shift_without_ocr_is_detected():
    # Same green torso saturation, no OCR votes at all, but the appearance
    # regime flips halfway: the appearance-only trigger must catch it.
    crops = [
        (f * 40, _crop(255, regime=0 if k < 10 else 1, seed=k), 0.9)
        for k, f in enumerate(range(20))
    ]
    cps = detect_track_change_points(
        crops,
        min_track_duration=300,
        min_subsegment_duration=120,
        appearance_extractor=_appearance(),
        jersey_extractor=_MarkerOCR(),
    )
    assert len(cps) == 1
    assert cps[0].left_frame == 360
    assert cps[0].right_frame == 400
    assert "appearance" in cps[0].reason


def test_transition_buffer_rows_are_never_dropped():
    rows = _dense_rows(599)
    crops = [
        (f * 40, _crop(27 if k < 10 else 10, regime=0, seed=k), 0.9)
        for k, f in enumerate(range(20))
    ]
    cps = detect_track_change_points(
        crops,
        min_track_duration=300,
        min_subsegment_duration=120,
        appearance_extractor=_appearance(),
        jersey_extractor=_MarkerOCR(),
    )
    assert len(cps) == 1
    subs = split_track_observations(rows, cps)
    flat = [r["frame_idx"] for sub in subs for r in sub]
    assert sorted(flat) == list(range(0, 600))
    cp = cps[0]
    buffer_frames = set(range(cp.transition_start + 1, cp.transition_end))
    covered = {r["frame_idx"] for sub in subs for r in sub}
    assert buffer_frames <= covered
    # The buffer forms its own pending subsegment, not merged into either side.
    transition_subs = [
        sub
        for sub in subs
        if sub[0]["frame_idx"] > cp.transition_start and sub[-1]["frame_idx"] < cp.transition_end
    ]
    assert len(transition_subs) == 1
    assert {r["frame_idx"] for r in transition_subs[0]} == buffer_frames


def test_cross_team_saturation_regression_still_splits():
    # Synthetic version of the clip-68 scenario: dark green -> bright yellow
    # with orthogonal appearance regimes and no OCR evidence.
    def yellow_crop(seed):
        rng = np.random.default_rng(seed)
        crop = np.full((100, 60, 3), (20, 210, 240), dtype=np.int16)
        crop += rng.integers(-15, 15, crop.shape, dtype=np.int16)
        crop = np.clip(crop, 0, 255).astype(np.uint8)
        crop[0, 0, :] = 255
        crop[0, 1, :] = (1, 1, 1)
        return crop

    crops = [(f * 40, _crop(255, regime=0, seed=k), 0.9) for k, f in enumerate(range(10))]
    crops += [(400 + k * 40, yellow_crop(100 + k), 0.9) for k in range(10)]
    cps = detect_track_change_points(
        crops,
        min_track_duration=300,
        min_subsegment_duration=120,
        appearance_extractor=_appearance(),
        jersey_extractor=_MarkerOCR(),
    )
    assert len(cps) == 1
    assert cps[0].left_frame == 360
    assert cps[0].right_frame == 400


def test_detection_is_deterministic():
    crops = [
        (f * 40, _crop(27 if k < 10 else 10, regime=0, seed=k), 0.9)
        for k, f in enumerate(range(20))
    ]
    kwargs = dict(
        min_track_duration=300,
        min_subsegment_duration=120,
        appearance_extractor=_appearance(),
        jersey_extractor=_MarkerOCR(),
    )
    first = detect_track_change_points(crops, **kwargs)
    second = detect_track_change_points(list(reversed(crops)), **kwargs)
    assert first == second
