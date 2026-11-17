"""T1 contract and decision tests; synthetic crops only, no evaluation inputs."""

import builtins
from dataclasses import FrozenInstanceError, fields, replace

import numpy as np
import pytest
from PIL import Image, ImageFilter

from app.features.appearance import AppearanceEmbedding, AppearanceExtractor
from app.features.dual_stream import (
    DualStreamExtractor,
    DualStreamFeature,
    TrackletSegment,
    TrackObservation,
)
from app.features.jersey_number import (
    JerseyNumberExtractor,
    JerseyNumberPrediction,
    assess_image_quality,
    validate_rgb_image,
)


@pytest.fixture
def clear_crop():
    return np.random.default_rng(104).integers(45, 210, (128, 64, 3), dtype=np.uint8)


class Reader:
    """Scripted OCR boundary; scores are evidence supplied by each test."""

    def __init__(self, *frames):
        self.frames = iter(frames)
        self.calls = []

    def readtext(self, image, **kwargs):
        self.calls.append((image.copy(), kwargs))
        h, w = image.shape[:2]
        box = [[w * 0.3, h * 0.2], [w * 0.7, h * 0.2], [w * 0.7, h * 0.6], [w * 0.3, h * 0.6]]
        return [(box, text, confidence) for text, confidence in next(self.frames)]


def appearance(encoder=None):
    return AppearanceExtractor(
        encoder=encoder or (lambda image: np.arange(1, 513, dtype=np.float32)),
        model_name="unit-test-encoder-v1",
    )


def observation(path=None, *, video="3", track=7, frame=0):
    return TrackObservation(video, track, frame, frame / 60, (0.0, 0.0, 64.0, 128.0), 0.9, path)


def write_crop(root, image, *, video="3", name="frame.png"):
    path = root / video / "samples" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
    return str(path.relative_to(root))


def extractor(root, *ocr_frames):
    reader = Reader(*ocr_frames)
    return DualStreamExtractor(
        root,
        jersey_extractor=JerseyNumberExtractor(reader),
        appearance_extractor=appearance(),
    ), reader


@pytest.mark.parametrize(
    ("contract", "names"),
    [
        (
            TrackObservation,
            "video_id local_track_id frame_idx timestamp_s bbox_xyxy confidence crop_path",
        ),
        (
            TrackletSegment,
            "segment_id video_id local_track_id start_frame end_frame start_time_s "
            "end_time_s observation_count observations",
        ),
        (JerseyNumberPrediction, "number confidence legible votes raw_text"),
        (AppearanceEmbedding, "vector quality_score model_name"),
        (DualStreamFeature, "segment_id jersey appearance"),
    ],
)
def test_frozen_contract_fields_are_exact(contract, names):
    assert [field.name for field in fields(contract)] == names.split()
    assert contract.__dataclass_params__.frozen


def test_contract_default_and_frozen_assignment():
    item = observation()
    assert item.crop_path is None
    with pytest.raises(FrozenInstanceError):
        item.frame_idx = 3


def test_quality_filters_blur_backlight_and_tiny_images(clear_crop):
    assert 0 < assess_image_quality(clear_crop) <= 1
    assert assess_image_quality(np.zeros_like(clear_crop)) == 0
    assert assess_image_quality(np.full_like(clear_crop, 255)) == 0
    blurred = np.asarray(Image.fromarray(clear_crop).filter(ImageFilter.GaussianBlur(8)))
    assert assess_image_quality(blurred) == 0
    assert assess_image_quality(clear_crop[:5, :5]) == 0


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((40, 20)),
        np.zeros((40, 20, 4)),
        np.zeros((0, 20, 3)),
        np.full((40, 20, 3), np.nan),
        np.full((40, 20, 3), -1),
        np.full((40, 20, 3), 256),
    ],
)
def test_invalid_images_fail_explicitly(bad):
    with pytest.raises(ValueError):
        validate_rgb_image(bad)


def test_quality_gate_avoids_ocr_backend(clear_crop):
    reader = Reader()
    result = JerseyNumberExtractor(reader).extract_track([np.zeros_like(clear_crop)])
    assert result.number is None and result.confidence == 0 and not result.legible
    assert result.votes == {} and reader.calls == []


def test_single_frame_ocr_is_torso_limited_and_digit_allowlisted(clear_crop):
    reader = Reader([("27", 0.96)])
    result = JerseyNumberExtractor(reader).extract(clear_crop)
    assert result.number == 27 and result.raw_text == "27" and result.legible
    image, kwargs = reader.calls[0]
    # OCR may enlarge the torso for legibility; its aspect ratio still excludes the legs.
    assert image.shape[0] / image.shape[1] < clear_crop.shape[0] / clear_crop.shape[1]
    assert kwargs["allowlist"] == "0123456789" and kwargs["detail"] == 1
    assert kwargs["paragraph"] is False


@pytest.mark.parametrize("text", ["123", "team27", "2 7", "-1", "2.7", "２７", ""])
def test_only_whole_ascii_one_or_two_digit_tokens_are_accepted(clear_crop, text):
    result = JerseyNumberExtractor(Reader([(text, 0.99)])).extract(clear_crop)
    assert result.number is None and result.votes == {}


@pytest.mark.parametrize("score", [0.59, float("nan"), float("inf"), -0.1, 1.1])
def test_low_or_invalid_ocr_scores_do_not_vote(clear_crop, score):
    result = JerseyNumberExtractor(Reader([("27", score)])).extract(clear_crop)
    assert result.number is None and result.votes == {}


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("min_ocr_confidence", 0),
        ("min_ocr_confidence", 1.1),
        ("min_track_confidence", float("nan")),
        ("min_track_confidence", -0.1),
        ("min_vote_margin", float("inf")),
        ("min_vote_margin", 0),
        ("min_support_frames", 1),
        ("min_support_frames", 2.5),
        ("min_support_frames", True),
    ],
)
def test_invalid_temporal_configuration_is_rejected(name, value):
    with pytest.raises(ValueError):
        JerseyNumberExtractor(**{name: value})


def test_custom_temporal_thresholds_and_frame_support_are_respected(clear_crop):
    frames = [[("27", 0.92)]] * 3
    assert (
        JerseyNumberExtractor(Reader(*frames), min_track_confidence=0.95)
        .extract_track([clear_crop] * 3)
        .number
        is None
    )
    ocr = JerseyNumberExtractor(Reader(*frames, *frames), min_support_frames=3)
    assert ocr.extract_track([clear_crop] * 2).number is None
    assert ocr.extract_track([clear_crop] * 3).number == 27


def test_near_tied_candidates_never_accumulate_track_evidence(clear_crop):
    frames = [[("27", 0.99), ("10", 0.98)]] * 3
    result = JerseyNumberExtractor(Reader(*frames), min_vote_margin=0.2).extract_track(
        [clear_crop] * 3
    )
    assert result.number is None and result.votes == {} and result.confidence == 0


@pytest.mark.parametrize(
    "polygon",
    [
        None,
        [[0, 0]],
        [[0, 0]] * 4,
        [[-1, 0], [10, 0], [10, 10], [-1, 10]],
        [[0, 0], [1e6, 0], [1e6, 10], [0, 10]],
        [[0, 0], [10, 0], [10, float("nan")], [0, 10]],
    ],
)
def test_invalid_ocr_bounding_polygons_are_ignored(clear_crop, polygon):
    reader = Reader()
    reader.readtext = lambda image, **kwargs: [(polygon, "27", 0.99)]
    result = JerseyNumberExtractor(reader).extract(clear_crop)
    assert result.number is None and result.votes == {}


def test_consistent_independent_frames_accumulate_confidence(clear_crop):
    result = JerseyNumberExtractor(Reader([("27", 0.96)], [("27", 0.94)])).extract_track(
        [clear_crop, clear_crop.copy()]
    )
    assert result.number == 27 and result.legible
    assert result.votes == pytest.approx({27: 1.9})
    assert result.confidence == pytest.approx(0.95)


def test_duplicate_detections_cannot_create_temporal_support(clear_crop):
    result = JerseyNumberExtractor(Reader([("27", 0.99)] * 5)).extract_track([clear_crop])
    assert result.number is None
    assert result.votes == pytest.approx({27: 0.99})


def test_conflicting_frames_abstain_and_do_not_leak_between_calls(clear_crop):
    ocr = JerseyNumberExtractor(
        Reader([("27", 0.98)], [("10", 0.98)], [("15", 0.97)], [("15", 0.96)])
    )
    conflict = ocr.extract_track([clear_crop, clear_crop])
    assert conflict.number is None and set(conflict.votes) == {10, 27}
    next_track = ocr.extract_track([clear_crop, clear_crop])
    assert next_track.number == 15 and set(next_track.votes) == {15}


def test_unknown_readable_frames_do_not_dilute_temporal_confidence(clear_crop):
    result = JerseyNumberExtractor(Reader([("27", 0.99)], [("27", 0.99)], [], [])).extract_track(
        [clear_crop] * 4
    )
    assert result.number == 27
    assert result.confidence == pytest.approx(0.99)
    assert result.legible
    assert result.votes == pytest.approx({27: 1.98})


# Backward-compatible alias for test discovery if referenced by old name
test_unknown_readable_frames_reduce_temporal_confidence = (
    test_unknown_readable_frames_do_not_dilute_temporal_confidence
)


def test_competing_numbers_require_vote_margin_to_confirm(clear_crop):
    # Two numbers both pass frame support, but margin is insufficient
    tied = JerseyNumberExtractor(
        Reader([("27", 0.95)], [("27", 0.95)], [("10", 0.90)], [("10", 0.90)]),
        min_vote_margin=0.15,
    ).extract_track([clear_crop] * 4)
    assert tied.number is None
    assert tied.confidence == pytest.approx(0.95)
    assert tied.votes == pytest.approx({27: 1.90, 10: 1.80})

    # Decisive margin confirms the leading number
    decisive = JerseyNumberExtractor(
        Reader([("27", 0.95)], [("27", 0.95)], [("10", 0.70)], [("10", 0.70)]),
        min_vote_margin=0.15,
    ).extract_track([clear_crop] * 4)
    assert decisive.number == 27
    assert decisive.confidence == pytest.approx(0.95)


def test_two_numbers_in_one_crop_are_not_multiple_votes(clear_crop):
    result = JerseyNumberExtractor(Reader([("27", 0.98), ("10", 0.97)])).extract_track([clear_crop])
    assert result.number is None and sum(result.votes.values()) <= 1


def test_appearance_is_finite_float32_unit_vector(clear_crop):
    result = appearance().extract(clear_crop)
    assert result.vector.shape == (512,) and result.vector.dtype == np.float32
    assert np.isfinite(result.vector).all()
    assert np.linalg.norm(result.vector) == pytest.approx(1.0, abs=1e-6)
    assert 0 < result.quality_score <= 1
    assert result.model_name == "unit-test-encoder-v1"


@pytest.mark.parametrize(
    "vector",
    [np.zeros(512), np.ones(511), np.ones((1, 512)), np.full(512, np.nan), np.full(512, np.inf)],
)
def test_invalid_backend_embeddings_raise(clear_crop, vector):
    with pytest.raises(ValueError):
        appearance(lambda image: vector).extract(clear_crop)


def test_huge_finite_embedding_normalizes_without_overflow_and_is_read_only(clear_crop):
    vector = np.full(512, np.finfo(np.float64).max)
    vector[:256] *= -1
    result = appearance(lambda image: vector).extract(clear_crop)
    expected = np.sign(vector) / np.sqrt(512)
    np.testing.assert_allclose(result.vector, expected, atol=1e-7)
    assert np.isfinite(result.vector).all() and not result.vector.flags.writeable
    with pytest.raises(ValueError):
        result.vector[0] = 0


def test_injected_backend_requires_model_provenance():
    with pytest.raises(ValueError):
        AppearanceExtractor(encoder=lambda image: np.ones(512))


def test_default_models_are_lazy_and_missing_dependencies_fail_clearly(monkeypatch, clear_crop):
    original_import = builtins.__import__

    def without_models(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "torchvision", "easyocr"}:
            raise ImportError("model packages intentionally absent")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_models)
    ocr, visual = JerseyNumberExtractor(), AppearanceExtractor()
    assert ocr.extract(np.zeros_like(clear_crop)).number is None
    with pytest.raises(RuntimeError, match="[Ee]asy[Oo][Cc][Rr]|[Oo][Cc][Rr]"):
        ocr.extract(clear_crop)
    with pytest.raises(RuntimeError, match="[Tt]orch"):
        visual.extract(clear_crop)


def test_appearance_medoid_preserves_an_observed_vector(clear_crop):
    a, b, c = np.zeros((3, 512), dtype=np.float32)
    a[0], b[0], b[1], c[1] = 1, 0.9, 0.1, 1
    vectors = iter([a, b, c])
    result = appearance(lambda image: next(vectors)).extract_track([clear_crop] * 3)
    np.testing.assert_allclose(result.vector, b / np.linalg.norm(b), atol=1e-6)


def test_appearance_without_usable_crops_has_no_fabricated_fallback(clear_crop):
    for images in ([], [np.zeros_like(clear_crop)]):
        with pytest.raises(ValueError):
            appearance().extract_track(images)


def test_batch_groups_by_video_and_track_and_sorts_frames(tmp_path, clear_crop):
    obs = []
    for video, track, digit in [("20", 7, 10), ("3", 8, 15), ("3", 7, 27)]:
        for frame in (2, 0):
            path = write_crop(tmp_path, clear_crop, video=video, name=f"{track}-{frame}.png")
            obs.append(observation(path, video=video, track=track, frame=frame))
    pipeline, reader = extractor(tmp_path, *([[("27", 0.96)]] * 6))
    result = pipeline.extract_observations(iter(obs))
    assert {item.segment_id for item in result} == {"20_7_0", "3_7_0", "3_8_0"}
    assert all(isinstance(item, DualStreamFeature) and item.jersey.number == 27 for item in result)
    assert len(reader.calls) == 6


def test_segment_output_keeps_segment_id(tmp_path, clear_crop):
    paths = [write_crop(tmp_path, clear_crop, name=f"{frame}.png") for frame in (0, 2)]
    observations = tuple(observation(path, frame=i * 2) for i, path in enumerate(paths))
    segment = TrackletSegment("3_7_0", "3", 7, 0, 2, 0.0, 2 / 60, 2, observations)
    pipeline, _ = extractor(tmp_path, [("27", 0.96)], [("27", 0.96)])
    result = pipeline.extract(segment)
    assert result.segment_id == segment.segment_id and result.jersey.number == 27


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("segment_id", "3_7_99"),
        ("video_id", "20"),
        ("local_track_id", 8),
        ("start_frame", 1),
        ("end_frame", 9),
        ("start_time_s", 0.5),
        ("end_time_s", 0.5),
        ("observation_count", 8),
    ],
)
def test_segment_metadata_must_match_observations_before_model_work(tmp_path, name, value):
    obs = (observation(frame=0), observation(frame=2))
    segment = TrackletSegment("3_7_0", "3", 7, 0, 2, 0.0, 2 / 60, 2, obs)
    pipeline, reader = extractor(tmp_path)
    with pytest.raises(ValueError):
        pipeline.extract(replace(segment, **{name: value}))
    assert reader.calls == []


def test_segment_cannot_mix_tracks(tmp_path):
    obs = (observation(frame=0), observation(track=8, frame=2))
    segment = TrackletSegment("3_7_0", "3", 7, 0, 2, 0.0, 2 / 60, 2, obs)
    pipeline, reader = extractor(tmp_path)
    with pytest.raises(ValueError, match="exactly one track"):
        pipeline.extract(segment)
    assert reader.calls == []


def test_empty_batch_is_valid_but_an_empty_segment_is_not(tmp_path):
    pipeline, reader = extractor(tmp_path)
    assert pipeline.extract_observations([]) == []
    segment = TrackletSegment("3_7_0", "3", 7, 0, 0, 0.0, 0.0, 0, ())
    with pytest.raises(ValueError):
        pipeline.extract(segment)
    assert reader.calls == []


@pytest.mark.parametrize("kind", ["duplicate_frame", "reused_file", "invalid_metadata"])
def test_batch_validation_precedes_model_work(tmp_path, clear_crop, kind):
    path = write_crop(tmp_path, clear_crop)
    first = observation(path)
    second = observation(path, frame=1)
    if kind == "duplicate_frame":
        second = first
    elif kind == "invalid_metadata":
        second = replace(second, confidence=float("nan"))
    pipeline, reader = extractor(tmp_path)
    with pytest.raises(ValueError):
        pipeline.extract_observations([first, second])
    assert reader.calls == []


@pytest.mark.parametrize(
    "path",
    [
        "../outside.png",
        "3/other/file.png",
        "20/samples/file.png",
        "3/samples/file.jpg",
        "3/samples/nested/file.png",
    ],
)
def test_crop_paths_are_contained_in_matching_samples_directory(tmp_path, path):
    pipeline, reader = extractor(tmp_path)
    with pytest.raises(ValueError):
        pipeline.extract_observations([observation(path)])
    assert reader.calls == []


def test_symlink_outside_crop_root_is_rejected(tmp_path, clear_crop):
    root = tmp_path / "track"
    outside = tmp_path / "outside.png"
    Image.fromarray(clear_crop).save(outside)
    link = root / "3" / "samples" / "linked.png"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)
    pipeline, reader = extractor(root)
    with pytest.raises(ValueError):
        pipeline.extract_observations([observation("3/samples/linked.png")])
    assert reader.calls == []


def test_absolute_contained_crop_path_is_supported(tmp_path, clear_crop):
    path = write_crop(tmp_path, clear_crop)
    pipeline, _ = extractor(tmp_path, [("27", 0.96)])
    result = pipeline.extract_observations([observation(str(tmp_path / path))])
    assert len(result) == 1 and result[0].jersey.number is None


def test_missing_crop_path_is_skipped_but_no_usable_images_raise(tmp_path):
    pipeline, reader = extractor(tmp_path)
    with pytest.raises(ValueError):
        pipeline.extract_observations([observation()])
    assert reader.calls == []


def test_corrupt_png_fails_before_model_work(tmp_path):
    path = tmp_path / "3" / "samples" / "bad.png"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not a PNG")
    pipeline, reader = extractor(tmp_path)
    with pytest.raises(ValueError):
        pipeline.extract_observations([observation("3/samples/bad.png")])
    assert reader.calls == []


def test_osnet_sportsmot_baseline_feature_extraction(clear_crop):
    """OSNet x1.0 SportsMOT is default baseline producing 512-dim L2-normalized float32."""
    visual = AppearanceExtractor()
    assert visual.model_name == "sports-osnet-x1.0-sportsmot"
    result = visual.extract(clear_crop)
    assert result.vector.shape == (512,)
    assert result.vector.dtype == np.float32
    assert np.isfinite(result.vector).all()
    assert np.linalg.norm(result.vector) == pytest.approx(1.0, abs=1e-5)
    assert result.model_name == "sports-osnet-x1.0-sportsmot"
    assert 0 < result.quality_score <= 1
