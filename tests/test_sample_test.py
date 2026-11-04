"""The sample boundary checks input integrity without mutating upstream evidence."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.adapters.sample_test import SampleTestError, load_sample_test
from app.contracts.evidence import EvidenceBatch, FileFeatureReference
from app.settings import load_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_ROOT = PROJECT_ROOT.parent / "test"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def arrays(root, clip="alpha"):
    with np.load(root / "target" / clip / "features.npz", allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def refresh_contract(root, clip="alpha"):
    directory = root / "target" / clip
    contract = read_json(directory / "feature_contract.json")
    contract["feature_sha256"] = sha256(directory / "features.npz")
    contract["protocol"]["samples_sha256"] = sha256(directory / "samples.jsonl")
    contract["sample_count"] = len(read_rows(directory / "samples.jsonl"))
    write_json(directory / "feature_contract.json", contract)


def replace_arrays(root, data, clip="alpha"):
    np.savez(root / "target" / clip / "features.npz", **data)
    refresh_contract(root, clip)


def settings_for(root, config_dir):
    config = (PROJECT_ROOT / "examples/sample-test.toml").read_text(encoding="utf-8")
    config = config.replace('input_root = "../../test"', f'input_root = "{root.as_posix()}"')
    config = config.replace('runtime_root = "../var"', 'runtime_root = "runtime-not-created"')
    path = config_dir / "settings.toml"
    path.write_text(config, encoding="utf-8")
    return load_settings(path)


def make_input(root, *, parts=3, dimension=5):
    root.mkdir()
    (root / "video").mkdir()
    annotation_rows = []
    for clip, track in [("alpha", 12), ("beta", 9)]:
        directory = root / "target" / clip
        (directory / "samples").mkdir(parents=True)
        (root / "video" / f"{clip}.mp4").write_bytes(b"original video fixture")
        (directory / "annotated.mp4").write_bytes(b"review video fixture")
        write_json(
            directory / "video_meta.json",
            {
                "fps": 2.0,
                "frame_count": 5,
                "height": 20,
                "input_path": f"video/{clip}.mp4",
                "processed_frames": 5,
                "tracker": "MOTIP SportsMOT",
                "tracker_config": {"det_thresh": 0.3},
                "video_id": clip,
                "width": 30,
            },
        )
        write_json(
            directory / "selection.json",
            {
                "video_id": clip,
                "goal_time_s": 1.5,
                "event_jersey_number": "00",
                "source_video": f"video/{clip}.mp4",
                "annotated_video": f"target/{clip}/annotated.mp4",
                "coordinate_system": (
                    "original pixels, XYXY; frame index zero-based; bounds inclusive"
                ),
                "track_segments": [
                    {
                        "track_id": track,
                        "start_frame": 0,
                        "end_frame": 4,
                        "quality_status": "needs_review" if clip == "beta" else "accepted",
                        "quality_review_reasons": "abrupt_motion_review" if clip == "beta" else "",
                    }
                ],
                "selection_note": "Human selection; identity remains unconfirmed.",
            },
        )
        observations = [
            {
                "association_score": None,
                "bbox_xyxy": [1.0, 2.0, 12.0, 16.0],
                "category_id": 0,
                "det_score": 0.9,
                "frame_idx": frame,
                "source": "motip_sportsmot",
                "timestamp_s": frame / 2,
                "track_id": track,
                "video_id": clip,
            }
            for frame in [0, 2, 4]
        ]
        write_rows(directory / "tracks.jsonl", observations)
        samples = []
        for i, frame in enumerate([4, 0]):
            image = directory / "samples" / f"sample-{i}.png"
            Image.new("RGB", (11, 14), color=(i * 20, 80, 90)).save(image)
            samples.append(
                {
                    "bbox_xyxy": [1.0, 2.0, 12.0, 16.0],
                    "crop_path": str(image.relative_to(root)),
                    "crop_sha256": sha256(image),
                    "crop_xyxy": [1, 2, 12, 16],
                    "det_score": 0.9,
                    "frame_idx": frame,
                    "sample_id": f"{clip}-track-{track}-frame-{frame}",
                    "sampling_quality": 9.0,
                    "sharpness_laplacian_var": 20.0,
                    "timestamp_s": frame / 2,
                    "track_id": track,
                    "video_id": clip,
                }
            )
        write_rows(directory / "samples.jsonl", samples)
        embeddings = np.zeros((2, parts, dimension), dtype=np.float32)
        embeddings[:, :, 0] = 1.0
        np.savez(
            directory / "features.npz",
            embeddings=embeddings,
            visibility_scores=np.ones((2, parts), dtype=np.float32),
            sample_ids=np.array([sample["sample_id"] for sample in samples]),
            track_ids=np.array([track, track], dtype=np.int64),
            frame_indices=np.array([4, 0], dtype=np.int64),
        )
        write_json(
            directory / "feature_contract.json",
            {
                "aggregation": "per-sample; no cross-track or cross-video identity merge",
                "color_convention": "OpenCV BGR, existing KPR adapter",
                "distance": "euclidean_sqrt_visibility_mean_div2",
                "feature_dimension": dimension,
                "feature_sha256": sha256(directory / "features.npz"),
                "min_common_parts": 2,
                "normalize_feature": True,
                "part_count": parts,
                "protocol": {
                    "checkpoint_sha256": "a" * 64,
                    "config_sha256": "b" * 64,
                    "prompt_mode": "none",
                    "samples_sha256": sha256(directory / "samples.jsonl"),
                },
                "sample_count": len(samples),
                "schema_version": 1,
                "status": "extracted_not_accuracy_evaluated",
                "test_embeddings": ["bn_foreg", "parts"],
            },
        )
        annotation_rows.append([clip, f"{clip}.mp4", "1.5", "00", "", "", f"T{track}", ""])
    with (root / "annotations.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "video_id",
                "filename",
                "goal_time_s",
                "jersey_number",
                "team_id",
                "person_gt_id",
                "track_id",
                "target_frame_ranges",
            ]
        )
        writer.writerows(annotation_rows)
    return root


@pytest.fixture
def input_data(tmp_path):
    root = make_input(tmp_path / "input")
    return root, settings_for(root, tmp_path)


def load(settings, *, clip_ids=None, batch_id="test-batch"):
    return load_sample_test(settings, batch_id=batch_id, clip_ids=clip_ids)


def business_keys(batch):
    evidence = batch.identity_evidence
    records = [*evidence.tracks, *evidence.samples, *(batch.events or [])]
    return {
        "clips": sorted(clip.clip_id for clip in batch.clips),
        "segments": sorted(segment.segment_id for segment in evidence.segments),
        "samples": sorted(sample.sample_id for sample in evidence.samples),
        "events": sorted((event.event_id, event.source_event_key) for event in batch.events),
        "subjects": sorted(
            subject.subject_id for event in batch.events for subject in event.subjects
        ),
        "spaces": sorted(space.feature_space_id for space in evidence.feature_spaces),
        "feature_files": sorted(file.feature_file_id for file in evidence.feature_files),
        "source_keys": sorted(
            (
                record.source.producer,
                record.source.record_type,
                record.source.source_record_id,
                record.source.evidence_version,
                record.source.content_hash,
            )
            for record in records
        ),
    }


def test_variable_shape_original_rows_missing_frames_and_event_only_numbers(input_data):
    root, settings = input_data
    batch = load(settings)
    evidence = batch.identity_evidence
    assert batch.schema_version == "2"
    assert batch.match_id == settings.match.match_id
    assert len(batch.clips) == 2
    assert len(evidence.segments) == 2
    assert len(evidence.samples) == 4
    assert len(evidence.feature_spaces) == 1
    assert evidence.feature_spaces[0].shape == [3, 5]
    assert all(track.observed_frame_indices == [0, 2, 4] for track in evidence.tracks)
    assert {track.track_id for track in evidence.tracks} == {"12", "9"}
    for file in evidence.feature_files:
        assert file.values_file == file.visibility_file
        assert file.sample_count == 2
    for sample in evidence.samples:
        feature = sample.features[0]
        assert isinstance(feature, FileFeatureReference)
        original = read_rows(root / "target" / sample.clip_id / "samples.jsonl")
        assert original[feature.row_index]["frame_idx"] == sample.frame_idx
        assert sample.appearance is None or sample.appearance.jersey_number is None
        assert sample.source.locations
        assert sample.source.content_hash
    assert all(event.subjects[0].event_jersey_number == "00" for event in batch.events)
    assert all(event.points is None and event.game_clock is None for event in batch.events)
    assert all(event.time.kind == "point" and event.outcome == "made" for event in batch.events)
    assert all(clip.source_mapping is None for clip in batch.clips)
    assert batch.warnings
    assert EvidenceBatch.model_validate_json(batch.model_dump_json()) == batch
    assert not settings.paths.runtime_root.exists()


def test_ids_are_stable_across_batches_csv_order_and_directory_order(input_data, monkeypatch):
    root, settings = input_data
    first = business_keys(load(settings))
    assert business_keys(load(settings, batch_id="another-batch")) == first
    csv_path = root / "annotations.csv"
    header, *rows = csv_path.read_text().splitlines()
    csv_path.write_text("\n".join([header, *reversed(rows)]) + "\n")
    original_iterdir = Path.iterdir

    def reverse_iterdir(path):
        return iter(reversed(list(original_iterdir(path))))

    monkeypatch.setattr(Path, "iterdir", reverse_iterdir)
    assert business_keys(load(settings)) == first


def test_feature_space_ignores_sample_count_paths_and_file_hashes(input_data, tmp_path):
    root, settings = input_data
    expected = load(settings).identity_evidence.feature_spaces[0].feature_space_id
    data = arrays(root, "beta")
    replace_arrays(root, {key: value[:1] for key, value in data.items()}, "beta")
    samples = root / "target" / "beta" / "samples.jsonl"
    write_rows(samples, read_rows(samples)[:1])
    refresh_contract(root, "beta")
    assert load(settings).identity_evidence.feature_spaces[0].feature_space_id == expected
    other_root = make_input(tmp_path / "other-input")
    assert (
        load(settings_for(other_root, tmp_path))
        .identity_evidence.feature_spaces[0]
        .feature_space_id
        == expected
    )


@pytest.mark.parametrize("field", ["checkpoint_sha256", "config_sha256", "prompt_mode"])
def test_feature_space_changes_with_model_config_or_preprocessing(input_data, field):
    root, settings = input_data
    expected = (
        load(settings, clip_ids=["alpha"]).identity_evidence.feature_spaces[0].feature_space_id
    )
    path = root / "target" / "alpha" / "feature_contract.json"
    contract = read_json(path)
    contract["protocol"][field] = "c" * 64 if field.endswith("sha256") else "different-prompt"
    write_json(path, contract)
    actual = load(settings, clip_ids=["alpha"]).identity_evidence.feature_spaces[0].feature_space_id
    assert actual != expected


def test_subset_references_only_selected_evidence(input_data):
    _, settings = input_data
    whole = load(settings)
    subset = load(settings, clip_ids=["beta"])
    assert [clip.clip_id for clip in subset.clips] == ["beta"]
    assert len(subset.events) == 1
    assert len(subset.identity_evidence.samples) == 2
    segment_ids = {segment.segment_id for segment in subset.identity_evidence.segments}
    assert all(sample.segment_id in segment_ids for sample in subset.identity_evidence.samples)
    file_ids = {file.feature_file_id for file in subset.identity_evidence.feature_files}
    assert all(
        feature.feature_file_id in file_ids
        for sample in subset.identity_evidence.samples
        for feature in sample.features
    )
    assert {event.event_id for event in subset.events} <= {event.event_id for event in whole.events}
    assert all(
        "/alpha/" not in file.path and "alpha.mp4" not in file.path for file in subset.sources
    )
    with pytest.raises(SampleTestError, match="absent|unknown|missing|not found|not.*exist"):
        load(settings, clip_ids=["absent"])


@pytest.mark.parametrize(
    "relative_path",
    [
        "annotations.csv",
        "target/alpha/selection.json",
        "target/alpha/video_meta.json",
        "target/alpha/tracks.jsonl",
        "target/alpha/samples.jsonl",
        "target/alpha/features.npz",
        "target/alpha/feature_contract.json",
        "target/alpha/samples/sample-0.png",
        "video/alpha.mp4",
    ],
)
def test_missing_required_files_fail_with_locator(input_data, relative_path):
    root, settings = input_data
    (root / relative_path).unlink()
    with pytest.raises(SampleTestError) as error:
        load(settings)
    assert Path(relative_path).name in str(error.value)


@pytest.mark.parametrize("kind", ["features", "samples", "image"])
def test_wrong_declared_hash_fails(input_data, kind):
    root, settings = input_data
    directory = root / "target" / "alpha"
    if kind == "image":
        (directory / "samples" / "sample-0.png").write_bytes(b"tampered image")
    elif kind == "features":
        with (directory / "features.npz").open("ab") as file:
            file.write(b"changed bytes")
    else:
        path = directory / "samples.jsonl"
        path.write_text(
            path.read_text().replace('"sampling_quality": 9.0', '"sampling_quality": 9.00', 1)
        )
    with pytest.raises(SampleTestError, match="[Hh]ash|sha256"):
        load(settings)


@pytest.mark.parametrize("key", ["sample_ids", "track_ids", "frame_indices"])
def test_npz_rows_must_correspond_to_jsonl(input_data, key):
    root, settings = input_data
    data = arrays(root)
    if key == "track_ids":
        data[key][0] += 1
    else:
        data[key] = data[key][::-1]
    replace_arrays(root, data)
    with pytest.raises(SampleTestError, match=key):
        load(settings)


@pytest.mark.parametrize(
    "mutation,field",
    [
        ("missing_visibility", "visibility"),
        ("missing_embeddings", "embeddings"),
        ("missing_sample_ids", "sample_ids"),
        ("wrong_index_shape", "sample_ids|shape"),
        ("boolean_track_ids", "track_ids|dtype|integer"),
        ("wrong_embedding_shape", "embeddings"),
        ("wrong_visibility_shape", "visibility"),
        ("float64_embeddings", "float32|dtype"),
        ("float64_visibility", "float32|dtype"),
        ("nonfinite_embeddings", "finite|NaN|Inf"),
        ("nonfinite_visibility", "finite|NaN|Inf"),
        ("visibility_outside_range", "visibility"),
        ("unnormalized_visible_part", "norm|L2|l2"),
        ("object_sample_ids", "pickle|object"),
        ("float_frame_indices", "frame_indices|dtype|integer"),
    ],
)
def test_invalid_npz_arrays_fail_after_file_hash_verification(input_data, mutation, field):
    root, settings = input_data
    data = arrays(root)
    if mutation == "missing_visibility":
        del data["visibility_scores"]
    elif mutation == "missing_embeddings":
        del data["embeddings"]
    elif mutation == "missing_sample_ids":
        del data["sample_ids"]
    elif mutation == "wrong_index_shape":
        data["sample_ids"] = data["sample_ids"][:, None]
    elif mutation == "boolean_track_ids":
        data["track_ids"] = data["track_ids"].astype(bool)
    elif mutation == "wrong_embedding_shape":
        data["embeddings"] = data["embeddings"][:, :, :-1]
    elif mutation == "wrong_visibility_shape":
        data["visibility_scores"] = data["visibility_scores"][:, :-1]
    elif mutation == "float64_embeddings":
        data["embeddings"] = data["embeddings"].astype(np.float64)
    elif mutation == "float64_visibility":
        data["visibility_scores"] = data["visibility_scores"].astype(np.float64)
    elif mutation == "nonfinite_embeddings":
        data["embeddings"][0, 0, 0] = np.nan
    elif mutation == "nonfinite_visibility":
        data["visibility_scores"][0, 0] = np.inf
    elif mutation == "visibility_outside_range":
        data["visibility_scores"][0, 0] = 1.1
    elif mutation == "unnormalized_visible_part":
        data["embeddings"][0, 0, 0] = 2.0
    elif mutation == "object_sample_ids":
        data["sample_ids"] = data["sample_ids"].astype(object)
    elif mutation == "float_frame_indices":
        data["frame_indices"] = data["frame_indices"].astype(np.float32)
    replace_arrays(root, data)
    with pytest.raises(SampleTestError, match=field):
        load(settings)


def test_invisible_parts_need_not_be_normalized(input_data):
    root, settings = input_data
    data = arrays(root)
    data["visibility_scores"][0, 0] = 0.0
    data["embeddings"][0, 0] = 0.0
    replace_arrays(root, data)
    assert len(load(settings).identity_evidence.samples) == 4


@pytest.mark.parametrize("kind", ["traversal", "absolute", "symlink"])
def test_referenced_image_cannot_escape_allowed_root(input_data, tmp_path, kind):
    root, settings = input_data
    outside = tmp_path / "outside.png"
    Image.new("RGB", (11, 14)).save(outside)
    if kind == "traversal":
        path = "../outside.png"
    elif kind == "absolute":
        path = str(outside)
    else:
        link = root / "escape.png"
        link.symlink_to(outside)
        path = "escape.png"
    index = root / "target" / "alpha" / "samples.jsonl"
    rows = read_rows(index)
    rows[0]["crop_path"] = path
    rows[0]["crop_sha256"] = sha256(outside)
    write_rows(index, rows)
    refresh_contract(root)
    with pytest.raises(SampleTestError, match="root|outside|escape|travers|allowed"):
        load(settings)


def test_npz_symlink_cannot_escape_allowed_root(input_data, tmp_path):
    root, settings = input_data
    npz = root / "target" / "alpha" / "features.npz"
    outside = tmp_path / "outside.npz"
    npz.rename(outside)
    npz.symlink_to(outside)
    with pytest.raises(SampleTestError, match="root|outside|escape|allowed"):
        load(settings)


@pytest.mark.parametrize("clip", ["../outside", "/absolute", "alpha/../../outside"])
def test_clip_selection_cannot_be_used_for_path_traversal(input_data, clip):
    _, settings = input_data
    with pytest.raises(SampleTestError):
        load(settings, clip_ids=[clip])


@pytest.mark.parametrize("kind", ["sample_time", "track_frame", "segment_frame", "event_time"])
def test_invalid_time_and_frame_ranges_are_fatal(input_data, kind):
    root, settings = input_data
    directory = root / "target" / "alpha"
    if kind == "sample_time":
        path = directory / "samples.jsonl"
        rows = read_rows(path)
        rows[0]["timestamp_s"] = 99.0
        write_rows(path, rows)
        refresh_contract(root)
    elif kind == "track_frame":
        path = directory / "tracks.jsonl"
        rows = read_rows(path)
        rows[0]["frame_idx"] = 5
        write_rows(path, rows)
    elif kind == "segment_frame":
        path = directory / "selection.json"
        value = read_json(path)
        value["track_segments"][0]["end_frame"] = 5
        write_json(path, value)
    else:
        path = root / "annotations.csv"
        path.write_text(path.read_text().replace("alpha,alpha.mp4,1.5,", "alpha,alpha.mp4,99,"))
    with pytest.raises(SampleTestError) as error:
        load(settings)
    assert path.name in str(error.value)


def test_jsonl_parse_errors_report_file_and_line(input_data):
    root, settings = input_data
    path = root / "target" / "alpha" / "tracks.jsonl"
    lines = path.read_text().splitlines()
    lines[1] = "{this is not JSON}"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(SampleTestError) as error:
        load(settings)
    assert "tracks.jsonl" in str(error.value)
    assert "2" in str(error.value)


def test_excluded_identity_directory_is_never_ingested(input_data):
    root, settings = input_data
    before = business_keys(load(settings))
    directory = root / "identity" / "alpha"
    directory.mkdir(parents=True)
    (directory / "samples.jsonl").write_text("malformed excluded evidence")
    (directory / "features.npz").write_bytes(b"not an archive")
    assert business_keys(load(settings)) == before


@pytest.mark.sample_data
def test_real_sample_statistics_semantics_and_read_only_tree():
    if not REAL_ROOT.is_dir():
        pytest.skip("read-only companion sample is unavailable")
    before = {
        str(path.relative_to(REAL_ROOT)): sha256(path)
        for path in REAL_ROOT.rglob("*")
        if path.is_file()
    }
    try:
        batch = load(load_settings(PROJECT_ROOT / "examples/sample-test.toml"))
        evidence = batch.identity_evidence
        assert (
            len(batch.clips),
            len(evidence.segments),
            len(evidence.samples),
            len(batch.events),
            len(evidence.feature_spaces),
        ) == (10, 11, 118, 10, 1)
        assert evidence.feature_spaces[0].shape == [6, 128]
        twenty = next(event for event in batch.events if event.clip_id == "20")
        assert twenty.time.kind == "point" and twenty.time.time_s == 3.5
        sixty_eight = [event for event in batch.events if event.clip_id == "68"]
        assert len(sixty_eight) == 1
        assert len(sixty_eight[0].subjects) == 1
        assert sixty_eight[0].subjects[0].role == "shooter"
        assert len(sixty_eight[0].subjects[0].references) == 2
        assert {
            (segment.track_id, segment.frame_start, segment.frame_end_exclusive)
            for segment in evidence.segments
            if segment.clip_id == "68"
        } == {("8", 0, 961), ("13", 1410, 1680)}
        assert all(
            sample.appearance is None or sample.appearance.jersey_number is None
            for sample in evidence.samples
        )
        assert all("identity/" not in source.path for source in batch.sources)
        warnings = " ".join(warning.message for warning in batch.warnings)
        assert "needs_review" in warnings or "abrupt_motion_review" in warnings
        assert EvidenceBatch.model_validate_json(batch.model_dump_json()) == batch
    finally:
        after = {
            str(path.relative_to(REAL_ROOT)): sha256(path)
            for path in REAL_ROOT.rglob("*")
            if path.is_file()
        }
        assert after == before


def test_record_fingerprints_change_only_with_their_own_annotation(input_data):
    root, settings = input_data
    first = load(settings)
    path = root / "annotations.csv"
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
        fieldnames = list(rows[0])
    next(row for row in rows if row["video_id"] == "beta")["jersey_number"] = "01"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    selection_path = root / "target" / "beta" / "selection.json"
    selection = read_json(selection_path)
    selection["event_jersey_number"] = "01"
    write_json(selection_path, selection)
    second = load(settings)
    before = {event.clip_id: event for event in first.events}
    after = {event.clip_id: event for event in second.events}
    for clip in before:
        assert before[clip].event_id == after[clip].event_id
        assert before[clip].source_event_key == after[clip].source_event_key
        assert before[clip].source.source_record_id == after[clip].source.source_record_id
    assert before["alpha"].source.content_hash == after["alpha"].source.content_hash
    assert before["beta"].source.content_hash != after["beta"].source.content_hash
    assert (
        next(source for source in first.sources if source.path.endswith("annotations.csv")).sha256
        != next(
            source for source in second.sources if source.path.endswith("annotations.csv")
        ).sha256
    )
    assert [sample.source.content_hash for sample in first.identity_evidence.samples] == [
        sample.source.content_hash for sample in second.identity_evidence.samples
    ]


def test_sample_fingerprint_excludes_unrelated_npz_rows(input_data):
    root, settings = input_data
    first = load(settings)
    data = arrays(root)
    data["embeddings"][0, :, :] = 0.0
    data["embeddings"][0, :, 1] = 1.0
    replace_arrays(root, data)
    second = load(settings)
    before = {
        sample.frame_idx: sample
        for sample in first.identity_evidence.samples
        if sample.clip_id == "alpha"
    }
    after = {
        sample.frame_idx: sample
        for sample in second.identity_evidence.samples
        if sample.clip_id == "alpha"
    }
    assert set(before) == set(after) == {0, 4}
    assert before[4].sample_id == after[4].sample_id
    assert before[4].source.content_hash != after[4].source.content_hash
    assert before[0].sample_id == after[0].sample_id
    assert before[0].source.content_hash == after[0].source.content_hash
    assert first.identity_evidence.feature_spaces == second.identity_evidence.feature_spaces
    assert (
        next(
            file
            for file in first.identity_evidence.feature_files
            if "alpha" in file.values_file.path
        ).values_file.sha256
        != next(
            file
            for file in second.identity_evidence.feature_files
            if "alpha" in file.values_file.path
        ).values_file.sha256
    )


@pytest.mark.parametrize(
    "relative_path",
    [
        "video/alpha.mp4",
        "target/alpha/video_meta.json",
        "target/alpha/annotated.mp4",
        "target/alpha",
    ],
)
def test_video_metadata_or_target_symlink_cannot_escape_root(input_data, tmp_path, relative_path):
    root, settings = input_data
    path = root / relative_path
    outside = tmp_path / "outside-evidence"
    directory = path.is_dir()
    path.rename(outside)
    path.symlink_to(outside, target_is_directory=directory)
    with pytest.raises(SampleTestError, match="root|outside|escape|allowed"):
        load(settings)


def test_overflowing_number_in_extra_json_data_reports_file_and_line(input_data):
    root, settings = input_data
    path = root / "target" / "alpha" / "tracks.jsonl"
    text = path.read_text()
    path.write_text(text.replace("{", '{"upstream_extra_score": 1e999, ', 1))
    with pytest.raises(SampleTestError) as error:
        load(settings)
    assert "tracks.jsonl:1" in str(error.value)


def test_corrupt_npz_zip_is_reported_as_adapter_error(input_data):
    root, settings = input_data
    path = root / "target" / "alpha" / "features.npz"
    path.write_bytes(b"PK\x03\x04corrupt zip archive")
    refresh_contract(root)
    with pytest.raises(SampleTestError) as error:
        load(settings)
    assert "features.npz" in str(error.value)


@pytest.mark.parametrize("field", ["track_id", "frame_idx"])
@pytest.mark.parametrize("value", [[12], True, "12"])
def test_sample_index_fields_have_strict_scalar_types(input_data, field, value):
    root, settings = input_data
    path = root / "target" / "alpha" / "samples.jsonl"
    rows = read_rows(path)
    rows[0][field] = value
    write_rows(path, rows)
    refresh_contract(root)
    with pytest.raises(SampleTestError) as error:
        load(settings)
    assert "samples.jsonl:1" in str(error.value)
    assert field in str(error.value)


def test_nul_in_crop_path_is_reported_as_adapter_error(input_data):
    root, settings = input_data
    path = root / "target" / "alpha" / "samples.jsonl"
    rows = read_rows(path)
    rows[0]["crop_path"] = "target/alpha/samples/invalid\x00.png"
    write_rows(path, rows)
    refresh_contract(root)
    with pytest.raises(SampleTestError):
        load(settings)


def test_selection_quality_changes_corresponding_sample_fingerprint_and_locator(input_data):
    root, settings = input_data
    first = load(settings)
    path = root / "target" / "alpha" / "selection.json"
    selection = read_json(path)
    selection["track_segments"][0]["quality_status"] = "needs_review"
    selection["track_segments"][0]["quality_review_reasons"] = "duplicate_box_candidate"
    write_json(path, selection)
    second = load(settings)
    before = {sample.sample_id: sample for sample in first.identity_evidence.samples}
    after = {sample.sample_id: sample for sample in second.identity_evidence.samples}
    assert before.keys() == after.keys()
    sources = {source.file_id: source for source in second.sources}
    for sample_id, original in before.items():
        updated = after[sample_id]
        if original.clip_id == "alpha":
            assert original.source.content_hash != updated.source.content_hash
            assert original.tracking_quality != updated.tracking_quality
            assert "needs_review" in updated.tracking_quality.notes
            assert "duplicate_box_candidate" in updated.tracking_quality.notes
            assert any(
                sources[location.file_id].path.endswith("target/alpha/selection.json")
                for location in updated.source.locations
            )
        else:
            assert original.source.content_hash == updated.source.content_hash
            assert original.tracking_quality == updated.tracking_quality
