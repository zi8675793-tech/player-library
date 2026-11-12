"""Machine ingestion never needs identity answers or a previously reviewed gallery."""

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.adapters.machine import MachineInputError, load_machine, load_machine_input
from app.contracts.api import InitializeRequest, UpdateRequest
from app.identity.prepare import resolve_feature
from app.settings import load_settings
from app.storage.database import connect
from app.storage.snapshot import read_identity_snapshot
from app.workflows.initialize import initialize_library
from app.workflows.update import update_library

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def json_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def refresh_contract(root, clip):
    directory = root / "identity" / clip
    path = directory / "feature_contract.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["feature_sha256"] = digest(directory / "features.npz")
    data["protocol"]["samples_sha256"] = digest(directory / "samples.jsonl")
    write_json(path, data)


def feature_arrays(root, clip="alpha"):
    with np.load(root / "identity" / clip / "features.npz", allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def save_arrays(root, data, clip="alpha"):
    np.savez(root / "identity" / clip / "features.npz", **data)
    refresh_contract(root, clip)


def make_machine_input(root, *, clips=("alpha", "beta"), track_count=2, parts=4, dimension=7):
    """Build only upstream tracking, image and feature files, with no answer data."""
    (root / "video").mkdir(parents=True)
    for clip in clips:
        track_dir = root / "track" / clip
        identity_dir = root / "identity" / clip
        (track_dir / "samples").mkdir(parents=True)
        identity_dir.mkdir(parents=True)
        (root / "video" / f"{clip}.mp4").write_bytes(f"video fixture {clip}".encode())
        write_json(
            track_dir / "video_meta.json",
            {
                "fps": 2.0,
                "frame_count": 5,
                "height": 20,
                "width": 30,
                "processed_frames": 5,
                "tracker": "MOTIP SportsMOT",
                "tracker_config": {"det_thresh": 0.3},
                "input_path": f"video/{clip}.mp4",
                "video_id": clip,
            },
        )
        observations, tracks, samples = [], [], []
        for track in range(track_count):
            frames = [0, 2, 4]
            tracks.append([track, 3, 1.5, 0, 4, "accepted", ""])
            for frame in frames:
                observations.append(
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
                )
            # Deliberately nonchronological feature order; row index must be preserved.
            for i, frame in enumerate([4, 0]):
                image = track_dir / "samples" / f"T{track}-B{i}.png"
                Image.new("RGB", (11, 14), color=(i * 30, track * 20, 90)).save(image)
                samples.append(
                    {
                        "bbox_xyxy": [1.0, 2.0, 12.0, 16.0],
                        "crop_path": str(image.relative_to(root)),
                        "crop_sha256": digest(image),
                        "crop_xyxy": [1, 2, 12, 16],
                        "det_score": 0.9,
                        "frame_idx": frame,
                        "sample_id": f"{clip}-T{track}-F{frame}",
                        "sampling_quality": 9.0,
                        "sharpness_laplacian_var": 20.0,
                        "timestamp_s": frame / 2,
                        "track_id": track,
                        "video_id": clip,
                    }
                )
        write_rows(track_dir / "tracks.jsonl", observations)
        with (track_dir / "tracks.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "track_id",
                    "observations",
                    "observed_seconds",
                    "start_frame",
                    "end_frame",
                    "quality_status",
                    "review_reasons",
                ]
            )
            writer.writerows(tracks)
        write_rows(identity_dir / "samples.jsonl", samples)
        embeddings = np.zeros((len(samples), parts, dimension), dtype=np.float32)
        for row in range(len(samples)):
            embeddings[row, :, row % dimension] = 1.0
        visibility = np.ones((len(samples), parts), dtype=np.float32)
        if samples:
            visibility[0, 0] = 0.25
        np.savez(
            identity_dir / "features.npz",
            embeddings=embeddings,
            visibility_scores=visibility,
            sample_ids=np.array([sample["sample_id"] for sample in samples], dtype=str),
            track_ids=np.array([sample["track_id"] for sample in samples], dtype=np.int64),
            frame_indices=np.array([sample["frame_idx"] for sample in samples], dtype=np.int64),
        )
        write_json(
            identity_dir / "feature_contract.json",
            {
                "aggregation": "per-sample; no cross-track or cross-video identity merge",
                "color_convention": "OpenCV BGR, existing KPR adapter",
                "distance": "euclidean_sqrt_visibility_mean_div2",
                "feature_dimension": dimension,
                "feature_sha256": digest(identity_dir / "features.npz"),
                "min_common_parts": 2,
                "normalize_feature": True,
                "part_count": parts,
                "protocol": {
                    "checkpoint_sha256": "a" * 64,
                    "config_sha256": "b" * 64,
                    "prompt_mode": "none",
                    "samples_sha256": digest(identity_dir / "samples.jsonl"),
                },
                "sample_count": len(samples),
                "schema_version": 1,
                "status": "extracted_not_accuracy_evaluated",
                "test_embeddings": ["bn_foreg", "parts"],
            },
        )
    return root


def machine_settings(root, config_dir):
    config = f'''[paths]
input_root = "{root.as_posix()}"
runtime_root = "{(config_dir / "runtime").as_posix()}"
[input]
adapter = "machine"
[match]
match_id = "machine-fixture"
[matching]
min_common_parts = 2
max_templates_per_player = 6
auto_assign_enabled = false
[playback]
before_event_s = 3.0
after_event_s = 2.0
[server]
host = "127.0.0.1"
port = 8000
'''
    path = config_dir / "machine.toml"
    path.write_text(config, encoding="utf-8")
    return load_settings(path)


@pytest.mark.parametrize(
    ("clips", "track_count", "parts", "dimension"),
    [
        (("empty",), 0, 4, 9),
        (("solo",), 1, 2, 3),
        (("alpha", "beta", "third"), 3, 4, 7),
    ],
)
def test_machine_discovery_preserves_variable_shape_and_feature_rows(
    tmp_path, clips, track_count, parts, dimension
):
    root = make_machine_input(
        tmp_path / "input", clips=clips, track_count=track_count, parts=parts, dimension=dimension
    )
    settings = machine_settings(root, tmp_path)
    before = {str(path): digest(path) for path in root.rglob("*") if path.is_file()}
    batch, values = load_machine_input(settings, batch_id="first")
    assert batch == load_machine(settings, batch_id="first")
    evidence = batch.identity_evidence
    assert sorted(clip.clip_id for clip in batch.clips) == sorted(clips)
    assert len(evidence.tracks) == len(evidence.segments) == len(clips) * track_count
    assert len(evidence.samples) == len(values) == len(clips) * track_count * 2
    assert all(space.shape == [parts, dimension] for space in evidence.feature_spaces)
    assert not batch.events
    assert not (root / "annotations.csv").exists()
    assert not (root / "target").exists()
    spaces = {space.feature_space_id: space for space in evidence.feature_spaces}
    files = {file.feature_file_id: file for file in evidence.feature_files}
    assets = {asset.asset_id: asset for asset in batch.assets}
    for track in evidence.tracks:
        assert track.observed_frame_indices == [0, 2, 4]
        assert track.source.locations
    for sample in evidence.samples:
        reference = sample.features[0]
        feature = values[(reference.feature_file_id, reference.row_index)]
        arrays = feature_arrays(root, sample.clip_id)
        np.testing.assert_array_equal(
            np.asarray(feature.values).reshape(parts, dimension),
            arrays["embeddings"][reference.row_index],
        )
        assert feature.visibility == arrays["visibility_scores"][reference.row_index].tolist()
        assert resolve_feature(sample, spaces[reference.feature_space_id], values) == feature
        assert sample.frame_idx == int(arrays["frame_indices"][reference.row_index])
        assert sample.time_s == sample.frame_idx / 2
        image = assets[sample.image_asset_id].file
        assert image.sha256 == digest(root / image.path)
        assert sample.tracking_quality.clean_review == "unreviewed"
        assert sample.appearance is None or sample.appearance.jersey_number is None
        assert files[reference.feature_file_id].values_file.sha256
    for source in batch.sources:
        assert source.sha256 == digest(root / source.path)
        assert source.size_bytes == (root / source.path).stat().st_size
        assert Path(source.path).parts[0] in {"track", "identity", "video"}
    assert before == {str(path): digest(path) for path in root.rglob("*") if path.is_file()}
    assert not settings.paths.runtime_root.exists()


def test_machine_discovery_and_ids_do_not_depend_on_batch_or_selection_order(tmp_path):
    root = make_machine_input(tmp_path / "input", clips=("z", "alpha", "another"))
    settings = machine_settings(root, tmp_path)
    first, _ = load_machine_input(settings, batch_id="one", clip_ids=["z", "alpha"])
    second, _ = load_machine_input(settings, batch_id="two", clip_ids=["alpha", "z"])
    assert first.model_dump(exclude={"batch_id"}) == second.model_dump(exclude={"batch_id"})
    whole = load_machine(settings, batch_id="whole")
    assert {sample.sample_id for sample in first.identity_evidence.samples} <= {
        sample.sample_id for sample in whole.identity_evidence.samples
    }


@pytest.mark.parametrize(
    "failure",
    [
        "features_hash",
        "samples_hash",
        "image_hash",
        "sample_ids",
        "track_ids",
        "frame_indices",
        "object",
        "dtype",
        "time",
        "bbox",
        "schema_bool",
    ],
)
def test_machine_rejects_bad_correspondence_and_hashes(tmp_path, failure):
    root = make_machine_input(tmp_path / "input", clips=("alpha",))
    directory = root / "identity" / "alpha"
    if failure == "features_hash":
        with (directory / "features.npz").open("ab") as file:
            file.write(b"changed")
    elif failure == "samples_hash":
        with (directory / "samples.jsonl").open("a", encoding="utf-8") as file:
            file.write("\n")
    elif failure == "image_hash":
        (root / "track" / "alpha" / "samples" / "T0-B0.png").write_bytes(b"different")
    elif failure == "schema_bool":
        path = directory / "feature_contract.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["schema_version"] = True
        write_json(path, data)
    elif failure in {"time", "bbox"}:
        rows = json_rows(directory / "samples.jsonl")
        if failure == "time":
            rows[0]["timestamp_s"] = 1.0
        else:
            rows[0]["bbox_xyxy"][0] = 2.0
        write_rows(directory / "samples.jsonl", rows)
        refresh_contract(root, "alpha")
    else:
        arrays = feature_arrays(root)
        if failure == "object":
            arrays["sample_ids"] = arrays["sample_ids"].astype(object)
        elif failure == "dtype":
            arrays["embeddings"] = arrays["embeddings"].astype(np.float64)
        else:
            arrays[failure] = arrays[failure][::-1]
            if failure == "track_ids":
                arrays[failure][0] = 99
        save_arrays(root, arrays)
    with pytest.raises(MachineInputError):
        load_machine_input(machine_settings(root, tmp_path), batch_id="bad")


def test_machine_repeat_import_is_idempotent_and_tracks_are_not_confirmed(tmp_path):
    root = make_machine_input(tmp_path / "input")
    settings = machine_settings(root, tmp_path)
    first, values = load_machine_input(settings, batch_id="one")
    connection = connect(tmp_path / "fresh.sqlite3")
    try:
        report = initialize_library(
            connection,
            InitializeRequest(batch=first, expected_revision=0),
            feature_values=values,
            settings=settings,
        )
        before = read_identity_snapshot(connection, first.match_id)
        second = load_machine(settings, batch_id="another-handoff")
        repeated = update_library(
            connection,
            UpdateRequest(batch=second, expected_revision=report.revision),
            feature_values=values,
            settings=settings,
        )
        after = read_identity_snapshot(connection, first.match_id)
        assert repeated.revision == report.revision
        assert not repeated.data.changed
        assert before == after
        assert not after.gallery.players
        assert not after.gallery.templates
        assert len(after.gallery.assignments) == len(after.gallery.evidence.segments)
        assert all(assignment.status == "unassigned" for assignment in after.gallery.assignments)
        assert all(assignment.player_id is None for assignment in after.gallery.assignments)
        assert all(resolution.decision == "pending" for resolution in report.data.resolutions)
    finally:
        connection.close()


@pytest.mark.parametrize(
    "reference",
    ["annotations.csv", "target/alpha/video.mp4", "../annotations.csv", "history.sqlite3"],
)
def test_machine_rejects_video_paths_outside_explicit_allowlist(tmp_path, reference):
    root = make_machine_input(tmp_path / "input", clips=("alpha",))
    path = root / "track" / "alpha" / "video_meta.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["input_path"] = reference
    write_json(path, data)
    with pytest.raises(MachineInputError):
        load_machine_input(machine_settings(root, tmp_path), batch_id="blocked")


@pytest.mark.parametrize(
    "reference",
    ["annotations.csv", "target/alpha/sample.png", "../answer.png", "identity/alpha/features.npz"],
)
def test_machine_rejects_image_paths_outside_explicit_allowlist(tmp_path, reference):
    root = make_machine_input(tmp_path / "input", clips=("alpha",))
    path = root / "identity" / "alpha" / "samples.jsonl"
    rows = json_rows(path)
    rows[0]["crop_path"] = reference
    write_rows(path, rows)
    refresh_contract(root, "alpha")
    with pytest.raises(MachineInputError):
        load_machine_input(machine_settings(root, tmp_path), batch_id="blocked")


@pytest.mark.parametrize("kind", ["crop", "feature", "video", "track_directory", "input_root"])
def test_machine_symlinks_cannot_turn_forbidden_files_into_inputs(tmp_path, kind):
    root = make_machine_input(tmp_path / "input", clips=("alpha",))
    forbidden = tmp_path / "target"
    forbidden.mkdir()
    if kind == "input_root":
        link = tmp_path / "linked-input"
        root.rename(forbidden / "input")
        link.symlink_to(forbidden / "input", target_is_directory=True)
        root = link
    elif kind == "track_directory":
        directory = root / "track" / "alpha"
        directory.rename(forbidden / "alpha")
        directory.symlink_to(forbidden / "alpha", target_is_directory=True)
    else:
        path = {
            "crop": root / "track" / "alpha" / "samples" / "T0-B0.png",
            "feature": root / "identity" / "alpha" / "features.npz",
            "video": root / "video" / "alpha.mp4",
        }[kind]
        destination = forbidden / path.name
        path.rename(destination)
        path.symlink_to(destination)
    with pytest.raises(MachineInputError):
        load_machine_input(machine_settings(root, tmp_path), batch_id="blocked")


def test_machine_never_opens_unlisted_answers_or_history(tmp_path, monkeypatch):
    root = make_machine_input(tmp_path / "input", clips=("alpha",))
    answer = root.parent / "annotations.csv"
    answer.write_text("wrong answer", encoding="utf-8")
    (root / "target").mkdir()
    (root / "target" / "selection.json").write_text("not input", encoding="utf-8")
    (root / "history.sqlite3").write_bytes(b"not a database")
    settings = machine_settings(root, tmp_path)
    original_open = Path.open
    original_os_open = os.open
    opened = []

    def guarded_open(path, *args, **kwargs):
        opened.append(path)
        assert path.name != "annotations.csv"
        assert "target" not in path.parts
        assert path.suffix != ".sqlite3"
        return original_open(path, *args, **kwargs)

    def guarded_os_open(reference, *args, **kwargs):
        path = Path(reference)
        opened.append(path)
        assert path.name != "annotations.csv"
        assert "target" not in path.parts
        assert path.suffix != ".sqlite3"
        return original_os_open(reference, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "open", guarded_open)
        scoped.setattr(os, "open", guarded_os_open)
        first = load_machine_input(settings, batch_id="predict")
    answer.write_text("changed identity labels and selected tracks", encoding="utf-8")
    (root / "target" / "selection.json").write_text("changed selection", encoding="utf-8")
    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "open", guarded_open)
        scoped.setattr(os, "open", guarded_os_open)
        second = load_machine_input(settings, batch_id="predict")
    assert opened
    assert first == second


@pytest.mark.parametrize("name", ["target", "repository-review"])
def test_forbidden_directory_cannot_become_the_input_root(tmp_path, name):
    root = make_machine_input(tmp_path / name, clips=("solo",))
    with pytest.raises(MachineInputError, match="forbidden"):
        load_machine_input(machine_settings(root, tmp_path), batch_id="blocked")
