"""T4 firewall and golden scoring tests; all answers and geometry are synthetic."""

import copy
import csv
import hashlib
import itertools
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from app.evaluation.golden_score import align_identities


def alignment_rows(counts):
    return [
        {"truth": truth, "identity": identity}
        for (truth, identity), count in counts.items()
        for _ in range(count)
    ]


def test_alignment_is_injective_global_optimum():
    counts = {("27", "a"): 10, ("27", "b"): 9, ("15", "a"): 9}
    aligned = align_identities(alignment_rows(counts))
    assert aligned == {"15": "a", "27": "b"}
    for size in range(1, 4):
        counts = {
            (str(i), str(j)): ((i + 3) * (j + 7)) % 5 for i in range(size) for j in range(size + 1)
        }
        aligned = align_identities(alignment_rows(counts))
        actual = sum(counts.get((truth, identity), 0) for truth, identity in aligned.items())
        exhaustive = max(
            sum(counts[str(i), permutation[i]] for i in range(size))
            for permutation in itertools.permutations(map(str, range(size + 1)), size)
        )
        assert actual == exhaustive


def test_alignment_ties_ignore_input_order_and_never_combine_fragments():
    rows = alignment_rows({("27", "a"): 35, ("27", "b"): 35})
    aligned = align_identities(rows)
    assert aligned == align_identities(list(reversed(rows)))
    assert sum(row["identity"] == aligned["27"] for row in rows) == 35


def test_alignment_never_assigns_missing_predictions_as_an_identity():
    assert align_identities(alignment_rows({("27", None): 70})) == {"27": None}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_case(
    tmp_path, *, identities=None, unprocessed=(), frame_step=1, fps=1, before_prepare=None
):
    """A complete tiny T3 handoff and 70 golden samples, never real answers."""
    from app.evaluation.prepare_benchmark import prepare_benchmark

    run, data, geometry = (tmp_path / name for name in ("run", "data", "geometry"))
    run.mkdir()
    predictions = {
        "schema": "t3-dual-stream-predictions-v1",
        "match_id": "synthetic-match",
        "revision": 1,
        "segments": [],
    }
    db_path = run / "player-library.sqlite3"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        "CREATE TABLE matches (match_id TEXT PRIMARY KEY, revision INTEGER);"
        "CREATE TABLE dual_stream_segments (match_id TEXT, segment_id TEXT, "
        "segment_json TEXT, feature_json TEXT);"
        "CREATE TABLE dual_stream_decisions (match_id TEXT, segment_id TEXT, "
        "revision INTEGER, json TEXT);"
        "CREATE TABLE assignments (match_id TEXT, segment_id TEXT, revision INTEGER, "
        "player_id TEXT, current INTEGER, json TEXT);"
    )
    connection.execute("INSERT INTO matches VALUES (?, ?)", ("synthetic-match", 1))
    clips, annotations, samples_by_clip = [], [], {}
    labels = [(f"v{i}", "27", 7) for i in range(10)] + [("c10", "10", 1), ("c15", "15", 1)]
    for clip, number, count in labels:
        identity = (identities or {}).get(clip, "p27" if number == "27" else "p" + number)
        observations, samples = [], []
        for index in range(count):
            frame = index * frame_step
            crop_path = f"track/{clip}/samples/sample-{index}.png"
            crop = data / crop_path
            crop.parent.mkdir(parents=True, exist_ok=True)
            crop.write_bytes(b"synthetic crop bytes")
            box = [0.0, 0.0, 10.0, 20.0]
            observations.append(
                {
                    "video_id": clip,
                    "track_id": 1,
                    "frame_idx": frame,
                    "timestamp_s": frame / fps,
                    "bbox_xyxy": box,
                    "confidence": 0.95,
                }
            )
            samples.append(
                {
                    "video_id": clip,
                    "track_id": 1,
                    "frame_idx": frame,
                    "bbox_xyxy": box,
                    "sample_id": f"{clip}-sample-{index}",
                    "crop_path": crop_path,
                    "crop_sha256": digest(crop),
                    "path": crop_path,
                }
            )
        frame_count = (count - 1) * frame_step + 1
        write_jsonl(data / "track" / clip / "tracks.jsonl", observations)
        write_jsonl(data / "identity" / clip / "samples.jsonl", samples)
        write_json(
            data / "track" / clip / "video_meta.json",
            {"video_id": clip, "fps": fps, "frame_count": frame_count},
        )
        video = data / "video" / f"{clip}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"synthetic raw video bytes")
        samples_by_clip[clip] = samples
        annotations.append(
            {"video_id": clip, "jersey_number": number, "person_gt_id": "", "team_id": ""}
        )
        target = data / "target" / clip
        write_json(
            target / "selection.json",
            {
                "video_id": clip,
                "event_jersey_number": number,
                "track_segments": [
                    {
                        "track_id": 1,
                        "start_frame": 0,
                        "end_frame": frame_count - 1,
                        "quality_status": "clean",
                    }
                ],
            },
        )
        write_jsonl(target / "samples.jsonl", samples)
        if clip in unprocessed:
            continue
        segment_id = clip + "_1_0"
        segment = {
            "segment_id": segment_id,
            "video_id": clip,
            "local_track_id": 1,
            "start_frame": 0,
            "end_frame": frame_count - 1,
            "start_time_s": 0.0,
            "end_time_s": (frame_count - 1) / fps,
            "observation_count": len(observations),
            "observations": [
                {
                    "video_id": clip,
                    "local_track_id": 1,
                    "frame_idx": row["frame_idx"],
                    "timestamp_s": row["timestamp_s"],
                    "bbox_xyxy": row["bbox_xyxy"],
                    "confidence": row["confidence"],
                    "crop_path": None,
                }
                for row in observations
            ],
        }
        decision = {
            "segment_id": segment_id,
            "outcome": "matched_existing" if identity else "insufficient_evidence",
            "target_player_id": identity,
            "best_distance": 0.01 if identity else {"$float": "+inf"},
            "competitive_margin": 0.4 if identity else 0.0,
            "evidence": {
                "reason": "matched_existing" if identity else "insufficient_evidence",
                "player_scores": {},
                "distance_threshold": 0.25,
                "required_margin": 0.04,
            },
        }
        feature = {
            "segment_id": segment_id,
            "jersey": {
                "number": int(number),
                "confidence": 0.99,
                "legible": True,
                "votes": {number: 2.97},
                "raw_text": number,
            },
            "appearance": {"quality_score": 0.9, "model_name": "synthetic-appearance"},
        }
        predictions["segments"].append(
            {"segment": segment, "decision": decision, "player_id": identity}
        )
        connection.execute(
            "INSERT INTO dual_stream_segments VALUES (?, ?, ?, ?)",
            ("synthetic-match", segment_id, json.dumps(segment), json.dumps(feature)),
        )
        connection.execute(
            "INSERT INTO dual_stream_decisions VALUES (?, ?, ?, ?)",
            ("synthetic-match", segment_id, 1, json.dumps(decision)),
        )
        legacy_identity = "pl_" + identity if identity else None
        assignment = {"player_id": legacy_identity, "segment_id": segment_id}
        connection.execute(
            "INSERT INTO assignments VALUES (?, ?, ?, ?, ?, ?)",
            ("synthetic-match", segment_id, 1, legacy_identity, 1, json.dumps(assignment)),
        )
        inputs = [
            {"path": relative, "sha256": digest(data / relative)}
            for relative in (
                f"track/{clip}/tracks.jsonl",
                f"track/{clip}/video_meta.json",
                f"video/{clip}.mp4",
            )
        ]
        clips.append({"clip": clip, "inputs": inputs, "derived_crops": []})
    connection.commit()
    connection.close()
    annotation_path = data / "annotations.csv"
    with annotation_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["video_id", "jersey_number", "person_gt_id", "team_id"]
        )
        writer.writeheader()
        writer.writerows(annotations)
    write_json(run / "predictions.json", predictions)
    write_json(run / "summary.json", {"final_revision": 1})
    write_json(run / "config.json", {"clips": [entry["clip"] for entry in clips]})
    write_json(run / "input-manifest.json", {"clips": clips, "labels_read": False})
    write_json(run / "code-hashes.json", {})
    if before_prepare is not None:
        before_prepare(run, data)
    write_json(
        run / "artifact-hashes.json",
        {path.name: digest(path) for path in run.iterdir() if path.is_file()},
    )
    freeze = prepare_benchmark(run, data, geometry)
    return {
        "run": run,
        "data": data,
        "geometry": geometry,
        "annotations": annotation_path,
        "freeze": freeze["freeze_sha256"],
        "samples": samples_by_clip,
    }


def score_case(case, out_dir):
    from app.evaluation.golden_score import score_benchmark

    return score_benchmark(
        case["geometry"],
        case["annotations"],
        out_dir,
        expected_freeze_sha256=case["freeze"],
    )


def test_prepare_never_opens_answers(tmp_path, monkeypatch):
    original = Path.open
    forbidden_reads = []

    def guarded(path, mode="r", *args, **kwargs):
        if ("r" in mode or "+" in mode) and (
            path.name == "annotations.csv" or "target" in path.parts
        ):
            forbidden_reads.append(str(path))
            raise AssertionError("prepare attempted to read an answer")
        return original(path, mode, *args, **kwargs)

    # The fixture creates synthetic labels, but the guard only forbids reading them.
    monkeypatch.setattr(Path, "open", guarded)
    case = synthetic_case(tmp_path)
    assert case["freeze"]
    assert forbidden_reads == []


def test_all_seventy_correct_samples_pass_five_acceptance_thresholds(tmp_path):
    case = synthetic_case(tmp_path)
    result = score_case(case, tmp_path / "scoring")
    assert result["total_denominator"] == 70
    assert result["correct_samples"] == 70
    assert result["coverage_rate"] == 1.0
    assert result["observed_exposure_seconds"] == pytest.approx(70.0)
    assert result["cross_identity_pollution"] == {"10号": 0, "15号": 0}
    assert result["physical_conflicts"] == 0
    assert result["fragmentation_count"] == 0
    assert result["passed"] is True
    for name in (
        "scoring_summary.json",
        "confusion_matrix.csv",
        "audit_failure_cases.jsonl",
        "acceptance_report.md",
    ):
        assert (tmp_path / "scoring" / name).is_file()


def test_empty_gallery_cannot_claim_purity_acceptance(tmp_path):
    identities = {f"v{i}": None for i in range(10)} | {"c10": None, "c15": None}
    case = synthetic_case(tmp_path, identities=identities)
    result = score_case(case, tmp_path / "scoring")
    assert result["total_denominator"] == 70
    assert result["correct_samples"] == 0
    assert result["observed_exposure_seconds"] == 0.0
    assert result["selected_identity"] is None
    assert result["purity_status"] == "no_target_identity"
    assert result["passed"] is False
    failures = [
        json.loads(line)
        for line in (tmp_path / "scoring/audit_failure_cases.jsonl").read_text().splitlines()
    ]
    assert len(failures) == 70


def test_unprocessed_clips_remain_in_fixed_denominator(tmp_path):
    case = synthetic_case(tmp_path, unprocessed=("v0", "v1"))
    result = score_case(case, tmp_path / "scoring")
    assert result["total_denominator"] == 70
    assert result["correct_samples"] == 56
    assert result["coverage_rate"] == pytest.approx(56 / 70)
    assert result["passed"] is False
    assert len((tmp_path / "scoring/audit_failure_cases.jsonl").read_text().splitlines()) == 14


def test_splits_cannot_be_stitched_into_full_sample_coverage(tmp_path):
    identities = {f"v{i}": "a" if i < 5 else "b" for i in range(10)}
    case = synthetic_case(tmp_path, identities=identities)
    result = score_case(case, tmp_path / "scoring")
    assert result["correct_samples"] == 35
    assert result["observed_exposure_seconds"] == pytest.approx(35.0)
    assert result["fragmentation_count"] == 1
    assert result["passed"] is False


def test_known_foreign_players_are_counted_as_pollution(tmp_path):
    case = synthetic_case(tmp_path, identities={"c10": "p27", "c15": "p27"})
    result = score_case(case, tmp_path / "scoring")
    assert result["correct_samples"] == 70
    assert result["cross_identity_pollution"] == {"10号": 1, "15号": 1}
    assert result["passed"] is False


def test_sparse_exposure_uses_observed_frames_instead_of_track_span(tmp_path):
    case = synthetic_case(tmp_path, frame_step=100, fps=10)
    result = score_case(case, tmp_path / "scoring")
    assert result["correct_samples"] == 70
    assert result["observed_exposure_seconds"] == pytest.approx(7.0)
    assert result["passed"] is False


@pytest.mark.parametrize(
    "target",
    [
        "predictions",
        "database",
        "run_manifest",
        "correspondence",
        "machine_sample",
        "freeze_manifest",
        "implementation",
    ],
)
def test_tampering_is_rejected_before_any_answer_read(tmp_path, monkeypatch, target):
    case = synthetic_case(tmp_path)
    path = {
        "predictions": case["run"] / "predictions.json",
        "database": case["run"] / "player-library.sqlite3",
        "run_manifest": case["run"] / "artifact-hashes.json",
        "correspondence": case["geometry"] / "correspondence.jsonl",
        "machine_sample": case["data"] / "track/v0/samples/sample-0.png",
        "freeze_manifest": case["geometry"] / "artifact-hashes.json",
    }.get(target)
    if target == "implementation":
        # Pin a temporary executable source in this synthetic receipt; do not edit
        # the shared running implementation to test its independent code gate.
        path = tmp_path / "temporary-scorer.py"
        path.write_text("pass\n")
        seal = case["geometry"] / "geometry-freeze.json"
        content = json.loads(seal.read_text())
        content["code_files"][str(path)] = digest(path)
        write_json(seal, content)
        case["freeze"] = digest(seal)
    with path.open("ab") as stream:
        stream.write(b"tampered")
    original = Path.open
    seen = []

    def guarded(path, mode="r", *args, **kwargs):
        if "r" in mode and (path == case["annotations"] or "target" in path.parts):
            seen.append(str(path))
            raise AssertionError("labels opened before fingerprint rejection")
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    with pytest.raises((ValueError, RuntimeError)):
        score_case(case, tmp_path / "scoring")
    assert not seen
    assert not (tmp_path / "scoring/scoring_summary.json").exists()


def test_external_freeze_digest_prevents_rewriting_the_seal(tmp_path):
    from app.evaluation.golden_freeze import verify_prepared

    case = synthetic_case(tmp_path)
    with pytest.raises((ValueError, RuntimeError)):
        verify_prepared(case["geometry"], "0" * 64)


def test_changed_denominator_refuses_to_publish_a_score(tmp_path):
    case = synthetic_case(tmp_path)
    selection = case["data"] / "target/v0/selection.json"
    content = json.loads(selection.read_text())
    content["track_segments"][0]["end_frame"] = 5
    write_json(selection, content)
    with pytest.raises(ValueError, match="70|denominator|scope|samples"):
        score_case(case, tmp_path / "scoring")
    assert not (tmp_path / "scoring/scoring_summary.json").exists()


def test_score_does_not_overwrite_previous_output(tmp_path):
    case = synthetic_case(tmp_path)
    destination = tmp_path / "scoring"
    score_case(case, destination)
    original = (destination / "scoring_summary.json").read_bytes()
    with pytest.raises((FileExistsError, ValueError)):
        score_case(case, destination)
    assert (destination / "scoring_summary.json").read_bytes() == original


def test_cli_runs_separate_prepare_and_score_processes(tmp_path):
    case = synthetic_case(tmp_path)
    destination = tmp_path / "cli-run"
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.evaluation.run_benchmark",
            "--run-dir",
            str(case["run"]),
            "--test-data",
            str(case["data"]),
            "--out-dir",
            str(destination),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    result = json.loads((destination / "scoring_summary.json").read_text())
    assert result["correct_samples"] == 70
    assert result["passed"] is True
    receipt = json.loads((destination / "prepare-receipt.json").read_text())
    score_gate = json.loads((destination / "pre-label-gate.json").read_text())
    assert receipt["process_id"] != score_gate["process_id"]


def add_second_same_frame_track(run, data, *, overlap=False, labelled=False):
    """Alter only a synthetic upstream fixture, before the prepare freeze."""
    prediction_path = run / "predictions.json"
    predictions = json.loads(prediction_path.read_text())
    original = next(r for r in predictions["segments"] if r["segment"]["video_id"] == "v0")
    extra = copy.deepcopy(original)
    sid = "v0_2_0"
    extra["segment"].update(segment_id=sid, local_track_id=2)
    extra["decision"]["segment_id"] = sid
    for observation in extra["segment"]["observations"]:
        observation["local_track_id"] = 2
        observation["bbox_xyxy"] = [0.2, 0.0, 10.2, 20.0] if overlap else [30.0, 0.0, 40.0, 20.0]
    predictions["segments"].append(extra)
    write_json(prediction_path, predictions)
    connection = sqlite3.connect(run / "player-library.sqlite3")
    feature = {"segment_id": sid, "jersey": None, "appearance": {"quality_score": 0.9}}
    connection.execute(
        "INSERT INTO dual_stream_segments VALUES (?, ?, ?, ?)",
        ("synthetic-match", sid, json.dumps(extra["segment"]), json.dumps(feature)),
    )
    connection.execute(
        "INSERT INTO dual_stream_decisions VALUES (?, ?, ?, ?)",
        ("synthetic-match", sid, 1, json.dumps(extra["decision"])),
    )
    connection.execute(
        "INSERT INTO assignments VALUES (?, ?, ?, ?, ?, ?)",
        ("synthetic-match", sid, 1, "pl_p27", 1, "{}"),
    )
    connection.commit()
    connection.close()
    track_path = data / "track/v0/tracks.jsonl"
    rows = [json.loads(line) for line in track_path.read_text().splitlines()]
    rows.extend(
        {**r, "track_id": 2, "bbox_xyxy": o["bbox_xyxy"]}
        for r, o in zip(rows[:], extra["segment"]["observations"])
    )
    write_jsonl(track_path, rows)
    manifest_path = run / "input-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for clip in manifest["clips"]:
        for record in clip["inputs"]:
            record["sha256"] = digest(data / record["path"])
    write_json(manifest_path, manifest)
    if labelled:
        selection_path = data / "target/v0/selection.json"
        selection = json.loads(selection_path.read_text())
        selection["track_segments"].append(
            {"track_id": 2, "start_frame": 0, "end_frame": 6, "quality_status": "clean"}
        )
        write_json(selection_path, selection)


def test_same_source_separated_people_conflict_and_exposure_is_deduplicated(tmp_path):
    case = synthetic_case(
        tmp_path,
        before_prepare=lambda run, data: add_second_same_frame_track(run, data, labelled=True),
    )
    result = score_case(case, tmp_path / "scoring")
    assert result["correct_samples"] == 70
    assert result["physical_conflicts"] == 7
    assert result["observed_exposure_seconds"] == 70
    assert result["checks"]["physical_conflicts"] is False
    assert result["passed"] is False


def test_overlapping_duplicates_stay_ambiguous_and_are_not_separated_conflicts(tmp_path):
    case = synthetic_case(
        tmp_path,
        before_prepare=lambda run, data: add_second_same_frame_track(run, data, overlap=True),
    )
    result = score_case(case, tmp_path / "scoring")
    assert result["correct_samples"] == 63
    assert result["total_denominator"] == 70
    assert result["physical_conflicts"] == 0
    assert result["selected_identity_pending_overlaps"] == 7
    assert result["passed"] is False


def test_prepare_active_barrier_blocks_secret_and_symlink_alias(tmp_path):
    from app.evaluation.golden_freeze import blind_inputs

    secret = tmp_path / "target/v0/selection.json"
    write_json(secret, {"synthetic": True})
    alias = tmp_path / "machine-looking.json"
    alias.symlink_to(secret)
    with blind_inputs():
        for path in (secret, alias):
            with pytest.raises(PermissionError, match="firewall"):
                path.read_bytes()


def test_prepare_rejects_export_disagreement_with_database(tmp_path):
    def corrupt_export(run, data):
        path = run / "predictions.json"
        predictions = json.loads(path.read_text())
        predictions["segments"][0]["segment"]["observations"][0]["bbox_xyxy"][0] = 0.1
        write_json(path, predictions)

    with pytest.raises(ValueError, match="database geometry mismatch"):
        synthetic_case(tmp_path, before_prepare=corrupt_export)
