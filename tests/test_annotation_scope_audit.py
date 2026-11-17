"""Annotation-only audit: closed scopes, frozen cohorts, upstream immutability."""

import copy
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from scripts.audit_target_scopes import (
    DENOMINATORS,
    PROTOCOL_A,
    PROTOCOL_B,
    audit_samples,
    input_hashes,
    safe_output,
    score_reviewed,
    validate_review_ranges,
    validate_scope,
    verify_hashes,
    write_confusion,
    write_json,
)


def test_script_supports_direct_cli_entrypoint():
    script = Path(__file__).resolve().parents[1] / "scripts/audit_target_scopes.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--expected-freeze-sha256" in result.stdout


def scope(start, end, truth="jersey:27", track=1):
    return {
        "track_id": track,
        "start_frame": start,
        "end_frame": end,
        "truth": truth,
        "reason": "visible number on synthetic evidence",
        "evidence_images": ["evidence/image.png"],
    }


def test_frame_must_be_in_declared_and_reviewed_scope():
    sample = {"track_id": 1, "frame_idx": 10}
    assert validate_scope(sample, [scope(0, 9)], [scope(0, 20)], []) == "outside_declared_segment"
    assert (
        validate_scope(sample, [scope(0, 20)], [scope(0, 9)], [])
        == "outside_or_overlapping_reviewed_range"
    )
    assert validate_scope(sample, [scope(0, 20)], [scope(0, 20)], []) == "valid"


def test_mixed_track_warns_until_explicitly_partitioned():
    observations = [
        {"track_id": 1, "frame_idx": 2, "suggested_truth": "jersey:10"},
        {"track_id": 1, "frame_idx": 12, "suggested_truth": "jersey:27"},
    ]
    sample = {"track_id": 1, "frame_idx": 12}
    declared = [scope(0, 20)]
    assert validate_scope(sample, declared, declared, observations) == "scope_suspicious"
    assert (
        validate_scope(sample, declared, [scope(0, 9, "jersey:10"), scope(10, 20)], observations)
        == "valid"
    )
    assert (
        validate_scope(sample, declared, [scope(0, 20, None)], observations) == "scope_suspicious"
    )


@pytest.mark.parametrize(
    "reviewed",
    [[scope(0, 10), scope(10, 20)], [scope(0, 9), scope(11, 20)], [scope(0, 21)]],
)
def test_ranges_cannot_overlap_leave_gaps_or_escape_selection(reviewed):
    with pytest.raises(ValueError):
        validate_review_ranges([scope(0, 20)], reviewed)


def frozen_cohorts():
    original, reviewed, correspondence, members = [], [], [], []
    for clip, (truth, count) in enumerate(DENOMINATORS.items(), 1):
        ranges = []
        for frame in range(count):
            corrected = (
                "jersey:10"
                if clip == 1 and frame < 7
                else "jersey:30"
                if clip == 1 and frame == 7
                else truth
            )
            ranges.append(scope(frame, frame, corrected))
        for frame in range(count):
            identity = "p27" if clip == 1 else "p15"
            if clip == 1 and frame == count - 1:
                identity = None
            row = {
                "sample_id": f"{clip}-{frame}",
                "video_id": str(clip),
                "track_id": "1",
                "frame_idx": frame,
                "box": [0, 0, 10, 20],
                "identity": identity,
                "status": "matched" if identity else "below_iou",
                "segment_id": f"{clip}_1_0",
                "truth": truth,
                "correct": identity is not None,
            }
            original.append(row)
            correspondence.append(row.copy())
            members.append(row.copy())
            reviewed.append(
                {
                    **row,
                    "clip": str(clip),
                    "bbox_xyxy": row["box"],
                    "original_truth": truth,
                    "suggested_truth": ranges[frame]["truth"],
                    "reviewed_scope_status": "valid",
                    "evidence_images": ["evidence/image.png"],
                    "reason": "visual judgement",
                    "allowed_ranges": [ranges[frame]],
                    "all_clip_ranges": ranges,
                    "track_segments": [scope(0, count - 1, truth)],
                }
            )
    return original, reviewed, correspondence, members


def test_dual_protocol_keeps_cohorts_geometry_and_losses_separate(tmp_path):
    original, reviewed, correspondence, members = frozen_cohorts()
    before = copy.deepcopy(original)
    summary, scored, frames, _, others = score_reviewed(
        original,
        reviewed,
        correspondence,
        members,
        [{"video_id": "1", "fps": 1}, {"video_id": "2", "fps": 1}],
        [],
    )
    assert original == before
    assert PROTOCOL_A != PROTOCOL_B
    assert summary["correct_samples"] == 61
    assert summary["total_denominator"] == 70
    assert summary["reviewed_target_samples"] == 62
    assert summary["player_15_metrics"]["correct_samples"] == 48
    assert summary["player_15_metrics"]["total_denominator"] == 48
    assert summary["observed_exposure_seconds"] == 61
    assert summary["known_other_samples"] == len(others) == 8
    assert summary["passed"] is False
    assert len(scored) == 118
    assert sum(r["correct"] for r in original[:70]) == 69
    assert all(
        a["identity"] == b["identity"] and a["box"] == b["box"]
        for a, b in zip(original, scored, strict=True)
    )
    assert all(not r["correct"] for r in scored[:8])
    assert sum(r["correct_target"] for r in frames) == 61
    path = tmp_path / "confusion.csv"
    write_confusion(path, scored)
    with path.open() as stream:
        matrix = list(csv.DictReader(stream))
    assert sum(int(r["total"]) for r in matrix) == 118
    assert {r["truth"]: int(r["total"]) for r in matrix} == {
        "jersey:10": 7,
        "jersey:15": 48,
        "jersey:27": 62,
        "jersey:30": 1,
    }


def test_unknown_scope_is_not_scored_and_missing_cohort_rejected():
    original, reviewed, correspondence, members = frozen_cohorts()
    reviewed[8]["reviewed_scope_status"] = "scope_suspicious"
    args = (
        correspondence,
        members,
        [{"video_id": "1", "fps": 60}, {"video_id": "2", "fps": 60}],
        [],
    )
    summary, scored, *_ = score_reviewed(original, reviewed, *args)
    assert summary["correct_samples"] == 60
    assert not scored[8]["correct"]
    assert scored[8]["outcome"] == "scope_suspicious"
    with pytest.raises(ValueError, match="cohorts differ"):
        score_reviewed(original[:-1], reviewed, *args)


def test_cached_valid_cannot_allow_mixed_range_samples_or_exposure():
    original, reviewed, correspondence, members = frozen_cohorts()
    for row in reviewed[:70]:
        row["all_clip_ranges"] = [scope(0, 69)]
        row["allowed_ranges"] = [scope(0, 69)]
    summary, scored, frames, *_ = score_reviewed(
        original,
        reviewed,
        correspondence,
        members,
        [{"video_id": "1", "fps": 1}, {"video_id": "2", "fps": 1}],
        [],
    )
    assert summary["correct_samples"] == 0
    assert summary["observed_exposure_seconds"] == 0
    assert summary["scope_suspicious_samples"] == 70
    assert all(row["outcome"] == "scope_suspicious" for row in scored[:70])
    assert all(row["truth"] is None for row in frames if row["video_id"] == "1")


def audit_fixture(tmp_path):
    data = tmp_path / "test"
    data.mkdir()
    annotations = tmp_path / "annotations.csv"
    annotations.write_text("video_id,jersey_number\n1,27\n2,15\n")
    review = {"clips": {}, "samples": []}
    for clip, (truth, count) in enumerate(DENOMINATORS.items(), 1):
        clip = str(clip)
        base = data / "target" / clip
        (base / "samples").mkdir(parents=True)
        crop = base / "samples/image.png"
        Image.new("RGB", (10, 20), "green").save(crop)
        digest = hashlib.sha256(crop.read_bytes()).hexdigest()
        selection = {
            "event_jersey_number": truth.split(":")[1],
            "track_segments": [scope(0, count - 1)],
        }
        (base / "selection.json").write_text(json.dumps(selection))
        rows = [
            {
                "video_id": clip,
                "track_id": 1,
                "frame_idx": f,
                "bbox_xyxy": [0, 0, 10, 20],
                "sample_id": f"{clip}-{f}",
                "crop_path": str(crop.relative_to(data)),
                "crop_sha256": digest,
            }
            for f in range(count)
        ]
        text = "".join(json.dumps(r) + "\n" for r in rows)
        (base / "samples.jsonl").write_text(text)
        machine = data / "track" / clip
        machine.mkdir(parents=True)
        (machine / "tracks.jsonl").write_text(text)
        review["clips"][clip] = {"allowed_ranges": [scope(0, count - 1, truth)]}
        review["samples"].extend(
            {
                "sample_id": r["sample_id"],
                "suggested_truth": truth,
                "confidence": 0.9,
                "reason": "visible number",
                "evidence_images": ["evidence/image.png"],
                "crop_sha256": digest,
            }
            for r in rows
        )
    return data, annotations, review


def test_real_file_audit_is_repeatable_and_preserves_original_bytes(tmp_path):
    data, annotations, review = audit_fixture(tmp_path)
    before = input_hashes(data, annotations)
    first = audit_samples(data, annotations, review)
    second = audit_samples(data, annotations, review)
    assert first == second
    assert len(first[0]) == 118
    assert all(r["raw_track_iou"] == 1 for r in first[0])
    assert input_hashes(data, annotations) == before
    verify_hashes(before)
    annotations.write_text("tampered")
    with pytest.raises(ValueError, match="Read-only input changed"):
        verify_hashes(before)


def test_changed_crop_outside_sample_and_wrong_geometry_rejected(tmp_path):
    data, annotations, review = audit_fixture(tmp_path)
    review["samples"][0]["crop_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="changed reviewed crop"):
        audit_samples(data, annotations, review)
    review["samples"][0]["crop_sha256"] = hashlib.sha256(
        (data / "target/1/samples/image.png").read_bytes()
    ).hexdigest()
    path = data / "target/1/samples.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["frame_idx"] = 70
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="unique selected-track row"):
        audit_samples(data, annotations, review)


def test_output_cannot_touch_sources_or_overwrite_history(tmp_path):
    upstream = tmp_path / "test"
    upstream.mkdir()
    with pytest.raises(ValueError, match="read-only source"):
        safe_output(upstream / "target/new", [upstream])
    alias = tmp_path / "alias"
    alias.symlink_to(upstream, target_is_directory=True)
    with pytest.raises(ValueError, match="read-only source"):
        safe_output(alias / "new", [upstream])
    with pytest.raises(FileExistsError):
        safe_output(upstream, [])
    artifact = tmp_path / "result.json"
    write_json(artifact, {"protocol": PROTOCOL_A})
    with pytest.raises(FileExistsError):
        write_json(artifact, {"protocol": PROTOCOL_B})
    assert json.loads(artifact.read_text())["protocol"] == PROTOCOL_A
