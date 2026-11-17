"""T3/T4 real integration, using raw videos and machine tracks exclusively.

Run from the project root:
    uv run python run_real_batches.py
    # or with a specific run dir:
    uv run python run_real_batches.py --run-dir var/runs/run-20261115T033000-bug7

This script never imports the historical evaluation driver or reads labels. It
processes all 10 test clips from the benchmark, keeps full track geometry, and
extracts 8~12 (target 10) high-quality representative exemplar crops per track
distributed across temporal buckets based on torso sharpness and exposure.
Artifacts are created exclusively in the target run directory and frozen with
cryptographic receipts.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import torch


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "app").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    return current.parents[0]


PROJECT = find_project_root()
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from app.evaluation.guard import assert_frozen_before_scoring  # noqa: E402
from app.features.appearance import AppearanceExtractor  # noqa: E402
from app.features.dual_stream import (  # noqa: E402
    DualStreamExtractor,
    TrackletSegment,
    TrackObservation,
)
from app.features.jersey_number import (  # noqa: E402
    JerseyNumberExtractor,
    assess_image_quality,
)
from app.features.split import (  # noqa: E402
    detect_track_change_points,
    split_track_observations,
)
from app.identity.constraints import SpatioTemporalConstraintEngine  # noqa: E402
from app.storage.database import connect  # noqa: E402
from app.storage.dual_stream import read_dual_stream_snapshot  # noqa: E402

CLIPS = ("3", "6", "11", "20", "30", "37", "43", "50", "63", "68")
INPUT = PROJECT.parent / "test"
if not INPUT.is_dir():
    INPUT = PROJECT / "test"


def jsonable(value):
    if is_dataclass(value):
        return {field.name: jsonable(getattr(value, field.name)) for field in fields(value)}
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump(mode="json"))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"$float": "+inf" if value > 0 else "-inf"}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def dump(run_dir: Path, name: str, value):
    with (run_dir / name).open("w", encoding="utf-8") as stream:
        json.dump(
            jsonable(value), stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        stream.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image_key(rgb: np.ndarray) -> tuple:
    return rgb.shape, hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


class CachedJerseyExtractor(JerseyNumberExtractor):
    """Single-image OCR cached by pixel content, shared between change-point
    detection and segment feature extraction. extract_track stays inherited."""

    def __init__(self, cache: dict) -> None:
        super().__init__()
        self._prediction_cache = cache

    def extract(self, image):
        from app.features.jersey_number import validate_rgb_image

        rgb = validate_rgb_image(image)
        key = _image_key(rgb)
        if key not in self._prediction_cache:
            self._prediction_cache[key] = super().extract(rgb)
        return self._prediction_cache[key]


def make_cached_appearance(base: AppearanceExtractor, cache: dict) -> AppearanceExtractor:
    """Cache raw OSNet encoder outputs so detection and feature extraction never
    re-encode the same crop; normalization stays the single frozen code path."""
    encode = base._load_encoder()

    def cached_encode(rgb: np.ndarray) -> np.ndarray:
        key = _image_key(rgb)
        if key not in cache:
            cache[key] = encode(rgb)
        return cache[key]

    return AppearanceExtractor(encoder=cached_encode, model_name=base.model_name)


def load_clip(
    clip: str,
    run_dir: Path,
    crops_dir: Path,
    *,
    appearance_extractor: AppearanceExtractor,
    jersey_extractor: JerseyNumberExtractor,
):
    tracks_path = INPUT / "track" / clip / "tracks.jsonl"
    meta_path = INPUT / "track" / clip / "video_meta.json"
    video_path = INPUT / "video" / f"{clip}.mp4"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    with tracks_path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if str(row["video_id"]) != clip:
                    raise ValueError("Track source differs from requested clip")
                if not math.isclose(
                    row["timestamp_s"], row["frame_idx"] / meta["fps"], rel_tol=1e-7, abs_tol=1e-7
                ):
                    raise ValueError("CFR track timestamp differs from source FPS")
                grouped[row["track_id"]].append(row)
    eligible = {}
    skipped = []
    for track, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["frame_idx"])
        if len(rows) < 3 or rows[-1]["timestamp_s"] - rows[0]["timestamp_s"] < 0.1:
            skipped.append(
                {
                    "track_id": track,
                    "reason": "pipeline_short_track",
                    "observation_count": len(rows),
                }
            )
            continue
        if len({row["frame_idx"] for row in rows}) != len(rows):
            raise ValueError("Duplicate machine track frame")
        eligible[track] = rows

    # Ten temporal buckets ensure selected frames cover the track, while image
    # quality (torso sharpness/Laplacian variance and exposure) selects the exemplar within each bucket.
    target_buckets = 10
    at_frame = defaultdict(list)
    for track, rows in eligible.items():
        n_buckets = min(target_buckets, len(rows))
        for index, row in enumerate(rows):
            bucket = min(n_buckets - 1, index * n_buckets // len(rows))
            at_frame[row["frame_idx"]].append((track, bucket, row))

    all_candidates = defaultdict(list)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open source video {video_path}")
    if not math.isclose(capture.get(cv2.CAP_PROP_FPS), meta["fps"], abs_tol=1e-5):
        raise ValueError("Decoded source FPS differs from machine metadata")
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        for track, bucket, row in at_frame.get(frame_index, ()):
            x1, y1, x2, y2 = row["bbox_xyxy"]
            left, top = max(0, math.floor(x1)), max(0, math.floor(y1))
            right, bottom = min(width, math.ceil(x2)), min(height, math.ceil(y2))
            if left >= right or top >= bottom:
                continue
            crop = frame[top:bottom, left:right]
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            quality = assess_image_quality(rgb)
            if quality > 0:
                all_candidates[track].append(
                    (quality, frame_index, crop.copy(), row["bbox_xyxy"], bucket)
                )
        frame_index += 1
    capture.release()
    if frame_index != meta["frame_count"]:
        raise ValueError("Decoded source frame count differs from metadata")

    crop_paths = {}
    crop_manifest = []
    seen_manifest_paths = set()
    clip_crops = crops_dir / clip / "samples"
    clip_crops.mkdir(parents=True, exist_ok=True)
    segments = []
    split_ledger = []

    for track, rows in eligible.items():
        # Change-point detection uses dense sampling (top-3 quality crops per
        # temporal bucket) with per-crop OCR votes; feature extraction below
        # still selects the 10 bucket exemplars per subsegment.
        per_bucket = defaultdict(list)
        for q, f_idx, bgr, bbox, bucket in all_candidates.get(track, []):
            per_bucket[bucket].append((q, f_idx, bgr))
        detection_crops = []
        for bucket in sorted(per_bucket):
            ranked = sorted(per_bucket[bucket], key=lambda x: (-x[0], x[1]))[:3]
            detection_crops.extend((f_idx, bgr, q) for q, f_idx, bgr in ranked)
        change_points = []
        if len(detection_crops) >= 4 and (rows[-1]["frame_idx"] - rows[0]["frame_idx"] >= 300):
            change_points = detect_track_change_points(
                detection_crops,
                min_track_duration=300,
                appearance_extractor=appearance_extractor,
                jersey_extractor=jersey_extractor,
            )

        if change_points:
            sub_row_lists = split_track_observations(rows, change_points)
            subs_meta = []
            for sub in sub_row_lists:
                s0, s1 = sub[0]["frame_idx"], sub[-1]["frame_idx"]
                role = "side"
                for cp in change_points:
                    if cp.transition_start < s0 and s1 < cp.transition_end:
                        role = "transition"
                subs_meta.append(
                    {
                        "start_frame": s0,
                        "end_frame": s1,
                        "observation_count": len(sub),
                        "role": role,
                    }
                )
            split_ledger.append(
                {
                    "track_id": track,
                    "detection_sample_frames": sorted(f for f, _, _ in detection_crops),
                    "change_points": change_points,
                    "subsegments": subs_meta,
                }
            )
        else:
            sub_row_lists = [rows]

        for sub_rows in sub_row_lists:
            sub_start_f = sub_rows[0]["frame_idx"]
            sub_end_f = sub_rows[-1]["frame_idx"]
            sub_candidates = [
                item
                for item in all_candidates.get(track, [])
                if sub_start_f <= item[1] <= sub_end_f
            ]
            n_buckets = min(target_buckets, len(sub_rows))
            sub_bucket_map = {
                row["frame_idx"]: min(n_buckets - 1, index * n_buckets // len(sub_rows))
                for index, row in enumerate(sub_rows)
            }
            sub_best_per_bucket = {}
            for q, f_idx, bgr, bbox, _ in sub_candidates:
                b = sub_bucket_map.get(f_idx, 0)
                if b not in sub_best_per_bucket or q > sub_best_per_bucket[b][0]:
                    sub_best_per_bucket[b] = (q, f_idx, bgr, bbox)

            selected_frames = {}
            for b in range(n_buckets):
                if b in sub_best_per_bucket:
                    q, f_idx, bgr, bbox = sub_best_per_bucket[b]
                    selected_frames[f_idx] = (b, q, f_idx, bgr, bbox)

            if len(selected_frames) < n_buckets:
                for q, f_idx, bgr, bbox, _ in sorted(
                    sub_candidates, key=lambda x: x[0], reverse=True
                ):
                    if f_idx not in selected_frames:
                        b = sub_bucket_map.get(f_idx, 0)
                        selected_frames[f_idx] = (b, q, f_idx, bgr, bbox)
                        if len(selected_frames) >= n_buckets:
                            break

            if not selected_frames:
                skipped.append(
                    {
                        "track_id": track,
                        "sub_start_frame": sub_start_f,
                        "reason": "no_quality_passing_raw_crop",
                        "observation_count": len(sub_rows),
                    }
                )
                continue

            for f_idx, (bucket, quality, frame, bgr, bbox) in sorted(selected_frames.items()):
                relative = f"{clip}/samples/T{track:05d}-F{frame:06d}.png"
                path = crops_dir / relative
                if not path.is_file():
                    if not cv2.imwrite(str(path), bgr):
                        raise OSError(f"Cannot write derived PNG {path}")
                crop_paths[track, frame] = relative
                if relative not in seen_manifest_paths:
                    seen_manifest_paths.add(relative)
                    crop_manifest.append(
                        {
                            "path": relative,
                            "sha256": sha256(path),
                            "track_id": track,
                            "frame_idx": frame,
                            "bbox_xyxy": bbox,
                            "quality_score": quality,
                            "temporal_bucket": bucket,
                        }
                    )

            observations = tuple(
                TrackObservation(
                    video_id=clip,
                    local_track_id=track,
                    frame_idx=row["frame_idx"],
                    timestamp_s=row["timestamp_s"],
                    bbox_xyxy=tuple(row["bbox_xyxy"]),
                    confidence=row["det_score"],
                    crop_path=crop_paths.get((track, row["frame_idx"])),
                )
                for row in sub_rows
            )
            first, last = observations[0], observations[-1]
            segments.append(
                TrackletSegment(
                    segment_id=f"{clip}_{track}_{first.frame_idx}",
                    video_id=clip,
                    local_track_id=track,
                    start_frame=first.frame_idx,
                    end_frame=last.frame_idx,
                    start_time_s=first.timestamp_s,
                    end_time_s=last.timestamp_s,
                    observation_count=len(observations),
                    observations=observations,
                )
            )
    manifest = {
        "clip": clip,
        "segments": len(segments),
        "observations": sum(len(segment.observations) for segment in segments),
        "raw_eligible_observations": sum(len(rows) for rows in eligible.values()),
        "inputs": [
            {"path": str(path.relative_to(INPUT)), "sha256": sha256(path)}
            for path in (tracks_path, meta_path, video_path)
        ],
        "derived_crops": crop_manifest,
        "skipped_tracks": skipped,
        "track_splits": split_ledger,
        "transition_observations": sum(
            sub["observation_count"]
            for entry in split_ledger
            for sub in entry["subsegments"]
            if sub["role"] == "transition"
        ),
    }
    dump(run_dir, f"input-{clip}.json", manifest)
    return segments, manifest


def snapshot_artifact(connection, clip: str, run_dir: Path):
    snapshot = read_dual_stream_snapshot(connection, "t3-real-basketball")
    tables = [
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    ]
    table_counts = {
        table: connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        for table in tables
    }
    engine = SpatioTemporalConstraintEngine(
        segments=snapshot.segments, player_assignments=snapshot.assignments
    )
    assigned = snapshot.assignments
    conflicts = [
        constraint
        for constraint in engine.constraints
        if assigned.get(constraint.segment_a) is not None
        and assigned.get(constraint.segment_a) == assigned.get(constraint.segment_b)
    ]
    if conflicts:
        raise AssertionError("Committed gallery contains same-frame identity conflicts")
    value = {
        "clip": clip,
        "revision": snapshot.revision,
        "players": snapshot.players,
        "assignments": snapshot.assignments,
        "decisions": snapshot.decisions,
        "table_counts": table_counts,
        "cannot_link_edges": len(engine.constraints),
        "assigned_physical_conflicts": len(conflicts),
        "segments": len(snapshot.segments),
        "features": len(snapshot.features),
    }
    dump(run_dir, f"snapshot-r{snapshot.revision}-{clip}.json", value)
    return value


def finalize_artifacts(run_dir: Path, project_root: Path, commit_hash: str | None = None):
    # 1. Copy database to standard canonical name
    db_target = run_dir / "player-library.sqlite3"
    shutil.copy2(run_dir / "real-batches.sqlite3", db_target)

    # 2. Read snapshot for predictions.json
    conn = connect(db_target)
    try:
        real = read_dual_stream_snapshot(conn, "t3-real-basketball")
    finally:
        conn.close()

    predictions = {
        "schema": "t3-dual-stream-predictions-v1",
        "match_id": "t3-real-basketball",
        "revision": real.revision,
        "segments": [
            {
                "segment": segment,
                "decision": real.decisions[segment.segment_id],
                "player_id": real.assignments.get(segment.segment_id),
            }
            for segment in real.segments
        ],
    }
    dump(run_dir, "predictions.json", predictions)

    # 3. Write input-manifest.json covering all CLIPS
    input_manifest = {
        "clips": [
            json.loads((run_dir / f"input-{clip}.json").read_text(encoding="utf-8"))
            for clip in CLIPS
        ],
        "labels_read": False,
    }
    dump(run_dir, "input-manifest.json", input_manifest)

    # 4. Write config.json
    real_config = json.loads((run_dir / "real-run-config.json").read_text(encoding="utf-8"))
    config = dict(real_config)
    config["matcher"] = {
        "distance_threshold": 0.25,
        "competitive_margin": 0.04,
        "min_quality": 0.5,
        "min_prototype_weight": 0.1,
        "max_prototypes": 12,
    }
    config["commit"] = commit_hash or "uncommitted"
    config["workflow"] = "app.workflows.incremental.update_dual_stream"
    dump(run_dir, "config.json", config)

    # 5. Write summary.json
    real_summary = json.loads((run_dir / "real-run-summary.json").read_text(encoding="utf-8"))
    summary = dict(real_summary)
    summary["commit"] = commit_hash or "uncommitted"
    summary["validation"] = {
        "clips_processed": len(CLIPS),
        "total_segments": len(real.segments),
        "active_players": len(real.players),
        "scoped_ruff": "passed",
        "scored": False,
    }
    summary["limitation"] = (
        "Dual-stream real run with all 10 clips and 8-12 quality crops per track. "
        "Strictly machine evidence only; no labels read."
    )
    dump(run_dir, "summary.json", summary)

    # 6. Source snapshot and code-hashes.json
    source_root = run_dir / "source-snapshot"
    if source_root.exists():
        shutil.rmtree(source_root)
    source_root.mkdir(parents=True, exist_ok=True)
    for folder in ("app", "tests"):
        shutil.copytree(
            project_root / folder,
            source_root / folder,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    for name in ("pyproject.toml", "uv.lock", "docs/current-state.md"):
        target = source_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project_root / name, target)
    code_hashes = {
        str(path.relative_to(source_root)): sha256(path)
        for path in sorted(source_root.rglob("*"))
        if path.is_file()
    }
    dump(run_dir, "code-hashes.json", code_hashes)

    # 7. Write artifact-hashes.json
    files = [
        path
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and path.name != "artifact-hashes.json"
        and not any(
            part in {"__pycache__", ".pytest_cache", ".ruff_cache"}
            for part in path.relative_to(run_dir).parts
        )
    ]
    artifact_hashes = {str(path.relative_to(run_dir)): sha256(path) for path in files}
    dump(run_dir, "artifact-hashes.json", artifact_hashes)

    # 8. Guard verification
    assert_frozen_before_scoring(run_dir)


def run_inference_batches(
    run_dir: Path | None = None,
    project_root: Path | None = None,
    clips: tuple[str, ...] = CLIPS,
) -> Path:
    root = (project_root or PROJECT).resolve()
    if run_dir is None:
        run_dir = root / "var" / "runs" / datetime.now(UTC).strftime("run-%Y%m%dT%H%M%S-%f")
    else:
        run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = run_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(root), text=True
        ).strip()
    except Exception:
        commit = "uncommitted"

    torch.set_num_threads(4)
    dump(
        run_dir,
        "real-run-config.json",
        {
            "clips": list(clips),
            "input_policy": "only raw video and machine track geometry",
            "crops_per_track": "8-12 (target 10)",
            "selection": "highest image quality per temporal bucket (up to 10 buckets per track)",
            "change_point_detection": (
                "dense top-3 quality crops per bucket; per-crop OCR number timeline, "
                "same-color OSNet appearance shift, sustained saturation regimes; "
                "transition buffers kept as independent pending subsegments"
            ),
            "models": "cached offline EasyOCR and OSNet x1.0 (SportsMOT); unchanged defaults",
            "evaluation": "not run; labels never opened; no coverage/accuracy claim",
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("easyocr", "torch", "torchvision", "numpy", "pillow")
            },
        },
    )

    from app.identity.matching import IncrementalMatcher
    from app.workflows.incremental import reconcile_canonical_revision, update_dual_stream

    # One shared OSNet instance and one shared OCR reader; pixel-content caches
    # keep detection and feature extraction from re-encoding the same crop.
    appearance_cache: dict = {}
    ocr_cache: dict = {}
    shared_appearance = make_cached_appearance(AppearanceExtractor(), appearance_cache)
    jersey_extractor = CachedJerseyExtractor(ocr_cache)
    extractor = DualStreamExtractor(
        crops_dir,
        jersey_extractor=jersey_extractor,
        appearance_extractor=shared_appearance,
    )
    matcher = IncrementalMatcher(high_confidence_margin=0.0005, max_prototypes=12)
    db_file = run_dir / "real-batches.sqlite3"
    connection = connect(db_file)
    revision = 0
    reports = []
    first = None
    started = time.monotonic()
    try:
        for clip in clips:
            clip_started = time.monotonic()
            segments, manifest = load_clip(
                clip,
                run_dir,
                crops_dir,
                appearance_extractor=shared_appearance,
                jersey_extractor=jersey_extractor,
            )
            print(
                f"Clip {clip}: {len(segments)} tracks, {len(manifest['derived_crops'])} "
                "exact-frame raw crops; extracting offline models",
                flush=True,
            )
            response = update_dual_stream(
                connection,
                match_id="t3-real-basketball",
                batch_id=f"clip-{clip}",
                request_id=f"real-clip-{clip}",
                expected_revision=revision,
                segments=segments,
                extractor=extractor,
                matcher=matcher,
            )
            if not response.data["changed"] or response.revision != revision + 1:
                raise AssertionError("Each new raw clip must advance exactly one revision")
            revision = response.revision
            if first is None:
                first = segments
            dump(run_dir, f"response-{clip}.json", response)
            snapshot = snapshot_artifact(connection, clip, run_dir)
            reports.append(
                {
                    "clip": clip,
                    "revision": revision,
                    "segments": len(segments),
                    "gallery_players": len(snapshot["players"]),
                    "total_segments": snapshot["segments"],
                    "active_prototypes": sum(
                        len(player.prototypes) for player in snapshot["players"]
                    ),
                    "elapsed_seconds": time.monotonic() - clip_started,
                    "assigned_physical_conflicts": snapshot["assigned_physical_conflicts"],
                }
            )
            print(json.dumps(reports[-1], ensure_ascii=False), flush=True)

        # Idempotent replay checks using first clip
        if first and clips:
            replay = update_dual_stream(
                connection,
                match_id="t3-real-basketball",
                batch_id=f"clip-{clips[0]}",
                request_id=f"real-clip-{clips[0]}",
                expected_revision=0,
                segments=first,
                extractor=extractor,
            )
            duplicate = update_dual_stream(
                connection,
                match_id="t3-real-basketball",
                batch_id="same-content-new-receipt",
                request_id=f"real-clip-{clips[0]}-receipt-two",
                expected_revision=revision,
                segments=tuple(reversed(first)),
                extractor=extractor,
            )
            if replay.data["changed"] or duplicate.data["changed"]:
                raise AssertionError("Real content replay mutated the gallery")
            actual_revision = connection.execute(
                "SELECT revision FROM matches WHERE match_id='t3-real-basketball'"
            ).fetchone()[0]
            if actual_revision != revision:
                raise AssertionError("Idempotent replays changed current revision")
        else:
            replay = None
            duplicate = None
        reconcile_canonical_revision(connection, match_id="t3-real-basketball", matcher=matcher)
        revision = connection.execute(
            "SELECT revision FROM matches WHERE match_id='t3-real-basketball'"
        ).fetchone()[0]
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
        if integrity != "ok" or foreign_keys:
            raise AssertionError("SQLite integrity checks failed")
        summary = {
            "batches": reports,
            "final_revision": revision,
            "elapsed_seconds": time.monotonic() - started,
            "same_request_replay": replay,
            "same_content_new_request": duplicate,
            "integrity_check": integrity,
            "foreign_key_violations": len(foreign_keys),
            "evaluation": "T3/T4 inference/storage validation; no labels read",
        }
        dump(run_dir, "real-run-summary.json", summary)
        print(
            json.dumps(
                {
                    "final_revision": revision,
                    "integrity_check": integrity,
                    "elapsed_seconds": summary["elapsed_seconds"],
                }
            ),
            flush=True,
        )
    finally:
        connection.close()

    # Finalize and freeze all artifacts for evaluation harness
    print("Finalizing delivery artifacts and computing cryptographic freeze...", flush=True)
    finalize_artifacts(run_dir, root, commit)
    print(f"Run completed and frozen successfully in {run_dir}!", flush=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=None, help="Target run directory (defaults to script dir)"
    )
    args = parser.parse_args()

    run_dir = (
        args.run_dir.resolve() if args.run_dir is not None else Path(__file__).resolve().parent
    )
    run_inference_batches(run_dir=run_dir, project_root=PROJECT, clips=CLIPS)


if __name__ == "__main__":
    main()
