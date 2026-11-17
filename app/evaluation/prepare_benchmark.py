"""Phase A: freeze all machine geometry with an active secret-file read barrier."""

import argparse
import math
import os
import shutil
import sqlite3
import struct
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from app.evaluation.golden_freeze import (
    blind_inputs,
    contained,
    read_json,
    read_jsonl,
    sha,
    verify_run,
    write_json,
    write_jsonl,
)
from app.evaluation.golden_geometry import GeometryRule, iou, match_boxes


def key(row):
    return (
        str(row["video_id"]),
        str(row.get("track_id", row.get("local_track_id"))),
        row["frame_idx"],
    )


def database_evidence(run, predictions):
    """Read SQLite in immutable mode; compare export to latest committed facts."""
    path = run / "player-library.sqlite3"
    if any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-journal")):
        raise ValueError("Database has live transaction sidecars")
    conn = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Database integrity failed")
        if conn.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("Database foreign keys failed")
        match = predictions["match_id"]
        revision = conn.execute(
            "SELECT revision FROM matches WHERE match_id=?", (match,)
        ).fetchone()
        if revision is None or revision[0] != predictions["revision"]:
            raise ValueError("Prediction/database revision mismatch")
        stored = {
            r["segment_id"]: r
            for r in conn.execute("SELECT * FROM dual_stream_segments WHERE match_id=?", (match,))
        }
        exports = {r["segment"]["segment_id"]: r for r in predictions["segments"]}
        if len(exports) != len(predictions["segments"]) or exports.keys() != stored.keys():
            raise ValueError("Prediction/database segment inventory mismatch")
        diagnostics = {}
        for sid, row in exports.items():
            db = stored[sid]
            if row["segment"] != read_payload(db["segment_json"]):
                raise ValueError(f"Prediction/database geometry mismatch: {sid}")
            latest = conn.execute(
                "SELECT json FROM dual_stream_decisions WHERE match_id=? AND segment_id=? "
                "AND revision<=? ORDER BY revision DESC LIMIT 1",
                (match, sid, revision[0]),
            ).fetchone()
            if latest is None or row["decision"] != read_payload(latest[0]):
                raise ValueError(f"Prediction/database decision mismatch: {sid}")
            decision = row["decision"]
            assigned = decision["outcome"] in ("matched_existing", "created_new_identity")
            pid = (
                (
                    decision.get("target_player_id")
                    or decision.get("evidence", {}).get("new_player_id")
                )
                if assigned
                else None
            )
            if pid != row["player_id"] or (assigned and not pid):
                raise ValueError("Invalid decision identity projection")
            projection = conn.execute(
                "SELECT player_id FROM assignments WHERE match_id=? AND segment_id=? AND current=1",
                (match, sid),
            ).fetchall()
            legacy = pid if pid is None or pid.startswith("pl_") else "pl_" + pid
            if len(projection) != 1 or projection[0][0] != legacy:
                raise ValueError(f"Assignment projection mismatch: {sid}")
            feature = read_payload(db["feature_json"])
            if "vector_blob" in db.keys():
                feature["appearance"]["vector"] = list(struct.unpack("<512f", db["vector_blob"]))
            diagnostics[sid] = {"decision": decision, "feature": feature}
        return diagnostics, {
            "integrity": "ok",
            "foreign_key_violations": 0,
            "revision": revision[0],
            "segments": len(stored),
            "assigned_segments": sum(r["player_id"] is not None for r in exports.values()),
        }
    finally:
        conn.close()


def read_payload(value):
    import json

    return json.loads(value)


def load_machine(data):
    frames, samples, sources, inputs = defaultdict(list), [], [], {}

    def pin(path):
        path = Path(path)
        inputs[str(path.resolve())] = sha(path)

    for folder in sorted((data / "track").iterdir()):
        if not folder.is_dir():
            continue
        clip = folder.name
        meta_path, tracks_path = folder / "video_meta.json", folder / "tracks.jsonl"
        pin(meta_path)
        pin(tracks_path)
        meta = read_json(meta_path)
        fps, frame_count = meta["fps"], meta["frame_count"]
        if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
            raise ValueError("Invalid machine video metadata")
        seen = {}
        for row in read_jsonl(tracks_path):
            locus = key(row)
            if locus in seen or locus[0] != clip or not 0 <= locus[2] < frame_count:
                raise ValueError("Duplicate/mismatched machine observation")
            iou(row["bbox_xyxy"], row["bbox_xyxy"])
            seen[locus] = row
            frames[clip, locus[2]].append(row)
        sample_path = data / "identity" / clip / "samples.jsonl"
        pin(sample_path)
        sample_keys = set()
        for row in read_jsonl(sample_path):
            locus = key(row)
            if (
                locus in sample_keys
                or locus not in seen
                or row["bbox_xyxy"] != seen[locus]["bbox_xyxy"]
            ):
                raise ValueError("Machine sample geometry or inventory mismatch")
            sample_keys.add(locus)
            crop = contained(data, row["crop_path"])
            if not crop.resolve().is_relative_to((data / "track").resolve()):
                raise ValueError("Sample is not a machine track crop")
            pin(crop)
            if sha(crop) != row["crop_sha256"]:
                raise ValueError("Machine crop fingerprint mismatch")
            samples.append({**row, "track_id": str(row["track_id"])})
        sources.append({"video_id": clip, "fps": fps, "frame_count": frame_count})
    if not sources:
        raise ValueError("No machine sources")
    return frames, samples, sources, inputs


def _prepare(run, data, out):
    if out.resolve().is_relative_to(run) or out.resolve().is_relative_to(data):
        raise ValueError("Prepare output must be outside immutable inputs and source run")
    initial_manifest = sha(run / "artifact-hashes.json")
    hashes = verify_run(run, initial_manifest)
    predictions = read_json(run / "predictions.json")
    if predictions.get("schema") != "t3-dual-stream-predictions-v1":
        raise ValueError("Unsupported T3 prediction schema")
    diagnostics, db_check = database_evidence(run, predictions)
    manifest = read_json(run / "input-manifest.json")
    if manifest.get("labels_read") is not False:
        raise ValueError("T3 did not declare isolated machine input")
    frames, samples, sources, inputs = load_machine(data)
    for clip in manifest["clips"]:
        for record in clip["inputs"]:
            path = contained(data, record["path"])
            if path.parts[len(data.parts)] not in {"video", "track"}:
                raise ValueError("T3 input outside raw machine whitelist")
            if sha(path) != record["sha256"]:
                raise ValueError("Original T3 machine input changed")
            inputs[str(path.resolve())] = record["sha256"]
    processed = {str(c["clip"]) for c in manifest["clips"]}
    if processed != set(map(str, read_json(run / "config.json")["clips"])):
        raise ValueError("T3 clip inventory mismatch")
    if not processed <= {s["video_id"] for s in sources}:
        raise ValueError("Missing processed machine source")
    for source in sources:
        source["processed"] = source["video_id"] in processed
    points = defaultdict(list)
    for row in predictions["segments"]:
        seg = row["segment"]
        if seg["video_id"] not in processed or seg["observation_count"] != len(seg["observations"]):
            raise ValueError("Invalid predicted segment source/count")
        seen = set()
        for obs in seg["observations"]:
            locus = key(obs)
            if locus in seen or locus[:2] != (seg["video_id"], str(seg["local_track_id"])):
                raise ValueError("Duplicate or mismatched prediction observation")
            seen.add(locus)
            points[locus[0], locus[2]].append(
                {
                    "video_id": locus[0],
                    "track_id": locus[1],
                    "frame_idx": locus[2],
                    "box": obs["bbox_xyxy"],
                    "identity": row["player_id"],
                    "segment_id": seg["segment_id"],
                }
            )
    rule = GeometryRule()
    correspondence, members, conflicts = [], [], []
    for frame_key in sorted(set(frames) | set(points)):
        raw, predicted = frames[frame_key], points[frame_key]
        forward = match_boxes([r["bbox_xyxy"] for r in raw], [r["box"] for r in predicted], rule)
        confirmed = {r["index"] for r in forward if r["status"] == "matched"}
        for observation, result in zip(raw, forward):
            pid, sid = None, None
            if frame_key[0] not in processed:
                status = "unprocessed"
            elif result["status"] == "matched":
                hit = predicted[result["index"]]
                pid, sid = hit["identity"], hit["segment_id"]
                status = "matched" if pid is not None else "unassigned"
            else:
                status = result["status"]
            locus = key(observation)
            correspondence.append(
                {
                    "video_id": locus[0],
                    "track_id": locus[1],
                    "frame_idx": locus[2],
                    "box": observation["bbox_xyxy"],
                    "identity": pid,
                    "segment_id": sid,
                    "status": status,
                    "geometry": result,
                }
            )
        members.extend({**r, "geometry_confirmed": i in confirmed} for i, r in enumerate(predicted))
        for i, a in enumerate(predicted):
            for b in predicted[i + 1 :]:
                if a["identity"] is None or a["identity"] != b["identity"]:
                    continue
                overlap = iou(a["box"], b["box"])
                conflicts.append(
                    {
                        "video_id": frame_key[0],
                        "frame_idx": frame_key[1],
                        "identity": a["identity"],
                        "segment_ids": [a["segment_id"], b["segment_id"]],
                        "boxes": [a["box"], b["box"]],
                        "iou": overlap,
                        "status": "physical_conflict"
                        if overlap <= rule.physical_separation_iou
                        else "overlap_pending",
                    }
                )
    out.mkdir(parents=True, exist_ok=False)
    rules = {
        "geometry": rule.as_dict(),
        "target": "jersey:27",
        "denominator": 70,
        "coverage_pass": 57,
        "coverage_excellent": 62,
        "max_fragments": 2,
        "min_exposure_seconds": 50.0,
        "max_known_pollution": 0,
        "max_physical_conflicts": 0,
        "identity_alignment": "global all-label injective maximum sample-count Hungarian; sorted ties; one ID per truth; no fragment sum",
        "duration": "same mapped identity, correct geometry-confirmed unique (video,frame) only / fps; no interval filling",
        "purity": "unlabelled observations stay unknown; empty gallery purity is null",
        "causality": "persisted machine evidence only; missing diagnostics cannot prove OCR/appearance/cannot-link causation",
    }
    write_json(out / "rules.json", rules)
    write_json(out / "sources.json", sources)
    write_json(out / "machine-samples.json", samples)
    write_json(out / "segment-diagnostics.json", diagnostics)
    write_json(out / "physical-conflicts.json", conflicts)
    write_json(out / "database-verification.json", db_check)
    write_jsonl(out / "correspondence.jsonl", correspondence)
    write_jsonl(out / "prediction-members.jsonl", members)
    code = {str(p.resolve()): sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    implementation = out / "implementation"
    implementation.mkdir()
    for path in code:
        shutil.copyfile(path, implementation / Path(path).name)
    geometry_hashes = {
        str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()
    }
    write_json(out / "artifact-hashes.json", geometry_hashes)
    geometry_hashes["artifact-hashes.json"] = sha(out / "artifact-hashes.json")
    # Detect changes while preparing, before sealing the new independent receipt.
    verify_run(run, initial_manifest)
    if hashes != read_json(run / "artifact-hashes.json"):
        raise ValueError("T3 artifact index changed during prepare")
    for path, digest in inputs.items():
        if sha(path) != digest:
            raise ValueError(f"Machine input changed during prepare: {path}")
    for path, digest in code.items():
        if sha(path) != digest:
            raise ValueError(f"Evaluation code changed during prepare: {path}")
    freeze = {
        "schema": "t4-golden-freeze-v1",
        "prepared_at": datetime.now(UTC).isoformat(),
        "run_dir": str(run),
        "test_data": str(data),
        "run_manifest_sha256": initial_manifest,
        "files": geometry_hashes,
        "code_files": code,
        "machine_inputs": [{"path": p, "sha256": h} for p, h in sorted(inputs.items())],
        "rules": rules,
        "labels_read": False,
        "counts": dict(Counter(r["status"] for r in correspondence)),
    }
    write_json(out / "geometry-freeze.json", freeze)
    return {
        "geometry_dir": str(out),
        "freeze_sha256": sha(out / "geometry-freeze.json"),
        "process_id": os.getpid(),
        "labels_read": False,
        "counts": freeze["counts"],
    }


def prepare_benchmark(run_dir, test_data, out_dir):
    with blind_inputs():
        return _prepare(
            Path(run_dir).resolve(), Path(test_data).resolve(), Path(out_dir).absolute()
        )


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_benchmark(args.run_dir, args.test_data, args.out_dir), ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
