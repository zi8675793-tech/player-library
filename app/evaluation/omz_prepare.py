"""Freeze geometry for ALL people before a separate scoring process reads labels.

Usage: python -m app.evaluation.omz_prepare --t1 RUN --data ../test --out NEW_DIR
"""

import argparse
import json
import shutil
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from app.evaluation.guard import assert_frozen_before_scoring
from app.evaluation.omz_geometry import GeometryRule, iou, match_boxes, synthetic_checks
from app.evaluation.omz_io import array_items, jsonl, read_json, sha, verify_t1, write_json


def key(row):
    return str(row["video_id"]), str(row["track_id"]), row["frame_idx"]


def load_machine(data, sources):
    frames, samples, files = defaultdict(list), [], []
    for source in sources:
        clip = Path(source["path"]).stem
        for relative in [
            f"track/{clip}/tracks.jsonl",
            f"track/{clip}/video_meta.json",
            f"identity/{clip}/samples.jsonl",
        ]:
            files.append({"path": str((data / relative).resolve()), "sha256": sha(data / relative)})
        for row in jsonl(data / f"track/{clip}/tracks.jsonl"):
            if str(row["video_id"]) != clip or not 0 <= row["frame_idx"] < source["decoded_frames"]:
                raise ValueError("Machine observation source/frame mismatch")
            frames[source["source"], row["frame_idx"]].append(row)
        for row in jsonl(data / f"identity/{clip}/samples.jsonl"):
            samples.append(row)
    observations = {key(r): r for rows in frames.values() for r in rows}
    if len(observations) != sum(map(len, frames.values())):
        raise ValueError("Duplicate machine observation key")
    for sample in samples:
        if key(sample) not in observations:
            raise ValueError("Sample has no exact real-frame machine observation")
        if sample["bbox_xyxy"] != observations[key(sample)]["bbox_xyxy"]:
            raise ValueError("Sample/observation geometry differs")
    return frames, samples, files


def retained_points(path, sources):
    points, tracks, excluded = defaultdict(list), [], Counter()
    for track in array_items(path, "tracks"):
        if not track["native_retained"]:
            excluded["non_retained_instances"] += 1
            continue
        if len({len(track[k]) for k in ("timestamps", "boxes", "origins")}) != 1:
            raise ValueError("Track parallel arrays disagree")
        if len(set(track["timestamps"])) != len(track["timestamps"]):
            raise ValueError("Duplicate native timestamps")
        tracks.append({k: v for k, v in track.items() if k != "appearance"})
        for frame, box, origin in zip(track["timestamps"], track["boxes"], track["origins"]):
            source = track["source"]
            if not 0 <= frame < sources[source]["decoded_frames"]:
                raise ValueError("Retained history outside real source frames")
            points[source, frame].append(
                {
                    "source": source,
                    "frame_idx": frame,
                    "box": box,
                    "origin": origin,
                    "instance": track["instance"],
                    "identity": str(track["native_id"]),
                }
            )
            excluded["retained_" + origin] += 1
    return points, tracks, dict(excluded)


def history_signature(tracks):
    return sorted(
        json.dumps([t["source"], t["native_id"], t["timestamps"], t["boxes"]]) for t in tracks
    )


def correspond_frame(old, raw_boxes, native, processed, rule):
    """Join two independent geometry-only bijections; never look up an old ID in OMZ."""
    real = [p for p in native if p["origin"] == "detection"]
    old_to_raw = match_boxes([r["bbox_xyxy"] for r in old], raw_boxes, rule)
    raw_to_native = match_boxes(raw_boxes, [p["box"] for p in real], rule)
    rows = []
    for observation, first in zip(old, old_to_raw):
        result = {
            "video_id": str(observation["video_id"]),
            "track_id": str(observation["track_id"]),
            "frame_idx": observation["frame_idx"],
            "box": observation["bbox_xyxy"],
            "identity": None,
            "instance": None,
            "old_to_raw": first,
        }
        if not processed:
            status = "unprocessed"
        elif first["status"] != "matched":
            status = (
                "detection_missing_or_geometry"
                if first["status"] in ("no_candidate", "below_iou")
                else "old_detection_" + first["status"]
            )
        else:
            second = raw_to_native[first["index"]]
            result["raw_to_native"] = second
            result["raw_box"] = raw_boxes[first["index"]]
            if second["status"] != "matched":
                status = (
                    "no_retained_real_detection"
                    if second["status"] in ("no_candidate", "below_iou")
                    else "native_" + second["status"]
                )
            else:
                status = "matched"
                point = real[second["index"]]
                result.update(
                    identity=point["identity"],
                    instance=point["instance"],
                    native_box=point["box"],
                    origin=point["origin"],
                )
        result["status"] = status
        rows.append(result)
    native_raw = {r["index"]: i for i, r in enumerate(raw_to_native) if r["status"] == "matched"}
    linked = {r["instance"]: r for r in rows if r["status"] == "matched"}
    members = []
    for index, point in enumerate(real):
        row = {
            **point,
            "raw_detection_confirmed": index in native_raw,
            "old_locus": None,
            "raw_box": None,
        }
        if index in native_raw:
            row["raw_box"] = raw_boxes[native_raw[index]]
        if point["instance"] in linked:
            link = linked[point["instance"]]
            row["old_locus"] = [link["video_id"], link["track_id"], link["frame_idx"]]
        members.append(row)
    return rows, members


def physical_pairs(members, rule):
    """Only same-source, same-frame, confirmed raw detections are physical evidence."""
    pairs = []
    for i, a in enumerate(members):
        for b in members[i + 1 :]:
            if (
                a["identity"] != b["identity"]
                or a["source"] != b["source"]
                or a["frame_idx"] != b["frame_idx"]
            ):
                continue
            if not (a["raw_detection_confirmed"] and b["raw_detection_confirmed"]):
                continue
            overlap = iou(a["raw_box"], b["raw_box"])
            pairs.append(
                {
                    "source": a["source"],
                    "frame_idx": a["frame_idx"],
                    "identity": a["identity"],
                    "instances": [a["instance"], b["instance"]],
                    "boxes": [a["raw_box"], b["raw_box"]],
                    "iou": overlap,
                    "status": "physical_conflict"
                    if overlap <= rule.physical_separation_iou
                    else "overlap_pending",
                }
            )
    return pairs


def prepare(t1, data, out, legacy):
    out.mkdir(parents=True, exist_ok=False)
    rule = GeometryRule()
    rules = {
        "schema_version": "omz-independent-evaluation-v1",
        "geometry": rule.as_dict(),
        "matching": "same video + exact zero-based real frame; old->raw->retained detection; "
        "mutual best IoU with two-sided margin; duplicates stay ambiguous",
        "identity_alignment": "one global predicted ID per truth class, injective maximum total "
        "sample count; deterministic lexicographic tie; reused for duration",
        "denominator": "all fixed target samples including missing, ambiguous, unprocessed; "
        "jersey:27 must have exactly 70; threshold 57",
        "duration": "only matched real observation frames / fps, same global ID as sample mapping; "
        "no interpolation or interval filling; report selected-range denominator too",
        "unknown": "unlabelled members are neither correct nor wrong; enumerate all",
        "label_scope": "annotations.csv and target selection ranges are proxy truth; "
        "existing old report was consulted for scope, this is not a blind dataset",
        "frozen_at": datetime.now(UTC).isoformat(),
    }
    write_json(out / "rules.json", rules)
    write_json(out / "synthetic-geometry.json", synthetic_checks(rule))
    manifest, validation = verify_t1(t1)
    write_json(out / "t1-verification-before.json", validation)
    for name in manifest["engine_files"]:
        if "/__pycache__/" in name and name.endswith(".pyc"):
            destination = out / "runtime-snapshot" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(t1.parent.parent / name, destination)
    legacy_checks = {}
    for name, path in legacy.items():
        hashes = assert_frozen_before_scoring(path)
        legacy_checks[name] = {
            "path": str(path.resolve()),
            "artifact_count": len(hashes),
            "artifact_index_sha256": sha(path / "artifact-hashes.json"),
        }
    write_json(out / "legacy-verification.json", legacy_checks)
    frames, samples, files = load_machine(data, manifest["sources"])
    write_json(out / "machine-inputs.json", files)
    write_json(out / "machine-samples.json", samples)
    write_json(out / "sources.json", manifest["sources"])
    b_frames = {(r["source"], r["frame_idx"]): r for r in jsonl(t1 / "B/frames.jsonl")}
    decode = {(r["source"], r["frame_idx"]): r for r in jsonl(t1 / "decode-audit.jsonl")}
    if b_frames.keys() != decode.keys():
        raise ValueError("Real frame ledgers differ")
    for k, frame in b_frames.items():
        if frame["decoded_sha256"] != decode[k]["decoded_sha256"]:
            raise ValueError("Decoded frame hash differs")
    a_frames = {
        (s, r["frame_id"]): r
        for s, source in enumerate(read_json(t1 / "A/detections.json"))
        for r in source
    }
    for k, frame in a_frames.items():
        if any(frame[f] != b_frames[k][f] for f in ("boxes", "scores")):
            raise ValueError("A/B prefix detections differ")
    summaries = {}
    for mode, native_path, detected in [
        ("A", "B/prefix.json", a_frames),
        ("B", "B/final.json", b_frames),
    ]:
        print(f"Reading frozen {mode} tracks", flush=True)
        points, tracks, exclusions = retained_points(t1 / native_path, manifest["sources"])
        if mode == "A":
            official = [
                {"source": s, "native_id": t["id"], **t}
                for s, ts in enumerate(read_json(t1 / "A/history.json"))
                for t in ts
            ]
            if history_signature(tracks) != history_signature(official):
                raise ValueError("A origin bridge requires EXACT native history equality")
        mode_out = out / mode
        mode_out.mkdir()
        counts, conflicts = Counter(), []
        with (
            (mode_out / "correspondence.jsonl").open("w") as ledger,
            (mode_out / "prediction-members.jsonl").open("w") as member_file,
        ):
            for frame_key in sorted(set(frames) | set(points)):
                raw = detected.get(frame_key)
                rows, members = correspond_frame(
                    frames.get(frame_key, []),
                    raw["boxes"] if raw else [],
                    points.get(frame_key, []),
                    raw is not None,
                    rule,
                )
                for row in rows:
                    ledger.write(json.dumps(row) + "\n")
                    counts[row["status"]] += 1
                for member in members:
                    member_file.write(json.dumps(member) + "\n")
                conflicts.extend(physical_pairs(members, rule))
        write_json(mode_out / "physical-conflicts.json", conflicts)
        summaries[mode] = {
            "correspondence": dict(counts),
            "origins": exclusions,
            "retained_instances": len(tracks),
            "global_ids": len({t["native_id"] for t in tracks}),
            "processed_frames": len(detected),
            "physical_pairs": len(conflicts),
        }
    write_json(out / "geometry-summary.json", summaries)
    implementation = out / "implementation"
    implementation.mkdir()
    for path in sorted(Path(__file__).parent.glob("omz_*.py")):
        shutil.copyfile(path, implementation / path.name)
    hashes = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
    write_json(
        out / "geometry-freeze.json",
        {
            "completed_at": datetime.now(UTC).isoformat(),
            "t1": str(t1.resolve()),
            "data": str(data.resolve()),
            "files": hashes,
        },
    )
    print(json.dumps(summaries, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path)
    args = parser.parse_args()
    legacy = (
        {name: args.legacy_root / name for name in ("cold", "forward", "reverse", "shuffled")}
        if args.legacy_root
        else {}
    )
    prepare(args.t1, args.data, args.out, legacy)


if __name__ == "__main__":
    main()
