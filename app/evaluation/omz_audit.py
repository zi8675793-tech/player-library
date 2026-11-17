"""Supplementary post-score audit and real-video review sheets; no score changes.

Run with an existing OpenCV-capable interpreter. This module is not an inference
entry and does not load any model. Geometry and scoring remain frozen inputs.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from app.evaluation.omz_io import jsonl, read_json, sha, write_json
from app.evaluation.omz_score import align_identities, identity_diagnostics, score_rows


def audit(geometry, scoring, out):
    import cv2
    from PIL import Image, ImageDraw

    out.mkdir(parents=True, exist_ok=False)
    frozen = read_json(geometry / "geometry-freeze.json")
    for name, expected in frozen["files"].items():
        if sha(geometry / name) != expected:
            raise ValueError("Frozen geometry changed")
    t1 = Path(frozen["t1"])
    sources = read_json(geometry / "sources.json")
    b_frames = {(r["source"], r["frame_idx"]): r for r in jsonl(t1 / "B/frames.jsonl")}
    rows = list(jsonl(scoring / "B/sample-ledger.jsonl"))
    all_frames = list(jsonl(scoring / "B/frame-ledger.jsonl"))
    comparison = read_json(scoring / "comparison.json")
    selected = comparison["B"]["alignment"]["jersey:27"]
    known_other = [r for r in all_frames if r["identity"] == selected and r["truth"] != "jersey:27"]
    write_json(out / "selected27-known-other.json", known_other)
    # Supplemental legacy input validation covers every source listed by all four freezes.
    expected_sources, legacy_counts = {}, {}
    for name, legacy in read_json(geometry / "legacy-verification.json").items():
        manifest = read_json(Path(legacy["path"]) / "input-manifest.json")
        root = Path(manifest["input_root"])
        legacy_counts[name] = len(manifest["sources"])
        for record in manifest["sources"]:
            path = root / record["path"]
            expected = record["sha256"]
            if path in expected_sources and expected_sources[path] != expected:
                raise ValueError("Legacy input versions differ")
            expected_sources[path] = expected
    for path, expected in expected_sources.items():
        if sha(path) != expected:
            raise ValueError(f"Legacy input changed: {path}")
    write_json(
        out / "legacy-input-verification.json",
        {
            "passed": True,
            "runs": legacy_counts,
            "unique_files": len(expected_sources),
            "files": [{"path": str(p), "sha256": v} for p, v in sorted(expected_sources.items())],
        },
    )
    # Reuse actual fixed truth loci in adversarial metric tests; never modify predictions.
    collapsed = [{**r, "identity": "all", "status": "matched"} for r in rows]
    collapsed_scored = score_rows(collapsed, align_identities(collapsed))
    fragmented = [
        {**r, "identity": f"unique-{i:04}", "status": "matched"} for i, r in enumerate(rows)
    ]
    fragment_scored = score_rows(fragmented, align_identities(fragmented))
    missing = [{**r, "identity": None, "status": "unprocessed"} for r in rows]
    checks = {
        "all_one": {
            "correct": sum(r["correct"] for r in collapsed_scored),
            "known_wrong": sum(r["outcome"] == "known_wrong_identity" for r in collapsed_scored),
            "mixed": identity_diagnostics(collapsed)["mixed_identities"],
        },
        "each_sample_separate": {
            "correct27": sum(r["correct"] and r["truth"] == "jersey:27" for r in fragment_scored)
        },
        "missing_all": {
            "denominator": len(missing),
            "correct": sum(r["correct"] for r in score_rows(missing, align_identities(missing))),
        },
    }
    if checks["all_one"]["known_wrong"] != 48 or checks["each_sample_separate"]["correct27"] != 1:
        raise AssertionError("Adversarial audit failed")
    write_json(out / "actual-scope-adversarial-checks.json", checks)
    review = [r for r in rows if r["truth"] == "jersey:27"] + known_other
    wanted = defaultdict(set)
    for row in rows + known_other:
        wanted[row["video_id"]].add(row["frame_idx"])
    clips = {Path(s["path"]).stem: s for s in sources}
    images, pixel_checks = {}, []
    for clip, frames in wanted.items():
        source = clips[clip]
        capture = cv2.VideoCapture(source["path"])
        try:
            for index in range(max(frames) + 1):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError("Early EOF during independent visual audit")
                if index not in frames:
                    continue
                actual = hashlib.sha256(frame.tobytes()).hexdigest()
                if actual != b_frames[source["source"], index]["decoded_sha256"]:
                    raise ValueError("Decoded pixels differ from T1")
                pixel_checks.append({"video_id": clip, "frame_idx": index, "sha256": actual})
                images[clip, index] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
    panels = []
    for number, row in enumerate(review, 1):
        image = images[row["video_id"], row["frame_idx"]].copy()
        draw = ImageDraw.Draw(image)
        box = row["box"]
        draw.rectangle(box, outline=(0, 255, 0), width=3)
        candidate = b_frames[clips[row["video_id"]]["source"], row["frame_idx"]]
        first = row.get("old_to_raw", {})
        if first.get("best_index") is not None:
            draw.rectangle(candidate["boxes"][first["best_index"]], outline=(255, 170, 0), width=3)
        if row.get("native_box"):
            draw.rectangle(row["native_box"], outline=(255, 40, 180), width=3)
        width, height = box[2] - box[0], box[3] - box[1]
        crop = image.crop(
            (
                max(0, int(box[0] - width * 0.6)),
                max(0, int(box[1] - height * 0.25)),
                min(image.width, int(box[2] + width * 0.6)),
                min(image.height, int(box[3] + height * 0.25)),
            )
        )
        crop.thumbnail((360, 250))
        panel = Image.new("RGB", (380, 330), "#141923")
        panel.paste(crop, ((380 - crop.width) // 2, 70))
        text = (
            f"{number:02} clip {row['video_id']} / frame {row['frame_idx']} / {row['truth']}\n"
            f"{row['outcome']} / ID {row['identity']}\n"
            f"old/raw IoU {first.get('best_iou', 0):.3f}"
        )
        ImageDraw.Draw(panel).multiline_text((8, 8), text, fill="white", spacing=3)
        panels.append(panel)
    for offset in range(0, len(panels), 16):
        sheet = Image.new("RGB", (1520, 1320), "#141923")
        for i, panel in enumerate(panels[offset : offset + 16]):
            sheet.paste(panel, ((i % 4) * 380, (i // 4) * 330))
        sheet.save(out / f"review-{offset // 16 + 1:02}.jpg", quality=92)
    breakdown = {}
    for mode in ("A", "B"):
        target = list(jsonl(scoring / mode / "player27-ledger.jsonl"))
        breakdown[mode] = {
            "detection_missing_or_geometry": dict(
                Counter(
                    r["old_to_raw"]["status"]
                    for r in target
                    if r["status"] == "detection_missing_or_geometry"
                )
            ),
            "no_retained_real_detection": dict(
                Counter(
                    r["raw_to_native"]["status"]
                    for r in target
                    if r["status"] == "no_retained_real_detection"
                )
            ),
        }
    write_json(out / "failure-breakdown.json", breakdown)
    write_json(
        out / "real-frame-audit.json",
        {
            "passed": True,
            "verified_real_frames": len(pixel_checks),
            "checks": pixel_checks,
            "panels": len(panels),
            "legend": "green=old reference; orange=best raw detector; magenta=matched native",
            "scope": "all 118 labelled sample positions plus selected27 known-other observation frames",
            "score_unchanged": True,
        },
    )
    print(
        json.dumps(
            {
                "pixel_frames": len(pixel_checks),
                "panels": len(panels),
                "legacy_files": len(expected_sources),
                "failures": breakdown,
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--scoring", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    audit(args.geometry.resolve(), args.scoring.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
