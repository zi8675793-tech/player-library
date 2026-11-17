"""Offline annotation audit. Reads answers only after verifying frozen predictions.

No inference, feature extraction, threshold changes, or upstream writes. Review
records are independent visual judgements, never predictions used as ground truth.
Protocol A calls the unmodified scorer. Protocol B preserves the original cohorts
and denominators, but counts only reviewed target identities in those cohorts.
"""

import argparse
import csv
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.golden_freeze import read_json, read_jsonl, sha, verify_prepared  # noqa: E402
from app.evaluation.golden_geometry import GeometryRule, iou  # noqa: E402
from app.evaluation.golden_score import align_identities, score_benchmark  # noqa: E402

PROTOCOL_A = "A-original-selected-range-v1"
PROTOCOL_B = "B-reviewed-per-sample-scope-v2"
DENOMINATORS = {"jersey:27": 70, "jersey:15": 48}


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_rows(path, rows):
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            )


def locus(row):
    return str(row["video_id"]), str(row["track_id"]), int(row["frame_idx"])


def input_hashes(test_data, annotations):
    """Include every upstream file, including crops, videos and unused answers."""
    paths = sorted(p.resolve() for p in Path(test_data).rglob("*") if p.is_file())
    paths.append(Path(annotations).resolve())
    return {str(p): sha(p) for p in sorted(set(paths))}


def verify_hashes(manifest):
    for name, expected in manifest.items():
        if sha(name) != expected:
            raise ValueError(f"Read-only input changed: {name}")


def safe_output(out, protected):
    out = Path(out).resolve()
    for path in protected:
        path = Path(path).resolve()
        if out == path or out.is_relative_to(path) or path.is_relative_to(out):
            raise ValueError(f"Output overlaps read-only source: {path}")
    if out.exists():
        raise FileExistsError(f"Audit output already exists: {out}")
    return out


def in_range(frame, scope):
    return scope["start_frame"] <= frame <= scope["end_frame"]


def validate_scope(sample, declared, allowed, observations):
    """Fail closed: a whole mixed track is suspicious until explicitly partitioned."""
    frame = sample["frame_idx"]
    track = str(sample["track_id"])
    declared = [s for s in declared if str(s["track_id"]) == track and in_range(frame, s)]
    allowed = [s for s in allowed if str(s["track_id"]) == track and in_range(frame, s)]
    if len(declared) != 1:
        return "outside_declared_segment"
    if len(allowed) != 1:
        return "outside_or_overlapping_reviewed_range"
    scope = allowed[0]
    if scope.get("truth") is None:
        return "scope_suspicious"
    identities = {
        r["suggested_truth"]
        for r in observations
        if str(r["track_id"]) == track
        and in_range(r["frame_idx"], scope)
        and r.get("suggested_truth") is not None
    }
    if identities - {scope["truth"]}:
        return "scope_suspicious"
    return "valid"


def validate_review_ranges(selection, ranges):
    for scope in ranges:
        start, end = scope["start_frame"], scope["end_frame"]
        if type(start) is not int or type(end) is not int or start > end:
            raise ValueError("Invalid review range")
        if not any(
            str(s["track_id"]) == str(scope["track_id"])
            and s["start_frame"] <= start <= end <= s["end_frame"]
            for s in selection
        ):
            raise ValueError("Review range escapes original selection")
        if not scope.get("reason") or not scope.get("evidence_images"):
            raise ValueError("Review range lacks image evidence or reason")
    for i, a in enumerate(ranges):
        for b in ranges[i + 1 :]:
            if str(a["track_id"]) == str(b["track_id"]) and max(
                a["start_frame"], b["start_frame"]
            ) <= min(a["end_frame"], b["end_frame"]):
                raise ValueError("Overlapping review ranges")
    for original in selection:
        covered = sum(
            s["end_frame"] - s["start_frame"] + 1
            for s in ranges
            if str(s["track_id"]) == str(original["track_id"])
            and original["start_frame"]
            <= s["start_frame"]
            <= s["end_frame"]
            <= original["end_frame"]
        )
        if covered != original["end_frame"] - original["start_frame"] + 1:
            raise ValueError("Review must partition every selected frame, including unknowns")


def audit_samples(test_data, annotations, review):
    root = Path(test_data).resolve()
    entries = list(csv.DictReader(Path(annotations).read_text(encoding="utf-8-sig").splitlines()))
    clip_labels = {str(r["video_id"]): "jersey:" + str(r["jersey_number"]) for r in entries}
    reviewed = review["samples"]
    if len({r["sample_id"] for r in reviewed}) != len(reviewed):
        raise ValueError("Duplicate reviewed sample")
    by_id = {r["sample_id"]: r for r in reviewed}
    rows, clips, raw_tracks = [], {}, {}
    for clip, original_truth in sorted(clip_labels.items(), key=lambda item: int(item[0])):
        base = root / "target" / clip
        selection = read_json(base / "selection.json")
        if "jersey:" + str(selection["event_jersey_number"]) != original_truth:
            raise ValueError("Selection and annotations disagree")
        ranges = review["clips"][clip]["allowed_ranges"]
        validate_review_ranges(selection["track_segments"], ranges)
        samples = read_jsonl(base / "samples.jsonl")
        track_rows = read_jsonl(root / "track" / clip / "tracks.jsonl")
        indexed = defaultdict(list)
        for raw in track_rows:
            indexed[raw["frame_idx"]].append(raw)
        raw_tracks[clip] = track_rows
        observations = []
        for sample in samples:
            decision = by_id.get(sample["sample_id"])
            if decision is None or decision["crop_sha256"] != sha(root / sample["crop_path"]):
                raise ValueError("Missing review or changed reviewed crop")
            if sample.get("crop_sha256") != decision["crop_sha256"]:
                raise ValueError("Upstream crop manifest mismatch")
            observations.append({**sample, **decision})
        declared_review = [{**s, "truth": original_truth} for s in selection["track_segments"]]
        for sample, decision in zip(samples, observations, strict=True):
            raw = [r for r in indexed[sample["frame_idx"]] if r["track_id"] == sample["track_id"]]
            if len(raw) != 1:
                raise ValueError("Fixed sample lacks a unique selected-track row")
            overlap = iou(sample["bbox_xyxy"], raw[0]["bbox_xyxy"])
            if overlap < GeometryRule().min_iou:
                raise ValueError("Fixed sample does not match selected raw track")
            declared_status = validate_scope(
                sample, selection["track_segments"], declared_review, observations
            )
            status = validate_scope(sample, selection["track_segments"], ranges, observations)
            if declared_status == "outside_declared_segment":
                raise ValueError("Sample outside declared original segment")
            true_scope = [
                s
                for s in ranges
                if s["track_id"] == sample["track_id"] and in_range(sample["frame_idx"], s)
            ]
            if status == "valid" and true_scope[0]["truth"] != decision["suggested_truth"]:
                raise ValueError("Per-sample truth conflicts with reviewed range")
            if not decision.get("reason") or not decision.get("evidence_images"):
                raise ValueError("Per-sample review lacks image evidence or reason")
            if not 0 <= decision["confidence"] <= 1:
                raise ValueError("Invalid review confidence")
            suspicious = decision["suggested_truth"] != original_truth or status != "valid"
            selected_segments = [
                s
                for s in selection["track_segments"]
                if s["track_id"] == sample["track_id"] and in_range(sample["frame_idx"], s)
            ]
            rows.append(
                {
                    **sample,
                    "clip": clip,
                    "original_truth": original_truth,
                    "suggested_truth": decision["suggested_truth"],
                    "confidence": decision["confidence"],
                    "reason": decision["reason"],
                    "evidence_images": decision["evidence_images"],
                    "crop_sha256": decision["crop_sha256"],
                    "raw_track_iou": overlap,
                    "raw_track_bbox": raw[0]["bbox_xyxy"],
                    "other_track_ious": [
                        {"track_id": r["track_id"], "iou": iou(sample["bbox_xyxy"], r["bbox_xyxy"])}
                        for r in indexed[sample["frame_idx"]]
                        if r["track_id"] != sample["track_id"]
                        and iou(sample["bbox_xyxy"], r["bbox_xyxy"]) >= GeometryRule().min_iou
                    ],
                    "track_segments": selected_segments,
                    "in_declared_segment": True,
                    "original_scope_status": declared_status,
                    "reviewed_scope_status": status,
                    "suspicious": suspicious,
                    "flags": (["identity_disagrees_with_original"] if suspicious else [])
                    + (
                        ["mixed_identity_original_segment"]
                        if declared_status == "scope_suspicious"
                        else []
                    ),
                    "allowed_ranges": true_scope,
                }
            )
        clips[clip] = {"selection": selection, "allowed_ranges": ranges}
    if set(by_id) != {r["sample_id"] for r in rows}:
        raise ValueError("Review sample set differs from fixed sample set")
    if Counter(r["original_truth"] for r in rows) != Counter(DENOMINATORS):
        raise ValueError("Fixed 70/48 cohorts changed")
    return rows, clips, raw_tracks


def contact_sheet(items, path, *, columns=4, cell=(260, 360)):
    sheet = Image.new("RGB", (columns * cell[0], math.ceil(len(items) / columns) * cell[1]), "#eee")
    draw = ImageDraw.Draw(sheet)
    for i, (label, image) in enumerate(items):
        image = image.copy().convert("RGB")
        image.thumbnail((cell[0] - 10, cell[1] - 50))
        x, y = (i % columns) * cell[0], (i // columns) * cell[1]
        sheet.paste(image, (x + (cell[0] - image.width) // 2, y + 40))
        draw.multiline_text((x + 5, y + 5), label, fill="black")
    sheet.save(path, quality=95)


def render_evidence(test_data, out, rows, clips, raw_tracks, review):
    """Visual evidence only; cv2 decodes videos, never runs a vision model."""
    import cv2

    root = Path(test_data)
    evidence = out / "evidence"
    evidence.mkdir()
    for row in rows:
        shutil.copyfile(root / row["crop_path"], evidence / (row["sample_id"] + ".png"))
    for clip, info in clips.items():
        selected = [r for r in rows if r["clip"] == clip]
        contact_sheet(
            [
                (
                    r["sample_id"] + f"\nT{r['track_id']} frame={r['frame_idx']}",
                    Image.open(root / r["crop_path"]),
                )
                for r in selected
            ],
            evidence / f"clip-{clip}-fixed-samples.jpg",
        )
        chosen = {
            (r["frame_idx"], r["track_id"]): r
            for r in raw_tracks[clip]
            if any(
                str(r["track_id"]) == str(s["track_id"]) and in_range(r["frame_idx"], s)
                for s in info["selection"]["track_segments"]
            )
        }
        frames = {key for key in chosen if key[0] % 30 == 0}
        for scope in info["selection"]["track_segments"]:
            frames.update(
                key
                for key in chosen
                if key[1] == scope["track_id"]
                and key[0] in [scope["start_frame"], scope["end_frame"]]
            )
        for frame in review["clips"][clip].get("extra_evidence_frames", []):
            frames.update(key for key in chosen if key[0] == frame)
        cap = cv2.VideoCapture(str(root / "video" / f"{clip}.mp4"))
        requests = sorted(frames)
        crops = []
        # Sequential decode makes the reviewed evidence reproducible and avoids slow seeks.
        requested = defaultdict(list)
        for frame, track in requests:
            requested[frame].append(track)
        current = 0
        while current <= max(requested):
            ok, frame_image = cap.read()
            if not ok:
                raise ValueError("Source video cannot decode requested evidence frame")
            for track in requested.get(current, []):
                raw = chosen[current, track]
                x1, y1 = map(math.floor, raw["bbox_xyxy"][:2])
                x2, y2 = map(math.ceil, raw["bbox_xyxy"][2:])
                image = Image.fromarray(cv2.cvtColor(frame_image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
                crops.append((f"{clip} T{track} F{current}", image))
            current += 1
        cap.release()
        for page in range(math.ceil(len(crops) / 32)):
            contact_sheet(
                crops[page * 32 : (page + 1) * 32],
                evidence / f"clip-{clip}-range-review-{page + 1}.jpg",
                columns=8,
                cell=(180, 250),
            )
    for row in rows:
        for name in row["evidence_images"]:
            if not (out / name).is_file():
                raise ValueError(f"Missing sample evidence: {name}")
    for clip in clips.values():
        for scope in clip["allowed_ranges"]:
            for name in scope["evidence_images"]:
                if not (out / name).is_file():
                    raise ValueError(f"Missing range evidence: {name}")


def score_reviewed(a_rows, audit_rows, correspondence, members, sources, conflicts):
    """Same frozen geometry and alignment rule; reviewed truths, fixed cohorts."""
    reviewed = {r["sample_id"]: r for r in audit_rows}
    if {r["sample_id"] for r in a_rows} != set(reviewed):
        raise ValueError("Protocol cohorts differ")
    observations_by_clip = defaultdict(list)
    for audit in audit_rows:
        observations_by_clip[audit["clip"]].append(audit)
    sample_rows = []
    for original in a_rows:
        audit = reviewed[original["sample_id"]]
        if locus(original) != locus(audit) or original["box"] != audit["bbox_xyxy"]:
            raise ValueError("Protocol geometry differs")
        scopes = audit["all_clip_ranges"]
        # Revalidate here as well: callers cannot bypass the guard by passing
        # a previously cached `valid` result for a mixed or changed range.
        scope_status = validate_scope(
            audit, audit["track_segments"], scopes, observations_by_clip[audit["clip"]]
        )
        containing = [
            s
            for s in scopes
            if str(s["track_id"]) == str(audit["track_id"]) and in_range(audit["frame_idx"], s)
        ]
        if scope_status == "valid" and containing[0]["truth"] != audit["suggested_truth"]:
            scope_status = "scope_suspicious"
        if audit["reviewed_scope_status"] != "valid":
            scope_status = audit["reviewed_scope_status"]
        row = {
            **original,
            "protocol": PROTOCOL_B,
            "original_truth": audit["original_truth"],
            "truth": audit["suggested_truth"],
            "scope_status": scope_status,
            "evidence_images": audit["evidence_images"],
            "reason": audit["reason"],
            "allowed_ranges": audit["allowed_ranges"],
        }
        sample_rows.append(row)
    alignment = align_identities(
        [r for r in sample_rows if r["truth"] and r["scope_status"] == "valid"]
    )
    for row in sample_rows:
        row["aligned_identity"] = alignment.get(row["truth"])
        row["correct_identity"] = (
            row["scope_status"] == "valid"
            and row["identity"] is not None
            and row["identity"] == row["aligned_identity"]
        )
        # Contaminated samples remain in their original 70/48 bucket as losses.
        row["correct"] = row["correct_identity"] and row["truth"] == row["original_truth"]
        row["outcome"] = (
            "scope_suspicious"
            if row["scope_status"] != "valid"
            else "annotation_identity_corrected"
            if row["truth"] != row["original_truth"]
            else "correct"
            if row["correct"]
            else "identity_fragment_or_misassignment"
            if row["identity"] is not None
            else row["status"]
        )
    ranges = {}
    for audit in audit_rows:
        for scope in audit["all_clip_ranges"]:
            key = audit["clip"], str(scope["track_id"]), scope["start_frame"], scope["end_frame"]
            incompatible = any(
                str(other["track_id"]) == str(scope["track_id"])
                and in_range(other["frame_idx"], scope)
                and other["suggested_truth"] is not None
                and other["suggested_truth"] != scope["truth"]
                for other in observations_by_clip[audit["clip"]]
            )
            ranges[key] = {**scope, "truth": None} if incompatible else scope
    declared = {}
    for (clip, track, start, end), scope in sorted(ranges.items()):
        for frame in range(start, end + 1):
            key = clip, track, frame
            if key in declared:
                raise ValueError("Overlapping exposure ranges")
            declared[key] = scope
    selected = alignment.get("jersey:27")
    fps = {str(s["video_id"]): float(s["fps"]) for s in sources}
    correspondence = {locus(r): r for r in correspondence}
    frames, others, correct_frames = [], [], set()
    for key, scope in sorted(declared.items()):
        observed = correspondence.get(key, {})
        correct = (
            scope["truth"] == "jersey:27"
            and selected is not None
            and observed.get("identity") == selected
            and observed.get("status") == "matched"
        )
        row = {
            "video_id": key[0],
            "track_id": key[1],
            "frame_idx": key[2],
            "truth": scope["truth"],
            "identity": observed.get("identity"),
            "status": observed.get("status", "missing_reference_geometry"),
            "correct_target": correct,
            "protocol": PROTOCOL_B,
            "evidence_images": scope["evidence_images"],
        }
        frames.append(row)
        if correct:
            correct_frames.add((key[0], key[2]))
        if (
            selected is not None
            and observed.get("identity") == selected
            and scope["truth"] not in {None, "jersey:27"}
        ):
            others.append(row)
    reverse = defaultdict(list)
    for key, row in correspondence.items():
        if row.get("identity") is not None and row["status"] == "matched":
            reverse[row.get("segment_id"), key[0], key[2]].append(key)
    purity = []
    for member in members:
        if selected is None or member.get("identity") != selected:
            continue
        possible = reverse[member.get("segment_id"), str(member["video_id"]), member["frame_idx"]]
        scope = declared.get(possible[0]) if len(possible) == 1 else None
        truth = scope["truth"] if scope else None
        kind = (
            "known_target"
            if truth == "jersey:27"
            else "known_other"
            if truth
            else "unlabelled_or_unresolved"
        )
        purity.append({**member, "truth": truth, "purity_class": kind, "protocol": PROTOCOL_B})
    metrics = {}
    for label, denominator in DENOMINATORS.items():
        cohort = [r for r in sample_rows if r["original_truth"] == label]
        if len(cohort) != denominator:
            raise ValueError("Fixed denominator changed")
        metrics[label] = {
            "total_denominator": denominator,
            "correct_samples": sum(r["correct"] for r in cohort),
            "coverage_rate": sum(r["correct"] for r in cohort) / denominator,
            "reviewed_target_samples": sum(
                r["truth"] == label and r["scope_status"] == "valid" for r in cohort
            ),
            "annotation_mismatch_samples": sum(r["truth"] != label for r in cohort),
            "scope_suspicious_samples": sum(r["scope_status"] != "valid" for r in cohort),
            "aligned_identity": alignment.get(label),
        }
    known_samples = [
        r
        for r in sample_rows
        if selected is not None
        and r["identity"] == selected
        and r["truth"] not in {None, "jersey:27"}
    ]
    target_conflicts = [
        r
        for r in conflicts
        if selected is not None
        and r.get("identity") == selected
        and r.get("status") == "physical_conflict"
    ]
    pending_overlaps = [
        r
        for r in conflicts
        if selected is not None
        and r.get("identity") == selected
        and r.get("status") == "overlap_pending"
    ]
    target_ids = {
        r["identity"]
        for r in sample_rows
        if r["truth"] == "jersey:27" and r["scope_status"] == "valid" and r["identity"] is not None
    }
    fragmentation = max(0, len(target_ids) - 1)
    unknown = sum(r["purity_class"] == "unlabelled_or_unresolved" for r in purity)
    exposure = math.fsum(1 / fps[clip] for clip, _ in sorted(correct_frames))
    checks = {
        "coverage": metrics["jersey:27"]["correct_samples"] >= 57,
        "known_pollution": not known_samples and not others,
        "physical_conflicts": not target_conflicts,
        "fragmentation": fragmentation <= 2,
        "exposure": exposure >= 50.0,
        "target_identity_exists": selected is not None,
        "no_pending_target_overlaps": not pending_overlaps,
    }
    summary = {
        "schema": "annotation-scope-dual-score-v2",
        "protocol": PROTOCOL_B,
        **metrics["jersey:27"],
        "player_15_metrics": metrics["jersey:15"],
        "alignment": alignment,
        "alignment_rule": "unchanged global maximum-weight injective Hungarian on reviewed valid fixed samples",
        "selected_identity": selected,
        "observed_exposure_seconds": exposure,
        "correct_observation_frames": len(correct_frames),
        "known_other_samples": len(known_samples),
        "known_other_observations": len(others),
        "cross_identity_pollution": dict(
            sorted(Counter(r["truth"] for r in known_samples).items())
        ),
        "physical_conflicts": len(target_conflicts),
        "all_identity_physical_conflicts": sum(
            r.get("status") == "physical_conflict" for r in conflicts
        ),
        "selected_identity_pending_overlaps": len(pending_overlaps),
        "fragmentation_count": fragmentation,
        "target_identity_count": len(target_ids),
        "selected_identity_unknown_observations": unknown,
        "checks": {**checks, "no_unresolved_target_members": unknown == 0},
        "passed": all(checks.values()),
        "per_clip": {
            clip: {
                "correct": sum(r["correct"] for r in sample_rows if r["video_id"] == clip),
                "total": sum(r["video_id"] == clip for r in sample_rows),
                "outcomes": dict(
                    sorted(
                        Counter(r["outcome"] for r in sample_rows if r["video_id"] == clip).items()
                    )
                ),
            }
            for clip in sorted(
                {r["video_id"] for r in sample_rows if r["original_truth"] == "jersey:27"}
            )
        },
        "certification_status": "visual_review_proposal_pending_human_signoff",
        "limits": review_limits(),
    }
    return summary, sample_rows, frames, purity, others


def review_limits():
    return [
        "Independent visual audit proposal; confidence values are reviewer judgements, not probabilities.",
        "Fixed original 70/48 cohorts and thresholds are unchanged; corrected other-person samples remain denominator losses.",
        "Exposure uses only reviewed 27 ranges and matched observed frame union at frozen fps; unknown transition frames are excluded.",
        "30-frame visual continuity checks do not certify every unsampled frame or exhaustive gallery purity; human final signoff remains required.",
        "Annotation changes must never be advertised as algorithm improvement or fed back into inference.",
    ]


def write_confusion(path, rows):
    identities = sorted({r["identity"] for r in rows if r["identity"] is not None})
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["truth", *identities, "unassigned_or_missing", "total"])
        for truth in sorted({r["truth"] or "unknown" for r in rows}):
            cohort = [r for r in rows if (r["truth"] or "unknown") == truth]
            counts = Counter(r["identity"] for r in cohort)
            writer.writerow([truth, *(counts[i] for i in identities), counts[None], len(cohort)])


def run_audit(
    test_data,
    annotations,
    geometry,
    frozen_run,
    review_path,
    out_dir,
    expected_hash,
    *,
    evidence_source=None,
):
    freeze = verify_prepared(geometry, expected_hash)
    if Path(freeze["run_dir"]).resolve() != Path(frozen_run).resolve():
        raise ValueError("Geometry references a different frozen prediction run")
    if Path(freeze["test_data"]).resolve() != Path(test_data).resolve():
        raise ValueError("Geometry references different upstream inputs")
    frozen_prediction_sha = sha(Path(frozen_run) / "predictions.json")
    review_sha = sha(review_path)
    review = read_json(review_path)
    if review["schema"] != "target-scope-visual-review-v2":
        raise ValueError("Unsupported visual review schema")
    before = input_hashes(test_data, annotations)
    rows, clips, raw_tracks = audit_samples(test_data, annotations, review)
    out = safe_output(
        out_dir,
        [
            test_data,
            annotations,
            frozen_run,
            geometry,
            review_path,
            Path(test_data).parent / "repository-review",
        ],
    )
    out.mkdir(parents=True)
    write_json(out / "inputs-before.json", before)
    shutil.copyfile(__file__, out / "audit_target_scopes.py")
    shutil.copyfile(review_path, out / "visual-review-input.json")
    if evidence_source:
        shutil.copytree(evidence_source, out / "evidence")
        for row in rows:
            shutil.copyfile(
                Path(test_data) / row["crop_path"], out / "evidence" / (row["sample_id"] + ".png")
            )
    else:
        render_evidence(test_data, out, rows, clips, raw_tracks, review)
    for row in rows:
        row["all_clip_ranges"] = clips[row["clip"]]["allowed_ranges"]
    evidence_files = {name for row in rows for name in row["evidence_images"]}
    evidence_files.update(
        name
        for info in clips.values()
        for scope in info["allowed_ranges"]
        for name in scope["evidence_images"]
    )
    for name in evidence_files:
        path = (out / name).resolve()
        if not path.is_relative_to(out) or not path.is_file():
            raise ValueError("Evidence missing or escapes audit directory")
    write_rows(out / "sample-truth-table.jsonl", rows)
    with (out / "sample-truth-table.csv").open("x", encoding="utf-8", newline="") as stream:
        fields = [
            "clip",
            "sample_id",
            "frame_idx",
            "bbox_xyxy",
            "track_id",
            "track_segments",
            "in_declared_segment",
            "raw_track_iou",
            "crop_path",
            "original_truth",
            "suggested_truth",
            "confidence",
            "suspicious",
            "flags",
            "original_scope_status",
            "reviewed_scope_status",
            "allowed_ranges",
            "reason",
            "evidence_images",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row[key], ensure_ascii=False)
                    if isinstance(row[key], (list, dict))
                    else row[key]
                    for key in fields
                }
            )
    write_json(
        out / "target-scope-v2.json",
        {
            "schema": "target-scope-proposal-v2",
            "protocol": PROTOCOL_B,
            "fixed_denominators": DENOMINATORS,
            "review_status": "pending_human_final_signoff",
            "clips": clips,
            "samples": rows,
            "input_sha256": before,
            "limits": review_limits(),
        },
    )
    write_json(
        out / "scope-validation.json",
        {
            "original_mixed_segment_samples": [
                r["sample_id"] for r in rows if r["original_scope_status"] == "scope_suspicious"
            ],
            "identity_or_sample_scope_suspicious": [
                r["sample_id"] for r in rows if r["suspicious"]
            ],
            "reviewed_scope_suspicious": [
                r["sample_id"] for r in rows if r["reviewed_scope_status"] != "valid"
            ],
            "legacy_exception": "Protocol A deliberately reproduces original scorer despite mixed-scope warnings; never used as corrected truth.",
        },
    )
    a_out, b_out = out / "protocol-A", out / "protocol-B"
    a_summary = score_benchmark(
        geometry,
        annotations,
        a_out,
        target_root=Path(test_data) / "target",
        expected_freeze_sha256=expected_hash,
    )
    write_json(
        a_out / "protocol.json",
        {
            "protocol": PROTOCOL_A,
            "geometry_freeze_sha256": expected_hash,
            "predictions_sha256": frozen_prediction_sha,
            "scorer_sha256": sha(
                Path(__file__).resolve().parents[1] / "app/evaluation/golden_score.py"
            ),
        },
    )
    a_rows = read_jsonl(a_out / "sample-ledger.jsonl")
    geometry = Path(geometry)
    b_summary, b_rows, frames, purity, others = score_reviewed(
        a_rows,
        rows,
        read_jsonl(geometry / "correspondence.jsonl"),
        read_jsonl(geometry / "prediction-members.jsonl"),
        read_json(geometry / "sources.json"),
        read_json(geometry / "physical-conflicts.json"),
    )
    b_out.mkdir()
    b_summary.update(
        {
            "geometry_freeze_sha256": expected_hash,
            "predictions_sha256": frozen_prediction_sha,
            "review_sha256": sha(review_path),
        }
    )
    write_json(b_out / "scoring_summary.json", b_summary)
    write_json(
        b_out / "truth-scope.json",
        {
            "protocol": PROTOCOL_B,
            "fixed_denominators": DENOMINATORS,
            "ranges": {clip: info["allowed_ranges"] for clip, info in clips.items()},
            "review_status": "pending_human_final_signoff",
        },
    )
    write_json(
        b_out / "truth-manifest.json",
        {
            "upstream_sha256": before,
            "review_sha256": review_sha,
            "evidence_sha256": {name: sha(out / name) for name in sorted(evidence_files)},
        },
    )
    write_json(
        b_out / "physical-conflicts.json",
        [
            row
            for row in read_json(geometry / "physical-conflicts.json")
            if row.get("identity") == b_summary["selected_identity"]
            and row.get("status") == "physical_conflict"
        ],
    )
    write_json(
        b_out / "protocol.json",
        {
            "protocol": PROTOCOL_B,
            "geometry_freeze_sha256": expected_hash,
            "predictions_sha256": frozen_prediction_sha,
            "review_sha256": sha(review_path),
        },
    )
    for name, content in [
        ("sample-ledger.jsonl", b_rows),
        ("player27-ledger.jsonl", [r for r in b_rows if r["original_truth"] == "jersey:27"]),
        ("frame-ledger.jsonl", frames),
        ("purity-ledger.jsonl", purity),
        ("known-other-observations.jsonl", others),
        ("audit_failure_cases.jsonl", [r for r in b_rows if not r["correct"]]),
    ]:
        write_rows(b_out / name, content)
    write_confusion(b_out / "confusion_matrix.csv", b_rows)
    differences = []
    for a, b in zip(a_rows, b_rows, strict=True):
        if a["sample_id"] != b["sample_id"]:
            raise ValueError("Protocol sample order differs")
        if a["truth"] != b["truth"] or a["correct"] != b["correct"]:
            differences.append(
                {
                    "sample_id": a["sample_id"],
                    "video_id": a["video_id"],
                    "frame_idx": a["frame_idx"],
                    "protocol_A_truth": a["truth"],
                    "protocol_B_truth": b["truth"],
                    "protocol_A_correct": a["correct"],
                    "protocol_B_correct": b["correct"],
                    "frozen_identity": a["identity"],
                    "geometry": a.get("geometry"),
                    "reason": b["reason"],
                    "evidence_images": b["evidence_images"],
                }
            )
    write_rows(out / "protocol-differences.jsonl", differences)
    a_frames = {locus(r): r for r in read_jsonl(a_out / "frame-ledger.jsonl")}
    write_rows(
        out / "frame-differences.jsonl",
        [
            {"protocol_A": a_frames[locus(r)], "protocol_B": r}
            for r in frames
            if a_frames[locus(r)]["truth"] != r["truth"]
            or a_frames[locus(r)]["correct_target"] != r["correct_target"]
        ],
    )
    summary = {
        "schema": "annotation-scope-audit-summary-v2",
        "protocol_A": {
            "protocol": PROTOCOL_A,
            "correct_samples": a_summary["correct_samples"],
            "total_denominator": 70,
            "player_15_metrics": a_summary["player_15_metrics"],
            "observed_exposure_seconds": a_summary["observed_exposure_seconds"],
            "known_other_samples": a_summary["known_other_samples"],
            "passed": a_summary["passed"],
        },
        "protocol_B": b_summary,
        "suspicious_by_original_label": {
            label: sum(r["suspicious"] for r in rows if r["original_truth"] == label)
            for label in DENOMINATORS
        },
        "mixed_segment_affected_samples": sum(
            r["original_scope_status"] == "scope_suspicious" for r in rows
        ),
        "difference_sample_ids": [r["sample_id"] for r in differences],
        "all_samples_in_declared_segment": all(r["in_declared_segment"] for r in rows),
        "all_selected_track_iou_exact": all(r["raw_track_iou"] == 1 for r in rows),
        "predictions_sha256": frozen_prediction_sha,
        "geometry_freeze_sha256": expected_hash,
        "limits": review_limits(),
    }
    verify_prepared(geometry, expected_hash)
    if sha(review_path) != review_sha:
        raise ValueError("Visual review changed during scoring")
    verify_hashes(before)
    after = input_hashes(test_data, annotations)
    if before != after:
        raise ValueError("Upstream input set or bytes changed")
    write_json(
        b_out / "post-score-gate.json",
        {
            "passed": True,
            "protocol": PROTOCOL_B,
            "geometry_freeze_sha256": expected_hash,
            "predictions_sha256": frozen_prediction_sha,
            "review_sha256": review_sha,
            "upstream_bytes_unchanged": True,
        },
    )
    write_json(out / "inputs-after.json", after)
    write_json(
        out / "input-integrity.json",
        {
            "files": len(before),
            "before_equals_after": True,
            "predictions_sha256": frozen_prediction_sha,
            "review_sha256": sha(review_path),
            "geometry_freeze_sha256": expected_hash,
        },
    )
    write_json(out / "summary.json", summary)
    write_json(
        out / "artifact-hashes.json",
        {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-data", type=Path, default=Path("../test"))
    parser.add_argument("--annotations", type=Path, default=Path("../annotations.csv"))
    parser.add_argument("--frozen-run", type=Path, required=True)
    parser.add_argument("--geometry-dir", type=Path, required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--evidence-source", type=Path, help="Reuse byte-identical reviewed evidence"
    )
    args = parser.parse_args()
    summary = run_audit(
        args.test_data,
        args.annotations,
        args.geometry_dir,
        args.frozen_run,
        args.review,
        args.out_dir,
        args.expected_freeze_sha256,
        evidence_source=args.evidence_source,
    )
    print(
        json.dumps(
            {
                "protocol_A": summary["protocol_A"],
                "protocol_B": summary["protocol_B"],
                "suspicious_by_original_label": summary["suspicious_by_original_label"],
                "difference_sample_ids": summary["difference_sample_ids"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
