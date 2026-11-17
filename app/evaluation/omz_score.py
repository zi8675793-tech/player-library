"""Independent scoring CLI. Reads labels only after frozen geometry and T1 gates.

Never imports or calls inference. All outputs go to a new directory.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from app.evaluation.guard import assert_frozen_before_scoring
from app.evaluation.omz_io import jsonl, read_json, sha, verify_t1, write_json
from app.evaluation.omz_prepare import key


def align_identities(rows):
    """Maximum-weight injective alignment (Hungarian), fixed once on all samples.

    Sorted labels/IDs + strict comparisons make ties reproducible. Dummy zero columns
    let a class remain unassigned. Multiple fragments can never be added for one label.
    """
    labels = sorted({r["truth"] for r in rows})
    identities = sorted({r["identity"] for r in rows if r["identity"] is not None})
    counts = Counter((r["truth"], r["identity"]) for r in rows if r["identity"] is not None)
    n, m = len(labels), len(identities) + len(labels)
    weights = [[counts[label, identity] for identity in identities] + [0] * n for label in labels]
    u, v, p, way = [0] * (n + 1), [0] * (m + 1), [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv, used = [float("inf")] * (m + 1), [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], float("inf"), 0
            for j in range(1, m + 1):
                if not used[j]:
                    current = -weights[i0 - 1][j - 1] - u[i0] - v[j]
                    if current < minv[j]:
                        minv[j], way[j] = current, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    aligned = dict.fromkeys(labels)
    for j in range(1, len(identities) + 1):
        if p[j] and weights[p[j] - 1][j - 1] > 0:
            aligned[labels[p[j] - 1]] = identities[j - 1]
    return aligned


def score_rows(rows, alignment):
    inverse = {pid: truth for truth, pid in alignment.items() if pid is not None}
    scored = []
    for row in rows:
        pid = row["identity"]
        correct = pid is not None and pid == alignment.get(row["truth"])
        if correct:
            reason = "correct"
        elif pid is not None:
            reason = "known_wrong_identity" if pid in inverse else "identity_fragment"
        else:
            reason = row["status"]
        scored.append(
            {
                **row,
                "aligned_identity": alignment.get(row["truth"]),
                "correct": correct,
                "outcome": reason,
            }
        )
    return scored


def identity_diagnostics(rows):
    by_identity, by_truth = defaultdict(Counter), defaultdict(Counter)
    for row in rows:
        if row["identity"] is not None:
            by_identity[row["identity"]][row["truth"]] += 1
            by_truth[row["truth"]][row["identity"]] += 1
    pairs = Counter(tp=0, fp=0, fn=0, tn=0)
    for i, a in enumerate(rows):
        for b in rows[i + 1 :]:
            if a["video_id"] == b["video_id"]:
                continue
            same_truth = a["truth"] == b["truth"]
            same_id = a["identity"] is not None and a["identity"] == b["identity"]
            pairs[
                ("tp" if same_truth else "fp") if same_id else ("fn" if same_truth else "tn")
            ] += 1
    return {
        "mixed_identities": {k: dict(v) for k, v in by_identity.items() if len(v) > 1},
        "truth_fragments": {k: dict(v) for k, v in by_truth.items()},
        "split_truth_count": sum(len(v) > 1 for v in by_truth.values()),
        "cross_clip_pairs": dict(pairs),
        "known_assigned_purity": (
            sum(max(v.values()) for v in by_identity.values())
            / sum(sum(v.values()) for v in by_identity.values())
            if by_identity
            else None
        ),
    }


def verify_geometry(geometry):
    frozen = read_json(geometry / "geometry-freeze.json")
    for name, digest in frozen["files"].items():
        path = (geometry / name).resolve()
        if not path.is_relative_to(geometry.resolve()) or sha(path) != digest:
            raise ValueError(f"Changed frozen geometry: {name}")
    for record in read_json(geometry / "machine-inputs.json"):
        if sha(record["path"]) != record["sha256"]:
            raise ValueError("Machine reference changed")
    # Verify the executable scorer, not just its archived copy.
    for name, digest in frozen["files"].items():
        if name.startswith("implementation/"):
            if sha(Path(__file__).parent / Path(name).name) != digest:
                raise ValueError(f"Evaluation implementation changed after freeze: {name}")
    _, verification = verify_t1(Path(frozen["t1"]), geometry / "runtime-snapshot")
    if (
        verification["artifact_index_sha256"]
        != read_json(geometry / "t1-verification-before.json")["artifact_index_sha256"]
    ):
        raise ValueError("T1 artifact index changed")
    legacy = read_json(geometry / "legacy-verification.json")
    for run in legacy.values():
        path = Path(run["path"])
        if sha(path / "artifact-hashes.json") != run["artifact_index_sha256"]:
            raise ValueError("Legacy freeze index changed")
        assert_frozen_before_scoring(path)
    return frozen, verification, legacy


def load_truth(workspace, sources, samples):
    """Only called below the complete gate. No truth is used in correspondence."""
    annotation_path = workspace / "annotations.csv"
    with annotation_path.open(newline="") as stream:
        annotations = list(csv.DictReader(stream))
    paths, ranges, target_samples, target_observations = [annotation_path], [], {}, {}
    source_by_clip = {Path(s["path"]).stem: s for s in sources}
    for annotation in annotations:
        clip = annotation["video_id"]
        base = workspace / "test/target" / clip
        paths.extend(
            base / n for n in ("selection.json", "video_meta.json", "samples.jsonl", "tracks.jsonl")
        )
        selection, meta = read_json(base / "selection.json"), read_json(base / "video_meta.json")
        if selection["event_jersey_number"] != annotation["jersey_number"]:
            raise ValueError("Conflicting annotation and selection")
        if meta["fps"] != source_by_clip[clip]["fps"]:
            raise ValueError("Truth/source fps differs")
        truth = "jersey:" + annotation["jersey_number"]
        for selected in selection["track_segments"]:
            ranges.append(
                {
                    "video_id": clip,
                    "track_id": str(selected["track_id"]),
                    "start_frame": selected["start_frame"],
                    "end_frame_exclusive": selected["end_frame"] + 1,
                    "truth": truth,
                    "fps": meta["fps"],
                    "quality_status": selected["quality_status"],
                }
            )
        for row in jsonl(base / "samples.jsonl"):
            target_samples[key(row)] = row
        for row in jsonl(base / "tracks.jsonl"):
            target_observations[key(row)] = row
    truth_loci = {}
    for selected in ranges:
        for frame in range(selected["start_frame"], selected["end_frame_exclusive"]):
            locus = selected["video_id"], selected["track_id"], frame
            if locus in truth_loci:
                raise ValueError("Overlapping truth ranges")
            truth_loci[locus] = selected["truth"]
    machine = {key(s): s for s in samples}
    selected_machine = {k for k in machine if k in truth_loci}
    if selected_machine != target_samples.keys():
        raise ValueError("Fixed target samples differ from machine reference scope")
    for locus, sample in target_samples.items():
        if sample["bbox_xyxy"] != machine[locus]["bbox_xyxy"]:
            raise ValueError("Truth sample geometry differs from frozen machine geometry")
    sample_truths = Counter(truth_loci[k] for k in selected_machine)
    if sample_truths["jersey:27"] != 70:
        raise ValueError("Fixed 70 denominator changed")
    return {
        "ranges": ranges,
        "loci": truth_loci,
        "samples": target_samples,
        "observations": target_observations,
        "manifest": [{"path": str(p), "sha256": sha(p)} for p in paths],
        "scope": {
            "samples_by_label": dict(sample_truths),
            "annotation_rows": len(annotations),
            "person_gt_ids_provided": sum(bool(a["person_gt_id"]) for a in annotations),
            "team_ids_provided": sum(bool(a["team_id"]) for a in annotations),
            "truth_type": "selected-range jersey proxy; not exhaustive person identity truth",
        },
    }


def write_rows(path, rows):
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    if rows:
        with path.with_suffix(".csv").open("w", newline="") as stream:
            fields = sorted({k for row in rows for k in row})
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                        for k, v in row.items()
                    }
                )


def legacy_correspondence(path, observations, fps):
    prediction = read_json(path / "predictions.json")
    assignments = {a["segment_id"]: a for a in prediction["assignments"]}
    segments = defaultdict(list)
    for segment in prediction["segments"]:
        if segment["active"]:
            segments[str(segment["clip_id"]), str(segment["track_id"])].append(segment)
    output = {}
    for locus, observation in observations.items():
        clip, track, frame = locus
        hits = [s for s in segments[clip, track] if s["start_s"] <= frame / fps[clip] < s["end_s"]]
        if len(hits) > 1:
            raise ValueError("Overlapping legacy active segments")
        pid = assignments.get(hits[0]["segment_id"], {}).get("player_id") if hits else None
        output[locus] = {
            "video_id": clip,
            "track_id": track,
            "frame_idx": frame,
            "identity": pid,
            "status": "matched" if pid else "legacy_unassigned",
            "box": observation["box"],
            "origin": "legacy_observation_record",
        }
    return output


def compact_spans(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source"], row["instance"], row["identity"], row["review_reason"]].append(row)
    spans = []
    for (source, instance, identity, reason), members in sorted(grouped.items()):
        current = None
        for member in sorted(members, key=lambda x: x["frame_idx"]):
            frame = member["frame_idx"]
            if current is None or current["end_frame_exclusive"] != frame:
                current = {
                    "source": source,
                    "instance": instance,
                    "identity": identity,
                    "reason": reason,
                    "start_frame": frame,
                    "end_frame_exclusive": frame + 1,
                    "count": 1,
                    "first_box": member["box"],
                }
                spans.append(current)
            else:
                current["end_frame_exclusive"] += 1
                current["count"] += 1
    return spans


def score_mode(name, observations, samples, truth, fps, out, members=None):
    out.mkdir()
    sample_rows = []
    for locus in sorted(truth["samples"]):
        row = observations.get(
            locus,
            {
                "video_id": locus[0],
                "track_id": locus[1],
                "frame_idx": locus[2],
                "identity": None,
                "status": "missing_reference_geometry",
            },
        )
        sample_rows.append(
            {**row, "sample_id": samples[locus]["sample_id"], "truth": truth["loci"][locus]}
        )
    alignment = align_identities(sample_rows)
    sample_rows = score_rows(sample_rows, alignment)
    frame_rows = []
    for locus, label in sorted(truth["loci"].items()):
        row = observations.get(
            locus,
            {
                "video_id": locus[0],
                "track_id": locus[1],
                "frame_idx": locus[2],
                "identity": None,
                "status": "missing_reference_geometry",
            },
        )
        frame_rows.append({**row, "truth": label, "exposure_s": 1 / fps[locus[0]]})
    frame_rows = score_rows(frame_rows, alignment)
    write_rows(out / "sample-ledger.jsonl", sample_rows)
    write_rows(out / "player27-ledger.jsonl", [r for r in sample_rows if r["truth"] == "jersey:27"])
    write_rows(out / "frame-ledger.jsonl", frame_rows)
    diagnostics = identity_diagnostics(sample_rows)
    # Frame confusion is a separate audit, not a second alignment optimized for duration.
    frame_confusion = defaultdict(Counter)
    for row in frame_rows:
        if row["identity"] is not None:
            frame_confusion[row["identity"]][row["truth"]] += 1
    diagnostics["mixed_identities_observations"] = {
        k: dict(v) for k, v in frame_confusion.items() if len(v) > 1
    }
    wrong = [
        r for r in sample_rows if r["outcome"] in ("known_wrong_identity", "identity_fragment")
    ]
    write_rows(out / "misassociation-and-fragments.jsonl", wrong)
    write_json(out / "identity-diagnostics.json", diagnostics)
    result = {"mode": name, "alignment": alignment, "labels": {}, "diagnostics": diagnostics}
    for label in sorted(alignment):
        ss = [r for r in sample_rows if r["truth"] == label]
        ff = [r for r in frame_rows if r["truth"] == label]
        result["labels"][label] = {
            "identity": alignment[label],
            "correct_samples": sum(r["correct"] for r in ss),
            "sample_denominator": len(ss),
            "sample_outcomes": dict(Counter(r["outcome"] for r in ss)),
            "correct_observation_frames": sum(r["correct"] for r in ff),
            "correct_seconds": sum(r["exposure_s"] for r in ff if r["correct"]),
            "selected_range_seconds": sum(r["exposure_s"] for r in ff),
            "available_reference_frames": sum(
                r["status"] != "missing_reference_geometry" for r in ff
            ),
            "reference_frame_denominator": len(ff),
            "frame_outcomes": dict(Counter(r["outcome"] for r in ff)),
            "per_clip": {
                clip: {
                    "correct": sum(r["correct"] for r in ss if r["video_id"] == clip),
                    "total": sum(r["video_id"] == clip for r in ss),
                }
                for clip in sorted({r["video_id"] for r in ss})
            },
        }
    if members is not None:
        pending, known = [], Counter()
        for member in members:
            locus = tuple(member["old_locus"]) if member["old_locus"] else None
            label = truth["loci"].get(locus)
            if label:
                known[label] += 1
            else:
                pending.append(
                    {
                        **member,
                        "review_reason": (
                            "raw_to_native_unresolved"
                            if not member["raw_detection_confirmed"]
                            else "old_geometry_unresolved"
                            if locus is None
                            else "outside_labelled_ranges"
                        ),
                    }
                )
        spans = compact_spans(pending)
        write_rows(out / "unlabelled-members.jsonl", pending)
        write_json(out / "review-ranges.json", spans)
        selected = alignment.get("jersey:27")
        write_json(
            out / "player27-review-ranges.json", [s for s in spans if s["identity"] == selected]
        )
        result["unknown_scope"] = {
            "real_native_points": len(members),
            "known_labelled_points": dict(known),
            "unresolved_or_unlabelled_points": len(pending),
            "review_spans": len(spans),
            "selected27_unknown_points": sum(p["identity"] == selected for p in pending),
            "selected27_known_other_points": sum(
                m["identity"] == selected
                and m["old_locus"] is not None
                and truth["loci"].get(tuple(m["old_locus"])) not in (None, "jersey:27")
                for m in members
            ),
            "unknown_by_reason": dict(Counter(p["review_reason"] for p in pending)),
        }
    result["sample_target_passed"] = result["labels"]["jersey:27"]["correct_samples"] >= 57
    result["production_replacement"] = False
    write_json(out / "report.json", result)
    return result


def score_frozen_predictions(geometry, workspace, out):
    # Nothing, including answer hashes, may be read before this gate succeeds.
    frozen, verification, legacy = verify_geometry(geometry)
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "pre-label-gate.json",
        {
            "verified_at": datetime.now(UTC).isoformat(),
            "t1": verification,
            "geometry_index_sha256": sha(geometry / "geometry-freeze.json"),
            "implementation_verified": True,
        },
    )
    sources = read_json(geometry / "sources.json")
    sample_list = read_json(geometry / "machine-samples.json")
    samples = {key(s): s for s in sample_list}
    truth = load_truth(workspace, sources, sample_list)
    write_json(out / "truth-manifest.json", truth["manifest"])
    write_json(out / "truth-scope.json", {**truth["scope"], "ranges": truth["ranges"]})
    fps = {Path(s["path"]).stem: s["fps"] for s in sources}
    report, observations = {}, {}
    for name in ("A", "B"):
        observations = {key(r): r for r in jsonl(geometry / name / "correspondence.jsonl")}
        for locus, reference in truth["observations"].items():
            if locus not in observations or observations[locus]["box"] != reference["bbox_xyxy"]:
                raise ValueError("Target observation geometry differs from frozen machine input")
        report[name] = score_mode(
            name,
            observations,
            samples,
            truth,
            fps,
            out / name,
            list(jsonl(geometry / name / "prediction-members.jsonl")),
        )
        report[name]["physical_conflicts"] = read_json(geometry / name / "physical-conflicts.json")
    for name, run in legacy.items():
        old = legacy_correspondence(Path(run["path"]), observations, fps)
        report["legacy_" + name] = score_mode(
            "legacy_" + name, old, samples, truth, fps, out / ("legacy_" + name)
        )
    write_json(out / "comparison.json", report)
    # Enforce read-only input/answer/implementation integrity after all computation.
    verify_geometry(geometry)
    for record in truth["manifest"]:
        if sha(record["path"]) != record["sha256"]:
            raise ValueError("Truth changed during evaluation")
    write_json(
        out / "post-score-gate.json",
        {
            "verified_at": datetime.now(UTC).isoformat(),
            "passed": True,
            "t1_run": frozen["t1"],
            "geometry_index_sha256": sha(geometry / "geometry-freeze.json"),
        },
    )
    print(json.dumps({k: v["labels"]["jersey:27"] for k, v in report.items()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    score_frozen_predictions(args.geometry.resolve(), args.workspace.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
