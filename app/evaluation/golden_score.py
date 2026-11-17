"""T4 scoring of immutable geometry; never imports or invokes inference.

All answer access is below verify_prepared. Labels describe selected jersey ranges,
not an exhaustive person identity annotation. Unknown predicted members are audited
and cannot silently establish purity or valid exposure.
"""

import csv
import hashlib
import io
import json
import math
import os
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from app.evaluation.golden_freeze import verify_prepared


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _write_rows(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _key(row):
    return str(row["video_id"]), str(row["track_id"]), int(row["frame_idx"])


def _unique(rows, description):
    result = {}
    for row in rows:
        locus = _key(row)
        if locus in result:
            raise ValueError(f"Duplicate {description} locus: {locus}")
        result[locus] = row
    return result


def align_identities(rows):
    """Frozen-rule global maximum-weight injective alignment on ALL samples.

    One label maps to at most one identity. Sorting and strict comparisons make
    ties deterministic. This permutation assessment never changes predictions,
    combines fragments, or re-aligns for duration, purity or one preferred player.
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


def _load_truth(annotations, target_root, machine_samples, sources):
    """Called only after all frozen artifacts/code/input hashes are verified."""
    annotations = Path(annotations).resolve()
    target_root = Path(target_root).resolve()
    source_ids = {str(s["video_id"]) for s in sources}
    frame_counts = {str(s["video_id"]): int(s["frame_count"]) for s in sources}
    answer_hashes = {}

    def answer_text(path):
        content = Path(path).read_bytes()
        answer_hashes[str(path)] = hashlib.sha256(content).hexdigest()
        return content.decode("utf-8-sig")

    entries = list(csv.DictReader(io.StringIO(answer_text(annotations), newline="")))
    required = {"video_id", "jersey_number"}
    if not entries or not required <= entries[0].keys():
        raise ValueError("Invalid annotation columns")
    truth_loci, ranges, annotation_clips = {}, [], {}
    target_sample_rows = {}
    for entry in entries:
        clip, number = str(entry["video_id"]), str(entry["jersey_number"])
        if clip in annotation_clips:
            if annotation_clips[clip] != number:
                raise ValueError("Conflicting jersey annotations for one video selection")
            continue
        annotation_clips[clip] = number
        if clip not in source_ids or Path(clip).name != clip or clip in {".", ".."}:
            raise ValueError("Annotated video not in frozen machine sources")
        base = target_root / clip
        selection_path = base / "selection.json"
        selection = json.loads(answer_text(selection_path))
        if str(selection["event_jersey_number"]) != number:
            raise ValueError("Annotation and selection jersey disagree")
        label = "jersey:" + number
        for segment in selection["track_segments"]:
            start, end = segment["start_frame"], segment["end_frame"]
            if (
                type(start) is not int
                or type(end) is not int
                or start < 0
                or end < start
                or end >= frame_counts[clip]
            ):
                raise ValueError("Invalid labelled frame range")
            record = {
                "video_id": clip,
                "track_id": str(segment["track_id"]),
                "start_frame": start,
                "end_frame": end,
                "truth": label,
                "quality_status": segment.get("quality_status"),
            }
            ranges.append(record)
            for frame in range(start, end + 1):
                locus = clip, record["track_id"], frame
                if locus in truth_loci:
                    raise ValueError("Overlapping labelled frame ranges")
                truth_loci[locus] = label
        sample_path = base / "samples.jsonl"
        if sample_path.is_file():
            selected = _unique(
                [
                    json.loads(line)
                    for line in answer_text(sample_path).splitlines()
                    if line.strip()
                ],
                "golden sample",
            )
            if any(k[0] != clip for k in selected):
                raise ValueError("Golden sample video differs from selection directory")
            target_sample_rows.update(selected)
    if set(annotation_clips) != source_ids:
        raise ValueError("Annotation scope must include every frozen machine video")
    selected_machine = {k: row for k, row in machine_samples.items() if k in truth_loci}
    if target_sample_rows:
        # If provided, the golden sample ledger is authoritative in its covered clips.
        covered = {k[0] for k in target_sample_rows}
        expected = {k: row for k, row in selected_machine.items() if k[0] in covered}
        if expected.keys() != target_sample_rows.keys():
            raise ValueError("Fixed golden samples differ from frozen machine samples")
        for locus, row in target_sample_rows.items():
            if row["bbox_xyxy"] != expected[locus]["bbox_xyxy"]:
                raise ValueError("Golden sample geometry differs from frozen machine geometry")
    counts = Counter(truth_loci[k] for k in selected_machine)
    if counts["jersey:27"] != 70:
        raise ValueError("Fixed 70 denominator changed or golden sample scope incomplete")
    target_clips = {k[0] for k in selected_machine if truth_loci[k] == "jersey:27"}
    return {
        "loci": truth_loci,
        "samples": selected_machine,
        "ranges": ranges,
        "manifest": [{"path": p, "sha256": digest} for p, digest in answer_hashes.items()],
        "scope": {
            "samples_by_label": dict(sorted(counts.items())),
            "target_clips": sorted(target_clips),
            "all_benchmark_clips": sorted(source_ids),
            "annotation_rows": len(entries),
            "person_gt_ids_provided": sum(bool(a.get("person_gt_id")) for a in entries),
            "team_ids_provided": sum(bool(a.get("team_id")) for a in entries),
            "truth_type": "selected-range jersey proxy; not exhaustive person identity truth",
        },
    }


def _number(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, dict) and value.get("$float") in {"+inf", "inf"}:
        return math.inf
    return None


def _failure(row, diagnostics, selected):
    """Separate observed decision causes from plausible feature limitations."""
    status = row["status"]
    segment = diagnostics.get(row.get("segment_id"), {})
    decision = segment.get("decision") or {}
    evidence = decision.get("evidence") or {}
    feature = dict(segment.get("feature") or {})
    if "appearance" in feature:
        appearance = dict(feature["appearance"])
        vector = appearance.pop("vector", None)
        if vector is not None:
            appearance["vector_dimensions"] = len(vector)
            appearance["vector_norm"] = math.sqrt(math.fsum(float(v) ** 2 for v in vector))
            appearance["full_vector_location"] = "geometry/segment-diagnostics.json"
        feature["appearance"] = appearance
    jersey = feature.get("jersey") or {}
    scores = evidence.get("player_scores") or {}
    blocking = sorted({s.get("blocked_by") for s in scores.values() if s.get("blocked_by")})
    support = []
    if jersey and jersey.get("number") is None and jersey.get("votes"):
        support.append("ocr_votes_present_but_no_accepted_number")
    if selected is None:
        support.append("no_globally_aligned_target_identity")
    if status == "unprocessed":
        primary, detail = "unprocessed_clip", "冻结 T3 运行未处理该视频；仍计入70分母。"
    elif status not in {"matched", "unassigned"}:
        primary, detail = "geometry_" + status, "冻结同帧几何对应未形成唯一可靠匹配。"
    elif row.get("identity") is not None:
        primary, detail = (
            "identity_fragment_or_misassignment",
            "样本归入其他档案；不允许缝合碎档提高命中。",
        )
    elif evidence.get("reason") == "batch_conflict" or "cannot_link" in blocking:
        primary, detail = "cannot_link_blocked", "冻结机器决策明确记录同源同帧排他或批内冲突。"
    elif "jersey_conflict" in blocking:
        primary, detail = "jersey_number_gate", "冻结机器决策记录号码冲突门禁。"
    else:
        distance = _number(decision.get("best_distance"))
        threshold = _number(evidence.get("distance_threshold"))
        margin = _number(decision.get("competitive_margin"))
        required = _number(evidence.get("required_margin"))
        if (
            distance is not None
            and math.isfinite(distance)
            and threshold is not None
            and distance > threshold
        ):
            primary, detail = "appearance_distance_exceeded", "冻结最佳外观距离超过准入门槛。"
        elif margin is not None and required is not None and scores and margin < required:
            primary, detail = "competitive_margin_gate", "冻结候选竞争边距不足。"
        elif not scores:
            primary, detail = (
                "safety_gate_no_admissible_identity",
                "图库未形成可准入身份；冻结记录未给出逐对聚类拒绝细节。",
            )
        else:
            primary, detail = (
                "insufficient_evidence_gate",
                "冻结记录为证据不足或无可比原型，保留待定。",
            )
    return {
        **row,
        "primary_cause": primary,
        "explanation": detail,
        "supporting_observations": support,
        "causal_limit": "OCR置信稀释、同色外观倒挂若无直接机器证据，仅能视为待验证假设。",
        "frozen_decision": decision,
        "frozen_feature": feature,
        "blocking_reasons": blocking,
    }


def _report(summary):
    lines = [
        "# T4 独立黄金评测验收报告",
        "",
        f"最终结论：**{'PASS' if summary['passed'] else 'FAIL（算法未达标）'}**。",
        "",
        "| 指标 | 实测 | 合格线 | 结论 |",
        "| :--- | :--- | :--- | :--- |",
    ]
    checks = summary["checks"]
    metrics = [
        (
            "27号正确样本",
            f"{summary['correct_samples']} / 70 ({summary['coverage_rate']:.1%})",
            "≥57 / 70",
            "coverage",
        ),
        ("已知异人混入样本", str(summary["known_other_samples"]), "0", "known_pollution"),
        ("同源同帧物理冲突", str(summary["physical_conflicts"]), "0", "physical_conflicts"),
        ("多余碎档", str(summary["fragmentation_count"]), "≤2", "fragmentation"),
        (
            "真实有效观测曝光",
            f"{summary['observed_exposure_seconds']:.4f} 秒",
            "≥50.0 秒",
            "exposure",
        ),
    ]
    lines.extend(
        f"| {name} | {value} | {limit} | {'PASS' if checks[key] else 'FAIL'} |"
        for name, value, limit, key in metrics
    )
    lines.extend(
        [
            "",
            f"目标全局身份：`{summary['selected_identity']}`。纯度审计状态：`{summary['purity_status']}`。",
            "空图库的零混入、零冲突和零碎档只是空集合计数，不能作为已证明身份纯度。",
            "",
            "## 冻结与防火墙",
            "",
            f"独立评分进程先校验外部固定指纹 `{summary['geometry_freeze_sha256']}`，再读取人工标签；评分后复核同一指纹与标签指纹。",
            "推理代码、数据库与原始上游输入均不修改；固定70分母包括未处理视频、待定、几何歧义和拆档。",
            "所有人工标签只用于评分；全体样本进行一次确定性的一对一全局身份排列对齐，禁止拼接身份或为曝光另选身份。",
            "",
            "## 逐样本归因",
            "",
        ]
    )
    lines.extend(
        f"- `{cause}`：{count} 个27号未命中。" for cause, count in summary["failure_counts"].items()
    )
    lines.extend(
        [
            "",
            "详见 `audit_failure_cases.jsonl` 中每个未命中样本的冻结决策、几何状态和特征证据。",
            "OCR置信稀释或同色外观倒挂若无直接证据，不伪造为确定病因；保留为待验证假设。",
            "",
            "## 纯度、时长与边界",
            "",
            "标签是被选时间段的球衣号码代理，未提供穷尽的全场人物真值。纯度账本区分已知异人、未标注成员及几何未解成员。",
            "曝光仅累计同源同帧唯一可靠几何对应、已知为27号且属于目标身份的实际观测帧；每源每帧最多计1/fps，禁止用段首尾跨度填充缺帧。",
            f"目标档案未知成员：{summary['selected_identity_unknown_observations']}，已知异人观测：{summary['known_other_observations']}。",
            "",
            "## 每片段固定分母",
            "",
            "| 片段 | 正确样本 | 分母 | 结果分布 |",
            "| :--- | ---: | ---: | :--- |",
        ]
    )
    for clip, metrics in summary["per_clip"].items():
        lines.append(
            f"| {clip} | {metrics['correct']} | {metrics['total']} | {json.dumps(metrics['outcomes'], ensure_ascii=False, sort_keys=True)} |"
        )
    lines.extend(
        [
            "",
            "工程交付与算法验收应分别记录：评测工具可测试通过，算法达标由上述客观指标决定。",
            "后续若需改善覆盖率，必须在新任务中授权修改推理模块，并使用新运行、新冻结和独立复评；本次不回流标签调参。",
            "",
        ]
    )
    return "\n".join(lines)


def score_benchmark(
    geometry_dir, annotations, out_dir, *, target_root=None, expected_freeze_sha256
):
    """Verify everything before the first answer open, then score one frozen run."""
    geometry = Path(geometry_dir).resolve()
    out = Path(out_dir).resolve()
    freeze = verify_prepared(geometry, expected_freeze_sha256)
    if out == geometry or out.is_relative_to(geometry):
        raise ValueError("Scoring output must be outside immutable prepared geometry")
    run = Path(freeze["run_dir"]).resolve()
    if out == run or out.is_relative_to(run):
        raise ValueError("Scoring output must not modify the immutable source run")
    if out.is_relative_to(Path(freeze["test_data"]).resolve()):
        raise ValueError("Scoring output must not modify read-only upstream inputs")
    output_names = {
        "pre-label-gate.json",
        "post-score-gate.json",
        "truth-manifest.json",
        "truth-scope.json",
        "sample-ledger.jsonl",
        "player27-ledger.jsonl",
        "frame-ledger.jsonl",
        "audit_failure_cases.jsonl",
        "purity-ledger.jsonl",
        "known-other-observations.jsonl",
        "physical-conflicts.json",
        "scoring_summary.json",
        "confusion_matrix.csv",
        "acceptance_report.md",
    }
    if out.exists() and any(
        (out / name).exists() or (out / name).is_symlink() for name in output_names
    ):
        raise FileExistsError("Scoring output already contains benchmark artifacts")
    out.mkdir(parents=True, exist_ok=True)
    _write_json(
        out / "pre-label-gate.json",
        {
            "verified_at": datetime.now(UTC).isoformat(),
            "passed": True,
            "geometry_freeze_sha256": expected_freeze_sha256,
            "run_dir": freeze.get("run_dir"),
            "answer_files_opened_before_gate": 0,
            "process_id": os.getpid(),
        },
    )
    sources = _json(geometry / "sources.json")
    samples = _unique(_json(geometry / "machine-samples.json"), "machine sample")
    correspondence = _unique(_rows(geometry / "correspondence.jsonl"), "correspondence")
    members = _rows(geometry / "prediction-members.jsonl")
    diagnostics = _json(geometry / "segment-diagnostics.json")
    conflicts = _json(geometry / "physical-conflicts.json")
    fps = {str(s["video_id"]): float(s["fps"]) for s in sources}
    if any(not math.isfinite(value) or value <= 0 for value in fps.values()):
        raise ValueError("Invalid frozen frame rate")
    annotations = Path(annotations).resolve()
    truth = _load_truth(annotations, target_root or annotations.parent / "target", samples, sources)
    sample_rows = []
    for locus, sample in sorted(truth["samples"].items()):
        observed = correspondence.get(locus)
        if observed is None:
            observed = {
                "video_id": locus[0],
                "track_id": locus[1],
                "frame_idx": locus[2],
                "box": sample["bbox_xyxy"],
                "identity": None,
                "segment_id": None,
                "status": "missing_reference_geometry",
            }
        if observed["box"] != sample["bbox_xyxy"]:
            raise ValueError("Frozen sample and correspondence geometry differ")
        if observed.get("identity") is not None and observed["status"] != "matched":
            raise ValueError("Identity attached to unresolved geometry")
        sample_rows.append(
            {**observed, "sample_id": sample["sample_id"], "truth": truth["loci"][locus]}
        )
    alignment = align_identities(sample_rows)
    selected = alignment.get("jersey:27")
    for row in sample_rows:
        row["aligned_identity"] = alignment.get(row["truth"])
        row["correct"] = row["identity"] is not None and row["identity"] == row["aligned_identity"]
        row["outcome"] = (
            "correct"
            if row["correct"]
            else "identity_fragment_or_misassignment"
            if row["identity"] is not None
            else row["status"]
        )
    target_rows = [r for r in sample_rows if r["truth"] == "jersey:27"]
    if len(target_rows) != 70:
        raise ValueError("Fixed 70 denominator changed")
    failures = [_failure(r, diagnostics, selected) for r in target_rows if not r["correct"]]
    target_ids = {r["identity"] for r in target_rows if r["identity"] is not None}
    fragment_count = max(0, len(target_ids) - 1)
    known_other_samples = [
        r
        for r in sample_rows
        if selected is not None and r["identity"] == selected and r["truth"] != "jersey:27"
    ]
    pollution = {"10号": 0, "15号": 0}
    for row in known_other_samples:
        label = row["truth"].removeprefix("jersey:") + "号"
        pollution[label] = pollution.get(label, 0) + 1
    correct_frames, known_other_frames = set(), []
    frame_rows = []
    for locus, label in sorted(truth["loci"].items()):
        row = correspondence.get(locus)
        correct = (
            row is not None
            and selected is not None
            and row.get("identity") == selected
            and row["status"] == "matched"
            and label == "jersey:27"
        )
        if correct:
            correct_frames.add((locus[0], locus[2]))
        if (
            row is not None
            and selected is not None
            and row.get("identity") == selected
            and label != "jersey:27"
        ):
            known_other_frames.append({**row, "truth": label})
        frame_rows.append(
            {
                "video_id": locus[0],
                "track_id": locus[1],
                "frame_idx": locus[2],
                "truth": label,
                "identity": row.get("identity") if row else None,
                "status": row["status"] if row else "missing_reference_geometry",
                "correct_target": correct,
            }
        )
    exposure = math.fsum(1.0 / fps[clip] for clip, _ in sorted(correct_frames))
    # A member is known only when a unique frozen geometric correspondence links
    # the same segment/identity to a labelled raw observation. Track ID alone is
    # never used to match a prediction to a person.
    reverse_known = defaultdict(list)
    for locus, row in correspondence.items():
        if row.get("identity") is not None and row["status"] == "matched":
            reverse_known[row.get("segment_id"), locus[0], locus[2]].append((locus, row))
    purity_rows = []
    for member in members:
        if selected is None or member.get("identity") != selected:
            continue
        locus = _key(member)
        possible = reverse_known.get((member.get("segment_id"), locus[0], locus[2]), [])
        linked = [(key, row) for key, row in possible if row.get("identity") == selected]
        label = truth["loci"].get(linked[0][0]) if len(linked) == 1 else None
        kind = (
            "known_target"
            if label == "jersey:27"
            else "known_other"
            if label
            else "unlabelled_or_unresolved"
        )
        purity_rows.append({**member, "truth": label, "purity_class": kind})
    unknown = sum(r["purity_class"] == "unlabelled_or_unresolved" for r in purity_rows)
    purity_status = (
        "no_target_identity"
        if selected is None
        else "unresolved_or_unlabelled_predictions"
        if unknown
        else "known_labels_only"
    )
    physical_conflicts = [c for c in conflicts if c.get("status") == "physical_conflict"]
    selected_conflicts = [
        c for c in physical_conflicts if selected is not None and c.get("identity") == selected
    ]
    pending_overlaps = [
        c
        for c in conflicts
        if c.get("status") == "overlap_pending"
        and selected is not None
        and c.get("identity") == selected
    ]
    all_conflicts = len(physical_conflicts)
    correct_count = sum(r["correct"] for r in target_rows)
    core_checks = {
        "coverage": correct_count >= 57,
        "known_pollution": not known_other_samples and not known_other_frames,
        "physical_conflicts": not selected_conflicts,
        "fragmentation": fragment_count <= 2,
        "exposure": exposure >= 50.0,
        "target_identity_exists": selected is not None,
        "no_pending_target_overlaps": not pending_overlaps,
    }
    checks = {
        **core_checks,
        "no_unresolved_target_members": unknown == 0,
    }
    j15_rows = [r for r in sample_rows if r["truth"] == "jersey:15"]
    j15_correct = sum(r["correct"] for r in j15_rows)
    j15_ids = {r["identity"] for r in j15_rows if r["identity"] is not None}
    j15_fragment_count = max(0, len(j15_ids) - 1)

    summary = {
        "schema": "t4-golden-score-v1",
        "target_player": "27号",
        "total_denominator": 70,
        "correct_samples": correct_count,
        "coverage_rate": correct_count / 70,
        "observed_exposure_seconds": exposure,
        "cross_identity_pollution": pollution,
        "known_other_samples": len(known_other_samples),
        "known_other_observations": len(known_other_frames),
        "physical_conflicts": len(selected_conflicts),
        "all_identity_physical_conflicts": all_conflicts,
        "selected_identity_pending_overlaps": len(pending_overlaps),
        "fragmentation_count": fragment_count,
        "target_identity_count": len(target_ids),
        "selected_identity": selected,
        "player_15_metrics": {
            "total_denominator": len(j15_rows),
            "correct_samples": j15_correct,
            "coverage_rate": (j15_correct / len(j15_rows)) if j15_rows else 0.0,
            "fragmentation_count": j15_fragment_count,
            "target_identity_count": len(j15_ids),
            "aligned_identity": alignment.get("jersey:15"),
        },
        "alignment": alignment,
        "alignment_rule": "global maximum-weight injective Hungarian on all fixed labelled samples; sorted ties; one pass",
        "purity_status": purity_status,
        "selected_identity_unknown_observations": unknown,
        "selected_identity_observations": len(purity_rows),
        "correct_observation_frames": len(correct_frames),
        "checks": checks,
        "passed": all(core_checks.values()),
        "excellent_target_reached": correct_count >= 62,
        "failure_counts": dict(sorted(Counter(r["primary_cause"] for r in failures).items())),
        "geometry_freeze_sha256": expected_freeze_sha256,
        "run_dir": freeze.get("run_dir"),
        "truth_scope": truth["scope"],
        "per_clip": {
            clip: {
                "correct": sum(r["correct"] for r in target_rows if r["video_id"] == clip),
                "total": sum(r["video_id"] == clip for r in target_rows),
                "outcomes": dict(
                    sorted(
                        Counter(r["outcome"] for r in target_rows if r["video_id"] == clip).items()
                    )
                ),
            }
            for clip in truth["scope"]["target_clips"]
        },
        "limits": [
            "Selected-range jersey labels do not certify exhaustive person identity purity.",
            "Zero pollution/conflicts on an empty gallery is vacuous; target_identity_exists must pass.",
            "Duration is union of correctly labelled observed frames, never track start/end span.",
            "Unrecorded OCR/appearance causal hypotheses are not asserted as proven failure causes.",
        ],
    }
    # Refuse a completed score if predictions/code/inputs or answers changed while
    # computing it. The external root protects against rewriting both hash files.
    verify_prepared(geometry, expected_freeze_sha256)
    for record in truth["manifest"]:
        if _sha(record["path"]) != record["sha256"]:
            raise ValueError("Truth changed during evaluation")
    _write_json(out / "truth-manifest.json", truth["manifest"])
    _write_json(out / "truth-scope.json", {**truth["scope"], "ranges": truth["ranges"]})
    _write_rows(out / "sample-ledger.jsonl", sample_rows)
    _write_rows(out / "player27-ledger.jsonl", target_rows)
    _write_rows(out / "frame-ledger.jsonl", frame_rows)
    _write_rows(out / "audit_failure_cases.jsonl", failures)
    _write_rows(out / "purity-ledger.jsonl", purity_rows)
    _write_rows(out / "known-other-observations.jsonl", known_other_frames)
    _write_json(out / "physical-conflicts.json", selected_conflicts)
    _write_json(out / "scoring_summary.json", summary)
    with (out / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        identities = sorted({r["identity"] for r in sample_rows if r["identity"] is not None})
        writer = csv.writer(stream)
        writer.writerow(["truth", *identities, "unassigned_or_missing", "total"])
        for label in sorted(alignment):
            rows = [r for r in sample_rows if r["truth"] == label]
            counts = Counter(r["identity"] for r in rows)
            writer.writerow(
                [label, *(counts[identity] for identity in identities), counts[None], len(rows)]
            )
    (out / "acceptance_report.md").write_text(_report(summary), encoding="utf-8")
    _write_json(
        out / "post-score-gate.json",
        {
            "verified_at": datetime.now(UTC).isoformat(),
            "passed": True,
            "geometry_freeze_sha256": expected_freeze_sha256,
            "truth_files": len(truth["manifest"]),
            "process_id": os.getpid(),
        },
    )
    return summary
