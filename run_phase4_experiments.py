"""Automated controlled runner for Phase 4 methodological improvement exploration.

Evaluates:
- Baseline: controlled-baseline-001 (track-consistency-v2 + complete_support_v2 margin=0.04)
- Phase 3 Best Numerical Variant: exp-m1b-margin-006 (margin=0.06)
- Candidate A: phase4-cand-a-track-v3 (track-consistency-v3: local windows + transition zone + scoped risks)
- Candidate B: phase4-cand-b-compensation (complete_support_v3: independent support compensation for missing geometric negatives)
- Combined A+B: phase4-cand-ab-combined (track-consistency-v3 + complete_support_v3)

Projects all outputs onto fixed evaluation units (var/evaluation/fixed_units_baseline_001.json)
and tabulates multi-layer funnel metrics, unknown reason transitions, and safety audit assertions.
"""

import json
import time
from pathlib import Path

from app.adapters.machine import load_machine_input
from app.contracts.discovery import DiscoveryReport
from app.contracts.evidence import EvidenceBatch
from app.contracts.identity import Assignment, Segment
from app.contracts.segmentation import SegmentationReport
from app.evaluation.metrics import evaluate_dual_track
from app.evaluation.units import EvaluationManifest, project_units_to_segments
from app.presentation.baseline import load_baseline_settings, run_baseline

PROJECT_ROOT = Path(__file__).resolve().parent
VAR_DIR = PROJECT_ROOT / "var"
FIXED_UNITS_PATH = VAR_DIR / "evaluation" / "fixed_units_baseline_001.json"
BASELINE_DIR = VAR_DIR / "identity-discovery" / "controlled-baseline-001"
M1B_DIR = VAR_DIR / "identity-discovery" / "exp-m1b-margin-006"

PHASE4_EXPERIMENTS = [
    {
        "id": "phase4-cand-a-track-v3",
        "name": "Candidate A",
        "method_layer": "segmentation",
        "version_tag": "track-consistency-v3",
        "config_path": "examples/experiments/cand-a-track-consistency-v3.toml",
        "fixed_segmentation": False,
        "description": "候选方案 A：局部分段与过渡区优化（局部稳定窗+过渡未知区+风险范围收敛）",
    },
    {
        "id": "phase4-cand-b-compensation",
        "name": "Candidate B",
        "method_layer": "discovery",
        "version_tag": "complete_support_v3",
        "config_path": "examples/experiments/cand-b-geometric-compensation.toml",
        "fixed_segmentation": True,
        "description": "候选方案 B：几何负证据不足时的独立支持补偿机制（严格多部位+近距离锚定+反例防范）",
    },
    {
        "id": "phase4-cand-ab-combined",
        "name": "Candidate A+B Combined",
        "method_layer": "combined",
        "version_tag": "track-consistency-v3 + complete_support_v3",
        "config_path": "examples/experiments/cand-ab-combined.toml",
        "fixed_segmentation": False,
        "description": "组合方案 A+B：局部分段优化与几何负证据独立补偿综合方案",
    },
]


def load_fixed_units() -> EvaluationManifest:
    with open(FIXED_UNITS_PATH, encoding="utf-8") as f:
        return EvaluationManifest.model_validate(json.load(f))


def evaluate_run_directory(manifest: EvaluationManifest, run_dir: Path) -> dict:
    with open(run_dir / "predictions.json", encoding="utf-8") as f:
        pred = json.load(f)
    with open(run_dir / "discovery.json", encoding="utf-8") as f:
        disc = json.load(f)
        disc_rep = DiscoveryReport.model_validate(disc["report"])
    with open(run_dir / "segmentation.json", encoding="utf-8") as f:
        seg_rep = SegmentationReport.model_validate(json.load(f))

    segments = [Segment.model_validate(s) for s in pred["segments"]]
    assignments = [Assignment.model_validate(a) for a in pred["assignments"]]

    t0 = time.perf_counter()
    projected = project_units_to_segments(manifest.units, segments, assignments, disc_rep)
    eval_rep = evaluate_dual_track(
        manifest,
        projected,
        segments=segments,
        segmentation_report=seg_rep,
        discovery_report=disc_rep,
        runtime_s=time.perf_counter() - t0,
    )

    unsupervised = eval_rep.unsupervised

    # Preparation layer metrics
    active_segments = [s for s in segments if s.active]
    supported_proposals = [
        p for t in seg_rep.tracks for p in t.proposals if p.status == "supported"
    ]
    supported_duration_s = sum(p.segment.end_s - p.segment.start_s for p in supported_proposals)
    qualified_count = disc_rep.summary.get("qualified_segments", 0)

    # Grouping layer metrics
    pair_statuses = disc_rep.summary.get("pair_statuses", {})
    groups_count = len(disc_rep.groups)
    member_segments_count = sum(len(g.member_segment_ids) for g in disc_rep.groups)
    unknown_segments_count = len(disc_rep.unknown)

    # Stopping condition checks
    bridge_reasons = sum(
        1 for u in disc_rep.unknown if "incompatible_positive_neighbors_bridge" in u.reasons
    )
    unknown_breakdown = unsupervised.unknown_reason_distribution
    bridge_projected_units = unknown_breakdown.get("incompatible_positive_neighbors_bridge", 0)

    # Check template witness integrity
    self_loop_detected = False
    for group in disc_rep.groups:
        for t in group.templates:
            if not t.witness_sample_ids or t.sample_id in t.witness_sample_ids:
                self_loop_detected = True

    stopping_condition_triggered = False
    stopping_reasons = []
    if bridge_reasons > 20 or bridge_projected_units > 2000:
        stopping_condition_triggered = True
        stopping_reasons.append(f"桥接冲突爆发(bridge={bridge_reasons})")
    if self_loop_detected:
        stopping_condition_triggered = True
        stopping_reasons.append("自循环准入检测异常")

    return {
        "sample_coverage": unsupervised.sample_coverage,
        "sample_coverage_pct": f"{unsupervised.sample_coverage * 100:.3f}%",
        "assigned_samples": unsupervised.assigned_samples,
        "total_samples": unsupervised.total_samples,
        "duration_coverage": unsupervised.duration_coverage,
        "duration_coverage_pct": f"{unsupervised.duration_coverage * 100:.3f}%",
        "assigned_duration_s": unsupervised.assigned_duration_s,
        "total_duration_s": unsupervised.total_duration_s,
        "observation_coverage_pct": f"{unsupervised.observation_coverage * 100:.3f}%",
        "assigned_observations": unsupervised.assigned_observations,
        # Prep layer
        "active_segments": len(active_segments),
        "supported_segments": len(supported_proposals),
        "supported_ratio_pct": f"{len(supported_proposals) / len(active_segments) * 100:.2f}%"
        if active_segments
        else "0.0%",
        "supported_duration_s": supported_duration_s,
        "qualified_segments": qualified_count,
        "qualified_ratio_pct": f"{qualified_count / len(active_segments) * 100:.2f}%"
        if active_segments
        else "0.0%",
        # Grouping layer
        "pairs_supported": pair_statuses.get("supported", 0),
        "pairs_conflict": pair_statuses.get("conflict", 0),
        "pairs_ambiguous": pair_statuses.get("ambiguous", 0),
        "pairs_insufficient": pair_statuses.get("insufficient", 0),
        "total_pairs": sum(pair_statuses.values()),
        "groups_count": groups_count,
        "member_segments_count": member_segments_count,
        "unknown_segments_count": unknown_segments_count,
        "templates_count": len(pred.get("templates", [])),
        # Unknown breakdown on projected units
        "unknown_breakdown": unknown_breakdown,
        "bridge_segments_in_disc": bridge_reasons,
        "stopping_triggered": stopping_condition_triggered,
        "stopping_reasons": stopping_reasons,
        "eval_report": eval_rep,
    }


def main():
    print("=" * 80)
    print("Phase 4 球员身份发现模块方法级受控探索 runner")
    print("=" * 80)

    manifest = load_fixed_units()
    print(
        f"已加载固定评估单元: {manifest.total_samples} 样本, {manifest.total_observations} 观测, 总时长 {manifest.total_duration_s:.3f}s"
    )

    # 1. Baseline
    print("\n[Step 0] 评估对照基线 controlled-baseline-001 ...")
    baseline_metrics = evaluate_run_directory(manifest, BASELINE_DIR)
    print(
        f"  基线结果: 样本覆盖={baseline_metrics['sample_coverage_pct']} ({baseline_metrics['assigned_samples']}), "
        f"时长覆盖={baseline_metrics['duration_coverage_pct']} ({baseline_metrics['assigned_duration_s']:.2f}s), "
        f"组数={baseline_metrics['groups_count']}, 成员段数={baseline_metrics['member_segments_count']}"
    )

    # 2. Phase 3 Best (M1b)
    print("\n[Step 1] 评估 Phase 3 最佳数值变体 exp-m1b-margin-006 ...")
    m1b_metrics = evaluate_run_directory(manifest, M1B_DIR)
    print(
        f"  M1b 结果: 样本覆盖={m1b_metrics['sample_coverage_pct']} ({m1b_metrics['assigned_samples']}), "
        f"时长覆盖={m1b_metrics['duration_coverage_pct']} ({m1b_metrics['assigned_duration_s']:.2f}s), "
        f"组数={m1b_metrics['groups_count']}, 成员段数={m1b_metrics['member_segments_count']}"
    )

    # Preload machine input to avoid repetitive file parsing
    print("\n[Init] 预加载机器证据输入与特征数据...")
    init_settings = load_baseline_settings(PROJECT_ROOT / "examples/machine-discovery.toml")
    t_load = time.perf_counter()
    preloaded_raw = load_machine_input(init_settings, batch_id="phase4-preload")
    print(
        f"  预加载完成: {len(preloaded_raw[0].identity_evidence.samples)} 样本, {len(preloaded_raw[1])} 特征行, 耗时 {time.perf_counter() - t_load:.2f}s"
    )

    # Baseline segmentation & batch for Candidate B (fixed segmentation)
    with open(BASELINE_DIR / "segmentation.json", encoding="utf-8") as f:
        baseline_seg_rep = SegmentationReport.model_validate(json.load(f))
    with open(BASELINE_DIR / "evidence.json", encoding="utf-8") as f:
        baseline_ev_batch = EvidenceBatch.model_validate(json.load(f))
    preloaded_matching_input = (baseline_ev_batch, baseline_seg_rep)

    results = {
        "baseline": baseline_metrics,
        "exp-m1b-margin-006": m1b_metrics,
    }

    # 3. Run Phase 4 Candidate experiments
    for exp in PHASE4_EXPERIMENTS:
        exp_id = exp["id"]
        exp_name = exp["name"]
        print(f"\n[Running {exp_name}] {exp['description']} (run_id: {exp_id}) ...")

        config_file = PROJECT_ROOT / exp["config_path"]
        settings = load_baseline_settings(config_file)
        run_output_dir = VAR_DIR / "identity-discovery" / exp_id

        t0 = time.perf_counter()
        if run_output_dir.exists():
            print(f"  注意: 运行目录 {run_output_dir} 已存在，读取现有产物进行受控评估。")
        else:
            if exp["fixed_segmentation"]:
                fixed_seg_path = BASELINE_DIR / "segmentation.json"
                out_path = run_baseline(
                    settings,
                    run_id=exp_id,
                    batch_id=f"run-{exp_id}",
                    discover_identities=True,
                    fixed_segmentation=fixed_seg_path,
                    preloaded_raw=preloaded_raw,
                    preloaded_batch_and_seg=preloaded_matching_input,
                )
            else:
                out_path = run_baseline(
                    settings,
                    run_id=exp_id,
                    batch_id=f"run-{exp_id}",
                    discover_identities=True,
                    preloaded_raw=preloaded_raw,
                )
            print(f"  执行完成: {out_path} 耗时: {time.perf_counter() - t0:.2f}s")

        eval_metrics = evaluate_run_directory(manifest, run_output_dir)
        results[exp_id] = eval_metrics
        print(f"  评估结果 ({exp_name}):")
        print(
            f"    准备层: 派生段={eval_metrics['active_segments']}, "
            f"支持段={eval_metrics['supported_segments']} ({eval_metrics['supported_ratio_pct']}), "
            f"合格段={eval_metrics['qualified_segments']} ({eval_metrics['qualified_ratio_pct']}), "
            f"支持时长={eval_metrics['supported_duration_s']:.2f}s"
        )
        print(
            f"    归人层: 支持对={eval_metrics['pairs_supported']}, "
            f"冲突对={eval_metrics['pairs_conflict']}, 歧义对={eval_metrics['pairs_ambiguous']}, 不足对={eval_metrics['pairs_insufficient']}, "
            f"组数={eval_metrics['groups_count']}, 成员段数={eval_metrics['member_segments_count']}, 未知段数={eval_metrics['unknown_segments_count']}"
        )
        print(
            f"    覆盖率: 样本覆盖={eval_metrics['sample_coverage_pct']} ({eval_metrics['assigned_samples']}), "
            f"时长覆盖={eval_metrics['duration_coverage_pct']} ({eval_metrics['assigned_duration_s']:.2f}s), "
            f"观测覆盖={eval_metrics['observation_coverage_pct']} ({eval_metrics['assigned_observations']})"
        )
        if eval_metrics["stopping_triggered"]:
            print(f"    [警告] 触发停止条件: {eval_metrics['stopping_reasons']}")
        else:
            print("    [安全] 未触发任何停止条件，安全边界稳定。")

        # Save individual evaluation report
        eval_file = VAR_DIR / "evaluation" / f"{exp_id}_eval.json"
        with open(eval_file, "w", encoding="utf-8") as f:
            f.write(eval_metrics["eval_report"].model_dump_json(indent=2))

    # Save all summary results to json
    all_summary_path = VAR_DIR / "evaluation" / "phase4_methodological_experiments_summary.json"
    with open(all_summary_path, "w", encoding="utf-8") as f:
        serializable_results = {
            k: {sk: sv for sk, sv in v.items() if sk != "eval_report"} for k, v in results.items()
        }
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    print(f"\n全部实验数据已汇总保存至: {all_summary_path}")

    return results


if __name__ == "__main__":
    main()
