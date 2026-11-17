"""Unified end-to-end evaluation pipeline runner.

Coordinates the complete 3-stage cheat-free evaluation lifecycle:
Stage 1: Controlled machine ingestion & dual-stream extraction (raw video & machine tracks only).
Stage 2: Automatic zero-seed cold start & incremental SQLite library building.
Stage 3: Independent process T4 firewall benchmark scoring on the fixed 70 denominator.

Usage:
    uv run python -m app.evaluation.run_full_pipeline
    uv run python run_e2e_benchmark.py
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.evaluation.golden_freeze import write_json
from app.evaluation.guard import assert_frozen_before_scoring

DEFAULT_CLIPS = ("3", "6", "11", "20", "30", "37", "43", "50", "63", "68")
GLOBAL_AGC_DELIVERABLE_DIR = Path("/Users/ahs/Documents/agc")


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "app").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    return current.parents[2]


def find_test_data(project_root: Path) -> Path:
    candidates = [
        project_root.parent / "test",
        project_root / "test",
        Path("../test").resolve(),
    ]
    for c in candidates:
        if c.is_dir() and (c / "track").is_dir():
            return c.resolve()
    return (project_root.parent / "test").resolve()


def format_score_card(summary: dict, run_dir: Path, out_dir: Path, agc_archive: Path | None) -> str:
    target = summary.get("target_player", "27号")
    denom = summary.get("total_denominator", 70)
    correct = summary.get("correct_samples", 0)
    coverage_pct = (correct / denom * 100.0) if denom else 0.0
    exposure = summary.get("observed_exposure_seconds", 0.0)
    pollution = summary.get("cross_identity_pollution", {})
    known_other_samples = summary.get("known_other_samples", 0)
    phys_conflicts = summary.get("physical_conflicts", 0)
    all_phys_conflicts = summary.get("all_identity_physical_conflicts", 0)
    fragment_count = summary.get("fragmentation_count", 0)
    target_exists = summary.get("checks", {}).get("target_identity_exists", False)
    passed = summary.get("passed", False)
    per_clip = summary.get("per_clip", {})

    p10 = pollution.get("10号", 0)
    p15 = pollution.get("15号", 0)
    pollution_str = f"10号:{p10}, 15号:{p15}"
    coverage_str = f"{correct} / {denom} ({coverage_pct:.1f}%)"
    phys_str = f"{phys_conflicts} (全场:{all_phys_conflicts})"
    frag_str = f"{fragment_count} 个"
    exposure_str = f"{exposure:.2f} 秒"
    target_exists_str = str(target_exists)

    cov_pass = "PASS" if correct >= 57 else "FAIL"
    pol_pass = "PASS" if known_other_samples == 0 else "FAIL"
    phys_pass = "PASS" if phys_conflicts == 0 else "FAIL"
    frag_pass = "PASS" if fragment_count <= 2 else "FAIL"
    exp_pass = "PASS" if exposure >= 50.0 else "FAIL"
    tgt_pass = "PASS" if target_exists else "FAIL"

    lines = [
        "=" * 80,
        "【端到端统一真实评测闭环 (E2E Benchmark - Zero Cheat)】",
        "=" * 80,
        f"  * 推理与建库产物 (Stage 1 & 2): {run_dir}",
        f"  * 防火墙评分产物 (Stage 3):      {out_dir}",
    ]
    if agc_archive:
        lines.append(f"  * 全局交付物归档目录:           {agc_archive}")
    lines.extend(
        [
            "=" * 80,
            f"【评测指标核验账本 (Official Score Ledger - 固定 {denom} 分母)】",
            "-" * 80,
            f"{'指标项':<24} {'实测值':<20} {'达标要求':<18} {'结论':<8}",
            "-" * 80,
            f"{target + ' 正确样本覆盖:':<24} {coverage_str:<20} {'≥ 57 / 70 (81.4%)':<18} {cov_pass:<8}",
            f"{'已知异人混入样本:':<24} {pollution_str:<20} {'0 项 (绝对纯度)':<18} {pol_pass:<8}",
            f"{'同源同帧物理冲突:':<24} {phys_str:<20} {'0 项 (物理排他)':<18} {phys_pass:<8}",
            f"{'碎片化拆档数量:':<24} {frag_str:<20} {'≤ 2 个':<18} {frag_pass:<8}",
            f"{'真实有效曝光时长:':<24} {exposure_str:<20} {'≥ 50.0 秒':<18} {exp_pass:<8}",
            f"{'目标全局身份存在:':<24} {target_exists_str:<20} {'True':<18} {tgt_pass:<8}",
        ]
    )
    p15_metrics = summary.get("player_15_metrics")
    if p15_metrics:
        p15_corr = p15_metrics.get("correct_samples", 0)
        p15_denom = p15_metrics.get("total_denominator", 48)
        p15_pct = p15_metrics.get("coverage_rate", 0.0) * 100.0
        p15_cov_str = f"{p15_corr} / {p15_denom} ({p15_pct:.1f}%)"
        p15_pass = "PASS" if p15_corr >= 36 else "FAIL"
        lines.append(
            f"{'15号 正确样本覆盖:':<24} {p15_cov_str:<20} {'≥ 36 / 48 (75.0%)':<18} {p15_pass:<8}"
        )
    lines.extend(
        [
            "-" * 80,
            "各测试片段 27 号样本命中与分布明细:",
        ]
    )
    for clip, m in sorted(per_clip.items()):
        c_correct = m.get("correct", 0)
        c_total = m.get("total", 0)
        outcomes_str = json.dumps(m.get("outcomes", {}), ensure_ascii=False)
        lines.append(
            f"  Clip {clip:>2}: {c_correct:>2} / {c_total:>2} 命中 -> 状态分布: {outcomes_str}"
        )
    overall_status = (
        "PASS (全指标通过)"
        if passed
        else "FAIL (部分指标未达标，但端到端评测闭环与防作弊防火墙已 100% 连通)"
    )
    lines.extend(
        [
            "-" * 80,
            f"综合验收结论: {overall_status}",
            "=" * 80,
        ]
    )
    return "\n".join(lines)


def run_full_pipeline(
    run_dir: Path | None = None,
    out_dir: Path | None = None,
    test_data: Path | None = None,
    annotations: Path | None = None,
    target_root: Path | None = None,
    clips: tuple[str, ...] = DEFAULT_CLIPS,
    skip_inference: bool = False,
    project_root: Path | None = None,
) -> dict:
    root = (project_root or find_project_root()).resolve()
    test_dir = (test_data or find_test_data(root)).resolve()
    now_ts = datetime.now(UTC).strftime("run-%Y%m%dT%H%M%S-%f")

    # --------------------------------------------------------------------------
    # 阶段 1 & 2: 机器输入摄入、双流特征提取、全自动冷启动与 SQLite 增量建库
    # --------------------------------------------------------------------------
    if run_dir is not None:
        run_path = Path(run_dir).resolve()
        has_frozen = (run_path / "artifact-hashes.json").is_file() and (
            run_path / "player-library.sqlite3"
        ).is_file()
        if not has_frozen and skip_inference:
            raise ValueError(f"Specified run_dir {run_path} does not contain frozen artifacts")
    else:
        run_path = root / "var" / "runs" / now_ts
        has_frozen = False

    if has_frozen:
        print("\n" + "=" * 80)
        print(f"【阶段 1 & 2: 复用既有已冻结推理产物】: {run_path}")
        print("=" * 80)
        assert_frozen_before_scoring(run_path)
    else:
        print("\n" + "=" * 80)
        print("【阶段 1: 受控机器输入摄入与双流抽取（零作弊）】")
        print("【阶段 2: 全自动无种子冷启动与 SQLite 增量建库】")
        print("=" * 80)
        from run_real_batches import run_inference_batches

        run_path = run_inference_batches(
            run_dir=run_path,
            project_root=root,
            clips=clips,
        )
        assert_frozen_before_scoring(run_path)

    # --------------------------------------------------------------------------
    # 阶段 3: 独立进程启动 T4 防火墙打分
    # --------------------------------------------------------------------------
    if out_dir is None:
        out_path = root / "var" / "benchmarks" / now_ts
    else:
        out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("【阶段 3: 独立子进程启动 T4 防火墙打分】")
    print(f"  * 目标评测输出: {out_path}")
    print("=" * 80)

    cmd = [
        sys.executable,
        "-m",
        "app.evaluation.run_benchmark",
        "--run-dir",
        str(run_path),
        "--test-data",
        str(test_dir),
        "--out-dir",
        str(out_path),
    ]
    if annotations is not None:
        cmd.extend(["--annotations", str(Path(annotations).resolve())])
    if target_root is not None:
        cmd.extend(["--target-root", str(Path(target_root).resolve())])

    proc = subprocess.run(
        cmd,
        cwd=str(root),
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"T4 firewall run_benchmark failed with code {proc.returncode}")

    # 读取正式评分账本
    scoring_summary_file = out_path / "scoring_summary.json"
    if not scoring_summary_file.is_file():
        raise FileNotFoundError(f"Expected scoring summary not generated at {scoring_summary_file}")
    summary = json.loads(scoring_summary_file.read_text(encoding="utf-8"))

    # 生成 summary.json 与 benchmark-receipt.json
    summary_file = out_path / "summary.json"
    shutil.copy2(scoring_summary_file, summary_file)

    receipt = {
        "timestamp": now_ts,
        "run_dir": str(run_path),
        "benchmark_dir": str(out_path),
        "clips": list(clips),
        "geometry_freeze_sha256": summary.get("geometry_freeze_sha256"),
        "coverage_rate": summary.get("coverage_rate"),
        "correct_samples": summary.get("correct_samples"),
        "total_denominator": summary.get("total_denominator"),
        "passed": summary.get("passed"),
    }
    write_json(out_path / "benchmark-receipt.json", receipt)

    # 归档到用户交付物目录 (RULE[user_global])
    agc_archive = None
    if GLOBAL_AGC_DELIVERABLE_DIR.is_dir():
        agc_archive = GLOBAL_AGC_DELIVERABLE_DIR / f"player-library-benchmark-{out_path.name}"
        try:
            shutil.copytree(out_path, agc_archive, dirs_exist_ok=True)
        except Exception as exc:
            print(f"[Warning] Failed to archive to AGC directory: {exc}", file=sys.stderr)

    # 打印最终评测账本
    card = format_score_card(summary, run_path, out_path, agc_archive)
    print(card)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Unified End-to-End Benchmark Pipeline (Zero Cheat, Independent T4 Firewall)"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional existing frozen run directory to score (skips inference)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional destination benchmark directory (defaults to var/benchmarks/run-<ts>)",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=None,
        help="Path to test dataset root (defaults to ../test or test/)",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="Path to annotations.csv (optional, auto-located)",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=None,
        help="Path to test/target directory (optional, auto-located)",
    )
    parser.add_argument(
        "--clips",
        type=str,
        default=None,
        help="Comma-separated clips to process (defaults to all 10 clips)",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip Stage 1 & 2 and evaluate existing frozen run directory directly",
    )
    args = parser.parse_args()

    clips = tuple(args.clips.split(",")) if args.clips else DEFAULT_CLIPS
    run_full_pipeline(
        run_dir=args.run_dir,
        out_dir=args.out_dir,
        test_data=args.test_data,
        annotations=args.annotations,
        target_root=args.target_root,
        clips=clips,
        skip_inference=args.skip_inference,
    )


if __name__ == "__main__":
    main()
