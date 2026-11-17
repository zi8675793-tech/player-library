"""Unit and integration tests for the unified end-to-end evaluation pipeline (Bug 8)."""

import json
import subprocess
import sys
from pathlib import Path

from app.evaluation.run_full_pipeline import (
    find_project_root,
    find_test_data,
    format_score_card,
    run_full_pipeline,
)


def test_project_root_and_test_data_resolution():
    root = find_project_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "app").is_dir()

    test_data = find_test_data(root)
    assert (test_data / "track").is_dir()
    assert (test_data / "video").is_dir()


def test_cli_help():
    root = find_project_root()
    script = root / "run_e2e_benchmark.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=str(root),
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0
    assert "--run-dir" in proc.stdout
    assert "--out-dir" in proc.stdout
    assert "--skip-inference" in proc.stdout
    assert "Unified End-to-End Benchmark Pipeline" in proc.stdout


def test_format_score_card_rendering():
    dummy_summary = {
        "target_player": "27号",
        "total_denominator": 70,
        "correct_samples": 12,
        "observed_exposure_seconds": 12.98,
        "cross_identity_pollution": {"10号": 0, "15号": 0},
        "known_other_samples": 0,
        "physical_conflicts": 0,
        "all_identity_physical_conflicts": 0,
        "fragmentation_count": 0,
        "checks": {"target_identity_exists": True},
        "passed": False,
        "per_clip": {
            "6": {"correct": 12, "total": 12, "outcomes": {"correct": 12}},
            "20": {"correct": 0, "total": 12, "outcomes": {"unassigned": 12}},
        },
    }
    card = format_score_card(
        dummy_summary,
        Path("/mock/run"),
        Path("/mock/benchmarks"),
        Path("/mock/agc"),
    )
    assert "27号 正确样本覆盖:" in card
    assert "12 / 70 (17.1%)" in card
    assert "已知异人混入样本:" in card
    assert "10号:0, 15号:0" in card
    assert "同源同帧物理冲突:" in card
    assert "碎片化拆档数量:" in card
    assert "Clip  6: 12 / 12" in card


def test_invalid_run_dir_without_frozen_artifacts_raises(tmp_path):
    empty_dir = tmp_path / "empty_run"
    empty_dir.mkdir()
    try:
        run_full_pipeline(run_dir=empty_dir, skip_inference=True)
        assert False, "Should have raised ValueError for non-frozen run"
    except ValueError as exc:
        assert "does not contain frozen artifacts" in str(exc)


def test_pipeline_on_frozen_real_run(tmp_path):
    root = find_project_root()
    frozen_run = root / "var" / "runs" / "run-20261115T033000-bug7"
    if not frozen_run.is_dir():
        return

    out_dir = tmp_path / "benchmarks_test"
    summary = run_full_pipeline(
        run_dir=frozen_run,
        out_dir=out_dir,
        project_root=root,
    )

    # 1. Check summary dictionary
    assert summary["schema"] == "t4-golden-score-v1"
    assert summary["total_denominator"] == 70
    assert summary["correct_samples"] == 12
    assert summary["cross_identity_pollution"]["10号"] == 0
    assert summary["cross_identity_pollution"]["15号"] == 0
    assert summary["physical_conflicts"] == 0
    assert summary["checks"]["known_pollution"] is True
    assert summary["checks"]["physical_conflicts"] is True

    # 2. Check generated files
    assert (out_dir / "scoring_summary.json").is_file()
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "benchmark-receipt.json").is_file()
    assert (out_dir / "acceptance_report.md").is_file()
    assert (out_dir / "confusion_matrix.csv").is_file()
    assert (out_dir / "sample-ledger.jsonl").is_file()
    assert (out_dir / "player27-ledger.jsonl").is_file()
    assert (out_dir / "frame-ledger.jsonl").is_file()
    assert (out_dir / "pre-label-gate.json").is_file()
    assert (out_dir / "post-score-gate.json").is_file()

    # 3. Check receipt metadata
    receipt = json.loads((out_dir / "benchmark-receipt.json").read_text(encoding="utf-8"))
    assert receipt["correct_samples"] == 12
    assert receipt["total_denominator"] == 70
    assert receipt["geometry_freeze_sha256"] is not None


def test_workspace_root_wrapper_exists():
    root = find_project_root()
    workspace_root = root.parent
    wrapper = workspace_root / "run_e2e_benchmark.py"
    assert wrapper.is_file()
    content = wrapper.read_text(encoding="utf-8")
    assert "run_e2e_benchmark.py" in content
