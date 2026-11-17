"""The 100-order audit report must never be reused without a fingerprint match.

A cached congruence report is only trustworthy when it was produced by the
current algorithm source against the current frozen evidence. These tests pin
the cache gate: stale or fingerprint-less reports must trigger a real re-run.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

MODULE_PATH = Path(__file__).resolve().parent / "test_order_congruence_100.py"
PRIMARY = PROJECT / "var" / "runs" / "run-phase2-split-final"

pytestmark = pytest.mark.skipif(
    not (PRIMARY / "player-library.sqlite3").is_file(),
    reason="frozen run evidence not available (var/ is not distributed)",
)


def _load_module():
    spec = importlib.util.spec_from_file_location("order_congruence_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_summary(fingerprint):
    return {
        "input_fingerprint": fingerprint,
        "congruence_metrics": {
            "exact_partition_matches": 100,
            "exact_match_ratio": 1.0,
            "co_assignment_jaccard_mean": 1.0,
            "co_assignment_jaccard_variance": 0.0,
            "nmi_mean": 1.0,
            "nmi_variance": 0.0,
        },
        "target_stability": {"player_27_stable": True, "player_15_stable": True},
    }


def test_fingerprint_less_cached_report_is_recomputed(tmp_path, monkeypatch):
    module = _load_module()
    out_json = tmp_path / "congruence-100-runs.json"
    stale = _valid_summary("legacy-report-without-fingerprint")
    del stale["input_fingerprint"]
    out_json.write_text(json.dumps(stale), encoding="utf-8")
    calls = []

    def fake_run(primary_dir, out, **kwargs):
        calls.append(str(primary_dir))
        summary = _valid_summary(module._audit_fingerprint(PRIMARY))
        out.write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(module, "execute_100_runs_audit", fake_run)
    summary = module.load_or_run_audit(PRIMARY, out_json)
    assert calls == [str(PRIMARY)]
    assert summary["input_fingerprint"] == module._audit_fingerprint(PRIMARY)


def test_mismatched_fingerprint_is_recomputed(tmp_path, monkeypatch):
    module = _load_module()
    out_json = tmp_path / "congruence-100-runs.json"
    out_json.write_text(json.dumps(_valid_summary("outdated-source-hash")), encoding="utf-8")
    calls = []

    def fake_run(primary_dir, out, **kwargs):
        calls.append(str(primary_dir))
        summary = _valid_summary(module._audit_fingerprint(PRIMARY))
        out.write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(module, "execute_100_runs_audit", fake_run)
    module.load_or_run_audit(PRIMARY, out_json)
    assert calls == [str(PRIMARY)]


def test_matching_fingerprint_is_reused(tmp_path, monkeypatch):
    module = _load_module()
    out_json = tmp_path / "congruence-100-runs.json"
    fingerprint = module._audit_fingerprint(PRIMARY)
    out_json.write_text(json.dumps(_valid_summary(fingerprint)), encoding="utf-8")

    def fake_run(*args, **kwargs):
        raise AssertionError("a fingerprint-matching report must not be recomputed")

    monkeypatch.setattr(module, "execute_100_runs_audit", fake_run)
    summary = module.load_or_run_audit(PRIMARY, out_json)
    assert summary["input_fingerprint"] == fingerprint


def test_fingerprint_covers_algorithm_source_and_frozen_evidence(tmp_path):
    module = _load_module()
    base = module._audit_fingerprint(PRIMARY)
    assert base == module._audit_fingerprint(PRIMARY)
    assert len(base) == 64
