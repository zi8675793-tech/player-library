"""Machine segmentation is frozen through the existing fresh-state CLI."""

import json
from pathlib import Path

import pytest
from test_machine_input import digest, machine_settings, make_machine_input

from app.contracts.evidence import EvidenceBatch
from app.presentation import baseline, cli
from app.settings import SettingsError, load_settings


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_segmentation_cli_freezes_before_after_report_and_exact_policy(tmp_path, monkeypatch):
    root = make_machine_input(tmp_path / "input", clips=("solo",), track_count=3)
    settings = machine_settings(root, tmp_path)
    monkeypatch.setattr(baseline, "PROJECT_VAR", settings.paths.runtime_root)
    assert (
        cli.main(
            [
                "baseline",
                "--config",
                str(tmp_path / "machine.toml"),
                "--run-id",
                "segmented",
                "--segment-tracks",
            ]
        )
        == 0
    )
    output = settings.paths.runtime_root / "segmented"
    raw = EvidenceBatch.model_validate_json((output / "raw-evidence.json").read_text())
    proposed = EvidenceBatch.model_validate_json((output / "evidence.json").read_text())
    assert raw.identity_evidence.tracks == proposed.identity_evidence.tracks
    assert raw.sources == proposed.sources
    assert raw.identity_evidence.feature_files == proposed.identity_evidence.feature_files
    assert raw.identity_evidence.feature_spaces == proposed.identity_evidence.feature_spaces
    summary = read_json(output / "summary.json")
    predictions = read_json(output / "predictions.json")
    report = read_json(output / "segmentation.json")
    assert predictions["segmentation"] == "machine_consistency_proposals_not_identity_verified"
    assert summary["segmentation_summary"] == report["summary"]
    assert predictions["segmentation_summary"] == report["summary"]
    assert summary["idempotency"]["changed"] is False
    assert summary["human_interventions"] == summary["players"] == 0
    assert all(assignment["player_id"] is None for assignment in predictions["assignments"])
    config = read_json(output / "config.json")
    assert config["segment_tracks_enabled"] is True
    assert config["segmentation"] == settings.segmentation.model_dump(mode="json")
    assert "app/identity/consistency.py" in config["implementation"]["files"]
    hashes = read_json(output / "artifact-hashes.json")
    assert {"raw-evidence.json", "segmentation.json"}.issubset(hashes)
    assert all(digest(output / name) == checksum for name, checksum in hashes.items())


def test_segmentation_is_reproducible_without_answer_or_legacy_reads(tmp_path, monkeypatch):
    root = make_machine_input(tmp_path / "input", clips=("alpha", "beta"))
    settings = machine_settings(root, tmp_path)
    monkeypatch.setattr(baseline, "PROJECT_VAR", settings.paths.runtime_root)
    original_open = Path.open

    def machine_only(path, *args, **kwargs):
        assert path.name != "annotations.csv"
        assert "target" not in path.parts
        assert "repository-review" not in path.parts
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", machine_only)
    first = baseline.run_baseline(settings, run_id="first", segment_tracks=True)
    second = baseline.run_baseline(settings, run_id="second", segment_tracks=True)
    for name in ("predictions.json", "evidence.json", "segmentation.json", "summary.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_optional_segmentation_section_rejects_unknown_policy_fields(tmp_path):
    settings = machine_settings(tmp_path / "unused-input", tmp_path)
    assert settings.segmentation is not None
    path = tmp_path / "machine.toml"
    with path.open("a", encoding="utf-8") as file:
        file.write("\n[segmentation]\nmisspelled_threshold = 0.2\n")
    with pytest.raises(SettingsError, match="misspelled_threshold"):
        load_settings(path)
