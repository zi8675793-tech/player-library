"""The validation CLI only forwards inputs and summarizes the returned evidence."""

from pathlib import Path

import pytest

from app.contracts.evidence import EvidenceBatch, EvidenceWarning, IdentityEvidence
from app.presentation import cli
from app.settings import SettingsError, load_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_validate_forwards_settings_batch_and_clip_selection(monkeypatch, capsys):
    settings = load_settings(PROJECT_ROOT / "examples/sample-test.toml")
    calls = []

    def load(config_path):
        calls.append(config_path)
        return settings

    def adapt(value, *, batch_id, clip_ids):
        calls.append((value, batch_id, clip_ids))
        return EvidenceBatch(
            match_id=value.match.match_id,
            batch_id=batch_id,
            producer="test",
            identity_evidence=IdentityEvidence(),
            warnings=[
                EvidenceWarning(code="needs_review", message="Review source quality", clip_id="x")
            ],
        )

    monkeypatch.setattr(cli, "load_settings", load)
    monkeypatch.setattr(cli, "load_sample_test", adapt)
    assert (
        cli.main(
            [
                "validate",
                "--config",
                "chosen.toml",
                "--batch-id",
                "handoff",
                "--clip-id",
                "x",
                "--clip-id",
                "y",
            ]
        )
        == 0
    )
    assert calls == ["chosen.toml", (settings, "handoff", ["x", "y"])]
    output = capsys.readouterr()
    assert output.err == ""
    assert "match=sample-match batch=handoff" in output.out
    assert "clips=0 segments=0 samples=0 events=0 feature_spaces=0" in output.out
    assert "warnings=1" in output.out
    assert "warning [needs_review] clip=x: Review source quality" in output.out
    assert "embeddings" not in output.out


@pytest.mark.parametrize("error_type", [SettingsError, cli.SampleTestError])
def test_validate_failure_has_no_success_summary(monkeypatch, capsys, error_type):
    def fail(*args, **kwargs):
        raise error_type("target/x/features.npz: missing visibility_scores")

    if error_type is SettingsError:
        monkeypatch.setattr(cli, "load_settings", fail)
    else:
        settings = load_settings(PROJECT_ROOT / "examples/sample-test.toml")
        monkeypatch.setattr(cli, "load_settings", lambda path: settings)
        monkeypatch.setattr(cli, "load_sample_test", fail)
    assert cli.main(["validate", "--config", "chosen.toml", "--batch-id", "b"]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "Validation failed:" in output.err
    assert "features.npz: missing visibility_scores" in output.err


def test_validate_does_not_treat_normalized_as_sample_test(monkeypatch, capsys):
    settings = load_settings(PROJECT_ROOT / "config.example.toml")
    monkeypatch.setattr(cli, "load_settings", lambda path: settings)
    monkeypatch.setattr(
        cli, "load_sample_test", lambda *a, **k: pytest.fail("unexpected adapter call")
    )
    assert cli.main(["validate", "--config", "chosen.toml", "--batch-id", "b"]) == 1
    assert "does not yet support adapter 'normalized'" in capsys.readouterr().err


def test_validate_requires_explicit_config_and_batch_id():
    with pytest.raises(SystemExit) as error:
        cli.main(["validate"])
    assert error.value.code == 2
