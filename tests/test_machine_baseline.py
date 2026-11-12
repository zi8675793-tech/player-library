"""The executable machine baseline freezes outputs from a genuinely fresh state."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_machine_input import digest, json_rows, machine_settings, make_machine_input, write_rows

from app.contracts.identity import Assignment, Segment
from app.presentation import baseline, cli
from app.settings import SettingsError


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_cli_runs_without_answers_from_empty_state_and_exports_scoring_interface(
    tmp_path, monkeypatch, capsys
):
    root = make_machine_input(
        tmp_path / "input", clips=("solo",), track_count=3, parts=3, dimension=5
    )
    settings = machine_settings(root, tmp_path)
    monkeypatch.setattr(baseline, "PROJECT_VAR", settings.paths.runtime_root)
    assert (
        cli.main(["baseline", "--config", str(tmp_path / "machine.toml"), "--run-id", "fresh"]) == 0
    )
    output = settings.paths.runtime_root / "fresh"
    assert output.is_dir()
    assert "fresh" in capsys.readouterr().out
    summary = read_json(output / "summary.json")
    assert (summary["clips"], summary["tracks"], summary["samples"]) == (1, 3, 6)
    assert summary["observations"] == 9
    assert summary["players"] == summary["active_templates"] == summary["human_interventions"] == 0
    assert summary["assignment_statuses"] == {"unassigned": 3}
    assert summary["idempotency"]["changed"] is False
    assert summary["identity_recognition_evaluated"] is False
    predictions = read_json(output / "predictions.json")
    assert predictions["segmentation"] == "provisional_raw_tracks_not_identity_verified"
    segments = [Segment.model_validate(segment) for segment in predictions["segments"]]
    assignments = [
        Assignment.model_validate(assignment) for assignment in predictions["assignments"]
    ]
    assert len(segments) == len(assignments) == 3
    assert {segment.segment_id for segment in segments} == {
        assignment.segment_id for assignment in assignments
    }
    assert all(assignment.player_id is None for assignment in assignments)
    manifest = read_json(output / "input-manifest.json")
    assert manifest["fingerprint"] == summary["input_fingerprint"]
    for source in manifest["sources"]:
        assert source["sha256"] == digest(root / source["path"])
        assert Path(source["path"]).parts[0] in {"track", "identity", "video"}
    hashes = read_json(output / "artifact-hashes.json")
    assert set(hashes) >= {
        "player-library.sqlite3",
        "evidence.json",
        "predictions.json",
        "diagnostics.json",
        "config.json",
        "input-manifest.json",
        "summary.json",
    }
    assert all(digest(output / name) == checksum for name, checksum in hashes.items())


def test_changed_answer_files_and_old_database_do_not_change_predictions(tmp_path, monkeypatch):
    root = make_machine_input(tmp_path / "input", clips=("first", "second"))
    settings = machine_settings(root, tmp_path)
    runtime = settings.paths.runtime_root
    runtime.mkdir()
    old_db = runtime / "player-library.sqlite3"
    old_db.write_bytes(b"old reviewed gallery cannot be opened")
    answer = root.parent / "annotations.csv"
    answer.write_text("answer version one", encoding="utf-8")
    (root / "target").mkdir()
    selection = root / "target" / "selection.json"
    selection.write_text("human grouping one", encoding="utf-8")
    before_db = digest(old_db)
    monkeypatch.setattr(baseline, "PROJECT_VAR", runtime)
    original_open = Path.open

    def deny_answers(path, *args, **kwargs):
        assert path != old_db
        assert path.name != "annotations.csv"
        assert "target" not in path.parts
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "open", deny_answers)
        first = baseline.run_baseline(settings, run_id="first-run")
    answer.write_text("answer version two: different labels and clip order", encoding="utf-8")
    selection.write_text("human grouping two", encoding="utf-8")
    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "open", deny_answers)
        second = baseline.run_baseline(settings, run_id="second-run")
    for name in (
        "predictions.json",
        "input-manifest.json",
        "evidence.json",
        "summary.json",
        "diagnostics.json",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert digest(old_db) == before_db
    with pytest.raises(ValueError, match="already exists"):
        baseline.run_baseline(settings, run_id="first-run")


@pytest.mark.parametrize("run_id", ["../escape", "/absolute", ".", "with/slash", ""])
def test_baseline_rejects_unsafe_run_names_before_creating_state(tmp_path, monkeypatch, run_id):
    root = make_machine_input(tmp_path / "input", clips=("solo",))
    settings = machine_settings(root, tmp_path)
    monkeypatch.setattr(baseline, "PROJECT_VAR", settings.paths.runtime_root)
    with pytest.raises(ValueError, match="run_id"):
        baseline.run_baseline(settings, run_id=run_id)
    assert not settings.paths.runtime_root.exists()


def test_baseline_cannot_write_outside_project_var(tmp_path, monkeypatch):
    root = make_machine_input(tmp_path / "input", clips=("solo",))
    settings = machine_settings(root, tmp_path)
    monkeypatch.setattr(baseline, "PROJECT_VAR", tmp_path / "other-var")
    with pytest.raises(ValueError, match="var"):
        baseline.run_baseline(settings, run_id="outside")
    assert not settings.paths.runtime_root.exists()


@pytest.mark.parametrize("kind", ["target", "answers", "history", "symlink"])
def test_baseline_config_cannot_open_forbidden_input(tmp_path, monkeypatch, kind):
    if kind == "target":
        (tmp_path / "target").mkdir()
        path = tmp_path / "target" / "config.toml"
    elif kind == "answers":
        path = tmp_path / "annotations.csv"
    elif kind == "history":
        path = tmp_path / "reviewed.sqlite3"
    else:
        original = tmp_path / "original.toml"
        original.write_text("not needed", encoding="utf-8")
        path = tmp_path / "linked.toml"
        path.symlink_to(original)
    if kind != "symlink":
        path.write_text("must never be read", encoding="utf-8")

    def no_open(*args, **kwargs):
        pytest.fail("forbidden configuration was opened")

    monkeypatch.setattr(Path, "open", no_open)
    with pytest.raises(SettingsError):
        baseline.load_baseline_settings(path)


def test_fresh_cli_process_never_imports_legacy_answer_adapter(tmp_path):
    root = make_machine_input(tmp_path / "input", clips=("solo",))
    settings = machine_settings(root, tmp_path)
    script = """
import sys
from pathlib import Path
from app.presentation import baseline, cli
baseline.PROJECT_VAR = Path(sys.argv[2])
assert cli.main(['baseline', '--config', sys.argv[1], '--run-id', 'isolated']) == 0
assert 'app.adapters.sample_test' not in sys.modules
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(tmp_path / "machine.toml"),
            str(settings.paths.runtime_root),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Baseline frozen" in result.stdout


def test_reordered_raw_observations_do_not_change_predictions(tmp_path, monkeypatch):
    root = make_machine_input(tmp_path / "input", clips=("solo",), track_count=3)
    settings = machine_settings(root, tmp_path)
    monkeypatch.setattr(baseline, "PROJECT_VAR", settings.paths.runtime_root)
    first = baseline.run_baseline(settings, run_id="original-order")
    path = root / "track" / "solo" / "tracks.jsonl"
    write_rows(path, list(reversed(json_rows(path))))
    csv = root / "track" / "solo" / "tracks.csv"
    lines = csv.read_text(encoding="utf-8").splitlines()
    csv.write_text("\n".join([lines[0], *reversed(lines[1:])]) + "\n", encoding="utf-8")
    second = baseline.run_baseline(settings, run_id="reversed-order")
    assert (first / "predictions.json").read_bytes() == (second / "predictions.json").read_bytes()
    assert (
        read_json(first / "input-manifest.json")["fingerprint"]
        != read_json(second / "input-manifest.json")["fingerprint"]
    )
