"""Configuration parsing is strict, location-independent, and read-only."""

from pathlib import Path

import pytest

from app.settings import SettingsError, load_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_TEXT = (PROJECT_ROOT / "config.example.toml").read_text()


def write_config(tmp_path: Path, text: str = CONFIG_TEXT) -> Path:
    config_path = tmp_path / "config.toml"
    config_path.write_text(text)
    return config_path


@pytest.mark.parametrize(
    ("relative_path", "adapter", "match_id", "input_root", "runtime_root"),
    [
        ("config.example.toml", "normalized", "example-match", "incoming", "var"),
        ("examples/sample-test.toml", "sample_test", "sample-match", "../test", "var"),
    ],
)
def test_existing_examples_parse(relative_path, adapter, match_id, input_root, runtime_root):
    settings = load_settings(PROJECT_ROOT / relative_path)
    assert settings.input.adapter == adapter
    assert settings.match.match_id == match_id
    assert settings.paths.input_root == (PROJECT_ROOT / input_root).resolve()
    assert settings.paths.runtime_root == (PROJECT_ROOT / runtime_root).resolve()
    assert settings.matching.auto_assign_enabled is False


def test_same_config_resolves_identically_from_different_working_directories(tmp_path, monkeypatch):
    config_dir = tmp_path / "config-dir"
    config_dir.mkdir()
    config_path = write_config(config_dir)
    first_cwd = tmp_path / "first"
    second_cwd = tmp_path / "second"
    first_cwd.mkdir()
    second_cwd.mkdir()
    monkeypatch.chdir(first_cwd)
    first = load_settings("../config-dir/config.toml")
    monkeypatch.chdir(second_cwd)
    second = load_settings(config_path)
    assert first == second
    assert first.paths.input_root == config_dir / "incoming"
    assert first.paths.runtime_root == config_dir / "var"


def test_load_does_not_create_roots_or_modify_input(tmp_path):
    config_path = write_config(tmp_path)
    before = config_path.read_bytes()
    children_before = set(tmp_path.iterdir())
    settings = load_settings(config_path)
    assert not settings.paths.input_root.exists()
    assert not settings.paths.runtime_root.exists()
    assert set(tmp_path.iterdir()) == children_before
    assert config_path.read_bytes() == before


def test_absolute_paths_are_preserved_and_dot_segments_normalized(tmp_path):
    target = tmp_path / "outside" / "nested"
    text = CONFIG_TEXT.replace('"./incoming"', f'"{target.as_posix()}/../input"')
    settings = load_settings(write_config(tmp_path, text))
    assert settings.paths.input_root == tmp_path / "outside" / "input"


@pytest.mark.parametrize(
    ("old", "new", "error_field"),
    [
        ('input_root = "./incoming"', 'input_root = "  "', "paths.input_root"),
        ('input_root = "./incoming"', "input_root = 7", "paths.input_root"),
        ('runtime_root = "./var"', "runtime_root = false", "paths.runtime_root"),
        ('adapter = "normalized"', 'adapter = "unknown"', "input.adapter"),
        ('match_id = "example-match"', 'match_id = "  "', "match.match_id"),
        ('match_id = "example-match"', "match_id = 12", "match.match_id"),
        ("min_common_parts = 2", "min_common_parts = 0", "matching.min_common_parts"),
        ("min_common_parts = 2", "min_common_parts = -1", "matching.min_common_parts"),
        ("min_common_parts = 2", "min_common_parts = true", "matching.min_common_parts"),
        ("min_common_parts = 2", "min_common_parts = 2.0", "matching.min_common_parts"),
        ("min_common_parts = 2", 'min_common_parts = "2"', "matching.min_common_parts"),
        (
            "max_templates_per_player = 6",
            "max_templates_per_player = 0",
            "matching.max_templates_per_player",
        ),
        ("auto_assign_enabled = false", "auto_assign_enabled = 0", "matching.auto_assign_enabled"),
        (
            "auto_assign_enabled = false",
            'auto_assign_enabled = "false"',
            "matching.auto_assign_enabled",
        ),
        ("before_event_s = 3.0", "before_event_s = -0.1", "playback.before_event_s"),
        ("before_event_s = 3.0", "before_event_s = true", "playback.before_event_s"),
        ("before_event_s = 3.0", 'before_event_s = "3.0"', "playback.before_event_s"),
        ('host = "127.0.0.1"', 'host = "  "', "server.host"),
        ('host = "127.0.0.1"', "host = 123", "server.host"),
        ("port = 8000", "port = 0", "server.port"),
        ("port = 8000", "port = 65536", "server.port"),
        ("port = 8000", "port = true", "server.port"),
        ("port = 8000", "port = 8000.0", "server.port"),
        ("port = 8000", 'port = "8000"', "server.port"),
        ("port = 8000", "", "server.port"),
        ('[server]\nhost = "127.0.0.1"\nport = 8000', "", "server"),
        ("port = 8000", "port = 8000\nunknown = 1", "server.unknown"),
        ("[paths]", "unknown = 1\n[paths]", "unknown"),
    ],
)
def test_invalid_settings_report_config_path_and_field(tmp_path, old, new, error_field):
    assert old in CONFIG_TEXT
    path = write_config(tmp_path, CONFIG_TEXT.replace(old, new))
    with pytest.raises(SettingsError) as error:
        load_settings(path)
    assert str(path) in str(error.value)
    assert error_field in str(error.value)


@pytest.mark.parametrize("field", ["before_event_s", "after_event_s"])
@pytest.mark.parametrize("value", ["nan", "+nan", "-nan", "inf", "+inf", "-inf"])
def test_non_finite_seconds_rejected(tmp_path, field, value):
    old_value = "3.0" if field == "before_event_s" else "2.0"
    text = CONFIG_TEXT.replace(f"{field} = {old_value}", f"{field} = {value}")
    with pytest.raises(SettingsError, match=rf"playback\.{field}"):
        load_settings(write_config(tmp_path, text))


def test_values_have_no_sample_count_or_feature_dimension_limit(tmp_path):
    text = CONFIG_TEXT.replace("min_common_parts = 2", "min_common_parts = 17").replace(
        "max_templates_per_player = 6", "max_templates_per_player = 10000"
    )
    text = text.replace("before_event_s = 3.0", "before_event_s = 0")
    settings = load_settings(write_config(tmp_path, text))
    assert settings.matching.min_common_parts == 17
    assert settings.matching.max_templates_per_player == 10000
    assert settings.playback.before_event_s == 0.0


@pytest.mark.parametrize("port", [1, 65535])
def test_valid_port_boundaries(tmp_path, port):
    settings = load_settings(write_config(tmp_path, CONFIG_TEXT.replace("8000", str(port))))
    assert settings.server.port == port


def test_syntax_error_is_clear(tmp_path):
    path = write_config(tmp_path, "[paths\n")
    with pytest.raises(SettingsError, match="Cannot read configuration") as error:
        load_settings(path)
    assert str(path) in str(error.value)
    assert error.value.__cause__ is not None


def test_missing_file_is_clear(tmp_path):
    path = tmp_path / "missing.toml"
    with pytest.raises(SettingsError, match="Cannot read configuration") as error:
        load_settings(path)
    assert str(path) in str(error.value)
    assert not path.exists()


def test_invalid_encoding_is_clear(tmp_path):
    path = tmp_path / "invalid.toml"
    path.write_bytes(b"\xff")
    with pytest.raises(SettingsError, match="Cannot read configuration"):
        load_settings(path)
