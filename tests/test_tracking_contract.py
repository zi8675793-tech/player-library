"""File freeze verification, without reading any evaluation data."""

import hashlib
import json

import pytest

from app.contracts.tracking import verify_tracking_bundle


def test_frozen_bundle_rejects_tampering(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": "omz-tracking-v1"}))
    hashes = {"manifest.json": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    (tmp_path / "artifact-hashes.json").write_text(json.dumps(hashes))
    assert verify_tracking_bundle(tmp_path)["schema_version"] == "omz-tracking-v1"
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="Frozen artifact changed"):
        verify_tracking_bundle(tmp_path)


def test_frozen_bundle_rejects_path_escape(tmp_path):
    (tmp_path / "artifact-hashes.json").write_text(json.dumps({"../outside": "bad"}))
    with pytest.raises(ValueError, match="escapes"):
        verify_tracking_bundle(tmp_path)


def test_unknown_tracking_schema(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": "other"}))
    (tmp_path / "artifact-hashes.json").write_text(
        json.dumps({"manifest.json": hashlib.sha256(manifest.read_bytes()).hexdigest()})
    )
    with pytest.raises(ValueError, match="Unsupported"):
        verify_tracking_bundle(tmp_path)
