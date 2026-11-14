"""Hash gates shared by the independent T4 processes; never imports inference."""

import contextvars
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

_blind = contextvars.ContextVar("t4_blind", default=False)


def _audit(event, args):
    if event != "open" or not _blind.get() or isinstance(args[0], int):
        return
    path = Path(os.fsdecode(args[0])).absolute()
    for candidate in (path, path.resolve()):
        if candidate.name == "annotations.csv" or "target" in candidate.parts:
            raise PermissionError(f"Prepare firewall denied secret input: {candidate}")


sys.addaudithook(_audit)


@contextmanager
def blind_inputs():
    token = _blind.set(True)
    try:
        yield
    finally:
        _blind.reset(token)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def contained(root, name):
    root = Path(root).resolve()
    path = root / name
    if Path(name).is_absolute() or not path.resolve().is_relative_to(root):
        raise ValueError(f"Artifact path escapes root: {name}")
    return path


def verify_run(run_dir, expected_manifest=None):
    run_dir = Path(run_dir).resolve()
    index = run_dir / "artifact-hashes.json"
    if expected_manifest is not None and sha(index) != expected_manifest:
        raise ValueError("T3 artifact manifest changed")
    hashes = read_json(index)
    required = {
        "predictions.json",
        "summary.json",
        "config.json",
        "input-manifest.json",
        "player-library.sqlite3",
        "code-hashes.json",
    }
    if not required <= hashes.keys():
        raise ValueError("T3 freeze missing required artifacts")
    for name, digest in hashes.items():
        if sha(contained(run_dir, name)) != digest:
            raise ValueError(f"T3 frozen artifact changed: {name}")
    for name, digest in read_json(run_dir / "code-hashes.json").items():
        if sha(contained(run_dir / "source-snapshot", name)) != digest:
            raise ValueError(f"T3 source snapshot changed: {name}")
    return hashes


def verify_prepared(geometry_dir, expected_freeze_sha256):
    """External prepare receipt pins the manifest itself, not only its children."""
    geometry_dir = Path(geometry_dir).resolve()
    index = geometry_dir / "geometry-freeze.json"
    if not expected_freeze_sha256 or sha(index) != expected_freeze_sha256:
        raise ValueError("Prepared freeze fingerprint mismatch")
    frozen = read_json(index)
    if frozen.get("schema") != "t4-golden-freeze-v1":
        raise ValueError("Unsupported prepared freeze schema")
    required = {
        "rules.json",
        "sources.json",
        "machine-samples.json",
        "correspondence.jsonl",
        "prediction-members.jsonl",
        "physical-conflicts.json",
        "segment-diagnostics.json",
        "database-verification.json",
        "artifact-hashes.json",
    }
    if not required <= frozen["files"].keys():
        raise ValueError("Incomplete geometry freeze")
    for name, digest in frozen["files"].items():
        if sha(contained(geometry_dir, name)) != digest:
            raise ValueError(f"Frozen geometry changed: {name}")
    ledger_hashes = read_json(geometry_dir / "artifact-hashes.json")
    if ledger_hashes != {k: v for k, v in frozen["files"].items() if k != "artifact-hashes.json"}:
        raise ValueError("Geometry artifact index is incomplete")
    for path, digest in frozen["code_files"].items():
        if sha(path) != digest:
            raise ValueError(f"Evaluation implementation changed: {path}")
    for record in frozen["machine_inputs"]:
        if sha(record["path"]) != record["sha256"]:
            raise ValueError(f"Machine input changed: {record['path']}")
    verify_run(frozen["run_dir"], frozen["run_manifest_sha256"])
    return frozen
