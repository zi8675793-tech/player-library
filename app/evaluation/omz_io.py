"""Read-only freeze validation and bounded-memory readers for OMZ evaluation."""

import hashlib
import json
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path

from app.contracts.tracking import verify_tracking_bundle


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def array_items(path, key):
    """Stream one top-level array without materializing large audit/event arrays.

    Token decoding, rather than a regex search, ensures strings cannot impersonate keys.
    """
    decoder = json.JSONDecoder()
    with Path(path).open() as stream:
        buffer = ""

        def trim():
            nonlocal buffer
            buffer = buffer.lstrip()
            while not buffer:
                block = stream.read(65536)
                if not block:
                    raise ValueError("Unexpected JSON EOF")
                buffer = block.lstrip()

        def token(char):
            nonlocal buffer
            trim()
            if not buffer.startswith(char):
                raise ValueError(f"Expected {char}")
            buffer = buffer[len(char) :]

        def value():
            nonlocal buffer
            trim()
            while True:
                try:
                    item, end = decoder.raw_decode(buffer)
                    buffer = buffer[end:]
                    return item
                except json.JSONDecodeError:
                    block = stream.read(65536)
                    if not block:
                        raise
                    buffer += block

        token("{")
        while True:
            name = value()
            token(":")
            if name == key:
                token("[")
                trim()
                while not buffer.startswith("]"):
                    yield value()
                    trim()
                    if buffer.startswith("]"):
                        break
                    token(",")
                return
            value()
            token(",")


def verify_t1(run, runtime_snapshot=None):
    run = Path(run).resolve()
    manifest = verify_tracking_bundle(run)
    hashes = read_json(run / "artifact-hashes.json")
    required = {
        "manifest.json",
        "B/final.json",
        "B/prefix.json",
        "B/frames.jsonl",
        "A/history.json",
        "A/detections.json",
        "official-source.tar",
    }
    if required - hashes.keys():
        raise ValueError("Incomplete T1 freeze")
    checks = []

    def check(path, expected, category):
        actual = sha(path)
        if actual != expected:
            raise ValueError(f"Changed {category}: {path}")
        checks.append({"path": str(path), "sha256": actual, "category": category})

    for source in manifest["sources"]:
        check(Path(source["path"]), source["sha256"], "input_video")
    for name, digest in manifest["implementation"].items():
        check(run / "implementation" / name, digest, "inference_implementation_snapshot")
    with tarfile.open(run / "official-source.tar") as archive:
        members = {m.name: m for m in archive.getmembers() if m.isfile() or m.issym() or m.islnk()}
        for name, digest in manifest["engine_files"].items():
            if name.startswith("omz/"):
                if name[4:] not in members:
                    if "/__pycache__/" not in name or not name.endswith(".pyc"):
                        raise ValueError(f"Unarchived official file: {name}")
                    base = Path(runtime_snapshot) if runtime_snapshot else run.parent.parent
                    check(base / name, digest, "runtime_bytecode")
                    continue
                content = archive.extractfile(members[name[4:]]).read()
                if hashlib.sha256(content).hexdigest() != digest:
                    raise ValueError(f"Changed official source/config: {name}")
            else:
                check(run / "pinned-models" / name.removeprefix("models/"), digest, "model")
    return manifest, {
        "verified_at": datetime.now(UTC).isoformat(),
        "run": str(run),
        "artifact_index_sha256": sha(run / "artifact-hashes.json"),
        "artifact_count": len(hashes),
        "official_source_files": len(members),
        "engine_commit": manifest["commit"],
        "checks": checks,
        "config_sha256": manifest["engine_files"][
            "omz/demos/multi_camera_multi_target_tracking_demo/python/configs/person.py"
        ],
        "passed": True,
    }


def snapshot_workspace(project, out):
    out.mkdir(parents=True)
    for name, args in [
        ("status.txt", ["status", "--short"]),
        ("diff.patch", ["diff", "--binary"]),
        ("HEAD.txt", ["rev-parse", "HEAD"]),
    ]:
        (out / name).write_bytes(subprocess.check_output(["git", *args], cwd=project))
    files = (
        subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=project)
        .decode()
        .split("\0")
    )
    write_json(
        out / "files.json",
        {p: sha(project / p) for p in sorted(set(files)) if p and (project / p).is_file()},
    )
