"""Finalize a stopped run after adding test evidence and handoff; never re-infer."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    result = json.loads((root / "verification.json").read_text())
    assert all(
        result[k]
        for k in ("history_exact_equal", "detections_exact_equal", "frame_ledger_exact_equal")
    )
    manifest = json.loads((root / "manifest.json").read_text())
    for source in manifest["sources"]:
        assert sha(Path(source["path"])) == source["sha256"]
    for name, digest in manifest["implementation"].items():
        snapshot = root / "implementation" / name
        if not snapshot.exists():
            source_file = Path(__file__).resolve().parents[2] / name
            assert sha(source_file) == digest, name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_file, snapshot)
        assert sha(snapshot) == digest, name
    summary = json.loads((root / "B/summary.json").read_text())
    assert summary["queues_drained"] and all(summary["ended"])
    assert summary["real_frames"] == [s["decoded_frames"] for s in manifest["sources"]]
    hashes = {
        str(p.relative_to(root)): sha(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "artifact-hashes.json"
    }
    (root / "artifact-hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(f"Frozen {len(hashes)} artifacts, {sum(summary['real_frames'])} real frames")


if __name__ == "__main__":
    main()
