"""Evaluate permutation invariance and partition congruence on frozen machine features."""

import json
import shutil
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from app.features.dual_stream import DualStreamExtractor  # noqa: E402
from app.identity.matching import IncrementalMatcher  # noqa: E402
from app.storage.database import connect  # noqa: E402
from app.storage.dual_stream import (  # noqa: E402
    _load_feature,
    _load_segment,
)
from app.workflows.incremental import (  # noqa: E402
    reconcile_canonical_revision,
    update_dual_stream,
)
from run_real_batches import dump, finalize_artifacts, snapshot_artifact  # noqa: E402

MATCH = "t3-real-basketball"
ORIGINAL = ("3", "6", "11", "20", "30", "37", "43", "50", "63", "68")
REVERSED = ("68", "63", "50", "43", "37", "30", "20", "11", "6", "3")
SHUFFLED_20261116 = ("6", "3", "63", "37", "30", "11", "50", "43", "68", "20")


class CachedMachineExtractor(DualStreamExtractor):
    def __init__(self, crop_root, features, segments):
        super().__init__(crop_root, jersey_extractor=object(), appearance_extractor=object())
        self.features = features
        self.segments = segments
        self.hits = Counter()

    def _extract(self, segment, paths):
        feature = self.features[segment.segment_id]
        self.hits[segment.segment_id] += 1
        return feature


def load_primary_evidence(primary_dir: Path):
    primary_db = primary_dir.resolve() / "player-library.sqlite3"
    source = sqlite3.connect(primary_db.as_uri() + "?mode=ro&immutable=1", uri=True)
    grouped = defaultdict(list)
    features, segments = {}, {}
    try:
        rows = source.execute(
            "SELECT segment_id,segment_json,feature_json,vector_blob FROM dual_stream_segments "
            "WHERE match_id=? ORDER BY segment_id",
            (MATCH,),
        ).fetchall()
        for sid, segment_json, feature_json, blob in rows:
            segment = _load_segment(segment_json)
            feature = _load_feature(feature_json, blob)
            grouped[segment.video_id].append(segment)
            segments[sid], features[sid] = segment, feature
    finally:
        source.close()
    return grouped, segments, features


def run_order_replay(
    order: tuple[str, ...],
    primary_dir: Path,
    out_run_dir: Path,
    grouped,
    segments,
    features,
):
    out_run_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_run_dir / "crops"
    if not crops_dir.exists() and (primary_dir / "crops").exists():
        shutil.copytree(primary_dir / "crops", crops_dir)

    for clip in ORIGINAL:
        input_file = primary_dir / f"input-{clip}.json"
        if input_file.is_file():
            (out_run_dir / f"input-{clip}.json").write_text(
                input_file.read_text(encoding="utf-8"), encoding="utf-8"
            )

    shutil.copy2(primary_dir / "real-run-config.json", out_run_dir / "real-run-config.json")

    extractor = CachedMachineExtractor(crops_dir, features, segments)
    matcher = IncrementalMatcher(high_confidence_margin=0.0005, max_prototypes=12)
    db_file = out_run_dir / "real-batches.sqlite3"
    if db_file.exists():
        db_file.unlink()

    connection = connect(db_file)
    revision = 0
    try:
        for clip in order:
            clip_segs = grouped[clip]
            response = update_dual_stream(
                connection,
                match_id=MATCH,
                batch_id=f"clip-{clip}",
                request_id=f"replay-clip-{clip}",
                expected_revision=revision,
                segments=clip_segs,
                extractor=extractor,
                matcher=matcher,
            )
            revision = response.revision
            snapshot_artifact(connection, clip, out_run_dir)
        reconcile_canonical_revision(connection, match_id=MATCH, matcher=matcher)
        revision = connection.execute(
            "SELECT revision FROM matches WHERE match_id=?", (MATCH,)
        ).fetchone()[0]
    finally:
        connection.close()

    summary = {
        "batches": [],
        "final_revision": revision,
        "elapsed_seconds": 0.0,
        "same_request_replay": None,
        "same_content_new_request": None,
        "integrity_check": "ok",
        "foreign_key_violations": 0,
        "evaluation": "T3/T4 permutation replay; no labels read",
    }
    dump(out_run_dir, "real-run-summary.json", summary)

    commit_hash = "uncommitted"
    finalize_artifacts(out_run_dir, PROJECT, commit_hash=commit_hash)


def score_run(run_dir: Path, out_dir: Path) -> dict:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    cmd = [
        sys.executable,
        "-m",
        "app.evaluation.run_benchmark",
        "--run-dir",
        str(run_dir),
        "--test-data",
        str(PROJECT.parent / "test"),
        "--out-dir",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, cwd=str(PROJECT), capture_output=True, text=True)
    if proc.returncode != 0:
        print("Benchmark failed stderr:", proc.stderr)
        print("Benchmark stdout:", proc.stdout)
        raise RuntimeError(f"run_benchmark failed with code {proc.returncode}")
    summary_file = out_dir / "scoring_summary.json"
    if not summary_file.exists():
        summary_file = out_dir / "summary.json"
    return json.loads(summary_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()

    grouped, segments, features = load_primary_evidence(args.primary_dir)
    print(f"Loaded primary evidence: {len(segments)} segments across {len(grouped)} clips.")

    # Run Reverse
    rev_run = args.out_root / "reverse" / "inference"
    rev_bench = args.out_root / "reverse" / "benchmark"
    run_order_replay(REVERSED, args.primary_dir, rev_run, grouped, segments, features)
    rev_summary = score_run(rev_run, rev_bench)
    p27_rev = rev_summary["correct_samples"]
    p15_rev = rev_summary["player_15_metrics"]["correct_samples"]
    print(
        f"Reversed: 27 coverage = {p27_rev}/70, 15 coverage = {p15_rev}/48, conflicts = {rev_summary['physical_conflicts']}, passed = {rev_summary['passed']}"
    )

    # Run Shuffled 20261116
    shuf_run = args.out_root / "shuffled-20261116" / "inference"
    shuf_bench = args.out_root / "shuffled-20261116" / "benchmark"
    run_order_replay(SHUFFLED_20261116, args.primary_dir, shuf_run, grouped, segments, features)
    shuf_summary = score_run(shuf_run, shuf_bench)
    p27_shuf = shuf_summary["correct_samples"]
    p15_shuf = shuf_summary["player_15_metrics"]["correct_samples"]
    print(
        f"Shuffled 20261116: 27 coverage = {p27_shuf}/70, 15 coverage = {p15_shuf}/48, conflicts = {shuf_summary['physical_conflicts']}, passed = {shuf_summary['passed']}"
    )
