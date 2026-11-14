"""Independent T4 answer-reading process, entered only with a trusted freeze hash."""

import argparse
import json
from pathlib import Path

from app.evaluation.golden_score import score_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-dir", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--target-root", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--expected-freeze-sha256", required=True)
    args = parser.parse_args()
    result = score_benchmark(
        args.geometry_dir,
        args.annotations,
        args.out_dir,
        target_root=args.target_root,
        expected_freeze_sha256=args.expected_freeze_sha256,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
