"""Run prepare and scoring in separate processes, relaying the external hash receipt."""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.evaluation.golden_freeze import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, default=Path("../test"))
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--target-root", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    annotations = args.annotations or args.test_data / "annotations.csv"
    if args.annotations is None and not annotations.is_file():
        annotations = args.test_data.parent / "annotations.csv"
    out = args.out_dir or Path("var/benchmarks") / datetime.now(UTC).strftime(
        "run-%Y%m%dT%H%M%S-%f"
    )
    out.mkdir(parents=True, exist_ok=False)
    prepare = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.evaluation.prepare_benchmark",
            "--run-dir",
            str(args.run_dir),
            "--test-data",
            str(args.test_data),
            "--out-dir",
            str(out / "geometry"),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    receipt = json.loads(prepare.stdout)
    write_json(out / "prepare-receipt.json", receipt)
    command = [
        sys.executable,
        "-m",
        "app.evaluation.score_benchmark",
        "--geometry-dir",
        str(out / "geometry"),
        "--annotations",
        str(annotations),
        "--out-dir",
        str(out),
        "--expected-freeze-sha256",
        receipt["freeze_sha256"],
    ]
    command.extend(["--target-root", str(args.target_root or args.test_data / "target")])
    score = subprocess.run(command, text=True, capture_output=True, check=True)
    print(score.stdout, end="")


if __name__ == "__main__":
    main()
