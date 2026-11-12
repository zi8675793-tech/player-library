"""Extract fixed evaluation units from machine handoff evidence.

Outputs a standard EvaluationManifest JSON file providing an immutable denominator
for sample coverage and duration coverage across any segmentation algorithm.
"""

import argparse
import json
from pathlib import Path

from app.adapters.machine import load_machine
from app.contracts.evidence import EvidenceBatch
from app.evaluation.units import EvaluationManifest, extract_evaluation_units
from app.settings import load_settings


def extract_manifest_from_batch(batch: EvidenceBatch) -> EvaluationManifest:
    """Extract an EvaluationManifest from an EvidenceBatch."""
    return extract_evaluation_units(batch)


def extract_manifest_from_raw_json(raw_json_path: str | Path) -> EvaluationManifest:
    """Extract an EvaluationManifest from a raw-evidence.json file."""
    with open(raw_json_path, encoding="utf-8") as f:
        data = json.load(f)
    batch = EvidenceBatch.model_validate(data)
    return extract_evaluation_units(batch)


def main():
    parser = argparse.ArgumentParser(
        description="Extract fixed machine evaluation units into an immutable manifest."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config",
        type=str,
        help="Path to TOML configuration file to load machine evidence via adapter.",
    )
    group.add_argument(
        "--raw-evidence",
        type=str,
        help="Path to an existing raw-evidence.json file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path where the fixed evaluation units JSON will be written.",
    )
    parser.add_argument(
        "--batch-id",
        type=str,
        default="fixed-evaluation-units",
        help="Batch ID to use when reading via config.",
    )

    args = parser.parse_args()

    if args.raw_evidence:
        manifest = extract_manifest_from_raw_json(args.raw_evidence)
    else:
        settings = load_settings(args.config)
        batch = load_machine(settings, batch_id=args.batch_id)
        manifest = extract_manifest_from_batch(batch)

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))
        f.write("\n")

    print(
        f"Extracted {manifest.total_samples} samples and {manifest.total_observations} observations"
    )
    print(
        f"Total track duration: {manifest.total_duration_s:.3f}s across {manifest.track_count} tracks"
    )
    print(f"Saved fixed evaluation units to: {out_path}")


if __name__ == "__main__":
    main()
