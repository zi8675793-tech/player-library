"""Read-only input validation; business commands will use workflows."""

import argparse
import sys
from collections.abc import Sequence

from app.adapters.sample_test import SampleTestError, load_sample_test
from app.settings import SettingsError, load_settings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate external evidence without writing state")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="read and validate a configured input batch")
    validate.add_argument("--config", required=True, help="TOML configuration file")
    validate.add_argument("--batch-id", required=True, help="caller-provided handoff identifier")
    validate.add_argument(
        "--clip-id", action="append", dest="clip_ids", help="select a clip; may be repeated"
    )
    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.config)
        if settings.input.adapter != "sample_test":
            raise SampleTestError(
                f"validate does not yet support adapter {settings.input.adapter!r}"
            )
        batch = load_sample_test(settings, batch_id=args.batch_id, clip_ids=args.clip_ids)
    except (SettingsError, SampleTestError) as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1

    evidence = batch.identity_evidence
    print(f"match={batch.match_id} batch={batch.batch_id}")
    print(
        f"clips={len(batch.clips or [])} "
        f"segments={len(evidence.segments) if evidence else 0} "
        f"samples={len(evidence.samples) if evidence else 0} "
        f"events={len(batch.events or [])} "
        f"feature_spaces={len(evidence.feature_spaces) if evidence else 0}"
    )
    print(f"warnings={len(batch.warnings)}")
    for warning in batch.warnings:
        location = f" clip={warning.clip_id}" if warning.clip_id is not None else ""
        print(f"warning [{warning.code}]{location}: {warning.message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
