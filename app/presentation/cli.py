"""Input validation and a fresh-state machine baseline through shared workflows."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from app.adapters._reading import InputReadError
from app.settings import SettingsError, load_settings

# Backward-compatible validation error name, without importing an answer reader
# into the machine-baseline execution path.
SampleTestError = InputReadError


def load_sample_test(*args, **kwargs):
    from app.adapters.sample_test import load_sample_test as adapt

    return adapt(*args, **kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate evidence or run a fresh machine baseline"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="read and validate a configured input batch")
    validate.add_argument("--config", required=True, help="TOML configuration file")
    validate.add_argument("--batch-id", required=True, help="caller-provided handoff identifier")
    validate.add_argument(
        "--clip-id", action="append", dest="clip_ids", help="select a clip; may be repeated"
    )
    baseline = commands.add_parser("baseline", help="freeze a machine-only, empty-state baseline")
    baseline.add_argument("--config", required=True, help="machine-input TOML configuration")
    baseline.add_argument("--run-id", required=True, help="new directory name under runtime_root")
    baseline.add_argument("--batch-id", default="baseline", help="handoff identifier")
    baseline.add_argument(
        "--segment-tracks",
        action="store_true",
        help="propose machine-evidence identity segments before empty-state import",
    )
    baseline.add_argument(
        "--discover-identities",
        action="store_true",
        help="discover experimental anonymous identities; includes machine segmentation",
    )
    baseline.add_argument(
        "--incremental-batches",
        action="store_true",
        help="partition machine clips into sequential batches and incrementally update identities",
    )
    baseline.add_argument(
        "--batch-size",
        type=int,
        default=3,
        help="number of clips per batch when using incremental batches (default: 3)",
    )
    baseline.add_argument(
        "--reverse-batches",
        action="store_true",
        help="reverse batch order to evaluate order sensitivity",
    )
    baseline.add_argument(
        "--chunk-order",
        type=str,
        default=None,
        help="comma-separated chunk indices e.g. 2,0,3,1",
    )
    baseline.add_argument(
        "--fixed-segmentation",
        type=str,
        default=None,
        help="path to a fixed segmentation.json report (reuses baseline segments without re-segmenting)",
    )
    baseline.add_argument(
        "--segmentation-version",
        choices=["track-consistency-v1", "track-consistency-v2", "track-consistency-v3"],
        default=None,
        help="override segmentation policy version (e.g. track-consistency-v3)",
    )
    baseline.add_argument(
        "--runtime-root",
        type=str,
        default=None,
        help="custom directory under var/ for output artifacts",
    )
    baseline.add_argument(
        "--calibration-profile",
        type=str,
        default=None,
        help="path to a frozen calibration-profile.json file",
    )

    diagnose = commands.add_parser(
        "diagnose-calibration",
        help="extract and diagnose distance calibration candidate evidence",
    )
    diagnose.add_argument("--config", required=True, help="machine-input TOML configuration")
    diagnose.add_argument(
        "--output-profile", required=True, help="output path for calibration-profile.json"
    )
    diagnose.add_argument(
        "--output-diagnostic",
        default=None,
        help="optional output path for calibration-diagnostic.json",
    )
    diagnose.add_argument(
        "--feature-space-id", default=None, help="explicit target feature space ID"
    )
    args = parser.parse_args(argv)

    if args.command == "diagnose-calibration":
        from app.adapters.machine import load_machine_input
        from app.identity.calibration import (
            build_distance_calibration_profile,
            save_calibration_profile,
            save_diagnostic_report,
        )

        try:
            settings = load_settings(args.config)
            raw_b, vals = load_machine_input(settings, batch_id="distance-calibration")
            profile, report = build_distance_calibration_profile(
                raw_b, vals, feature_space_id=args.feature_space_id
            )
            out_prof = save_calibration_profile(profile, Path(args.output_profile))
            print(f"Calibration profile saved: {out_prof}")
            if args.output_diagnostic:
                out_diag = save_diagnostic_report(report, Path(args.output_diagnostic))
                print(f"Diagnostic report saved: {out_diag}")
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"Calibration diagnosis failed: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.command == "baseline":
        from app.presentation.baseline import load_baseline_settings, run_baseline

        try:
            chunk_order = (
                [int(x.strip()) for x in args.chunk_order.split(",") if x.strip()]
                if args.chunk_order
                else None
            )
            output = run_baseline(
                load_baseline_settings(args.config),
                run_id=args.run_id,
                batch_id=args.batch_id,
                segment_tracks=args.segment_tracks,
                discover_identities=args.discover_identities,
                incremental_batches=args.incremental_batches,
                batch_size=args.batch_size,
                reverse_batches=args.reverse_batches,
                chunk_order=chunk_order,
                fixed_segmentation=args.fixed_segmentation,
                segmentation_version=args.segmentation_version,
                runtime_root=args.runtime_root,
                calibration_profile=args.calibration_profile,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"Baseline failed: {exc}", file=sys.stderr)
            return 1
        print(f"Baseline frozen: {output}")
        print("Inspect summary.json for evidence coverage, pending results and limitations.")
        return 0

    try:
        settings = load_settings(args.config)
        if settings.input.adapter == "machine":
            from app.adapters.machine import load_machine

            batch = load_machine(settings, batch_id=args.batch_id, clip_ids=args.clip_ids)
        elif settings.input.adapter == "sample_test":
            batch = load_sample_test(settings, batch_id=args.batch_id, clip_ids=args.clip_ids)
        else:
            raise SampleTestError(
                f"validate does not yet support adapter {settings.input.adapter!r}"
            )
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
