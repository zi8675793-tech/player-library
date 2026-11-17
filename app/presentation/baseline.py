"""Fresh-state machine-evidence baseline. No answer or legacy adapter dependency."""

import hashlib
import importlib.metadata
import json
import platform
import re
import sqlite3
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from app.adapters._reading import fingerprint
from app.adapters.machine import load_machine_input
from app.contracts.api import UpdateRequest
from app.contracts.discovery import DiscoveryReport
from app.evaluation.funnel import build_coverage_funnel, build_loss_breakdown
from app.evaluation.units import extract_evaluation_units, project_units_to_segments
from app.settings import Settings, SettingsError, load_settings
from app.storage.database import connect
from app.storage.matching_diagnostics import read_matching_diagnostics
from app.workflows.read import get_discovery_reports, get_identity_state, get_review_queue
from app.workflows.update import update_library

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_VAR = PROJECT_ROOT / "var"
BASELINE_VERSION = "machine-evidence-v1"


def load_baseline_settings(config_path: str | Path) -> Settings:
    """Only an explicitly selected TOML configuration may precede machine reads."""
    path = Path(config_path).absolute()
    canonical = path.resolve()
    if (
        path.suffix != ".toml"
        or canonical.suffix != ".toml"
        or {"target", "repository-review"}.intersection(canonical.parts)
        or any(parent.is_symlink() for parent in (path, *path.parents))
    ):
        raise SettingsError("baseline requires a regular TOML config outside forbidden paths")
    return load_settings(path)


def _json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation() -> dict:
    # Fixed code roots, never test data, review artifacts, old databases or eval scripts.
    paths = sorted((PROJECT_ROOT / "app").rglob("*.py"))
    paths += sorted((PROJECT_ROOT / "app/storage/migrations").glob("*.sql"))
    paths += [PROJECT_ROOT / "pyproject.toml", PROJECT_ROOT / "uv.lock"]
    hashes = {str(path.relative_to(PROJECT_ROOT)): _sha(path) for path in paths}
    return {
        "files": hashes,
        "fingerprint": fingerprint(hashes),
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
        "packages": {
            name: importlib.metadata.version(name) for name in ("numpy", "pydantic", "pillow")
        },
    }


def run_baseline(
    settings: Settings,
    *,
    run_id: str,
    batch_id: str = "baseline",
    segment_tracks: bool = False,
    discover_identities: bool = False,
    incremental_batches: bool = False,
    batch_size: int = 3,
    reverse_batches: bool = False,
    chunk_order: Sequence[int] | None = None,
    fixed_segmentation: str | Path | None = None,
    preloaded_raw: tuple | None = None,
    preloaded_batch_and_seg: tuple | None = None,
    runtime_root: Path | str | None = None,
    calibration_profile: Path | str | None = None,
    segmentation_version: str | None = None,
) -> Path:
    """Import into a new database, exercise idempotency and freeze reviewable outputs.

    No database input option exists. Reusing a run directory is an error. Exported
    predictions use the existing Segment and Assignment contracts without ID remapping.
    """
    if settings.input.adapter != "machine":
        raise ValueError("baseline requires input.adapter = 'machine'")
    if settings.matching.auto_assign_enabled:
        raise ValueError("this baseline requires automatic confirmation disabled")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError("run_id must be a single safe directory name")
    runtime = (
        Path(runtime_root).resolve()
        if runtime_root is not None
        else settings.paths.runtime_root.resolve()
    )
    if PROJECT_VAR.is_symlink() or not runtime.is_relative_to(PROJECT_VAR):
        raise ValueError("baseline runtime_root must be inside this project's var directory")
    output = runtime / run_id
    if output.exists() or output.is_symlink():
        raise ValueError(f"baseline run already exists; select a new run_id: {output}")

    bound_profile = None
    if calibration_profile is not None:
        from app.identity.calibration import load_calibration_profile
        from app.settings import bind_calibration_profile

        bound_profile = load_calibration_profile(calibration_profile)
        settings = bind_calibration_profile(settings, bound_profile)
    elif settings.paths.calibration_profile is not None:
        from app.identity.calibration import load_calibration_profile

        bound_profile = load_calibration_profile(settings.paths.calibration_profile)

    if bound_profile is not None:
        disc = settings.discovery
        for attr in ("intra_segment_near_distance", "inter_segment_tau", "tau"):
            if abs(getattr(disc, attr) - bound_profile.near_distance) > 1e-6:
                raise ValueError(
                    f"Effective discovery policy {attr}={getattr(disc, attr)} conflicts with "
                    f"bound calibration_profile.near_distance={bound_profile.near_distance}"
                )
        for attr in ("candidate_far_distance", "negative_evidence_distance", "far_distance"):
            if abs(getattr(disc, attr) - bound_profile.far_distance) > 1e-6:
                raise ValueError(
                    f"Effective discovery policy {attr}={getattr(disc, attr)} conflicts with "
                    f"bound calibration_profile.far_distance={bound_profile.far_distance}"
                )

    if segmentation_version is not None:
        from app.contracts.segmentation import SegmentationPolicy

        seg_data = settings.segmentation.model_dump(mode="python")
        seg_data["version"] = segmentation_version
        settings = settings.model_copy(
            update={"segmentation": SegmentationPolicy.model_validate(seg_data)}
        )

    if incremental_batches:
        from app.adapters._machine_reading import MachineInputReader
        from app.contracts.evidence import EvidenceBatch, IdentityEvidence
        from app.identity.consistency import segment_tracks as propose_segments
        from app.workflows.incremental import update_incremental

        reader = MachineInputReader(settings.paths.input_root)
        discovered = sorted(reader.discover_clips())
        chunks = [discovered[i : i + batch_size] for i in range(0, len(discovered), batch_size)]
        if reverse_batches:
            chunks = list(reversed(chunks))
        elif chunk_order is not None:
            chunks = [chunks[i] for i in chunk_order]

        runtime.mkdir(parents=True, exist_ok=True)
        output.mkdir(exist_ok=False)
        database = output / "player-library.sqlite3"
        database.touch(exist_ok=False)
        connection = connect(database)

        all_raw_batches = []
        all_derived_batches = []
        all_seg_reports = []
        all_values = {}
        batch_summaries = []
        current_rev = 0

        try:
            for idx, chunk in enumerate(chunks, 1):
                chunk_batch_id = f"{batch_id}-batch-{idx}"
                raw_b, vals = load_machine_input(settings, batch_id=chunk_batch_id, clip_ids=chunk)
                all_raw_batches.append(raw_b)
                all_values.update(vals)

                seg_res = propose_segments(raw_b, vals, policy=settings.segmentation)
                all_derived_batches.append(seg_res.batch)
                all_seg_reports.append(seg_res.report)

                first = update_incremental(
                    connection,
                    UpdateRequest(batch=seg_res.batch, expected_revision=current_rev),
                    segmentation_report=seg_res.report,
                    feature_values=vals,
                    settings=settings,
                )
                before_state = get_identity_state(connection, settings.match.match_id)

                # Verify replay idempotency for this batch
                replay = update_incremental(
                    connection,
                    UpdateRequest(
                        batch=seg_res.batch.model_copy(
                            update={"batch_id": f"{chunk_batch_id}-replay"}
                        ),
                        expected_revision=first.revision,
                    ),
                    segmentation_report=seg_res.report,
                    feature_values=vals,
                    settings=settings,
                )
                after_state = get_identity_state(connection, settings.match.match_id)
                if (
                    replay.data.get("changed", True)
                    or replay.revision != first.revision
                    or before_state != after_state
                ):
                    raise RuntimeError(
                        f"duplicate evidence in batch {idx} changed state or revision"
                    )

                current_rev = first.revision
                batch_summaries.append(
                    {
                        "batch_index": idx,
                        "batch_id": chunk_batch_id,
                        "clips": chunk,
                        "revision": current_rev,
                        "summary": first.data.get("summary", {}),
                    }
                )

            state = get_identity_state(connection, settings.match.match_id)
            stored_reports = get_discovery_reports(connection, settings.match.match_id)
            matching_audit = read_matching_diagnostics(connection, settings.match.match_id)
            queue = get_review_queue(connection, settings.match.match_id)
        except Exception as exc:
            _json(output / "failure.json", {"error": type(exc).__name__, "message": str(exc)})
            raise
        finally:
            connection.close()

        # Combine all raw and derived batches for unified reporting
        all_sources = sorted(
            {s.path: s for b in all_raw_batches for s in b.sources}.values(),
            key=lambda s: s.path,
        )
        all_clips = sorted(
            {c.clip_id: c for b in all_raw_batches for c in b.clips or []}.values(),
            key=lambda c: c.clip_id,
        )
        combined_raw = EvidenceBatch(
            match_id=settings.match.match_id,
            batch_id=f"{batch_id}-all-raw",
            producer=all_raw_batches[0].producer,
            sources=all_sources,
            clips=all_clips,
            identity_evidence=IdentityEvidence(
                tracks=[t for b in all_raw_batches for t in b.identity_evidence.tracks],
                observations=[o for b in all_raw_batches for o in b.identity_evidence.observations],
                segments=[s for b in all_raw_batches for s in b.identity_evidence.segments],
                samples=[s for b in all_raw_batches for s in b.identity_evidence.samples],
                feature_spaces=all_raw_batches[0].identity_evidence.feature_spaces,
                feature_files=[
                    ff for b in all_raw_batches for ff in b.identity_evidence.feature_files
                ],
            ),
        )
        combined_derived = EvidenceBatch(
            match_id=settings.match.match_id,
            batch_id=f"{batch_id}-all-derived",
            producer=all_derived_batches[0].producer,
            sources=all_sources,
            clips=all_clips,
            identity_evidence=IdentityEvidence(
                tracks=[t for b in all_derived_batches for t in b.identity_evidence.tracks],
                observations=[
                    o for b in all_derived_batches for o in b.identity_evidence.observations
                ],
                segments=[s for b in all_derived_batches for s in b.identity_evidence.segments],
                samples=[s for b in all_derived_batches for s in b.identity_evidence.samples],
                feature_spaces=all_derived_batches[0].identity_evidence.feature_spaces,
                feature_files=[
                    ff for b in all_derived_batches for ff in b.identity_evidence.feature_files
                ],
            ),
        )

        gallery = state.data
        segments = sorted(gallery.evidence.segments, key=lambda s: s.segment_id)
        assignments = sorted(gallery.assignments, key=lambda a: a.segment_id)
        input_fingerprint = fingerprint([s.model_dump(mode="json") for s in all_sources])

        config = settings.model_dump(mode="json")
        config["baseline_version"] = "machine-incremental-v1"
        config["initial_state"] = "empty_database_no_gallery_no_human_seeds"
        config["implementation"] = _implementation()
        config["incremental_batches_enabled"] = True
        config["batch_size"] = batch_size
        config["batch_count"] = len(chunks)
        config["reverse_batches"] = reverse_batches
        config["chunk_order"] = list(chunk_order) if chunk_order is not None else None
        config["batch_chunks"] = chunks

        # Consolidate all identity groups across all stored discovery reports
        all_identity_groups = []
        seen_group_ids = set()
        for report_item in stored_reports.data:
            rep = report_item["report"]
            mapping = report_item.get("player_ids_by_group", {})
            for g in rep.get("groups", []):
                gid = g["group_id"]
                if gid not in seen_group_ids:
                    seen_group_ids.add(gid)
                    all_identity_groups.append({**g, "player_id": mapping.get(gid)})

        active_segment_ids = {s.segment_id for s in segments if s.active}
        active_assignments = [a for a in assignments if a.segment_id in active_segment_ids]

        matched_existing_count = sum(
            1
            for a in active_assignments
            if any(e.kind == "automatic_incremental_match" for e in a.evidence)
        )
        created_new_identity_count = sum(
            1
            for a in active_assignments
            if any(
                e.kind in {"automatic_discovery_incremental", "automatic_discovery_experimental"}
                for e in a.evidence
            )
        )
        insufficient_evidence_count = sum(1 for a in active_assignments if a.status == "unassigned")
        assigned_segments_count = sum(1 for a in active_assignments if a.player_id is not None)
        unassigned_segments_count = sum(1 for a in active_assignments if a.player_id is None)

        deferred_reasons: dict[str, int] = Counter()
        for a in active_assignments:
            if a.status == "unassigned":
                reasons_found = False
                for ev in a.evidence:
                    if ev.kind == "insufficient_evidence" and ev.description:
                        for reason in ev.description.split("; "):
                            clean_r = reason.strip()
                            if clean_r:
                                deferred_reasons[clean_r] += 1
                                reasons_found = True
                if not reasons_found:
                    deferred_reasons[a.unassigned_reason or "unknown"] += 1

        all_unknown_segments = [
            {
                "segment_id": a.segment_id,
                "reasons": [
                    clean_r
                    for ev in a.evidence
                    if ev.kind == "insufficient_evidence" and ev.description
                    for clean_r in (r.strip() for r in ev.description.split("; "))
                    if clean_r
                ]
                or [a.unassigned_reason or "unknown"],
            }
            for a in active_assignments
            if a.player_id is None
        ]

        supported_segment_ids = {
            prop.segment.segment_id
            for report in all_seg_reports
            for t in report.tracks
            for prop in t.proposals
            if prop.status == "supported"
        }
        total_supported_count = len(supported_segment_ids)
        assigned_supported_count = sum(
            1
            for a in active_assignments
            if a.segment_id in supported_segment_ids and a.player_id is not None
        )
        total_active_segments = len(active_segment_ids)

        supported_segment_coverage = (
            assigned_supported_count / total_supported_count if total_supported_count > 0 else 0.0
        )
        all_segment_coverage = (
            assigned_segments_count / total_active_segments if total_active_segments > 0 else 0.0
        )

        predictions = {
            "schema_version": "baseline-predictions-v1",
            "match_id": settings.match.match_id,
            "revision": state.revision,
            "segmentation": "machine_consistency_proposals_not_identity_verified",
            "identity_status": "experimental",
            "calibration": "unvalidated",
            "segments": [s.model_dump(mode="json") for s in segments],
            "assignments": [a.model_dump(mode="json") for a in assignments],
            "players": [p.model_dump(mode="json") for p in gallery.players],
            "templates": [t.model_dump(mode="json") for t in gallery.templates],
            "identity_groups": all_identity_groups,
            "unknown_segments": all_unknown_segments,
            "batches": batch_summaries,
        }
        if stored_reports.data:
            predictions["discovery_report_ids"] = [
                r["report"]["report_id"] for r in stored_reports.data
            ]
        if bound_profile is not None:
            predictions["distance_calibration_fingerprint"] = bound_profile.fingerprint
            predictions["distance_calibration_id"] = bound_profile.profile_id

        summary = {
            "baseline_version": "machine-incremental-v1",
            "match_id": settings.match.match_id,
            "revision": state.revision,
            "input_fingerprint": input_fingerprint,
            "batch_count": len(chunks),
            "batches": batch_summaries,
            "clips": len(all_clips),
            "tracks": len(combined_raw.identity_evidence.tracks),
            "observations": len(combined_raw.identity_evidence.observations),
            "provisional_segments": len(segments),
            "active_segments": total_active_segments,
            "samples": len(combined_raw.identity_evidence.samples),
            "feature_rows": len(all_values),
            "players": len(gallery.players),
            "active_players": len(gallery.players),
            "templates": len(gallery.templates),
            "active_templates": sum(t.active for t in gallery.templates),
            "assigned_segments": assigned_segments_count,
            "unassigned_segments": unassigned_segments_count,
            "matched_existing": matched_existing_count,
            "created_new_identity": created_new_identity_count,
            "insufficient_evidence": insufficient_evidence_count,
            "deferred_reasons": dict(deferred_reasons),
            "supported_segment_coverage": supported_segment_coverage,
            "all_segment_coverage": all_segment_coverage,
            "coverage_note": (
                "Coverage metrics report the fraction of segments assigned to identities; "
                "they must not be conflated with identity recognition accuracy."
            ),
            "assignment_statuses": dict(Counter(a.status for a in assignments)),
            "assignment_methods": dict(Counter(a.method for a in assignments)),
            "human_interventions": 0,
            "identity_recognition_evaluated": False,
            "identity_status": "experimental",
            "calibration": "unvalidated",
            "limitation": (
                "Open-set incremental machine discovery without independent calibration; "
                "automatic templates have no human approval."
            ),
        }

        combined_seg_report = None
        if all_seg_reports:
            from app.contracts.segmentation import SegmentationReport

            combined_tracks = [t for r in all_seg_reports for t in r.tracks]
            combined_duplicate_candidates = [
                d for r in all_seg_reports for d in r.duplicate_candidates
            ]
            combined_summary = {
                "tracks": len(combined_tracks),
                "proposals": sum(len(t.proposals) for t in combined_tracks),
                "supported_proposals": sum(
                    1 for t in combined_tracks for p in t.proposals if p.status == "supported"
                ),
                "unknown_proposals": sum(
                    1 for t in combined_tracks for p in t.proposals if p.status == "unknown"
                ),
                "findings": sum(len(t.findings) for t in combined_tracks),
                "exclusions": sum(len(t.exclusions) for t in combined_tracks),
                "risks": sum(len(t.risks) for t in combined_tracks),
            }
            combined_seg_report = SegmentationReport(
                schema_version="track-consistency-report-v1",
                policy=all_seg_reports[0].policy,
                policy_fingerprint=all_seg_reports[0].policy_fingerprint,
                input_fingerprint=input_fingerprint,
                derived_evidence_fingerprint=all_seg_reports[-1].derived_evidence_fingerprint,
                tracks=combined_tracks,
                duplicate_candidates=combined_duplicate_candidates,
                summary=combined_summary,
            )

        eval_manifest = extract_evaluation_units(combined_raw)
        eval_projected = project_units_to_segments(eval_manifest.units, segments, assignments)
        cov_funnel = build_coverage_funnel(
            eval_manifest,
            eval_projected,
            combined_raw,
            segments=segments,
            assignments=assignments,
            segmentation_report=combined_seg_report,
            discovery_report=None,
            templates=gallery.templates,
        )
        loss_breakdown = build_loss_breakdown(
            eval_manifest,
            eval_projected,
            combined_raw,
            segments=segments,
            assignments=assignments,
            batch_summaries=batch_summaries,
        )
        summary["coverage_funnel"] = cov_funnel.model_dump(mode="json")
        summary["loss_summary"] = {
            "unassigned_samples": loss_breakdown.total_unassigned_samples,
            "unassigned_observations": loss_breakdown.total_unassigned_observations,
            "unassigned_duration_s": loss_breakdown.total_unassigned_duration_s,
            "unassigned_segments": loss_breakdown.total_unassigned_segments,
        }
        if bound_profile is not None:
            summary["distance_calibration"] = {
                "profile_id": bound_profile.profile_id,
                "feature_space_id": bound_profile.feature_space_id,
                "calibration_version": bound_profile.calibration_version,
                "status": bound_profile.status,
                "fingerprint": bound_profile.fingerprint,
                "near_distance": bound_profile.near_distance,
                "far_distance": bound_profile.far_distance,
            }

        _json(output / "raw-evidence.json", combined_raw.model_dump(mode="json"))
        _json(output / "evidence.json", combined_derived.model_dump(mode="json"))
        _json(output / "predictions.json", predictions)
        _json(
            output / "discovery.json",
            {
                "reports": [r["report"] for r in stored_reports.data],
                "player_ids_by_group": {
                    k: v for r in stored_reports.data for k, v in r["player_ids_by_group"].items()
                },
            },
        )
        _json(
            output / "funnel.json",
            {
                "coverage_funnel": cov_funnel.model_dump(mode="json"),
                "loss_breakdown": loss_breakdown.model_dump(mode="json"),
            },
        )
        _json(
            output / "diagnostics.json",
            {
                "batches": batch_summaries,
                "review_queue": queue.model_dump(mode="json"),
                "matching_attempts": matching_audit,
            },
        )
        _json(output / "config.json", config)
        _json(
            output / "input-manifest.json",
            {
                "schema_version": "machine-input-manifest-v1",
                "input_root": str(settings.paths.input_root),
                "fingerprint": input_fingerprint,
                "sources": [s.model_dump(mode="json") for s in all_sources],
            },
        )
        if bound_profile is not None:
            from app.identity.calibration import save_calibration_profile

            save_calibration_profile(bound_profile, output / "calibration-profile.json")

        _json(output / "summary.json", summary)
        _json(
            output / "artifact-hashes.json",
            {path.name: _sha(path) for path in sorted(output.iterdir()) if path.is_file()},
        )
        return output

    segment_tracks = segment_tracks or discover_identities or (fixed_segmentation is not None)
    if preloaded_raw is not None:
        raw_batch, values = preloaded_raw
    else:
        raw_batch, values = load_machine_input(settings, batch_id=batch_id)
    batch = raw_batch
    segmentation_report = None
    if preloaded_batch_and_seg is not None:
        batch, segmentation_report = preloaded_batch_and_seg
    elif fixed_segmentation is not None:
        fixed_path = Path(fixed_segmentation).resolve()
        with fixed_path.open("r", encoding="utf-8") as f:
            from app.contracts.segmentation import SegmentationReport

            segmentation_report = SegmentationReport.model_validate(json.load(f))
        evidence_path = fixed_path.parent / "evidence.json"
        if not evidence_path.exists():
            raise ValueError(
                f"evidence.json not found alongside fixed segmentation at {fixed_path}"
            )
        with evidence_path.open("r", encoding="utf-8") as f:
            from app.contracts.evidence import EvidenceBatch

            batch = EvidenceBatch.model_validate(json.load(f))
    elif segment_tracks:
        from app.identity.consistency import segment_tracks as propose_segments

        result = propose_segments(raw_batch, values, policy=settings.segmentation)
        batch = result.batch
        segmentation_report = result.report
    evidence = batch.identity_evidence
    if evidence is None:
        raise ValueError("machine input did not provide identity evidence")
    policies = [settings.matching.policy_for(s.feature_space_id) for s in evidence.feature_spaces]
    config = settings.model_dump(mode="json")
    config["effective_policies"] = [p.model_dump(mode="json") for p in policies]
    config["baseline_version"] = BASELINE_VERSION
    config["initial_state"] = "empty_database_no_gallery_no_human_seeds"
    config["implementation"] = _implementation()
    config["segment_tracks_enabled"] = segment_tracks
    config["identity_discovery_enabled"] = discover_identities
    if fixed_segmentation is not None:
        config["fixed_segmentation"] = str(fixed_segmentation)

    runtime.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    database = output / "player-library.sqlite3"
    # Exclusive reservation prevents accepting a pre-existing file or symlink.
    database.touch(exist_ok=False)
    connection = connect(database)
    try:
        first = update_library(
            connection,
            UpdateRequest(batch=batch, expected_revision=0),
            feature_values=values,
            settings=settings,
        )
        before = get_identity_state(connection, batch.match_id)
        replay = update_library(
            connection,
            UpdateRequest(
                batch=batch.model_copy(update={"batch_id": f"{batch_id}-replay"}),
                expected_revision=first.revision,
            ),
            feature_values=values,
            settings=settings,
        )
        state = get_identity_state(connection, batch.match_id)
        if replay.data.changed or replay.revision != first.revision or before != state:
            raise RuntimeError("duplicate evidence changed state or revision")
        discovery = None
        discovery_replay = None
        if discover_identities:
            from app.workflows.discover import discover_library

            discovery = discover_library(
                connection,
                batch,
                segmentation_report=segmentation_report,
                feature_values=values,
                settings=settings,
                expected_revision=state.revision,
            )
            discovered_state = get_identity_state(connection, batch.match_id)
            discovery_replay = discover_library(
                connection,
                batch.model_copy(update={"batch_id": f"{batch_id}-discovery-replay"}),
                segmentation_report=segmentation_report,
                feature_values=values,
                settings=settings,
                expected_revision=discovery.revision,
            )
            state = get_identity_state(connection, batch.match_id)
            if (
                discovery_replay.data["changed"]
                or discovery_replay.revision != discovery.revision
                or discovered_state != state
            ):
                raise RuntimeError("duplicate discovery changed identity state or revision")
            stored = get_discovery_reports(connection, batch.match_id)
            if not stored.data:
                raise RuntimeError("missing immutable discovery support")
            # Unified cold-start routing commits a primary discovery report plus an
            # association-only reassessment report; the primary (lowest revision)
            # carries every cold-start identity group.
            discovery = stored.data[0]
        matching_audit = read_matching_diagnostics(connection, settings.match.match_id)
        queue = get_review_queue(connection, batch.match_id)
    except Exception as exc:
        _json(output / "failure.json", {"error": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        connection.close()

    gallery = state.data
    segments = sorted(gallery.evidence.segments, key=lambda s: s.segment_id)
    assignments = sorted(gallery.assignments, key=lambda a: a.segment_id)
    segmentation_semantics = (
        "machine_consistency_proposals_not_identity_verified"
        if segment_tracks
        else "provisional_raw_tracks_not_identity_verified"
    )
    predictions = {
        "schema_version": "baseline-predictions-v1",
        "match_id": batch.match_id,
        "revision": state.revision,
        "segmentation": segmentation_semantics,
        "segments": [s.model_dump(mode="json") for s in segments],
        "assignments": [a.model_dump(mode="json") for a in assignments],
        "players": [p.model_dump(mode="json") for p in gallery.players],
    }
    sources = [s.model_dump(mode="json") for s in sorted(batch.sources, key=lambda s: s.path)]
    input_fingerprint = fingerprint(sources)
    if bound_profile is not None:
        predictions["distance_calibration_fingerprint"] = bound_profile.fingerprint
        predictions["distance_calibration_id"] = bound_profile.profile_id

    summary = {
        "baseline_version": BASELINE_VERSION,
        "match_id": batch.match_id,
        "revision": state.revision,
        "input_fingerprint": input_fingerprint,
        "clips": len(batch.clips or []),
        "tracks": len(evidence.tracks),
        "observations": sum(len(t.observed_frame_indices or []) for t in evidence.tracks),
        "provisional_segments": len(segments),
        "active_segments": sum(segment.active for segment in segments),
        "samples": len(evidence.samples),
        "feature_rows": len(values),
        "feature_shapes": [s.shape for s in evidence.feature_spaces],
        "source_files": len(sources),
        "players": len(gallery.players),
        "active_templates": sum(t.active for t in gallery.templates),
        "assignment_statuses": dict(Counter(a.status for a in assignments)),
        "resolution_decisions": dict(Counter(r.decision for r in first.data.resolutions)),
        "candidates": sum(len(a.candidates) for a in first.data.associations),
        "events": len(batch.events or []),
        "human_interventions": 0,
        "idempotency": {"changed": replay.data.changed, "revision": replay.revision},
        "identity_recognition_evaluated": False,
        "limitation": "Input baseline only; provisional tracks do not establish identity.",
    }
    if segmentation_report is not None:
        predictions["segmentation_summary"] = segmentation_report.summary
        summary["segmentation"] = segmentation_semantics
        summary["segmentation_summary"] = segmentation_report.summary
        summary["limitation"] = (
            "Machine consistency proposals support downstream identity discovery; "
            "they do not verify identity or establish player assignments."
        )
        _json(output / "raw-evidence.json", raw_batch.model_dump(mode="json"))
        _json(output / "segmentation.json", segmentation_report.model_dump(mode="json"))
    if discovery is not None:
        report = discovery["report"]
        mapping = discovery["player_ids_by_group"]
        groups = [{**group, "player_id": mapping[group["group_id"]]} for group in report["groups"]]
        predictions.update(
            {
                "identity_status": "experimental",
                "calibration": "unvalidated",
                "discovery_report_id": report["report_id"],
                "identity_groups": groups,
                "unknown_segments": report["unknown"],
                "templates": [t.model_dump(mode="json") for t in gallery.templates],
                "discovery_policy_fingerprint": report["policy_fingerprint"],
            }
        )
        summary.update(
            {
                "discovery": report["summary"],
                "identity_status": "experimental",
                "calibration": "unvalidated",
                "discovery_idempotency": {
                    "changed": discovery_replay.data["changed"],
                    "revision": discovery_replay.revision,
                },
                "limitation": (
                    "Experimental machine identity predictions without independent calibration; "
                    "automatic templates have no human approval."
                ),
            }
        )
        _json(output / "discovery.json", discovery)

    eval_manifest = extract_evaluation_units(raw_batch)
    disc_rep_model = (
        DiscoveryReport.model_validate(discovery["report"]) if discovery is not None else None
    )
    eval_projected = project_units_to_segments(
        eval_manifest.units,
        segments,
        assignments,
        discovery_report=disc_rep_model,
    )
    cov_funnel = build_coverage_funnel(
        eval_manifest,
        eval_projected,
        raw_batch,
        segments=segments,
        assignments=assignments,
        segmentation_report=segmentation_report,
        discovery_report=disc_rep_model,
        templates=gallery.templates,
    )
    loss_breakdown = build_loss_breakdown(
        eval_manifest,
        eval_projected,
        raw_batch,
        segments=segments,
        assignments=assignments,
        discovery_report=disc_rep_model,
    )
    summary["coverage_funnel"] = cov_funnel.model_dump(mode="json")
    summary["loss_summary"] = {
        "unassigned_samples": loss_breakdown.total_unassigned_samples,
        "unassigned_observations": loss_breakdown.total_unassigned_observations,
        "unassigned_duration_s": loss_breakdown.total_unassigned_duration_s,
        "unassigned_segments": loss_breakdown.total_unassigned_segments,
    }

    _json(output / "evidence.json", batch.model_dump(mode="json"))
    _json(output / "predictions.json", predictions)
    _json(
        output / "funnel.json",
        {
            "coverage_funnel": cov_funnel.model_dump(mode="json"),
            "loss_breakdown": loss_breakdown.model_dump(mode="json"),
        },
    )
    _json(
        output / "diagnostics.json",
        {
            "import": first.model_dump(mode="json"),
            "replay": replay.model_dump(mode="json"),
            "review_queue": queue.model_dump(mode="json"),
            "matching_attempts": matching_audit,
            "warnings": [w.model_dump(mode="json") for w in batch.warnings],
        },
    )
    _json(output / "config.json", config)
    _json(
        output / "input-manifest.json",
        {
            "schema_version": "machine-input-manifest-v1",
            "input_root": str(settings.paths.input_root),
            "fingerprint": input_fingerprint,
            "sources": sources,
        },
    )
    if bound_profile is not None:
        summary["distance_calibration"] = {
            "profile_id": bound_profile.profile_id,
            "feature_space_id": bound_profile.feature_space_id,
            "calibration_version": bound_profile.calibration_version,
            "status": bound_profile.status,
            "fingerprint": bound_profile.fingerprint,
            "near_distance": bound_profile.near_distance,
            "far_distance": bound_profile.far_distance,
        }
        from app.identity.calibration import save_calibration_profile

        save_calibration_profile(bound_profile, output / "calibration-profile.json")

    _json(output / "summary.json", summary)
    _json(
        output / "artifact-hashes.json",
        {path.name: _sha(path) for path in sorted(output.iterdir()) if path.is_file()},
    )
    return output
