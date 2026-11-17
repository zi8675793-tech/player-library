"""Incremental workflows for the legacy and frozen T1/T2 identity domains.

``update_dual_stream`` is the T3 entry point: supply machine T1 segments and a
DualStreamExtractor, with match_id, batch_id, request_id and expected_revision.
It commits extraction, matching, historical review and audit in one transaction.
``update_incremental`` retains the existing legacy EvidenceBatch interface.
"""

import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from app.contracts.api import SuccessResponse, UpdateRequest
from app.contracts.evidence import EvidenceBatch, InlineFeature
from app.contracts.segmentation import SegmentationReport
from app.features.dual_stream import DualStreamExtractor, TrackletSegment
from app.identity.constraints import SpatioTemporalConstraintEngine
from app.identity.fingerprints import evidence_fingerprint
from app.identity.incremental import plan_incremental_identities
from app.identity.matching import (
    DecisionOutcome,
    IncrementalDecision,
    IncrementalMatcher,
    IncrementalPlan,
    PlayerProfile,
    _has_conflicting_numbers,
    _has_matching_ocr,
    _is_digit_compatible,
    _jersey,
    observed_numbers,
    recompute_canonical,
)
from app.settings import Settings
from app.storage.database import transaction
from app.storage.discovery import put_discovery_report, read_discovery_reports
from app.storage.dual_stream import (
    find_dual_stream_operation,
    put_dual_stream_evidence,
    put_dual_stream_operation,
    put_dual_stream_plan,
    read_dual_stream_snapshot,
    to_json,
)
from app.storage.features import save_feature_value
from app.storage.incremental_operations import (
    get_incremental_operation,
    put_incremental_operation,
)
from app.storage.repositories import Repository, RevisionConflict, StorageConflict
from app.storage.snapshot import read_identity_snapshot
from app.workflows._consistency import (
    validate_evidence_context,
    validate_segment_frames,
    validate_templates,
)
from app.workflows.ingest import policy_for


def _hash(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _stored_value(model):
    """Compare descriptors at their persisted float32 precision."""
    value = model.model_dump(mode="json")

    def normalize(item):
        if isinstance(item, dict):
            if item.get("kind") == "inline":
                for key in ("values", "visibility"):
                    if item[key] is not None:
                        item[key] = np.asarray(item[key], dtype=np.float32).tolist()
            for child in item.values():
                normalize(child)
        elif isinstance(item, list):
            for child in item:
                normalize(child)

    normalize(value)
    return value


def _validate_reassessment_evidence(repo, snapshot, batch, feature_values, segmentation_report):
    """Read-only validation: reassessment is not an alternative evidence importer."""
    # Receipts retain the original segment state even after a human split/review.
    # Ignore only handoff ID and entity ordering, just like the evidence binding.
    incoming = evidence_fingerprint(batch, snapshot.feature_values)
    if not any(
        evidence_fingerprint(EvidenceBatch.model_validate_json(row[0]), snapshot.feature_values)
        == incoming
        for row in repo.connection.execute(
            "SELECT json FROM batches WHERE match_id=?", (batch.match_id,)
        )
    ):
        raise StorageConflict("reassessment requires an identical persisted evidence batch")

    collections = [
        (batch.assets, snapshot.gallery.assets),
        (batch.clips or [], repo.clips(batch.match_id)),
    ]
    for field in type(batch.identity_evidence).model_fields:
        if field != "segments":
            collections.append(
                (getattr(batch.identity_evidence, field), getattr(snapshot.gallery.evidence, field))
            )
    for incoming_records, persisted_records in collections:
        stored = {_hash(_stored_value(record)) for record in persisted_records}
        if any(_hash(_stored_value(record)) not in stored for record in incoming_records):
            raise StorageConflict("reassessment evidence differs from current persisted evidence")
    for key, feature in (feature_values or {}).items():
        old = snapshot.feature_values.get(key)
        if old is None or _stored_value(old) != _stored_value(feature):
            raise StorageConflict("reassessment feature rows must equal persisted evidence")
    for track in segmentation_report.tracks:
        for proposal in track.proposals:
            sid = proposal.segment.segment_id
            if snapshot.segment_proposals.get(sid) != proposal or (
                snapshot.proposal_evidence_fingerprints.get(sid)
                != segmentation_report.derived_evidence_fingerprint
            ):
                raise StorageConflict(
                    "reassessment requires identical persisted segmentation proposals"
                )


def _complete_with_reassessment(
    connection,
    primary,
    request,
    *,
    segmentation_report,
    feature_values,
    settings,
    input_fp: str | None = None,
):
    if input_fp is None:
        input_fp = evidence_fingerprint(request.batch, feature_values or {})

    comp = get_incremental_operation(connection, request.batch.match_id, input_fp, "complete")
    if comp:
        res = dict(comp["result"])
        summary = dict(res.get("summary") or {})
        summary["idempotent_replay"] = True
        res["summary"] = summary
        res["changed"] = False
        return SuccessResponse(
            data=res,
            revision=comp["committed_revision"],
        )

    if settings.discovery.support_mode != "constrained_manifold":
        final_data = {
            **primary.data,
            "committed_revision": primary.revision,
            "final_revision": primary.revision,
            "reassessment": {"changed": False, "committed_revision": primary.revision},
        }
        with transaction(connection):
            put_incremental_operation(
                connection,
                request.batch.match_id,
                input_fp,
                "complete",
                "completed",
                primary.revision,
                primary.revision,
                input_fp,
                final_data,
            )
        return primary

    # A reassessment may have committed before the completion record was written.
    # Its operation row is atomic with the revision advance, so it is the durable
    # marker that the association pass already happened. Reuse it instead of
    # re-running at the stale primary revision (which would raise RevisionConflict).
    reassess_op = get_incremental_operation(
        connection, request.batch.match_id, input_fp, "incremental_reassessment"
    )
    if reassess_op:
        secondary = SuccessResponse(
            data=reassess_op["result"],
            revision=reassess_op["committed_revision"],
        )
    else:
        secondary = reassess_unassigned(
            connection,
            request.model_copy(update={"expected_revision": primary.revision}),
            segmentation_report=segmentation_report,
            feature_values=feature_values,
            settings=settings,
            input_fp=input_fp,
        )
    second_changed = secondary.data["changed"]
    final_plan = secondary.data.get("plan") if second_changed else primary.data.get("plan")
    final_summary = (
        secondary.data.get("summary") if second_changed else primary.data.get("summary", {})
    )
    final_data = {
        **primary.data,
        "plan": final_plan,
        "summary": final_summary,
        "primary_summary": primary.data.get("summary", {}),
        "primary_committed_revision": primary.revision,
        "committed_revision": secondary.revision,
        "final_revision": secondary.revision,
        "changed": primary.data["changed"] or secondary.data["changed"],
        "reassessment": secondary.data,
        "report": primary.data.get("report")
        or (final_plan.get("discovery_report") if isinstance(final_plan, dict) else None),
        "player_ids_by_group": primary.data.get("player_ids_by_group", {}),
        "sample_sources": primary.data.get("sample_sources", {}),
    }
    with transaction(connection):
        put_incremental_operation(
            connection,
            request.batch.match_id,
            input_fp,
            "complete",
            "completed",
            primary.data.get("primary_committed_revision", primary.revision),
            secondary.revision,
            input_fp,
            final_data,
        )
    return SuccessResponse(
        data=final_data,
        revision=secondary.revision,
    )


def update_incremental(
    connection: sqlite3.Connection,
    request: UpdateRequest,
    *,
    segmentation_report: SegmentationReport,
    feature_values: Mapping[tuple[str, int], InlineFeature] | None = None,
    settings: Settings,
) -> SuccessResponse[dict]:
    """Commit core discovery, then one separately committed association-only pass.

    A second-pass failure is propagated without rolling back the primary commit.
    Replaying the original request at the current revision resumes that pass.
    """
    input_fp = evidence_fingerprint(request.batch, feature_values or {})

    # 1. Complete operation already exists? Return idempotent success response
    comp = get_incremental_operation(connection, request.batch.match_id, input_fp, "complete")
    if comp:
        res = dict(comp["result"])
        summary = dict(res.get("summary") or {})
        summary["idempotent_replay"] = True
        res["summary"] = summary
        res["changed"] = False
        return SuccessResponse(
            data=res,
            revision=comp["committed_revision"],
        )

    # 2. Primary commit already succeeded, but reassessment was interrupted? Resume!
    primary_op = get_incremental_operation(
        connection, request.batch.match_id, input_fp, "incremental_primary"
    )
    if primary_op:
        primary = SuccessResponse(
            data=primary_op["result"],
            revision=primary_op["committed_revision"],
        )
        return _complete_with_reassessment(
            connection,
            primary,
            request.model_copy(update={"expected_revision": primary_op["committed_revision"]}),
            segmentation_report=segmentation_report,
            feature_values=feature_values,
            settings=settings,
            input_fp=input_fp,
        )

    # 3. Normal execution: run primary, then reassessment
    primary = _update_incremental(
        connection,
        request,
        segmentation_report=segmentation_report,
        feature_values=feature_values,
        settings=settings,
        input_fp=input_fp,
    )
    return _complete_with_reassessment(
        connection,
        primary,
        request,
        segmentation_report=segmentation_report,
        feature_values=feature_values,
        settings=settings,
        input_fp=input_fp,
    )


def reassess_unassigned(
    connection: sqlite3.Connection,
    request: UpdateRequest,
    *,
    segmentation_report: SegmentationReport,
    feature_values: Mapping[tuple[str, int], InlineFeature] | None = None,
    settings: Settings,
    input_fp: str | None = None,
) -> SuccessResponse[dict]:
    """Associate persisted unassigned evidence only; never import or admit templates."""
    return _update_incremental(
        connection,
        request,
        segmentation_report=segmentation_report,
        feature_values=feature_values,
        settings=settings,
        association_only=True,
        input_fp=input_fp,
    )


def _update_incremental(
    connection: sqlite3.Connection,
    request: UpdateRequest,
    *,
    segmentation_report: SegmentationReport,
    feature_values: Mapping[tuple[str, int], InlineFeature] | None = None,
    settings: Settings,
    association_only: bool = False,
    input_fp: str | None = None,
) -> SuccessResponse[dict]:
    """Incrementally process a batch of machine evidence against a fixed snapshot.

    All evidence, assignments, templates and discovery support commit atomically.
    Idempotent on re-submission of the same batch.
    """
    from app.evaluation.guard import assert_inference_isolation, assert_no_ground_truth_in_evidence

    assert_inference_isolation("update_incremental")
    if hasattr(request.batch, "sources"):
        assert_no_ground_truth_in_evidence(request.batch.sources)

    if settings.input.adapter != "machine":
        raise ValueError("incremental update requires input.adapter = 'machine'")
    batch, expected = request.batch, request.expected_revision
    if batch.identity_evidence is None:
        raise ValueError("incremental update requires identity evidence")

    repo = Repository(connection)
    with transaction(connection):
        repo.ensure_match(batch.match_id)
        current_rev = repo.revision(batch.match_id)
        if current_rev != expected:
            raise RevisionConflict(
                f"stale match {batch.match_id!r}: expected revision {expected}, current is {current_rev}"
            )

        # Read fixed snapshot at revision R BEFORE any writes
        snapshot = read_identity_snapshot(connection, batch.match_id)
        gallery = snapshot.gallery

        # Check idempotency / replay: if all incoming evidence is already stored
        existing_clips = {c.clip_id for c in repo.clips(batch.match_id)}
        batch_clips = {c.clip_id for c in batch.clips or []}
        existing_tracks = {(t.clip_id, t.track_id) for t in gallery.evidence.tracks}
        batch_tracks = {(t.clip_id, t.track_id) for t in batch.identity_evidence.tracks}
        existing_segments = {s.segment_id for s in gallery.evidence.segments}
        batch_segments = {s.segment_id for s in batch.identity_evidence.segments}
        existing_samples = {s.sample_id for s in gallery.evidence.samples}
        batch_samples = {s.sample_id for s in batch.identity_evidence.samples}
        existing_obs = {o.observation_id for o in gallery.evidence.observations}
        batch_obs = {o.observation_id for o in batch.identity_evidence.observations}

        all_evidence_already_stored = (
            batch_clips.issubset(existing_clips)
            and batch_tracks.issubset(existing_tracks)
            and batch_segments.issubset(existing_segments)
            and batch_samples.issubset(existing_samples)
            and batch_obs.issubset(existing_obs)
            and all(
                (file_id, row) in snapshot.feature_values
                for (file_id, row) in (feature_values or {}).keys()
            )
        )

        if input_fp is None:
            input_fp = evidence_fingerprint(batch, feature_values or snapshot.feature_values)
        existing_reports = read_discovery_reports(connection, batch.match_id)
        if association_only:
            _validate_reassessment_evidence(
                repo, snapshot, batch, feature_values, segmentation_report
            )
            changed = False
        else:
            changed = repo.store_batch(batch)
        plan_key = _hash(
            {
                "input": input_fp,
                "policy": settings.discovery.model_dump(mode="json"),
                "segmentation": segmentation_report.model_dump(
                    mode="json", exclude={"input_fingerprint"}
                ),
                "algorithm": "safe-matching-v3-diagnostics",
                "mode": "association_only" if association_only else "discovery",
                "snapshot_revision": gallery.revision,
            }
        )
        already_planned_report = any(
            r["report"].get("summary", {}).get("plan_key") == plan_key
            or (
                r["report"].get("input_fingerprint") == input_fp
                and r["report"].get("summary", {}).get("association_only", False)
                == association_only
            )
            for r in existing_reports
        )

        if all_evidence_already_stored and already_planned_report:
            return SuccessResponse(
                data={
                    "changed": False,
                    "committed_revision": expected,
                    "summary": {"idempotent_replay": True, "input_fingerprint": input_fp},
                },
                revision=expected,
            )

        # Store incoming batch evidence

        # Store segmentation proposals for the incoming batch
        if not association_only and segmentation_report and segmentation_report.tracks:
            for t in segmentation_report.tracks:
                for prop in t.proposals:
                    repo.put_segmentation_proposal(
                        batch.match_id,
                        prop,
                        evidence_fingerprint=segmentation_report.derived_evidence_fingerprint,
                    )

        # Store feature values
        files = {item.feature_file_id: item for item in batch.identity_evidence.feature_files}
        spaces = {item.feature_space_id: item for item in batch.identity_evidence.feature_spaces}
        all_features = dict(snapshot.feature_values)
        if feature_values and not association_only:
            for (file_id, row), value in feature_values.items():
                file = files.get(file_id)
                if file is None:
                    # check snapshot files
                    file = next(
                        (f for f in gallery.evidence.feature_files if f.feature_file_id == file_id),
                        None,
                    )
                if file is None:
                    raise KeyError(f"unknown feature file {file_id!r}")
                space = spaces.get(file.feature_space_id)
                if space is None:
                    space = next(
                        (
                            s
                            for s in gallery.evidence.feature_spaces
                            if s.feature_space_id == file.feature_space_id
                        ),
                        None,
                    )
                if space is None:
                    raise KeyError(f"unknown feature space {file.feature_space_id!r}")
                policy = policy_for(file.feature_space_id, [], settings)
                from app.identity.distance import validate_feature

                validate_feature(space, value, policy)
                changed = (
                    save_feature_value(connection, batch.match_id, file, space, row, value)
                    or changed
                )
                all_features[file_id, row] = value

        # Plan incremental decisions against the fixed snapshot
        plan = plan_incremental_identities(
            snapshot,
            batch,
            segmentation_report=segmentation_report,
            feature_values=all_features,
            policy=settings.discovery,
            constraints=snapshot.constraints,
            association_only=association_only,
        )
        plan.discovery_report.summary["plan_key"] = plan_key
        from app.storage.matching_diagnostics import put_matching_diagnostics

        put_matching_diagnostics(
            connection,
            batch.match_id,
            plan_key,
            expected,
            "incremental_reassessment" if association_only else "incremental_primary",
            plan.matching_diagnostics,
        )

        # Check if plan changes anything
        new_assignments = []
        for decision in plan.decisions:
            old = next(
                (a for a in gallery.assignments if a.segment_id == decision.segment_id),
                None,
            )
            if old is None or (old.player_id, old.status) != (
                decision.assignment.player_id,
                decision.assignment.status,
            ):
                new_assignments.append(decision)

        if not plan.new_players and not plan.new_templates and not new_assignments and not changed:
            return SuccessResponse(
                data={
                    "changed": False,
                    "committed_revision": expected,
                    "plan": plan.model_dump(mode="json"),
                    "summary": {**plan.summary, "idempotent_replay": True},
                },
                revision=expected,
            )

        committed = expected + 1

        # Put new players
        for player in plan.new_players:
            changed = repo.put_player(player) or changed

        # Put assignments
        for decision in plan.decisions:
            assignment = decision.assignment.model_copy(update={"revision": committed})
            changed = (
                repo.put_assignment(assignment, committed_revision=committed, replace=True)
                or changed
            )

        # Put new templates
        for template in plan.new_templates:
            tpl = template.model_copy(update={"revision": committed})
            changed = repo.put_template(tpl, committed_revision=committed, replace=True) or changed

        # Deactivate revoked templates if any
        for tid in plan.deactivated_template_ids:
            old_tpl = next((t for t in gallery.templates if t.template_id == tid), None)
            if old_tpl and old_tpl.active:
                deactivated = old_tpl.model_copy(update={"active": False, "revision": committed})
                changed = (
                    repo.put_template(deactivated, committed_revision=committed, replace=True)
                    or changed
                )

        # Persist discovery report
        put_discovery_report(
            connection,
            plan.discovery_report,
            plan.player_map,
            committed,
            plan.sample_sources,
        )

        # Consistency check: all template support_pair_ids must resolve to valid supported DiscoveryPairs
        all_supported_pairs = {
            p.pair_id for p in plan.discovery_report.pairs if p.status == "supported"
        }
        for g in plan.discovery_report.groups:
            for tpl_support in g.templates:
                if not tpl_support.support_pair_ids:
                    raise ValueError(f"template {tpl_support.sample_id} lacks support_pair_ids")
                for pid in tpl_support.support_pair_ids:
                    if pid not in all_supported_pairs:
                        raise ValueError(
                            f"template support_pair_id {pid!r} cannot be resolved to a supported DiscoveryPair"
                        )

        # Advance revision
        repo.advance_revision(
            batch.match_id,
            expected,
            actor="identity.incremental",
            reason="association_only_reassessment"
            if association_only
            else "incremental_batch_update",
            changes={
                "batch_id": batch.batch_id,
                "created_players": [p.player_id for p in plan.new_players],
                "matched_existing": [
                    d.segment_id for d in plan.decisions if d.category == "matched_existing"
                ],
                "created_new_identity": [
                    d.segment_id for d in plan.decisions if d.category == "created_new_identity"
                ],
                "insufficient_evidence": [
                    d.segment_id for d in plan.decisions if d.category == "insufficient_evidence"
                ],
                "new_templates": [t.template_id for t in plan.new_templates],
            },
        )

        # Re-read and validate full snapshot state
        final_state = read_identity_snapshot(connection, batch.match_id)
        validate_evidence_context(repo, final_state.gallery)
        validate_segment_frames(final_state.gallery.evidence.segments)
        validate_templates(
            final_state.gallery,
            all_features,
            lambda space_id: policy_for(space_id, [], settings),
            discovery_reports=read_discovery_reports(connection, batch.match_id),
        )

        if repo.check_segment_overlaps(batch.match_id) or repo.check_alias_cycles(batch.match_id):
            raise StorageConflict("segment overlap or player alias cycle")

        primary_data = {
            "changed": True,
            "committed_revision": committed,
            "plan": plan.model_dump(mode="json"),
            "report": plan.discovery_report.model_dump(mode="json"),
            "player_ids_by_group": plan.player_map,
            "sample_sources": plan.sample_sources,
            "summary": plan.summary,
        }
        put_incremental_operation(
            connection,
            batch.match_id,
            input_fp,
            "incremental_reassessment" if association_only else "incremental_primary",
            "completed",
            expected,
            committed,
            plan_key,
            primary_data,
        )
        return SuccessResponse(
            data=primary_data,
            revision=committed,
        )


def _dual_stream_input(segments, extractor):
    """Bind geometry to actual crop bytes, independently of batch/order/locators."""
    from app.evaluation.guard import assert_no_ground_truth_in_evidence

    assert_no_ground_truth_in_evidence([str(extractor.crop_root)])
    hashes = {}
    seen_paths = set()
    root = Path(extractor.crop_root).resolve()
    for segment in sorted(segments, key=lambda item: item.segment_id):
        value = asdict(segment)
        observations = []
        for observation in sorted(segment.observations, key=lambda item: item.frame_idx):
            record = asdict(observation)
            supplied = observation.crop_path
            record.pop("crop_path")
            record["crop_sha256"] = None
            if supplied is not None:
                assert_no_ground_truth_in_evidence([supplied])
                path = Path(supplied)
                candidate = path if path.is_absolute() else root / path
                expected = root / segment.video_id / "samples"
                if (
                    candidate.parent != expected
                    or candidate.suffix.lower() != ".png"
                    or candidate.resolve() != candidate
                    or not candidate.is_file()
                ):
                    raise ValueError("crop_path must reference a regular PNG inside video/samples")
                if candidate in seen_paths:
                    raise ValueError("Duplicate crop file cannot witness independent observations")
                seen_paths.add(candidate)
                digest = hashlib.sha256()
                with candidate.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                record["crop_sha256"] = digest.hexdigest()
            observations.append(record)
        if not any(item["crop_sha256"] for item in observations):
            raise ValueError(f"No crop images for segment {segment.segment_id}")
        value["observations"] = observations
        hashes[segment.segment_id] = _hash(value)
    return hashes, _hash({"algorithm": "dual-stream-input-v1", "segments": hashes})


def _dual_stream_owner(decision):
    if decision.outcome == DecisionOutcome.CREATED_NEW:
        return decision.evidence["new_player_id"]
    return decision.target_player_id


def _dual_stream_gallery(players, plan, revision):
    """Apply additions in memory; provisional galleries never escape the transaction."""
    gallery = {player.player_id: player for player in players}
    for player in plan.new_players:
        gallery[player.player_id] = replace(
            player,
            created_at_revision=revision,
            prototypes=tuple(
                replace(proto, created_at_revision=revision) for proto in player.prototypes
            ),
        )
    for prototype in plan.updated_prototypes:
        player = gallery[prototype.player_id]
        if not any(p.feature.segment_id == prototype.feature.segment_id for p in player.prototypes):
            gallery[player.player_id] = replace(
                player,
                prototypes=(*player.prototypes, replace(prototype, created_at_revision=revision)),
            )
    return gallery


def _dual_stream_reassess(snapshot, primary, segments, features, matcher, base_engine=None):
    """Revisit pending evidence and tentative assignments with self evidence removed.

    The second gallery is provisional R+1 state inside the *same* write lock.
    T2 receives that revision for validation; all resulting changes commit at
    R+1, alongside the primary plan. There is no separately committed pass.
    """
    if base_engine is None:
        base_engine = SpatioTemporalConstraintEngine(segments=segments)

    revision = primary.to_revision
    gallery = _dual_stream_gallery(snapshot.players, primary, revision)
    decisions = {decision.segment_id: decision for decision in primary.decisions}
    assignments = dict(snapshot.assignments)

    def update_assignments(items):
        for decision in items:
            owner = _dual_stream_owner(decision)
            if owner is None:
                assignments.pop(decision.segment_id, None)
            else:
                assignments[decision.segment_id] = owner

    update_assignments(primary.decisions)
    # Only historical deferred queries may see the provisional updated gallery.
    # New queries keep their fixed Revision-R decisions for this entire batch.
    deferred_history = [sid for sid in primary.deferred_segments if sid in snapshot.decisions]
    if deferred_history:
        engine = base_engine.with_player_assignments(assignments)
        pending = matcher.plan(
            [features[sid] for sid in deferred_history],
            tuple(gallery.values()),
            segments=segments,
            batch_id=primary.batch_id,
            from_revision=revision,
            player_assignments=assignments,
            engine=engine,
            feature_index=features,
        )
        gallery = _dual_stream_gallery(tuple(gallery.values()), pending, revision)
        for decision in pending.decisions:
            decisions[decision.segment_id] = replace(
                decision,
                evidence={**decision.evidence, "reassessment": "leave_one_out"},
            )
        update_assignments(pending.decisions)

    reassessed = set(snapshot.decisions) & set(decisions)
    for sid, old_owner in sorted(snapshot.assignments.items()):
        old_player = gallery.get(old_owner)
        old_confirmed_num = old_player.confirmed_jersey_number if old_player else None
        sid_feat = features[sid]
        sid_jersey = sid_feat.jersey
        sid_num = sid_jersey.number if sid_jersey is not None else None

        old_has_physical_number_match = old_confirmed_num is not None and (
            sid_num == old_confirmed_num or _has_matching_ocr(sid_feat, old_confirmed_num)
        )
        is_tentative_soft_assignment = (sid_num is None) and (not old_has_physical_number_match)

        def _held_out_protos(p):
            protos = tuple(proto for proto in p.prototypes if proto.feature.segment_id != sid)
            if (
                p.player_id == old_owner
                and is_tentative_soft_assignment
                and old_confirmed_num is not None
            ):
                anchors = tuple(
                    proto
                    for proto in protos
                    if _jersey(proto.feature) == old_confirmed_num
                    or _has_matching_ocr(proto.feature, old_confirmed_num)
                )
                if anchors:
                    return anchors
            return protos

        # Each historical query is held out from every profile, and from the
        # historical assignment map. Other segments retain their physical bans.
        held_out = tuple(
            replace(player, prototypes=_held_out_protos(player)) for player in gallery.values()
        )
        held_out_assignments = {key: owner for key, owner in assignments.items() if key != sid}
        engine = base_engine.with_player_assignments(held_out_assignments)
        review = matcher.plan(
            [features[sid]],
            held_out,
            segments=segments,
            batch_id=primary.batch_id,
            from_revision=revision,
            player_assignments=held_out_assignments,
            engine=engine,
            feature_index=features,
        )
        reassessed.add(sid)
        decision = review.decisions[0]
        owner = _dual_stream_owner(decision)
        if owner == old_owner:
            continue
        old_player = gallery.get(old_owner)
        remaining_protos = (
            [p for p in old_player.prototypes if p.feature.segment_id != sid] if old_player else []
        )
        old_score = decision.evidence.get("player_scores", {}).get(old_owner, {})
        old_dist = old_score.get("distance")
        old_blocked = old_score.get("blocked_by")

        old_numbers = (
            observed_numbers(
                replace(old_player, prototypes=tuple(remaining_protos)),
                held_out_assignments,
                features,
            )
            if old_player
            else set()
        )
        jersey_ok = (
            (sid_num is None)
            or (len(old_numbers) == 0)
            or (sid_num in old_numbers)
            or any(_is_digit_compatible(sid_num, n) for n in old_numbers)
        )

        old_still_valid = (
            len(remaining_protos) >= 1
            and isinstance(old_dist, (int, float))
            and math.isfinite(old_dist)
            and old_dist <= matcher.distance_threshold + 1e-7
            and old_blocked is None
            and jersey_ok
        )
        if owner is None:
            if old_still_valid:
                continue
            # Physical anchor protection: single-prototype archive with physical number match
            if len(remaining_protos) == 0 and old_has_physical_number_match:
                continue
        elif old_still_valid:
            old_has_matching_ocr = (
                old_player is not None
                and old_player.confirmed_jersey_number is not None
                and _has_matching_ocr(features[sid], old_player.confirmed_jersey_number)
            )
            if old_has_matching_ocr:
                continue
            if isinstance(old_dist, (int, float)) and math.isfinite(old_dist):
                if old_dist - decision.best_distance < matcher.competitive_margin:
                    continue
        # Reassignment/demotion must retract the source from its previous owner;
        # its immutable historical prototype record remains in storage.
        gallery = {
            player.player_id: replace(
                player,
                prototypes=tuple(p for p in player.prototypes if p.feature.segment_id != sid),
            )
            for player in gallery.values()
        }
        gallery = _dual_stream_gallery(tuple(gallery.values()), review, revision)
        decisions[sid] = replace(
            decision,
            evidence={
                **decision.evidence,
                "reassessment": "leave_one_out",
                "previous_player_id": old_owner,
            },
        )
        update_assignments([decision])

    # Canonical Recompute on Cumulative Evidence
    canon_matcher = IncrementalMatcher(
        distance_threshold=matcher.distance_threshold,
        competitive_margin=matcher.competitive_margin,
        min_quality=matcher.min_quality,
        min_prototype_weight=matcher.min_prototype_weight,
        max_prototypes=matcher.max_prototypes,
        high_confidence_distance=matcher.high_confidence_distance,
        high_confidence_margin=matcher.high_confidence_margin,
    )
    canon_assignments, canon_gallery, canon_decisions = recompute_canonical(
        segments, features, canon_matcher, base_engine=base_engine
    )

    cpid_to_gpid = {}
    gallery_by_num = {
        p.confirmed_jersey_number: p.player_id
        for p in gallery.values()
        if p.confirmed_jersey_number is not None
    }
    claimed_gpids = set()

    for cpid, cplayer in canon_gallery.items():
        if cplayer.confirmed_jersey_number is not None:
            if cplayer.confirmed_jersey_number in gallery_by_num:
                gpid = gallery_by_num[cplayer.confirmed_jersey_number]
                cpid_to_gpid[cpid] = gpid
                claimed_gpids.add(gpid)
            else:
                new_p = PlayerProfile(
                    cplayer.player_id,
                    cplayer.display_name,
                    cplayer.confirmed_jersey_number,
                    (),
                    "active",
                    revision,
                )
                gallery[cplayer.player_id] = new_p
                cpid_to_gpid[cpid] = cplayer.player_id
                gallery_by_num[cplayer.confirmed_jersey_number] = cplayer.player_id
                claimed_gpids.add(cplayer.player_id)

    # For unnumbered cplayers, exact ID match first if available and unclaimed
    for cpid, cplayer in canon_gallery.items():
        if (
            cplayer.confirmed_jersey_number is None
            and cpid in gallery
            and cpid not in claimed_gpids
            and gallery[cpid].confirmed_jersey_number is None
        ):
            cpid_to_gpid[cpid] = cpid
            claimed_gpids.add(cpid)

    # For remaining unnumbered cplayers, find best unclaimed overlap
    for cpid, cplayer in canon_gallery.items():
        if cplayer.confirmed_jersey_number is not None or cpid in cpid_to_gpid:
            continue
        canon_members = {s for s, cp in canon_assignments.items() if cp == cpid}
        canon_numbers = observed_numbers(cplayer, canon_assignments, features)
        best_overlap, best_pid = 0, None
        for gplayer in gallery.values():
            if gplayer.player_id in claimed_gpids:
                continue
            if gplayer.confirmed_jersey_number is None:
                if _has_conflicting_numbers(
                    canon_numbers | observed_numbers(gplayer, assignments, features)
                ):
                    continue
                g_members = {s for s, gp in assignments.items() if gp == gplayer.player_id}
                overlap = len(canon_members & g_members)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_pid = gplayer.player_id
        if best_pid is not None and best_overlap > 0:
            cpid_to_gpid[cpid] = best_pid
            claimed_gpids.add(best_pid)
        else:
            new_p = PlayerProfile(
                cplayer.player_id,
                cplayer.display_name,
                None,
                (),
                "active",
                revision,
            )
            gallery[cplayer.player_id] = new_p
            cpid_to_gpid[cpid] = cplayer.player_id
            claimed_gpids.add(cplayer.player_id)

    # Apply canonical assignments to reconcile any greedy misassignments
    for sid in (s.segment_id for s in segments):
        if sid not in snapshot.decisions:
            continue
        target_cpid = canon_assignments.get(sid)
        target_gpid = cpid_to_gpid.get(target_cpid) if target_cpid else None
        curr_gpid = assignments.get(sid)
        if target_gpid != curr_gpid:
            old_owner = curr_gpid
            if target_gpid is not None:
                assignments[sid] = target_gpid
                dec = canon_decisions.get(sid)
                best_d = dec.best_distance if dec else 0.1
                margin = dec.competitive_margin if dec else 0.05
                ev = dict(dec.evidence) if dec else {}
                ev["reassessment"] = "leave_one_out"
                ev["canonical_recompute"] = True
                if old_owner is not None:
                    ev["previous_player_id"] = old_owner
                decisions[sid] = IncrementalDecision(
                    sid,
                    DecisionOutcome.MATCHED_EXISTING,
                    target_gpid,
                    best_d,
                    margin,
                    ev,
                )
            else:
                assignments.pop(sid, None)
                decisions[sid] = IncrementalDecision(
                    sid,
                    DecisionOutcome.INSUFFICIENT,
                    None,
                    math.inf,
                    0.0,
                    {"reassessment": "leave_one_out", "previous_player_id": old_owner},
                )
            reassessed.add(sid)

    # Active Gallery Dynamic Regeneration
    existing_proto_map = {
        (p.player_id, proto.feature.segment_id): proto
        for p in snapshot.players
        for proto in p.prototypes
    }

    by_assigned_player = defaultdict(list)
    for sid, pid in assignments.items():
        by_assigned_player[pid].append(sid)

    original_ids = {player.player_id for player in snapshot.players}
    for pid in list(gallery.keys()):
        if pid not in original_ids and not by_assigned_player[pid]:
            del gallery[pid]

    for pid, player in list(gallery.items()):
        assigned_sids = set(by_assigned_player[pid])
        if not assigned_sids:
            gallery[pid] = replace(player, prototypes=())
            continue
        valid_protos = [p for p in player.prototypes if p.feature.segment_id in assigned_sids]
        available = max(0, matcher.max_prototypes - len(valid_protos))
        existing_sids = {p.feature.segment_id for p in valid_protos}
        if available > 0:
            candidates = sorted(
                (sid for sid in assigned_sids if sid not in existing_sids),
                key=lambda sid: (-features[sid].appearance.quality_score, sid),
            )
            for sid in candidates[:available]:
                key = (pid, sid)
                if key in existing_proto_map:
                    valid_protos.append(existing_proto_map[key])
                else:
                    valid_protos.append(matcher._prototype(pid, features[sid], revision))
        gallery[pid] = replace(player, prototypes=tuple(valid_protos))

    original_prototypes = {
        proto.prototype_id for player in snapshot.players for proto in player.prototypes
    }
    ordered = tuple(decisions[sid] for sid in sorted(decisions))
    plan = IncrementalPlan(
        primary.batch_id,
        primary.from_revision,
        revision,
        ordered,
        tuple(gallery[pid] for pid in sorted(gallery.keys() - original_ids)),
        tuple(
            proto
            for pid in sorted(original_ids)
            for proto in gallery[pid].prototypes
            if proto.prototype_id not in original_prototypes
        ),
        tuple(d.segment_id for d in ordered if d.outcome == DecisionOutcome.INSUFFICIENT),
    )
    return plan, tuple(gallery[pid] for pid in sorted(gallery)), assignments, sorted(reassessed)


def _segment_conflict_strength(player, segment_id, assignments, features, engine, decisions):
    """Deterministic evidence ordering for jersey-number conflict repair.

    Confirmed-number agreement first, then the number of assigned segments that
    support the same number, then observed frames, box area, distance and
    finally the segment_id. Returned as ascending keys so the strongest anchor
    sorts first without relying on ``reverse=True`` for a lexicographic tie.
    """
    feature = features.get(segment_id)
    number = _jersey(feature) if feature is not None else None
    confirmed = player.confirmed_jersey_number
    confirmed_match = 0
    if confirmed is not None and number is not None:
        if number == confirmed:
            confirmed_match = 3
        elif feature is not None and _has_matching_ocr(feature, confirmed):
            confirmed_match = 2
        elif _is_digit_compatible(number, confirmed):
            confirmed_match = 1
    support = 0
    if number is not None:
        support = sum(
            1
            for other, owner in assignments.items()
            if owner == player.player_id
            and other != segment_id
            and _jersey(features.get(other)) is not None
            and _is_digit_compatible(_jersey(features.get(other)), number)
        )
    segment = engine.segments_by_id.get(segment_id)
    frames = segment.observation_count if segment is not None else 0
    area = 0.0
    if segment is not None:
        for observation in segment.observations:
            box = observation.bbox_xyxy
            area += (box[2] - box[0]) * (box[3] - box[1])
    decision = decisions.get(segment_id)
    distance = decision.best_distance if decision is not None else math.inf
    if not isinstance(distance, (int, float)) or not math.isfinite(distance):
        distance = math.inf
    return (-confirmed_match, -support, -frames, -area, distance, segment_id)


def _rebuild_player_prototypes(player, assignments, features, matcher, revision):
    """Keep only prototypes whose segment is still assigned, then refill by quality."""
    assigned = sorted(
        sid for sid, pid in assignments.items() if pid == player.player_id and sid in features
    )
    kept = tuple(
        prototype for prototype in player.prototypes if prototype.feature.segment_id in assigned
    )
    used = {prototype.feature.segment_id for prototype in kept}
    if matcher is None:
        return player
    capacity = max(0, matcher.max_prototypes - len(kept))
    candidates = sorted(
        (
            sid
            for sid in assigned
            if sid not in used
            and features[sid].appearance.quality_score > 0
            and features[sid].appearance.quality_score
            >= max(matcher.min_quality, matcher.min_prototype_weight)
        ),
        key=lambda sid: (-features[sid].appearance.quality_score, sid),
    )
    prototypes = list(kept)
    prototypes.extend(
        matcher._prototype(player.player_id, features[sid], revision)
        for sid in candidates[:capacity]
    )
    return replace(player, prototypes=tuple(prototypes))


def _validate_dual_stream_commit(
    players,
    assignments,
    segments,
    features,
    *,
    decisions=None,
    revision=None,
    matcher=None,
):
    """Repair jersey-number conflicts deterministically; fail closed on real corruption.

    A profile can accumulate numbers that prototype capacity never surfaced.
    Instead of aborting a legal batch, keep the strongest anchor evidence and
    return every conflicting segment to ``insufficient_evidence``. Only genuine
    internal inconsistency (unknown player, orphan prototype, cannot-link)
    still raises.
    """
    engine = SpatioTemporalConstraintEngine(segments=segments)
    profiles = {player.player_id: player for player in players}
    decisions = dict(decisions or {})
    demoted: list[str] = []

    def check_structure(profiles, assignments):
        engine_now = SpatioTemporalConstraintEngine(
            segments=segments, player_assignments=assignments
        )
        grouped = {}
        for sid, pid in assignments.items():
            grouped.setdefault(pid, []).append(sid)
        for pid, ids in grouped.items():
            if pid not in profiles:
                raise StorageConflict("Assignment references an unknown dual-stream player")
            for i, left in enumerate(ids):
                if any(engine_now.pair_relation(left, right) != "clear" for right in ids[i + 1 :]):
                    raise StorageConflict("Dual-stream plan violates a physical cannot-link")
        for player in profiles.values():
            for prototype in player.prototypes:
                if assignments.get(prototype.feature.segment_id) != player.player_id:
                    raise StorageConflict("Active prototype has no matching current assignment")

    check_structure(profiles, assignments)

    if revision is None:
        revision = max((player.created_at_revision for player in players), default=1)
    repaired: dict[str, PlayerProfile] = dict(profiles)
    assignments = dict(assignments)
    for player in players:
        assigned = sorted(sid for sid, pid in assignments.items() if pid == player.player_id)
        numbers = observed_numbers(player, assignments, features)
        if not _has_conflicting_numbers(numbers):
            continue
        if player.confirmed_jersey_number is not None:
            accepted = {player.confirmed_jersey_number}
        else:
            accepted: set[int] = set()
        ordered = sorted(
            assigned,
            key=lambda sid: _segment_conflict_strength(
                player, sid, assignments, features, engine, decisions
            ),
        )
        conflicting = set()
        for sid in ordered:
            feature = features.get(sid)
            number = _jersey(feature) if feature is not None else None
            if number is None:
                continue
            candidate = accepted | {number}
            if _has_conflicting_numbers(candidate):
                conflicting.add(sid)
                demoted.append(sid)
            else:
                accepted = candidate
        if not conflicting:
            continue
        for sid in sorted(conflicting):
            owner = assignments.pop(sid)
            old = decisions.get(sid)
            reason = "jersey_number_conflict_repair"
            evidence = dict(old.evidence) if old is not None else {}
            evidence.update(
                {
                    "reason": reason,
                    "conflict_repair": {
                        "previous_player_id": owner,
                        "kept_numbers": sorted(accepted),
                        "observed_numbers": sorted(numbers),
                    },
                }
            )
            if old is not None:
                decisions[sid] = replace(
                    old,
                    outcome=DecisionOutcome.INSUFFICIENT,
                    target_player_id=None,
                    evidence=evidence,
                )
            else:
                decisions[sid] = IncrementalDecision(
                    sid, DecisionOutcome.INSUFFICIENT, None, math.inf, 0.0, evidence
                )
        repaired[player.player_id] = _rebuild_player_prototypes(
            player, assignments, features, matcher, revision
        )

    # A newly created profile that lost every segment is not a historical fact.
    dropped = {
        pid
        for pid, player in repaired.items()
        if player.created_at_revision == revision
        and not any(owner == pid for owner in assignments.values())
    }
    for pid in dropped:
        del repaired[pid]
    players = tuple(repaired[pid] for pid in sorted(repaired))

    check_structure({player.player_id: player for player in players}, assignments)
    for player in players:
        if _has_conflicting_numbers(observed_numbers(player, assignments, features)):
            raise StorageConflict("Dual-stream profile contains conflicting jersey numbers")
    return players, assignments, decisions, demoted


def _rebuild_dual_stream_plan(plan, players, decisions):
    """Reassemble a plan after the conflict repair changed assignments."""
    by_id = {player.player_id: player for player in players}
    prototype_ids = {proto.prototype_id for p in players for proto in p.prototypes}
    ordered = tuple(decisions[sid] for sid in sorted(decisions))
    return IncrementalPlan(
        plan.batch_id,
        plan.from_revision,
        plan.to_revision,
        ordered,
        tuple(by_id[p.player_id] for p in plan.new_players if p.player_id in by_id),
        tuple(proto for proto in plan.updated_prototypes if proto.prototype_id in prototype_ids),
        tuple(d.segment_id for d in ordered if d.outcome == DecisionOutcome.INSUFFICIENT),
    )


def update_dual_stream(
    connection: sqlite3.Connection,
    *,
    match_id: str,
    batch_id: str,
    request_id: str,
    expected_revision: int,
    segments: Sequence[TrackletSegment],
    extractor: DualStreamExtractor,
    matcher: IncrementalMatcher | None = None,
    actor: str = "automatic",
) -> SuccessResponse[dict]:
    """Run T1/T2 and historical leave-one-out review in one BEGIN IMMEDIATE.

    Inputs are machine-only T1 segments with exact frame/crop associations.
    Content replay precedes revision rejection, so a lost successful response can
    be retried at its original expected_revision, even after later commits.
    Replays report the current revision and retain the original commit receipt.
    A changed request_id payload, modified historical evidence, model failure or
    audit failure rolls back the entire transaction, including match creation.
    """
    from app.evaluation.guard import assert_inference_isolation

    assert_inference_isolation("update_dual_stream")
    for name, value in (("match_id", match_id), ("batch_id", batch_id), ("request_id", request_id)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty string")
    if not isinstance(actor, str) or not actor.strip():
        raise ValueError("actor must be a nonempty string")
    if type(expected_revision) is not int or expected_revision < 0:
        raise ValueError("expected_revision must be a nonnegative integer")
    ordered = tuple(sorted(segments, key=lambda item: item.segment_id))
    if not ordered:
        raise ValueError("A dual-stream batch requires at least one segment")
    matcher = matcher if matcher is not None else IncrementalMatcher()
    repo = Repository(connection)
    with transaction(connection):
        # Both validation and fixed-snapshot planning run under the write lock.
        SpatioTemporalConstraintEngine(segments=ordered)
        input_hashes, fingerprint = _dual_stream_input(ordered, extractor)
        repo.ensure_match(match_id)
        snapshot = read_dual_stream_snapshot(connection, match_id)
        prior = find_dual_stream_operation(connection, match_id, request_id=request_id)
        if prior is not None and prior["input_fingerprint"] != fingerprint:
            raise StorageConflict("request_id was already used for different input content")
        if prior is None:
            prior = find_dual_stream_operation(connection, match_id, input_fingerprint=fingerprint)
        payload = {
            "batch_id": batch_id,
            "input_hashes": input_hashes,
            "matcher": {
                key: getattr(matcher, key)
                for key in (
                    "distance_threshold",
                    "competitive_margin",
                    "min_quality",
                    "min_prototype_weight",
                    "max_prototypes",
                    "high_confidence_distance",
                    "high_confidence_margin",
                )
                if hasattr(matcher, key)
            },
        }
        for sid, digest in input_hashes.items():
            if sid in snapshot.input_hashes and snapshot.input_hashes[sid] != digest:
                raise StorageConflict(f"Immutable segment content changed: {sid}")
        fresh = [segment for segment in ordered if segment.segment_id not in snapshot.input_hashes]
        if prior is not None or not fresh:
            data = (
                dict(prior["result"])
                if prior
                else {
                    "committed_revision": snapshot.revision,
                    "input_fingerprint": fingerprint,
                }
            )
            data["changed"] = False
            data["summary"] = {**data.get("summary", {}), "idempotent_replay": True}
            put_dual_stream_operation(connection, match_id, request_id, fingerprint, payload, data)
            return SuccessResponse(data=data, revision=snapshot.revision)
        if snapshot.revision != expected_revision:
            raise RevisionConflict(
                f"stale match {match_id!r}: expected revision {expected_revision}, "
                f"current is {snapshot.revision}"
            )

        extracted = tuple(extractor.extract(segment) for segment in fresh)
        new_features = {feature.segment_id: feature for feature in extracted}
        if set(new_features) != {segment.segment_id for segment in fresh}:
            raise StorageConflict("Extractor returned features for different segments")
        all_segments = {segment.segment_id: segment for segment in snapshot.segments}
        all_segments.update((segment.segment_id, segment) for segment in fresh)
        metadata = tuple(all_segments[sid] for sid in sorted(all_segments))
        features = {**snapshot.features, **new_features}
        pending = [
            snapshot.features[sid]
            for sid, decision in sorted(snapshot.decisions.items())
            if decision.outcome == DecisionOutcome.INSUFFICIENT and sid not in new_features
        ]
        base_engine = SpatioTemporalConstraintEngine(segments=metadata)
        primary = matcher.plan(
            extracted,
            snapshot.players,
            segments=metadata,
            batch_id=batch_id,
            from_revision=snapshot.revision,
            pending_features=pending,
            player_assignments=snapshot.assignments,
            engine=base_engine.with_player_assignments(snapshot.assignments),
            feature_index=features,
        )
        plan, players, assignments, reassessed = _dual_stream_reassess(
            snapshot, primary, metadata, features, matcher, base_engine=base_engine
        )
        players, assignments, decisions, repaired = _validate_dual_stream_commit(
            players,
            assignments,
            metadata,
            features,
            decisions={d.segment_id: d for d in plan.decisions},
            revision=plan.to_revision,
            matcher=matcher,
        )
        if repaired:
            plan = _rebuild_dual_stream_plan(plan, players, decisions)
            reassessed = sorted(set(reassessed) | set(repaired))
        # A crop changed during inference cannot be committed under its old hash.
        if _dual_stream_input(ordered, extractor)[0] != input_hashes:
            raise StorageConflict("Input crop bytes changed during feature extraction")
        put_dual_stream_evidence(
            connection,
            match_id,
            fresh,
            extracted,
            {segment.segment_id: input_hashes[segment.segment_id] for segment in fresh},
        )
        put_dual_stream_plan(connection, match_id, plan, players=players)
        summary = {
            outcome.value: sum(d.outcome == outcome for d in plan.decisions)
            for outcome in DecisionOutcome
        }
        summary["reassessed_segments"] = reassessed
        summary["cannot_link_edges"] = len(base_engine.constraints)
        data = {
            "changed": True,
            "committed_revision": plan.to_revision,
            "input_fingerprint": fingerprint,
            "plan": to_json(plan),
            "summary": summary,
        }
        repo.advance_revision(
            match_id,
            expected_revision,
            actor=actor,
            reason="dual_stream_incremental",
            changes={"input_fingerprint": fingerprint, "batch_id": batch_id, **summary},
        )
        put_dual_stream_operation(connection, match_id, request_id, fingerprint, payload, data)
        return SuccessResponse(data=data, revision=plan.to_revision)


def reconcile_canonical_revision(
    connection: sqlite3.Connection,
    *,
    match_id: str,
    matcher: IncrementalMatcher | None = None,
    actor: str = "automatic",
) -> SuccessResponse[dict]:
    """Execute final canonical convergence on all accumulated evidence.

    Reconciles all accumulated tracklet segments to the deterministic canonical
    recompute baseline, ensuring 100% partition congruence regardless of arrival order.
    """
    from app.evaluation.guard import assert_inference_isolation

    assert_inference_isolation("reconcile_canonical_revision")
    matcher = (
        matcher
        if matcher is not None
        else IncrementalMatcher(high_confidence_margin=0.0005, max_prototypes=12)
    )
    with transaction(connection):
        repo = Repository(connection)
        snapshot = read_dual_stream_snapshot(connection, match_id)
        rev = snapshot.revision
        segments = snapshot.segments
        features = snapshot.features
        if not segments:
            return SuccessResponse(
                data={"changed": False, "reason": "empty_segments"}, revision=rev
            )
        base_engine = SpatioTemporalConstraintEngine(segments=segments)
        canon_assign, canon_gallery, canon_decisions = recompute_canonical(
            segments, features, matcher, base_engine=base_engine
        )

        cpid_to_gpid = {}
        gallery_by_num = {
            p.confirmed_jersey_number: p.player_id
            for p in snapshot.players
            if p.confirmed_jersey_number is not None
        }
        gallery = {p.player_id: p for p in snapshot.players}
        claimed_gpids = set()

        for cpid, cplayer in canon_gallery.items():
            if cplayer.confirmed_jersey_number is not None:
                if cplayer.confirmed_jersey_number in gallery_by_num:
                    gpid = gallery_by_num[cplayer.confirmed_jersey_number]
                    cpid_to_gpid[cpid] = gpid
                    claimed_gpids.add(gpid)
                else:
                    gallery[cplayer.player_id] = cplayer
                    cpid_to_gpid[cpid] = cplayer.player_id
                    gallery_by_num[cplayer.confirmed_jersey_number] = cplayer.player_id
                    claimed_gpids.add(cplayer.player_id)

        for cpid, cplayer in canon_gallery.items():
            if (
                cplayer.confirmed_jersey_number is None
                and cpid in gallery
                and cpid not in claimed_gpids
            ):
                sp = gallery[cpid]
                if sp.confirmed_jersey_number is None:
                    cpid_to_gpid[cpid] = cpid
                    claimed_gpids.add(cpid)

        for cpid, cplayer in canon_gallery.items():
            if cpid in cpid_to_gpid:
                continue
            canon_members = {s for s, cp in canon_assign.items() if cp == cpid}
            canon_numbers = observed_numbers(cplayer, canon_assign, features)
            best_overlap, best_pid = 0, None
            for gplayer in snapshot.players:
                if (
                    gplayer.player_id in claimed_gpids
                    or gplayer.confirmed_jersey_number is not None
                ):
                    continue
                if _has_conflicting_numbers(
                    canon_numbers | observed_numbers(gplayer, snapshot.assignments, features)
                ):
                    continue
                g_members = {s for s, gp in snapshot.assignments.items() if gp == gplayer.player_id}
                overlap = len(canon_members & g_members)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_pid = gplayer.player_id
            if best_pid is not None and best_overlap > 0:
                cpid_to_gpid[cpid] = best_pid
                claimed_gpids.add(best_pid)
            else:
                gallery[cplayer.player_id] = cplayer
                cpid_to_gpid[cpid] = cplayer.player_id
                claimed_gpids.add(cplayer.player_id)

        assignments: dict[str, str] = {}
        decisions: dict[str, IncrementalDecision] = {}
        by_assigned = defaultdict(list)
        for sid in (s.segment_id for s in segments):
            target_cpid = canon_assign.get(sid)
            target_gpid = cpid_to_gpid.get(target_cpid) if target_cpid else None
            old_owner = snapshot.assignments.get(sid)
            if target_gpid is not None:
                assignments[sid] = target_gpid
                by_assigned[target_gpid].append(sid)
                dec = canon_decisions.get(sid)
                best_d = dec.best_distance if dec else 0.1
                margin = dec.competitive_margin if dec else 0.05
                ev = dict(dec.evidence) if dec else {}
                ev["reassessment"] = "canonical_convergence"
                if old_owner is not None:
                    ev["previous_player_id"] = old_owner
                decisions[sid] = IncrementalDecision(
                    sid,
                    DecisionOutcome.MATCHED_EXISTING,
                    target_gpid,
                    best_d,
                    margin,
                    ev,
                )
            else:
                decisions[sid] = IncrementalDecision(
                    sid,
                    DecisionOutcome.INSUFFICIENT,
                    None,
                    math.inf,
                    0.0,
                    {
                        "reassessment": "canonical_convergence",
                        "previous_player_id": old_owner,
                    },
                )

        existing_proto_map = {
            (p.player_id, proto.feature.segment_id): proto
            for p in snapshot.players
            for proto in p.prototypes
        }
        for pid, player in list(gallery.items()):
            assigned_sids = by_assigned[pid]
            if not assigned_sids:
                gallery[pid] = replace(player, prototypes=())
                continue
            ranked = sorted(
                assigned_sids,
                key=lambda sid: (-features[sid].appearance.quality_score, sid),
            )
            chosen = []
            used_videos = set()
            for sid in ranked:
                v = base_engine.segments_by_id[sid].video_id
                if v not in used_videos and len(chosen) < matcher.max_prototypes:
                    chosen.append(sid)
                    used_videos.add(v)
            for sid in ranked:
                if sid not in chosen and len(chosen) < matcher.max_prototypes:
                    chosen.append(sid)
            protos = tuple(
                existing_proto_map[(pid, sid)]
                if (pid, sid) in existing_proto_map
                else matcher._prototype(pid, features[sid], rev + 1)
                for sid in chosen
            )
            gallery[pid] = replace(player, prototypes=protos)

        final_players = tuple(gallery[pid] for pid in sorted(gallery))
        original_ids = {p.player_id for p in snapshot.players}
        for pid in list(gallery.keys()):
            if pid not in original_ids:
                gallery[pid] = replace(gallery[pid], created_at_revision=rev + 1)
        final_players = tuple(gallery[pid] for pid in sorted(gallery))
        final_players, assignments, decisions, repaired = _validate_dual_stream_commit(
            final_players,
            assignments,
            segments,
            features,
            decisions=decisions,
            revision=rev + 1,
            matcher=matcher,
        )
        by_id = {p.player_id: p for p in final_players}
        new_players = tuple(by_id[pid] for pid in sorted(by_id.keys() - original_ids))
        plan = IncrementalPlan(
            "canonical-convergence",
            rev,
            rev + 1,
            tuple(decisions[sid] for sid in sorted(decisions)),
            new_players,
            tuple(p for pl in final_players if pl.player_id in original_ids for p in pl.prototypes),
            tuple(
                d.segment_id
                for d in decisions.values()
                if d.outcome == DecisionOutcome.INSUFFICIENT
            ),
        )
        put_dual_stream_plan(connection, match_id, plan, players=final_players)
        summary = {
            outcome.value: sum(d.outcome == outcome for d in plan.decisions)
            for outcome in DecisionOutcome
        }
        summary["converged_segments"] = len(segments)
        if repaired:
            summary["conflict_repairs"] = sorted(repaired)
        data = {
            "changed": True,
            "committed_revision": plan.to_revision,
            "plan": to_json(plan),
            "summary": summary,
        }
        repo.advance_revision(
            match_id,
            rev,
            actor=actor,
            reason="canonical_convergence",
            changes={"batch_id": "canonical-convergence", **summary},
        )
        return SuccessResponse(data=data, revision=plan.to_revision)
