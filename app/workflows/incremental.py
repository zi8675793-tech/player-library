"""Workflow orchestration for open-set incremental batch updates with atomic commits."""

import hashlib
import json
import sqlite3
from collections.abc import Mapping

import numpy as np

from app.contracts.api import SuccessResponse, UpdateRequest
from app.contracts.evidence import EvidenceBatch, InlineFeature
from app.contracts.segmentation import SegmentationReport
from app.identity.fingerprints import evidence_fingerprint
from app.identity.incremental import plan_incremental_identities
from app.settings import Settings
from app.storage.database import transaction
from app.storage.discovery import put_discovery_report, read_discovery_reports
from app.storage.features import save_feature_value
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
    connection, primary, request, *, segmentation_report, feature_values, settings
):
    if settings.discovery.support_mode != "constrained_manifold":
        return primary
    secondary = reassess_unassigned(
        connection,
        request.model_copy(update={"expected_revision": primary.revision}),
        segmentation_report=segmentation_report,
        feature_values=feature_values,
        settings=settings,
    )
    second_changed = secondary.data["changed"]
    final_plan = secondary.data.get("plan") if second_changed else primary.data.get("plan")
    final_summary = (
        secondary.data.get("summary") if second_changed else primary.data.get("summary", {})
    )
    return SuccessResponse(
        data={
            **primary.data,
            "plan": final_plan,
            "summary": final_summary,
            "primary_summary": primary.data.get("summary", {}),
            "primary_committed_revision": primary.revision,
            "committed_revision": secondary.revision,
            "final_revision": secondary.revision,
            "changed": primary.data["changed"] or secondary.data["changed"],
            "reassessment": secondary.data,
        },
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
    primary = _update_incremental(
        connection,
        request,
        segmentation_report=segmentation_report,
        feature_values=feature_values,
        settings=settings,
    )
    return _complete_with_reassessment(
        connection,
        primary,
        request,
        segmentation_report=segmentation_report,
        feature_values=feature_values,
        settings=settings,
    )


def reassess_unassigned(
    connection: sqlite3.Connection,
    request: UpdateRequest,
    *,
    segmentation_report: SegmentationReport,
    feature_values: Mapping[tuple[str, int], InlineFeature] | None = None,
    settings: Settings,
) -> SuccessResponse[dict]:
    """Associate persisted unassigned evidence only; never import or admit templates."""
    return _update_incremental(
        connection,
        request,
        segmentation_report=segmentation_report,
        feature_values=feature_values,
        settings=settings,
        association_only=True,
    )


def _update_incremental(
    connection: sqlite3.Connection,
    request: UpdateRequest,
    *,
    segmentation_report: SegmentationReport,
    feature_values: Mapping[tuple[str, int], InlineFeature] | None = None,
    settings: Settings,
    association_only: bool = False,
) -> SuccessResponse[dict]:
    """Incrementally process a batch of machine evidence against a fixed snapshot.

    All evidence, assignments, templates and discovery support commit atomically.
    Idempotent on re-submission of the same batch.
    """
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
                "algorithm": "safe-matching-v2",
                "mode": "association_only" if association_only else "discovery",
                "snapshot_revision": gallery.revision,
            }
        )
        already_planned_report = any(
            r["report"].get("summary", {}).get("plan_key") == plan_key for r in existing_reports
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

        return SuccessResponse(
            data={
                "changed": True,
                "committed_revision": committed,
                "plan": plan.model_dump(mode="json"),
                "summary": plan.summary,
            },
            revision=committed,
        )
