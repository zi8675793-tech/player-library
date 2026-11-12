"""Cold-start experimental identity discovery with durable evidence and stable IDs."""

import hashlib
import json
from uuid import NAMESPACE_URL, uuid5

import numpy as np

from app.contracts.api import SuccessResponse
from app.contracts.evidence import EvidenceBatch
from app.contracts.identity import Assignment, AssignmentEvidence, Player, Template
from app.identity.discovery import discover_identities
from app.storage.database import transaction
from app.storage.discovery import put_discovery_report, read_discovery_reports, sample_binding
from app.storage.repositories import Repository, RevisionConflict, StorageConflict
from app.storage.snapshot import read_identity_snapshot


def _unordered_records(model):
    """Entity list ordering is not a business change or an identity signal."""
    payload = model.model_dump(mode="json")
    for sample in payload.get("samples", []):
        for feature in sample["features"]:
            if feature["kind"] == "inline":
                feature["values"] = np.asarray(feature["values"], dtype=np.float32).tolist()
                if feature["visibility"] is not None:
                    feature["visibility"] = np.asarray(
                        feature["visibility"], dtype=np.float32
                    ).tolist()
    return {
        key: sorted(json.dumps(item, sort_keys=True) for item in value)
        if isinstance(value, list)
        else value
        for key, value in payload.items()
    }


def _finish_reassessment(connection, batch, segmentation_report, feature_values, settings, primary):
    if settings.discovery.support_mode != "constrained_manifold":
        return primary
    from app.contracts.api import UpdateRequest
    from app.workflows.incremental import update_incremental

    secondary = update_incremental(
        connection,
        UpdateRequest(batch=batch, expected_revision=primary.revision),
        segmentation_report=segmentation_report,
        feature_values=feature_values,
        settings=settings,
    )
    return SuccessResponse(
        data={
            **primary.data,
            "committed_revision": secondary.revision,
            "reassessment": secondary.data,
            "changed": primary.data.get("changed", False) or secondary.data.get("changed", False),
        },
        revision=secondary.revision,
    )


def _discover_core(
    connection,
    batch: EvidenceBatch,
    *,
    segmentation_report,
    feature_values,
    settings,
    expected_revision: int,
) -> SuccessResponse[dict]:
    """Discover once from current machine evidence; never replace a human identity.

    Proposed automatic assignments are experimental predictions with real persistent
    player IDs. They are not manual confirmations and do not enter official events.
    All support comes from the fixed raw evidence plan, never newly admitted templates.
    """
    if settings.input.adapter != "machine":
        raise ValueError("automatic discovery requires machine input")
    report = discover_identities(
        batch,
        feature_values,
        segmentation_report=segmentation_report,
        policy=settings.discovery,
    )
    repo = Repository(connection)
    with transaction(connection):
        revision = repo.revision(batch.match_id)
        existing = read_discovery_reports(connection, batch.match_id)
        for old in existing:
            if old["report"]["report_id"] == report.report_id:
                if old["report"] != report.model_dump(mode="json"):
                    raise StorageConflict("discovery report ID has different evidence")
                return SuccessResponse(
                    data={**old, "changed": False, "committed_revision": revision},
                    revision=revision,
                )
        if revision != expected_revision:
            raise RevisionConflict("stale automatic discovery request")
        state = read_identity_snapshot(connection, batch.match_id)
        if (
            existing
            or state.gallery.players
            or state.gallery.templates
            or state.constraints
            or state.template_reviews
            or state.gallery.risk_reviews
            or repo.clean_reviews(batch.match_id)
            or repo.review_records(batch.match_id)
            or any(a.method in {"manual", "seed"} for a in state.gallery.assignments)
        ):
            raise StorageConflict(
                "cold-start discovery requires an unreviewed empty identity library"
            )
        if batch.identity_evidence is None or _unordered_records(
            state.gallery.evidence
        ) != _unordered_records(batch.identity_evidence):
            raise StorageConflict("discovery must use the exact currently ingested evidence")
        if state.feature_values != feature_values:
            raise StorageConflict("discovery feature rows must equal the persisted machine input")
        player_map = {
            group.group_id: "pl_"
            + uuid5(
                NAMESPACE_URL, f"{batch.match_id}\0auto-discovery\0{group.anchor_segment_id}"
            ).hex
            for group in report.groups
        }
        committed = revision + 1
        for group in report.groups:
            player_id = player_map[group.group_id]
            repo.put_player(Player(match_id=batch.match_id, player_id=player_id))
            evidence = [
                AssignmentEvidence(
                    kind="automatic_discovery_experimental", reference_id=report.report_id
                ),
                *[
                    AssignmentEvidence(kind="automatic_pair_support", reference_id=pair_id)
                    for pair_id in group.support_pair_ids
                ],
            ]
            for segment_id in group.member_segment_ids:
                repo.put_assignment(
                    Assignment(
                        match_id=batch.match_id,
                        segment_id=segment_id,
                        player_id=player_id,
                        status="proposed",
                        method="automatic",
                        evidence=evidence,
                        revision=committed,
                    ),
                    committed_revision=committed,
                    replace=True,
                )
            for support in group.templates:
                key = f"{batch.match_id}\0{player_id}\0{support.feature_space_id}\0{support.sample_id}"
                repo.put_template(
                    Template(
                        template_id="tpl_auto_" + hashlib.sha256(key.encode()).hexdigest()[:24],
                        match_id=batch.match_id,
                        player_id=player_id,
                        sample_id=support.sample_id,
                        feature_space_id=support.feature_space_id,
                        review_status="unreviewed",
                        active=True,
                        admission="automatic_experimental",
                        automatic_support_id=report.report_id,
                        revision=committed,
                        prototype_id=support.prototype_id,
                        time_bucket=support.time_bucket,
                        reliable_parts=support.reliable_parts,
                        quality_score=support.quality_score,
                        risks=support.risks,
                        source_component=support.source_component,
                    ),
                    committed_revision=committed,
                )
        support_ids = {
            sid
            for group in report.groups
            for support in group.templates
            for sid in [support.sample_id, *support.witness_sample_ids]
        }
        support_pair_ids = {
            pair_id
            for group in report.groups
            for support in group.templates
            for pair_id in support.support_pair_ids
        }
        support_ids.update(
            sample_id
            for pair in report.pairs
            if pair.pair_id in support_pair_ids
            for comparison in pair.comparisons
            for sample_id in (comparison.left_sample_id, comparison.right_sample_id)
        )
        bindings = {
            sample.sample_id: sample_binding(sample, batch.assets)
            for sample in batch.identity_evidence.samples
            if sample.sample_id in support_ids
        }
        put_discovery_report(connection, report, player_map, committed, bindings)
        repo.advance_revision(
            batch.match_id,
            revision,
            actor="identity.discovery",
            reason="automatic_identity_discovery_experimental",
            changes={
                "report_id": report.report_id,
                "policy_fingerprint": report.policy_fingerprint,
                "created_players": list(player_map.values()),
                "calibration": "unvalidated",
                "human_approvals": 0,
            },
        )
        primary = SuccessResponse(
            data={
                "report": report.model_dump(mode="json"),
                "player_ids_by_group": player_map,
                "committed_revision": committed,
                "sample_sources": bindings,
                "changed": True,
            },
            revision=committed,
        )
    return primary


def discover_library(
    connection, batch, *, segmentation_report, feature_values, settings, expected_revision
):
    primary = _discover_core(
        connection,
        batch,
        segmentation_report=segmentation_report,
        feature_values=feature_values,
        settings=settings,
        expected_revision=expected_revision,
    )
    return _finish_reassessment(
        connection, batch, segmentation_report, feature_values, settings, primary
    )
