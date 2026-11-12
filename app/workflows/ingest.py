"""Shared ingestion, fixed-snapshot matching and one atomic business commit."""

import hashlib
import sqlite3
from collections.abc import Mapping, Sequence

from app.contracts.api import (
    ImportRequest,
    InitializeRequest,
    MatchingPreviewRequest,
    SuccessResponse,
)
from app.contracts.common import SourceKey
from app.contracts.evidence import EvidenceBatch, InlineFeature
from app.contracts.identity import (
    Assignment,
    AssignmentEvidence,
    ManualTemplateReview,
    MatchingPolicy,
    Player,
    ReviewRecord,
    Segment,
    Template,
)
from app.contracts.workflows import WorkflowReport
from app.events.validate import validate_events
from app.identity import associate_segments, plan_gallery, prepare_evidence, resolve_associations
from app.identity.prepare import _validate_segments, _within, active_risks
from app.settings import Settings
from app.storage.database import connect, transaction
from app.storage.discovery import read_discovery_reports
from app.storage.features import save_feature_value
from app.storage.repositories import Repository, RevisionConflict, StorageConflict
from app.storage.snapshot import read_identity_snapshot
from app.storage.workflow_plans import put_workflow_plan
from app.workflows._consistency import (
    validate_evidence_context,
    validate_segment_frames,
    validate_templates,
)
from app.workflows._review_ops import current_evidence, stable_player_id, validate_constraints


def policy_for(space_id: str, policies: Sequence[MatchingPolicy], settings: Settings | None):
    matches = [policy for policy in policies if policy.feature_space_id == space_id]
    if len(matches) > 1:
        raise ValueError("duplicate policy for feature space")
    policy = (
        matches[0]
        if matches
        else (
            settings.matching.policy_for(space_id)
            if settings
            else MatchingPolicy(feature_space_id=space_id)
        )
    )
    if policy.auto_assign_enabled:
        raise ValueError("workflow v1 requires automatic confirmation disabled")
    return policy


def apply_gallery_changes(templates, plans, revision):
    """Interpret only executable gallery actions; waiting is never activation."""
    result = {template.template_id: template.model_copy(deep=True) for template in templates}
    for plan in plans:
        for change in plan.changes:
            if change.action == "deactivate":
                old = result.get(change.template_id)
                if old is not None:
                    result[old.template_id] = old.model_copy(
                        update={"active": False, "revision": revision}
                    )
            elif change.action == "activate":
                review = change.review
                if review is None or review.review_status != "approved":
                    raise ValueError("gallery activation requires explicit sample approval")
                key = f"{plan.match_id}\0{change.player_id}\0{change.feature_space_id}\0{change.sample_id}"
                template_id = (
                    change.template_id or "tpl_" + hashlib.sha256(key.encode()).hexdigest()[:24]
                )
                result[template_id] = Template(
                    template_id=template_id,
                    match_id=plan.match_id,
                    player_id=change.player_id,
                    sample_id=change.sample_id,
                    feature_space_id=change.feature_space_id,
                    review_status="approved",
                    active=True,
                    revision=revision,
                    approval=ReviewRecord(
                        actor=review.actor, reason=review.reason, revision=review.revision
                    ),
                )
    return list(result.values())


def _is_descendant_or_same(
    sample_segment_id: str | None, target_segment: Segment, segments: dict[str, Segment]
) -> bool:
    if sample_segment_id is None or sample_segment_id == target_segment.segment_id:
        return True
    origin = segments.get(sample_segment_id)
    if origin is not None and not origin.active:
        ancestor = target_segment
        seen = set()
        while ancestor.supersedes_segment_id and ancestor.segment_id not in seen:
            seen.add(ancestor.segment_id)
            if ancestor.supersedes_segment_id == origin.segment_id:
                return True
            ancestor = segments.get(ancestor.supersedes_segment_id)
            if ancestor is None:
                break
    return False


def _seed_state(repo, state, request, policies, settings):
    """Build a temporary R+1 human anchor view without reading mixed committed state."""
    match = request.batch.match_id
    revision = request.expected_revision + 1
    gallery = state.gallery.model_copy(deep=True)
    gallery.revision = revision
    created, clean, reviews, records = [], repo.clean_reviews(match), [], []
    players = {player.player_id: player for player in gallery.players}
    assignments = {assignment.segment_id: assignment for assignment in gallery.assignments}
    segments = {segment.segment_id: segment for segment in gallery.evidence.segments}
    samples = {sample.sample_id: sample for sample in gallery.evidence.samples}
    used_segments = set()
    for seed in request.seeds:
        player_id = stable_player_id(match, "seed", seed.request_id)
        if player_id in players:
            raise StorageConflict("initial anchors already exist; use update_library")
        if used_segments.intersection(seed.segment_ids):
            raise ValueError("seed identities cannot claim the same segment")
        used_segments.update(seed.segment_ids)
        supports = []
        for sample_id in seed.clean_sample_ids:
            sample = samples.get(sample_id)
            if sample is None:
                raise KeyError(f"unknown clean seed sample {sample_id!r}")
            located = [
                segments[sid]
                for sid in seed.segment_ids
                if sid in segments and _within(sample, segments[sid])
            ]
            if len(located) != 1 or not _is_descendant_or_same(
                sample.segment_id, located[0], segments
            ):
                raise ValueError("seed clean sample must belong to exactly one seed segment")
            if sample.tracking_quality and (
                sample.tracking_quality.usability == "unusable" or sample.tracking_quality.occluded
            ):
                raise ValueError("seed clean sample is unusable or occluded")
            supports.append(located[0].segment_id)
            clean[sample_id] = {
                "source_key": samples[sample_id].source.model_dump(
                    include=set(SourceKey.model_fields)
                ),
                "review_status": "approved",
                "actor": seed.actor,
                "reason": seed.reason,
                "revision": revision,
            }
        if set(supports) != set(seed.segment_ids):
            raise ValueError("each seed segment needs explicit clean sample support")
        player = Player(
            match_id=match,
            player_id=player_id,
            display_name=seed.display_name,
            team_id=seed.team_id,
            jersey_number=seed.jersey_number,
        )
        players[player_id] = player
        created.append(player_id)
        for segment_id in seed.segment_ids:
            segment = segments.get(segment_id)
            if segment is None or not segment.active:
                raise ValueError("seed requires an active known identity segment")
            if active_risks(gallery.evidence, segment, gallery.risk_reviews):
                raise ValueError("seed identity has unresolved risk; review it independently first")
            if (old := assignments.get(segment_id)) and (
                old.status == "confirmed"
                or (old.method in {"manual", "seed"} and old.unassigned_reason != "unknown")
            ):
                raise ValueError("seed cannot replace an existing human decision")
            assignments[segment_id] = Assignment(
                match_id=match,
                segment_id=segment_id,
                player_id=player_id,
                status="confirmed",
                method="seed",
                revision=revision,
                evidence=[
                    AssignmentEvidence(
                        kind="human_seed",
                        reference_id=seed.request_id,
                        description=f"{seed.actor}: {seed.reason}",
                    )
                ],
            )
        for sample_id in seed.template_sample_ids:
            if not samples[sample_id].features:
                raise ValueError("seed template sample has no feature declaration")
            for feature in samples[sample_id].features:
                reviews.append(
                    ManualTemplateReview(
                        match_id=match,
                        sample_id=sample_id,
                        player_id=player_id,
                        feature_space_id=feature.feature_space_id,
                        review_status="approved",
                        actor=seed.actor,
                        reason=seed.reason,
                        revision=revision,
                    )
                )
        records.append({"kind": "seed", **seed.model_dump(mode="json"), "player_id": player_id})
    gallery.players, gallery.assignments = list(players.values()), list(assignments.values())
    gallery.evidence = current_evidence(gallery.evidence, clean_reviews=clean)
    plans = []
    for space in gallery.evidence.feature_spaces:
        policy = policy_for(space.feature_space_id, policies, settings)
        prepared = prepare_evidence(
            EvidenceBatch(
                match_id=match,
                batch_id="seed-plan",
                producer="workflows",
                identity_evidence=gallery.evidence,
                assets=gallery.assets,
            ),
            policy,
            feature_values=state.feature_values,
            risk_reviews=gallery.risk_reviews,
        )
        scoped_reviews = [
            review for review in reviews if review.feature_space_id == space.feature_space_id
        ]
        plan = plan_gallery(
            prepared, gallery, policy, reviews=scoped_reviews, feature_values=state.feature_values
        )
        for review in scoped_reviews:
            activated = [
                change
                for change in plan.changes
                if change.sample_id == review.sample_id and change.action == "activate"
            ]
            if not activated:
                raise ValueError(
                    f"seed template {review.sample_id!r} is not eligible for activation"
                )
        # A seed identity cannot silently confirm a known unresolved mixed segment.
        for segment in prepared.segments:
            if segment.segment.segment_id in used_segments and segment.risks:
                raise ValueError("seed identity has unresolved risk; review it independently first")
        plans.append(plan)
    if reviews and not plans:
        raise ValueError("seed template requires a stored feature space")
    gallery.templates = apply_gallery_changes(gallery.templates, plans, revision)
    return gallery, created, clean, reviews, records, plans


def _calculate(
    gallery, feature_values, constraints, template_reviews, policies, settings, batch_id
):
    batch = EvidenceBatch(
        match_id=gallery.match_id,
        batch_id=batch_id,
        producer="workflows",
        assets=gallery.assets,
        identity_evidence=gallery.evidence,
    )
    spaces = [space.feature_space_id for space in gallery.evidence.feature_spaces] or [
        "missing-feature-space"
    ]
    associations, resolutions, plans, diagnostics = [], [], [], []
    assignments = {item.segment_id: item for item in gallery.assignments}
    computed = []
    for space_id in spaces:
        policy = policy_for(space_id, policies, settings)
        prepared = prepare_evidence(
            batch, policy, feature_values=feature_values, risk_reviews=gallery.risk_reviews
        )
        associated = associate_segments(prepared, gallery, policy, feature_values=feature_values)
        resolved = resolve_associations(
            prepared, gallery, associated, policy, constraints=constraints
        )
        computed.append((space_id, policy, prepared, associated, resolved))
    conflicts = set()
    tracks = {(track.clip_id, track.track_id) for track in gallery.evidence.tracks}
    for segment in gallery.evidence.segments:
        if not segment.active or (segment.clip_id, segment.track_id) not in tracks:
            continue
        old = assignments.get(segment.segment_id)
        if old is not None and (
            old.method in {"manual", "seed"}
            or any(e.kind == "automatic_discovery_experimental" for e in old.evidence)
        ):
            # The v1 human-gallery matcher has no evidence to erase an explicit
            # discovery prediction. Open-set rediscovery is a separate use case.
            continue
        proposals = [
            result.assignment
            for *_, resolved in computed
            for result in resolved
            if result.segment_id == segment.segment_id and result.assignment is not None
        ]
        owners = {proposal.player_id for proposal in proposals if proposal.player_id is not None}
        if len(owners) == 1:
            assignments[segment.segment_id] = proposals[0]
        else:
            if len(owners) > 1:
                conflicts.add(segment.segment_id)
            # An earlier proposed assignment cannot survive a newly pending/conflicting plan.
            assignments[segment.segment_id] = Assignment(
                match_id=gallery.match_id,
                segment_id=segment.segment_id,
                status="unassigned",
                method="automatic",
                unassigned_reason="ambiguous" if owners else "unknown",
                revision=gallery.revision,
                evidence=[
                    AssignmentEvidence(
                        kind="cross_space_conflicting_proposals" if owners else "pending_identity"
                    )
                ],
            )
    for space_id, policy, prepared, associated, resolved in computed:
        resolved = [
            result.model_copy(
                update={
                    "decision": "pending",
                    "player_id": None,
                    "assignment": assignments[result.segment_id],
                    "requires_review": True,
                    "automatic_confirmation_eligible": False,
                    "reasons": [*result.reasons, "cross_space_conflicting_proposals"],
                }
            )
            if result.segment_id in conflicts
            else result
            for result in resolved
        ]
        # Historical approvals belonging to the previous owner are history, not new approvals.
        usable = {
            item.sample.sample_id: segment.segment.segment_id
            for segment in prepared.segments
            for item in segment.samples
        }
        scoped_reviews = [
            review
            for review in template_reviews
            if review.feature_space_id == space_id
            and review.sample_id in usable
            and (owner := assignments.get(usable[review.sample_id]))
            and owner.status == "confirmed"
            and owner.player_id == review.player_id
            and review.revision
            >= max(
                (
                    template.revision
                    for template in gallery.templates
                    if (template.sample_id, template.player_id, template.feature_space_id)
                    == (review.sample_id, review.player_id, review.feature_space_id)
                ),
                default=0,
            )
        ]
        plan = plan_gallery(
            prepared,
            gallery,
            policy,
            final_assignments=list(assignments.values()),
            reviews=scoped_reviews,
            feature_values=feature_values,
        )
        for segment, association, resolution in zip(
            prepared.segments, associated, resolved, strict=True
        ):
            diagnostics.append(
                {
                    "segment_id": segment.segment.segment_id,
                    "feature_space_id": space_id,
                    "prepared": segment.model_dump(mode="json"),
                    "association": association.model_dump(mode="json"),
                    "resolution": resolution.model_dump(mode="json"),
                    "gallery_changes": [
                        change.model_dump(mode="json")
                        for change in plan.changes
                        if any(
                            item.sample.sample_id == change.sample_id for item in segment.samples
                        )
                    ],
                }
            )
        associations.extend(associated)
        resolutions.extend(resolved)
        plans.append(plan)
    # Under v1 policies no space can auto-confirm, so different-space candidates stay separate.
    return list(assignments.values()), associations, resolutions, plans, diagnostics


def ingest(
    connection: sqlite3.Connection,
    request: ImportRequest,
    *,
    feature_values: Mapping[tuple[str, int], InlineFeature] | None = None,
    policies: Sequence[MatchingPolicy] = (),
    settings: Settings | None = None,
    initialize: bool = False,
    events_only: bool = False,
) -> SuccessResponse[WorkflowReport]:
    """Ingest standard evidence and parsed values. No input paths are interpreted here."""
    request = type(request).model_validate(request.model_dump())
    batch, expected = request.batch, request.expected_revision
    if events_only and (batch.identity_evidence is not None or feature_values):
        raise ValueError("ingest_events accepts only event and clip evidence")
    if initialize and not isinstance(request, InitializeRequest):
        raise TypeError("initialize requires InitializeRequest")
    repo = Repository(connection)
    with transaction(connection):
        repo.ensure_match(batch.match_id)
        # Explicit request retry is accepted before the revision check; it performs no new write.
        payload = request.model_dump(mode="json", exclude={"expected_revision"})
        if feature_values:
            payload["feature_values"] = {
                f"{k[0]}:{k[1]}": v.model_dump(mode="json")
                for k, v in sorted(feature_values.items())
            }
        replay = None
        if initialize and request.request_id:
            replay = repo.get_operation(batch.match_id, "initialize", request.request_id, payload)
        if replay is not None:
            return SuccessResponse[WorkflowReport].model_validate(replay)
        if repo.revision(batch.match_id) != expected:
            raise RevisionConflict(f"stale match {batch.match_id!r}: expected revision {expected}")
        before = read_identity_snapshot(connection, batch.match_id)
        if initialize and before.gallery.players:
            # Even without a whole-request key, stable seed keys cannot produce a second library.
            same_seeds = request.seeds and all(
                repo.get_operation(
                    batch.match_id, "seed", seed.request_id, seed.model_dump(mode="json")
                )
                is not None
                for seed in request.seeds
            )
            if not same_seeds:
                raise StorageConflict("identity anchors already exist; use update_library")
        new_events = [event for event in batch.events or [] if not repo.is_recorded_event(event)]
        changed = repo.store_batch(batch)
        staged = read_identity_snapshot(
            connection, batch.match_id
        )  # All formal facts are still <= R.
        files = {item.feature_file_id: item for item in staged.gallery.evidence.feature_files}
        spaces = {item.feature_space_id: item for item in staged.gallery.evidence.feature_spaces}
        for (file_id, row), value in (feature_values or {}).items():
            file = files.get(file_id)
            if file is None:
                raise KeyError(f"unknown feature file {file_id!r}")
            policy = policy_for(file.feature_space_id, policies, settings)
            from app.identity.distance import validate_feature

            validate_feature(spaces[file.feature_space_id], value, policy)
            changed = (
                save_feature_value(
                    connection, batch.match_id, file, spaces[file.feature_space_id], row, value
                )
                or changed
            )
        state = read_identity_snapshot(connection, batch.match_id)  # No R+1 assignments yet.
        raw = state.gallery
        validate_evidence_context(repo, raw)
        validate_segment_frames(raw.evidence.segments)
        validate_events(
            new_events,
            match_id=batch.match_id,
            clips=repo.clips(batch.match_id),
            tracks=raw.evidence.tracks,
            segments=raw.evidence.segments,
        )
        if repo.check_segment_overlaps(batch.match_id) or repo.check_alias_cycles(batch.match_id):
            raise ValueError("formal segment overlap or player alias cycle")
        _validate_segments(
            raw.evidence.segments
        )  # Includes frame order even without features/ReID.
        gallery = raw.model_copy(deep=True)
        gallery.evidence = current_evidence(
            raw.evidence, clean_reviews=repo.clean_reviews(batch.match_id)
        )
        created, clean, reviews, records, seed_plans = [], {}, [], [], []
        if initialize and request.seeds and not before.gallery.players:
            gallery, created, clean, reviews, records, seed_plans = _seed_state(
                repo, state, request, policies, settings
            )
            changed = True
        if initialize and before.gallery.players and changed:
            raise StorageConflict(
                "initialization retry cannot import new evidence; use update_library"
            )
        assigned, associated, resolved, plans, diagnostics = gallery.assignments, [], [], [], []
        identity_input = (
            batch.identity_evidence is not None or bool(feature_values) or bool(created)
        )
        if changed and identity_input and not events_only:
            assigned, associated, resolved, plans, diagnostics = _calculate(
                gallery,
                state.feature_values,
                state.constraints,
                [*state.template_reviews, *reviews],
                policies,
                settings,
                batch.batch_id,
            )
        final = gallery.model_copy(
            deep=True,
            update={
                "revision": expected + 1,
                "assignments": assigned,
                "templates": apply_gallery_changes(gallery.templates, plans, expected + 1),
            },
        )
        validate_constraints(final, state.constraints)
        validate_templates(
            final,
            state.feature_values,
            lambda space_id: policy_for(space_id, policies, settings),
            discovery_reports=read_discovery_reports(connection, batch.match_id),
        )
        # Final writes happen after computation; never call the formal snapshot reader in this interval.
        for player in final.players:
            changed = repo.put_player(player) or changed
        for assignment in final.assignments:
            changed = (
                repo.put_assignment(assignment, committed_revision=expected + 1, replace=True)
                or changed
            )
        for template in final.templates:
            changed = (
                repo.put_template(template, committed_revision=expected + 1, replace=True)
                or changed
            )
        for sample_id, review in clean.items():
            changed = repo.put_clean_review(batch.match_id, sample_id, review) or changed
        for review in reviews:
            changed = repo.put_template_review(review) or changed
        if records:
            repo.record_review(batch.match_id, expected + 1, request.actor, records)
            changed = True
        for diagnostic in diagnostics:
            put_workflow_plan(
                connection,
                batch.match_id,
                diagnostic["segment_id"],
                diagnostic["feature_space_id"],
                expected + 1,
                diagnostic,
            )
        if repo.revision(batch.match_id) != expected:
            raise RevisionConflict("revision changed during plan commit")
        report = WorkflowReport(
            match_id=batch.match_id,
            batch_id=batch.batch_id,
            inserted_count=int(changed),
            duplicate_count=int(not changed),
            review_count=sum(result.requires_review for result in resolved),
            changed=changed,
            initialized=bool(final.players),
            created_player_ids=created,
            based_revision=expected,
            matching_revision=gallery.revision,
            associations=associated,
            resolutions=resolved,
            gallery_plans=[*seed_plans, *plans],
            warnings=[warning.code for warning in batch.warnings],
        )
        revision = expected
        if changed:
            revision = repo.advance_revision(
                batch.match_id,
                expected,
                actor=request.actor if initialize else batch.producer,
                reason=request.reason if initialize else "ingest",
                changes={
                    "batch_id": batch.batch_id,
                    "created_players": created,
                    "segments": [item.segment_id for item in assigned],
                    "features": len(feature_values or {}),
                    "events": [event.event_id for event in batch.events or []],
                },
            )
        response = SuccessResponse(data=report, revision=revision)
        if initialize:
            for seed in request.seeds:
                if not repo.get_operation(
                    batch.match_id, "seed", seed.request_id, seed.model_dump(mode="json")
                ):
                    repo.put_operation(
                        batch.match_id,
                        "seed",
                        seed.request_id,
                        seed.model_dump(mode="json"),
                        {"player_id": stable_player_id(batch.match_id, "seed", seed.request_id)},
                    )
            if request.request_id:
                repo.put_operation(
                    batch.match_id,
                    "initialize",
                    request.request_id,
                    payload,
                    response.model_dump(mode="json"),
                )
        return response


def preview_matching(
    connection: sqlite3.Connection,
    request: MatchingPreviewRequest,
    *,
    feature_values=None,
    policies=(),
    settings=None,
) -> SuccessResponse[WorkflowReport]:
    """Compute against a consistent private SQLite copy; the source connection is read-only."""
    if connection.in_transaction:
        raise RuntimeError("preview requires a connection outside a transaction")
    clone = connect(":memory:")
    try:
        connection.backup(clone)
        return ingest(
            clone, request, feature_values=feature_values, policies=policies, settings=settings
        )
    finally:
        clone.close()
