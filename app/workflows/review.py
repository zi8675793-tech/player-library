"""Atomic human identity, risk, segment and independent template review use case."""

import hashlib
import json
import sqlite3
from collections.abc import Sequence

from app.contracts.api import (
    ConfirmAssignmentCommand,
    CreatePlayerCommand,
    MergePlayersCommand,
    RestoreRevisionCommand,
    ReviewRequest,
    ReviewRiskCommand,
    ReviewSampleCommand,
    ReviewTemplateCommand,
    SetAssignmentCommand,
    SetConstraintCommand,
    SetPlayerRoleCommand,
    SplitSegmentCommand,
    SuccessResponse,
)
from app.contracts.evidence import EvidenceBatch, GallerySnapshot
from app.contracts.identity import (
    Assignment,
    AssignmentEvidence,
    AssignmentMethod,
    AssignmentStatus,
    GalleryAction,
    ManualConstraint,
    ManualTemplateReview,
    MatchingPolicy,
    Player,
    ReviewRecord,
    RiskReview,
    Template,
)
from app.identity.gallery import plan_gallery
from app.identity.prepare import _within, active_risks, prepare_evidence
from app.settings import Settings
from app.storage.database import transaction
from app.storage.discovery import read_discovery_reports
from app.storage.repositories import Repository, RevisionConflict
from app.storage.snapshot import read_identity_snapshot
from app.storage.workflow_plans import put_workflow_plan, read_workflow_plans
from app.workflows._consistency import (
    validate_evidence_context,
    validate_segment_frames,
    validate_templates,
)
from app.workflows._review_ops import current_evidence, validate_constraints


class UnsupportedReviewCommand(NotImplementedError):
    """The command is recognized but has no implemented business use case in v1."""


def _request_key(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _same_review(left, right) -> bool:
    return left is not None and left.model_dump(exclude={"revision"}) == right.model_dump(
        exclude={"revision"}
    )


def _assignment_fact(assignment: Assignment) -> dict:
    value = assignment.model_dump(exclude={"revision"})
    for evidence in value["evidence"]:
        if evidence["kind"] in {"manual_review", "manual_merge"}:
            evidence["reference_id"] = None
    return value


def _template_id(match_id: str, sample_id: str, player_id: str, space_id: str) -> str:
    key = f"{match_id}\0{player_id}\0{space_id}\0{sample_id}"
    return "tpl_" + hashlib.sha256(key.encode()).hexdigest()[:24]


def apply_review(
    connection: sqlite3.Connection,
    request: ReviewRequest,
    *,
    policies: Sequence[MatchingPolicy] = (),
    settings: Settings | None = None,
) -> SuccessResponse[dict]:
    """Execute supported typed commands in one revision; matching retries are no-ops.

    No official snapshot is read between writing next-revision records and advancing
    the match revision. All command and gallery calculations use temporary copies.
    """
    request = ReviewRequest.model_validate(request.model_dump())
    repo = Repository(connection)
    payload = request.model_dump(mode="json", exclude={"expected_revision", "request_id"})
    operation_id = request.request_id or _request_key(payload)
    with transaction(connection):
        previous = repo.get_operation(request.match_id, "review", operation_id, payload)
        if previous is not None:
            return SuccessResponse[dict].model_validate(previous)
        if repo.revision(request.match_id) != request.expected_revision:
            raise RevisionConflict("stale review expected_revision")
        revision = request.expected_revision + 1

        restore_commands = [c for c in request.commands if isinstance(c, RestoreRevisionCommand)]
        if restore_commands:
            if len(request.commands) > 1:
                raise ValueError(
                    "restore_revision cannot be combined with other commands in a single request"
                )
            command = restore_commands[0]
            if command.target_revision >= request.expected_revision:
                raise ValueError(
                    f"target revision {command.target_revision} must be strictly less than current revision {request.expected_revision}"
                )
            known_revisions = {row.revision for row in repo.revisions(request.match_id)}
            if command.target_revision not in known_revisions:
                raise ValueError(
                    f"target revision {command.target_revision} does not exist in revisions for match {request.match_id}"
                )

            from app.workflows._restore import restore_state

            restore_state(
                repo, request.match_id, command.target_revision, revision, policies, settings
            )

            repo.record_review(
                request.match_id,
                revision,
                request.actor,
                [command.model_dump(mode="json") for command in request.commands],
            )
            repo.advance_revision(
                request.match_id,
                request.expected_revision,
                actor=request.actor,
                reason=f"restore_revision:{command.target_revision}: {command.reason}",
                changes={
                    "commands": [c.kind for c in request.commands],
                    "target_revision": command.target_revision,
                    "restored": True,
                },
            )

            response = SuccessResponse[dict](
                revision=revision,
                data={
                    "match_id": request.match_id,
                    "target_revision": command.target_revision,
                    "restored_revision": revision,
                    "player_ids": [],
                    "gallery_plans": [],
                },
            )
            repo.put_operation(
                request.match_id,
                "review",
                operation_id,
                payload,
                response.model_dump(mode="json"),
            )
            return response

        state = read_identity_snapshot(connection, request.match_id)
        original = state.gallery
        existing_players = {
            item.player_id: item.model_copy(deep=True)
            for item in (
                Player.model_validate_json(row[0])
                for row in connection.execute(
                    "SELECT json FROM players WHERE match_id=? ORDER BY player_id",
                    (request.match_id,),
                )
            )
        }
        players = existing_players
        segments = {
            item.segment_id: item.model_copy(deep=True) for item in original.evidence.segments
        }
        assignments = {item.segment_id: item.model_copy(deep=True) for item in original.assignments}
        templates = {item.template_id: item.model_copy(deep=True) for item in original.templates}
        reviews = {
            (item.sample_id, item.player_id, item.feature_space_id): item
            for item in state.template_reviews
        }
        constraints = {item.constraint_id: item for item in state.constraints}
        risk_reviews = {(item.risk_id, item.segment_id): item for item in original.risk_reviews}
        samples = {sample.sample_id: sample for sample in original.evidence.samples}
        policy_map = {policy.feature_space_id: policy for policy in policies}
        changed_players, changed_segments, changed_assignments = set(), set(), set()
        new_reviews, new_constraints, new_risk_reviews = [], [], []
        derived_risks = {}
        invalidated_segments = set()
        invalidated_templates = set()

        def require_segment(segment_id):
            segment = segments.get(segment_id)
            if segment is None or not segment.active:
                raise ValueError("review requires a current active identity segment")
            return segment

        def set_assignment(segment_id, player_id, reason, evidence, unassigned_reason=None):
            require_segment(segment_id)
            if player_id is not None and (
                player_id not in players or players[player_id].alias_of is not None
            ):
                raise ValueError("assignment player is missing or merged in this match")
            old = assignments.get(segment_id)
            new = Assignment(
                match_id=request.match_id,
                segment_id=segment_id,
                player_id=player_id,
                status="confirmed" if player_id is not None else "unassigned",
                method="manual",
                unassigned_reason=(unassigned_reason or "unknown") if player_id is None else None,
                revision=revision,
                evidence=[
                    *evidence,
                    AssignmentEvidence(
                        kind="manual_review",
                        reference_id=operation_id,
                        description=f"{request.actor}: {reason}",
                    ),
                ],
            )
            if old is None or _assignment_fact(old) != _assignment_fact(new):
                changed_assignments.add(segment_id)
            else:
                new = old
            if old and (old.player_id != new.player_id or old.status != new.status):
                invalidated_segments.add(segment_id)
            assignments[segment_id] = new

        for command in request.commands:
            if isinstance(command, CreatePlayerCommand):
                if command.player.alias_of is not None:
                    raise UnsupportedReviewCommand(
                        "creating aliases requires the unsupported merge use case"
                    )
                old = players.get(command.player.player_id)
                if old is not None and old != command.player:
                    raise ValueError("create_player cannot overwrite an existing player")
                if old is None:
                    players[command.player.player_id] = command.player
                    changed_players.add(command.player.player_id)
                for segment_id in command.segment_ids:
                    set_assignment(
                        segment_id, command.player.player_id, command.reason, command.evidence
                    )
            elif isinstance(command, (ConfirmAssignmentCommand, SetAssignmentCommand)):
                set_assignment(
                    command.segment_id,
                    command.player_id,
                    command.reason,
                    command.evidence,
                    getattr(command, "unassigned_reason", None),
                )
            elif isinstance(command, SplitSegmentCommand):
                old = require_segment(command.segment_id)
                for segment in command.segments:
                    if segment.segment_id in segments:
                        raise ValueError("replacement segment_id already exists")
                    if (
                        not segment.active
                        or (segment.clip_id, segment.track_id) != (old.clip_id, old.track_id)
                        or segment.start_s < old.start_s
                        or segment.end_s > old.end_s
                    ):
                        raise ValueError(
                            "split replacement must be active and within the original segment range"
                        )
                    if old.frame_start is not None and (
                        segment.frame_start is None
                        or segment.frame_start < old.frame_start
                        or segment.frame_end_exclusive > old.frame_end_exclusive
                    ):
                        raise ValueError(
                            "split frame ranges must be supplied within the original frame range"
                        )
                segments[old.segment_id] = old.model_copy(update={"active": False})
                changed_segments.add(old.segment_id)
                invalidated_segments.add(old.segment_id)
                for segment in command.segments:
                    segments[segment.segment_id] = segment
                    changed_segments.add(segment.segment_id)
                    assignments[segment.segment_id] = Assignment(
                        match_id=request.match_id,
                        segment_id=segment.segment_id,
                        method="manual",
                        revision=revision,
                        unassigned_reason="unknown",
                        evidence=[
                            AssignmentEvidence(
                                kind="manual_split",
                                reference_id=old.segment_id,
                                description=command.reason,
                            )
                        ],
                    )
                    changed_assignments.add(segment.segment_id)
            elif isinstance(command, (ReviewSampleCommand, ReviewTemplateCommand)):
                if isinstance(command, ReviewTemplateCommand):
                    template = templates.get(command.template_id)
                    if template is None:
                        raise ValueError("unknown template_id in this match")
                    sample_id, player_id, feature_space_id = (
                        template.sample_id,
                        template.player_id,
                        template.feature_space_id,
                    )
                    template_id = template.template_id
                else:
                    sample_id, player_id, feature_space_id = (
                        command.sample_id,
                        command.player_id,
                        command.feature_space_id,
                    )
                    template_id = None
                review = ManualTemplateReview(
                    match_id=request.match_id,
                    sample_id=sample_id,
                    player_id=player_id,
                    feature_space_id=feature_space_id,
                    template_id=template_id,
                    review_status=command.review_status,
                    actor=request.actor,
                    reason=command.reason,
                    revision=revision,
                )
                key = (sample_id, player_id, feature_space_id)
                if not _same_review(reviews.get(key), review):
                    reviews[key] = review
                    new_reviews.append(review)
            elif isinstance(command, ReviewRiskCommand):
                segment = require_segment(command.segment_id)
                known_risks = {risk.risk_id for risk in original.evidence.risks}
                if command.risk_id not in known_risks:
                    risk = next(
                        (
                            risk
                            for risk in active_risks(original.evidence, segment)
                            if risk.risk_id == command.risk_id
                        ),
                        None,
                    )
                    if risk is None:
                        raise ValueError("risk review requires a current observed risk")
                    derived_risks[risk.risk_id] = risk
                by_sample = {sample.sample_id: sample for sample in original.evidence.samples}
                if any(
                    sample_id not in by_sample or not _within(by_sample[sample_id], segment)
                    for sample_id in command.clean_sample_ids
                ):
                    raise ValueError("risk clean support must be inside the reviewed segment")
                review = RiskReview(
                    match_id=request.match_id,
                    risk_id=command.risk_id,
                    segment_id=command.segment_id,
                    resolution=command.resolution,
                    clean_sample_ids=command.clean_sample_ids,
                    actor=request.actor,
                    reason=command.reason,
                    revision=revision,
                )
                key = (review.risk_id, review.segment_id)
                if not _same_review(risk_reviews.get(key), review):
                    risk_reviews[key] = review
                    new_risk_reviews.append(review)
            elif isinstance(command, SetConstraintCommand):
                require_segment(command.segment_id)
                constraint = ManualConstraint(
                    match_id=request.match_id,
                    constraint_id=command.constraint_id,
                    segment_id=command.segment_id,
                    kind=command.constraint_kind,
                    player_id=command.player_id,
                    related_segment_id=command.related_segment_id,
                    actor=request.actor,
                    reason=command.reason,
                    revision=revision,
                )
                if not _same_review(constraints.get(constraint.constraint_id), constraint):
                    constraints[constraint.constraint_id] = constraint
                    new_constraints.append(constraint)
            elif isinstance(command, MergePlayersCommand):
                if command.player_id not in players:
                    raise ValueError(f"target player {command.player_id!r} not found in match")
                for alias_id in command.alias_player_ids:
                    if alias_id not in players:
                        raise ValueError(f"alias player {alias_id!r} not found in match")
                    alias_player = players[alias_id].model_copy(
                        update={"alias_of": command.player_id}
                    )
                    players[alias_id] = alias_player
                    changed_players.add(alias_id)

                for segment_id, assignment in list(assignments.items()):
                    if assignment.player_id in command.alias_player_ids:
                        new_assignment = assignment.model_copy(
                            update={
                                "player_id": command.player_id,
                                "method": AssignmentMethod.MANUAL,
                                "revision": revision,
                                "evidence": [
                                    *assignment.evidence,
                                    AssignmentEvidence(
                                        kind="manual_review",
                                        reference_id=operation_id,
                                        description=f"{request.actor}: {command.reason}",
                                    ),
                                ],
                            }
                        )
                        assignments[segment_id] = new_assignment
                        changed_assignments.add(segment_id)

                migrated_template_ids = set()
                for template in templates.values():
                    if template.player_id in command.alias_player_ids:
                        template.player_id = command.player_id
                        migrated_template_ids.add(template.template_id)
                        invalidated_templates.add(template.template_id)

                spaces = {
                    t.feature_space_id
                    for t in templates.values()
                    if t.player_id == command.player_id
                }
                for space_id in sorted(spaces):
                    policy = policy_map.get(space_id) or (
                        settings.matching.policy_for(space_id)
                        if settings
                        else MatchingPolicy(feature_space_id=space_id)
                    )
                    active_approved = [
                        t
                        for t in templates.values()
                        if t.player_id == command.player_id
                        and t.feature_space_id == space_id
                        and t.active
                        and t.review_status == "approved"
                    ]
                    if len(active_approved) > policy.max_templates_per_player:

                        def template_priority(t: Template):
                            s = samples.get(t.sample_id)
                            score = (
                                s.tracking_quality.score
                                if s and s.tracking_quality and s.tracking_quality.score is not None
                                else 0.0
                            )
                            time_s = s.time_s if s else 0.0
                            return (t.revision, score, time_s, t.template_id)

                        ranked = sorted(active_approved, key=template_priority, reverse=True)
                        discard = ranked[policy.max_templates_per_player :]
                        for t in discard:
                            t.active = False
                            invalidated_templates.add(t.template_id)

                for t_id in migrated_template_ids:
                    templates[t_id].revision = revision
            elif isinstance(command, SetPlayerRoleCommand):
                if command.player_id not in players:
                    raise ValueError(f"unknown player_id {command.player_id}")
                old_player = players[command.player_id]
                if old_player.role != command.role:
                    players[command.player_id] = old_player.model_copy(
                        update={"role": command.role}
                    )
                    changed_players.add(command.player_id)
            else:
                raise UnsupportedReviewCommand(f"unsupported review command: {command.kind}")

        if len(
            {(item.sample_id, item.player_id, item.feature_space_id) for item in new_reviews}
        ) != len(new_reviews):
            raise ValueError("duplicate template review target in one request")
        if len({(item.risk_id, item.segment_id) for item in new_risk_reviews}) != len(
            new_risk_reviews
        ):
            raise ValueError("duplicate risk review target in one request")
        if len({item.constraint_id for item in new_constraints}) != len(new_constraints):
            raise ValueError("duplicate constraint target in one request")

        for player_id in sorted(changed_players):
            repo.put_player(players[player_id], update=True)
        if repo.check_alias_cycles(request.match_id):
            raise ValueError("player alias cycle")

        raw_evidence = original.evidence.model_copy(
            update={"segments": list(segments.values())}, deep=True
        )
        validate_segment_frames(raw_evidence.segments)
        final_evidence = current_evidence(
            raw_evidence, clean_reviews=repo.clean_reviews(request.match_id)
        )
        # Retire old ownership before a fresh independent approval can qualify again.
        automatic_reports = {
            item["report"]["report_id"]: item["report"]
            for item in read_discovery_reports(connection, request.match_id)
        }
        for template in templates.values():
            sample = samples.get(template.sample_id)
            old_scope = sample.segment_id if sample else None
            if sample and old_scope is None:
                candidates = [
                    segment.segment_id
                    for segment in original.evidence.segments
                    if segment.active and _within(sample, segment)
                ]
                old_scope = candidates[0] if len(candidates) == 1 else None
            if template.active and old_scope in invalidated_segments:
                template.active = False
                invalidated_templates.add(template.template_id)
            if template.active and template.admission == "automatic_experimental":
                report = automatic_reports.get(template.automatic_support_id)
                support = next(
                    (
                        support
                        for group in (report["groups"] if report else [])
                        for support in group["templates"]
                        if support["sample_id"] == template.sample_id
                        and support["feature_space_id"] == template.feature_space_id
                    ),
                    None,
                )
                witness_scopes = {
                    samples[sid].segment_id
                    for sid in (support["witness_sample_ids"] if support else [])
                    if sid in samples
                }
                if support is not None:
                    witness_scopes.update(
                        negative_segment_id
                        for pair in report["pairs"]
                        if pair["pair_id"] in support["support_pair_ids"]
                        for direction in pair["directions"]
                        for negative_segment_id in direction.get("competitor_segment_ids", [])
                    )
                if invalidated_segments.intersection(witness_scopes):
                    template.active = False
                    invalidated_templates.add(template.template_id)

        temporary = GallerySnapshot(
            match_id=request.match_id,
            revision=revision,
            players=list(players.values()),
            assignments=list(assignments.values()),
            templates=list(templates.values()),
            evidence=final_evidence,
            assets=original.assets,
            risk_reviews=list(risk_reviews.values()),
        )
        validate_evidence_context(repo, temporary.model_copy(update={"evidence": raw_evidence}))
        validate_constraints(temporary, list(constraints.values()))
        policy_map = {policy.feature_space_id: policy for policy in policies}
        plans = []
        prepared_by_space = {}
        for space in final_evidence.feature_spaces:
            policy = policy_map.get(space.feature_space_id) or (
                settings.matching.policy_for(space.feature_space_id)
                if settings
                else MatchingPolicy(feature_space_id=space.feature_space_id)
            )
            prepared = prepare_evidence(
                EvidenceBatch(
                    match_id=request.match_id,
                    batch_id="review-plan",
                    producer="workflows.review",
                    identity_evidence=final_evidence,
                    assets=original.assets,
                ),
                policy,
                feature_values=state.feature_values,
                risk_reviews=temporary.risk_reviews,
            )
            prepared_by_space[space.feature_space_id] = prepared
            located = {
                item.sample.sample_id: segment
                for segment in prepared.segments
                for item in segment.samples
            }
            relevant = []
            for review in reviews.values():
                if (
                    review.feature_space_id != space.feature_space_id
                    or review.sample_id not in located
                ):
                    continue
                matching_templates = [
                    template
                    for template in templates.values()
                    if (template.sample_id, template.player_id, template.feature_space_id)
                    == (review.sample_id, review.player_id, review.feature_space_id)
                ]
                if review not in new_reviews and any(
                    template.revision > review.revision for template in matching_templates
                ):
                    continue
                segment = located[review.sample_id].segment
                owner = assignments.get(segment.segment_id)
                if review.review_status == "approved" and (
                    owner is None
                    or owner.status != AssignmentStatus.CONFIRMED
                    or owner.player_id != review.player_id
                ):
                    if review in new_reviews:
                        raise ValueError(
                            "template approval requires the sample's current confirmed player"
                        )
                    continue
                relevant.append(review)
            plan = plan_gallery(
                prepared,
                temporary,
                policy,
                final_assignments=list(assignments.values()),
                reviews=relevant,
                feature_values=state.feature_values,
            )
            plans.append(plan)
            for change in plan.changes:
                if change.action == GalleryAction.DEACTIVATE:
                    template = templates[change.template_id]
                    template.active = False
                    if change.review is not None:
                        template.review_status = change.review.review_status
                    invalidated_templates.add(template.template_id)
                elif change.action == GalleryAction.ACTIVATE:
                    template_id = change.template_id or _template_id(
                        request.match_id,
                        change.sample_id,
                        change.player_id,
                        change.feature_space_id,
                    )
                    templates[template_id] = Template(
                        template_id=template_id,
                        match_id=request.match_id,
                        sample_id=change.sample_id,
                        player_id=change.player_id,
                        feature_space_id=change.feature_space_id,
                        review_status="approved",
                        active=True,
                        revision=revision,
                        approval=ReviewRecord(
                            actor=change.review.actor,
                            reason=change.review.reason,
                            revision=change.review.revision,
                        ),
                    )
                    invalidated_templates.add(template_id)
        # A rejected/unreviewed sample may be unusable; its old template still exits.
        for review in new_reviews:
            if review.review_status != "approved":
                for template in templates.values():
                    if (template.sample_id, template.player_id, template.feature_space_id) == (
                        review.sample_id,
                        review.player_id,
                        review.feature_space_id,
                    ):
                        template.active = False
                        template.review_status = review.review_status
                        invalidated_templates.add(template.template_id)
        final = temporary.model_copy(update={"templates": list(templates.values())}, deep=True)
        validate_templates(
            final,
            state.feature_values,
            lambda space_id: (
                policy_map.get(space_id)
                or (
                    settings.matching.policy_for(space_id)
                    if settings
                    else MatchingPolicy(feature_space_id=space_id)
                )
            ),
            discovery_reports=read_discovery_reports(connection, request.match_id),
        )
        for player_id in sorted(changed_players):
            repo.put_player(players[player_id], update=True)
        for segment_id in sorted(changed_segments):
            repo.replace_segment(segments[segment_id], committed_revision=revision)
        for segment_id in sorted(changed_assignments):
            repo.put_assignment(assignments[segment_id], committed_revision=revision, replace=True)
        for risk in derived_risks.values():
            repo.put_derived_risk(risk)
        for review in new_risk_reviews:
            repo.put_risk_review(review)
        for constraint in new_constraints:
            repo.put_constraint(constraint)
        for review in new_reviews:
            repo.put_template_review(review)
        for template_id in sorted(invalidated_templates):
            repo.put_template(templates[template_id], committed_revision=revision, replace=True)
        if repo.check_segment_overlaps(request.match_id):
            raise ValueError("active identity segments overlap")
        if repo.check_alias_cycles(request.match_id):
            raise ValueError("player alias cycle")
        changed = bool(
            changed_players
            or changed_segments
            or changed_assignments
            or new_reviews
            or new_risk_reviews
            or new_constraints
            or invalidated_templates
        )
        if changed:
            old_plans = {
                (item["segment_id"], item["feature_space_id"]): item
                for item in read_workflow_plans(connection, request.match_id)
            }
            for plan in plans:
                for segment in prepared_by_space[plan.feature_space_id].segments:
                    key = (segment.segment.segment_id, plan.feature_space_id)
                    previous = old_plans.get(key, {})
                    diagnostic_revision = previous.get(
                        "diagnostic_revision", previous.get("committed_revision", 0)
                    )
                    body = {k: v for k, v in previous.items() if k != "committed_revision"}
                    sample_ids = {item.sample.sample_id for item in segment.samples}
                    body.update(
                        {
                            "segment_id": segment.segment.segment_id,
                            "feature_space_id": plan.feature_space_id,
                            "diagnostic_revision": diagnostic_revision,
                            "prepared": segment.model_dump(mode="json"),
                            "gallery_changes": [
                                change.model_dump(mode="json")
                                for change in plan.changes
                                if change.sample_id in sample_ids
                            ],
                        }
                    )
                    put_workflow_plan(connection, request.match_id, key[0], key[1], revision, body)
            repo.record_review(
                request.match_id,
                revision,
                request.actor,
                [command.model_dump(mode="json") for command in request.commands],
            )
            repo.advance_revision(
                request.match_id,
                request.expected_revision,
                actor=request.actor,
                reason="apply_review",
                changes={
                    "commands": [command.kind for command in request.commands],
                    "player_ids": sorted(changed_players),
                    "segments": sorted(changed_segments | changed_assignments),
                    "templates": sorted(invalidated_templates),
                },
            )
        response = SuccessResponse[dict](
            revision=revision if changed else request.expected_revision,
            data={
                "match_id": request.match_id,
                "player_ids": sorted(changed_players),
                "gallery_plans": [plan.model_dump(mode="json") for plan in plans],
            },
        )
        repo.put_operation(
            request.match_id, "review", operation_id, payload, response.model_dump(mode="json")
        )
        return response
