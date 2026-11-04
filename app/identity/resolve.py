"""Produce revision-scoped decisions without creating players or executing reviews."""

from collections.abc import Sequence

from app.contracts.evidence import GallerySnapshot, PreparedEvidence, PreparedSegment
from app.contracts.identity import (
    Assignment,
    AssignmentEvidence,
    AssignmentMethod,
    AssignmentStatus,
    AssociationResult,
    ConstraintKind,
    ManualConstraint,
    MatchingPolicy,
    NewIdentityRequest,
    Resolution,
    ResolutionDecision,
    UnassignedReason,
)


def _index(records, key, label):
    result = {}
    for record in records:
        value = key(record)
        if value in result:
            raise ValueError(f"duplicate {label}: {value}")
        result[value] = record
    return result


def _manual(assignment: Assignment | None) -> bool:
    return assignment is not None and assignment.method in {
        AssignmentMethod.MANUAL,
        AssignmentMethod.SEED,
    }


def _reasons(segment: PreparedSegment, association: AssociationResult) -> list[str]:
    reasons = list(segment.reasons) + list(association.blocking_reasons)
    if segment.risks:
        reasons.append("unresolved_identity_risk")
    if association.strong_disagreement_sample_ids:
        reasons.append("strong_sample_disagreement")
    return sorted(set(reasons))


def resolve_associations(
    prepared: PreparedEvidence,
    snapshot: GallerySnapshot,
    associations: Sequence[AssociationResult],
    policy: MatchingPolicy,
    *,
    manual_assignments: Sequence[Assignment] = (),
    constraints: Sequence[ManualConstraint] = (),
    new_identity_requests: Sequence[NewIdentityRequest] = (),
) -> list[Resolution]:
    """Resolve all segments together so scoped human constraints are order independent.

    ``manual_assignments`` and requests are already accepted human context supplied
    by the caller. This function neither accepts ReviewCommands nor allocates IDs.
    The assignment revision is the input snapshot revision, never a claimed commit.
    """
    if prepared.match_id != snapshot.match_id:
        raise ValueError("prepared evidence and gallery must have the same match_id")
    if (prepared.feature_space_id, prepared.policy_fingerprint) != (
        policy.feature_space_id,
        policy.fingerprint,
    ):
        raise ValueError("prepared evidence does not match this policy")
    # Revalidate public model instances as callers may have used model_copy/update.
    policy = MatchingPolicy.model_validate(policy.model_dump())
    by_segment = _index(associations, lambda item: item.segment_id, "association")
    segments = _index(prepared.segments, lambda item: item.segment.segment_id, "segment")
    if set(by_segment) != set(segments):
        raise ValueError("associations must correspond exactly to prepared segments")
    assignments = _index(snapshot.assignments, lambda item: item.segment_id, "assignment")
    updates = _index(manual_assignments, lambda item: item.segment_id, "manual assignment")
    players = {p.player_id for p in snapshot.players}
    for assignment in updates.values():
        if assignment.match_id != prepared.match_id or assignment.segment_id not in segments:
            raise ValueError("manual assignment has an invalid scope")
        if not _manual(assignment):
            raise ValueError("manual_assignments must contain human decisions")
        if assignment.player_id is not None and assignment.player_id not in players:
            raise ValueError("manual assignment must reference an existing snapshot player")
    assignments.update(updates)
    requests = _index(new_identity_requests, lambda item: item.segment_id, "new identity request")
    _index(constraints, lambda item: item.constraint_id, "constraint")
    known_segments = set(segments) | {s.segment_id for s in snapshot.evidence.segments}
    for item in [*constraints, *requests.values()]:
        if item.match_id != prepared.match_id or item.segment_id not in known_segments:
            raise ValueError("human context has an invalid match/segment scope")
    for constraint in constraints:
        if (
            constraint.related_segment_id is not None
            and constraint.related_segment_id not in known_segments
        ):
            raise ValueError("constraint references an unknown related segment")

    results: dict[str, Resolution] = {}
    preserved = set()
    for segment_id in sorted(segments):
        segment = segments[segment_id]
        association = by_segment[segment_id]
        if (
            association.match_id != prepared.match_id
            or association.feature_space_id != policy.feature_space_id
            or association.policy_fingerprint != policy.fingerprint
            or association.policy_version != policy.policy_version
            or association.gallery_revision != snapshot.revision
        ):
            raise ValueError("association does not match policy/gallery snapshot")
        reasons = _reasons(segment, association)
        base = dict(
            match_id=prepared.match_id,
            segment_id=segment_id,
            feature_space_id=policy.feature_space_id,
            policy_version=policy.policy_version,
            gallery_revision=snapshot.revision,
        )
        human = assignments.get(segment_id)
        applicable = [c for c in constraints if c.segment_id == segment_id]
        forbidden = {c.player_id for c in applicable if c.kind == ConstraintKind.FORBID_PLAYER}
        if _manual(human):
            preserved.add(segment_id)
            if human.player_id in forbidden or any(
                c.kind == ConstraintKind.BLOCK_AUTOMATIC for c in applicable
            ):
                reasons.append("manual_constraint_conflict")
            if human.player_id is not None:
                if association.first_player_id not in {None, human.player_id}:
                    reasons.append("manual_candidate_conflict")
                result = Resolution(
                    **base,
                    decision=ResolutionDecision.MATCH_EXISTING,
                    player_id=human.player_id,
                    assignment=human.model_copy(deep=True),
                    reasons=sorted(set(["manual_decision_preserved", *reasons])),
                    requires_review=bool(reasons) or human.status != AssignmentStatus.CONFIRMED,
                )
            else:
                rejected = human.unassigned_reason in {
                    UnassignedReason.OTHER_PERSON,
                    UnassignedReason.REJECTED,
                }
                result = Resolution(
                    **base,
                    decision=ResolutionDecision.REJECTED
                    if rejected
                    else ResolutionDecision.PENDING,
                    assignment=human.model_copy(deep=True),
                    reasons=sorted(set(["manual_decision_preserved", *reasons])),
                    requires_review=bool(reasons),
                )
            results[segment_id] = result
            continue

        if not segment.segment.active:
            reasons.append("inactive_segment")
        if any(c.kind == ConstraintKind.BLOCK_AUTOMATIC for c in applicable):
            reasons.append("manual_block_automatic")
        if association.first_player_id in forbidden:
            reasons.append("forbidden_player")

        request = requests.get(segment_id)
        if request is not None:
            supports = {sample.sample.sample_id: sample for sample in segment.samples}
            if len(set(request.support_sample_ids)) != len(request.support_sample_ids):
                raise ValueError("new identity support_sample_ids must be unique")
            if any(s not in supports for s in request.support_sample_ids):
                raise ValueError("new identity support must be usable evidence in its segment")
            clean = all(supports[s].clean_approved for s in request.support_sample_ids)
            # Missing templates/calibration do not invalidate independent human support.
            blocked = bool(
                segment.risks
                or association.strong_disagreement_sample_ids
                or not segment.segment.active
                or not clean
            )
            if not blocked:
                results[segment_id] = Resolution(
                    **base,
                    decision=ResolutionDecision.CREATE_NEW_PROPOSAL,
                    evidence=[
                        AssignmentEvidence(kind="manual_new_identity_support", reference_id=s)
                        for s in sorted(request.support_sample_ids)
                    ],
                    reasons=["manual_new_identity_proposal"],
                )
                continue
            reasons.append("new_identity_support_requires_review")

        # Enforce the prerequisites here as well as preserving associate's diagnostics.
        if len(association.candidates) < 2:
            reasons.append("fewer_than_two_candidates")
        if association.common_bucket_count < policy.min_query_buckets:
            reasons.append("insufficient_common_buckets")
        if association.common_query_coverage < policy.min_common_query_coverage:
            reasons.append("insufficient_common_coverage")
        if set(association.covered_regions) != {0, 1, 2}:
            reasons.append("insufficient_temporal_coverage")
        if association.margin is None:
            reasons.append("missing_candidate_margin")
        if reasons:
            results[segment_id] = Resolution(
                **base, decision=ResolutionDecision.PENDING, reasons=sorted(set(reasons))
            )
            continue
        if not policy.calibrated:
            results[segment_id] = Resolution(
                **base, decision=ResolutionDecision.PENDING, reasons=["policy_uncalibrated"]
            )
            continue
        if (
            association.first_distance is None
            or association.first_distance > policy.tau
            or association.margin < policy.delta
            or association.support_rate is None
            or association.support_rate < policy.min_support_rate
        ):
            results[segment_id] = Resolution(
                **base, decision=ResolutionDecision.PENDING, reasons=["matching_gates_not_met"]
            )
            continue
        evidence = [
            AssignmentEvidence(
                kind="calibrated_gallery_match",
                reference_id=template_id,
                feature_space_id=policy.feature_space_id,
                distance=association.first_distance,
            )
            for template_id in association.candidates[0].template_ids
        ]
        results[segment_id] = Resolution(
            **base,
            decision=ResolutionDecision.MATCH_EXISTING,
            player_id=association.first_player_id,
            assignment=Assignment(
                match_id=prepared.match_id,
                segment_id=segment_id,
                player_id=association.first_player_id,
                status=(
                    AssignmentStatus.CONFIRMED
                    if policy.auto_assign_enabled
                    else AssignmentStatus.PROPOSED
                ),
                method=AssignmentMethod.AUTOMATIC,
                evidence=evidence,
                revision=snapshot.revision,
            ),
            evidence=evidence,
            reasons=["calibrated_match"],
            requires_review=not policy.auto_assign_enabled,
            automatic_confirmation_eligible=policy.auto_assign_enabled,
        )

    # CANNOT_LINK is explicit evidence; timestamps alone never create exclusivity.
    preliminary_results = dict(results)
    for constraint in sorted(constraints, key=lambda c: c.constraint_id):
        if constraint.kind != ConstraintKind.CANNOT_LINK:
            continue
        left, right = constraint.segment_id, constraint.related_segment_id
        targets = []
        for segment_id in (left, right):
            result = preliminary_results.get(segment_id)
            current = assignments.get(segment_id)
            targets.append(
                result.player_id
                if result is not None and result.player_id is not None
                else current.player_id
                if current is not None and current.status == AssignmentStatus.CONFIRMED
                else None
            )
        if targets[0] is None or targets[0] != targets[1]:
            continue
        for segment_id in (left, right):
            result = results.get(segment_id)
            if result is None:
                continue
            reasons = sorted(set([*result.reasons, "cannot_link_conflict"]))
            if segment_id in preserved:
                results[segment_id] = result.model_copy(
                    deep=True, update={"reasons": reasons, "requires_review": True}
                )
            else:
                results[segment_id] = Resolution(
                    match_id=result.match_id,
                    segment_id=result.segment_id,
                    decision=ResolutionDecision.PENDING,
                    reasons=reasons,
                    feature_space_id=result.feature_space_id,
                    policy_version=result.policy_version,
                    gallery_revision=result.gallery_revision,
                )
    return [results[key] for key in sorted(results)]
