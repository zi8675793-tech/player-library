"""Plan gallery changes from final identity and independent human sample reviews."""

from collections import defaultdict
from collections.abc import Mapping, Sequence

from app.contracts.evidence import GallerySnapshot, InlineFeature, PreparedEvidence
from app.contracts.identity import (
    Assignment,
    AssignmentStatus,
    GalleryAction,
    GalleryChange,
    GalleryPlan,
    ManualTemplateReview,
    MatchingPolicy,
    TemplateReviewStatus,
)
from app.identity._eligibility import eligible_template, prepare_gallery
from app.identity.prepare import _within, group_duplicate_samples, merge_assets


def plan_gallery(
    prepared: PreparedEvidence,
    snapshot: GallerySnapshot,
    policy: MatchingPolicy,
    *,
    final_assignments: Sequence[Assignment] = (),
    reviews: Sequence[ManualTemplateReview] = (),
    feature_values: Mapping[tuple[str, int], InlineFeature] | None = None,
) -> GalleryPlan:
    """Return activation, revocation, review and capacity plans without mutation.

    Assignments are final accepted context or calculated plans supplied by a caller;
    only confirmed assignments qualify. They overlay the fixed snapshot. No IDs or
    approvals are created here. Capacity overflow proposes human replacement choices.
    """
    if prepared.match_id != snapshot.match_id:
        raise ValueError("prepared evidence and gallery must have the same match_id")
    policy = MatchingPolicy.model_validate(policy.model_dump())
    if (prepared.feature_space_id, prepared.policy_fingerprint) != (
        policy.feature_space_id,
        policy.fingerprint,
    ):
        raise ValueError("prepared evidence does not match the selected policy")
    gallery = prepare_gallery(snapshot, policy, feature_values)
    if (
        prepared.feature_space is not None
        and gallery.feature_space is not None
        and prepared.feature_space != gallery.feature_space
    ):
        raise ValueError("same feature_space_id has conflicting semantic contracts")
    assignments = {item.segment_id: item for item in snapshot.assignments}
    seen_assignments = set()
    for assignment in final_assignments:
        if assignment.segment_id in seen_assignments:
            raise ValueError("duplicate final assignment segment_id")
        seen_assignments.add(assignment.segment_id)
        if assignment.match_id != snapshot.match_id:
            raise ValueError("final assignment has an invalid match scope")
        assignments[assignment.segment_id] = assignment
    players = {player.player_id for player in snapshot.players if player.alias_of is None}
    segments = {item.segment.segment_id: item for item in gallery.segments}
    segments.update({item.segment.segment_id: item for item in prepared.segments})
    if any(key not in segments for key in seen_assignments):
        raise ValueError("final assignment references an unknown segment")
    if any(
        item.player_id is not None and item.player_id not in players for item in final_assignments
    ):
        raise ValueError("final assignment references a missing or merged player")
    superseded = {
        item.segment.supersedes_segment_id
        for item in segments.values()
        if item.segment.active and item.segment.supersedes_segment_id is not None
    }
    # Query preparation supplies the current range/sample interpretation. Original
    # samples are retained in the input; they are never rewritten to fit a split.
    sample_segments = {}
    for segment in [*gallery.segments, *prepared.segments]:
        for item in segment.samples:
            sample_segments[item.sample.sample_id] = (segment, item)
    review_map = {}
    templates_by_id = {template.template_id: template for template in snapshot.templates}
    for review in reviews:
        if (
            review.match_id != snapshot.match_id
            or review.feature_space_id != policy.feature_space_id
        ):
            raise ValueError("template review has an invalid match/space scope")
        if review.sample_id not in sample_segments:
            raise ValueError("template review requires usable, located evidence")
        if review.sample_id in review_map:
            raise ValueError("duplicate template review sample_id")
        if review.template_id is not None:
            existing = templates_by_id.get(review.template_id)
            if existing is None or (
                existing.sample_id,
                existing.feature_space_id,
                existing.player_id,
            ) != (review.sample_id, review.feature_space_id, review.player_id):
                raise ValueError("template review does not match its referenced template")
        previous = [
            template
            for template in snapshot.templates
            if (template.sample_id, template.feature_space_id, template.player_id)
            == (review.sample_id, review.feature_space_id, review.player_id)
        ]
        if previous and review.revision < max(template.revision for template in previous):
            raise ValueError("stale template review predates the current template revision")
        review_map[review.sample_id] = review
    changes: list[GalleryChange] = []
    retained = defaultdict(list)
    active_keys = set()
    revoked_template_ids = set()
    original_samples = {sample.sample_id: sample for sample in snapshot.evidence.samples}
    for template in sorted(snapshot.templates, key=lambda item: item.template_id):
        if not template.active:
            continue
        selected_space = template.feature_space_id == policy.feature_space_id
        reasons = []
        if selected_space:
            _, reasons = eligible_template(
                template, snapshot, policy, feature_values, prepared_gallery=gallery
            )
        original_sample = original_samples.get(template.sample_id)
        original_segment_id = original_sample.segment_id if original_sample is not None else None
        if original_sample is not None and original_segment_id is None:
            matches = [
                segment.segment_id
                for segment in snapshot.evidence.segments
                if _within(original_sample, segment) and segment.active
            ]
            if len(matches) == 1:
                original_segment_id = matches[0]
        located = sample_segments.get(template.sample_id)
        current_segment = located[0] if located is not None else segments.get(original_segment_id)
        if selected_space and located is None:
            reasons.append("template_sample_no_longer_usable")
        if current_segment is None:
            reasons.append("unresolved_template_identity_scope")
        else:
            segment = current_segment
            if (
                original_sample is not None
                and located is None
                and not _within(original_sample, segment.segment)
            ):
                reasons.append("template_sample_outside_current_scope")
            assignment = assignments.get(segment.segment.segment_id)
            if (
                not segment.segment.active
                or segment.segment.segment_id in superseded
                or original_segment_id in superseded
            ):
                reasons.append("template_segment_no_longer_active")
            if segment.segment.segment_id != original_segment_id:
                reasons.append("template_identity_scope_changed")
            if segment.risks:
                reasons.append("template_segment_risk")
            if (
                assignment is None
                or assignment.status != AssignmentStatus.CONFIRMED
                or assignment.player_id != template.player_id
            ):
                reasons.append("template_ownership_invalidated")
        review = review_map.get(template.sample_id) if selected_space else None
        if review is not None and review.review_status != TemplateReviewStatus.APPROVED:
            reasons.append("template_review_not_approved")
        if reasons:
            revoked_template_ids.add(template.template_id)
            changes.append(
                GalleryChange(
                    action=GalleryAction.DEACTIVATE,
                    template_id=template.template_id,
                    sample_id=template.sample_id,
                    player_id=template.player_id,
                    feature_space_id=template.feature_space_id,
                    reasons=sorted(set(reasons)),
                    review=review,
                )
            )
        elif selected_space:
            retained[template.player_id].append(template)
            active_keys.add((template.sample_id, template.player_id))

    pending = defaultdict(list)
    query_sample_ids = {
        item.sample.sample_id for segment in prepared.segments for item in segment.samples
    }
    assets = merge_assets(snapshot.assets, prepared.assets)
    component_by_sample = {}
    for group in group_duplicate_samples([item for _, item in sample_segments.values()], assets):
        component = min(item.sample.sample_id for item in group)
        for item in group:
            component_by_sample[item.sample.sample_id] = component
    retained_owners = defaultdict(set)
    for player_id, templates in retained.items():
        for template in templates:
            retained_owners[component_by_sample[template.sample_id]].add(player_id)
    for sample_id in sorted(query_sample_ids | set(review_map)):
        segment, item = sample_segments[sample_id]
        assignment = assignments.get(segment.segment.segment_id)
        if (
            assignment is None
            or assignment.status != AssignmentStatus.CONFIRMED
            or assignment.player_id not in players
        ):
            continue
        player_id = assignment.player_id
        review = review_map.get(sample_id)
        if review is not None and review.review_status == TemplateReviewStatus.REJECTED:
            continue
        if review is not None and review.player_id != player_id:
            raise ValueError("template approval does not match the final confirmed owner")
        if (sample_id, player_id) in active_keys:
            continue
        reasons = []
        if not segment.segment.active or segment.segment.segment_id in superseded:
            reasons.append("inactive_or_superseded_segment")
        if segment.risks:
            reasons.append("unresolved_identity_risk")
        quality = item.sample.tracking_quality
        if quality is not None and quality.occluded is True:
            reasons.append("occluded_sample")
        if item.visible_parts < policy.min_common_parts:
            reasons.append("insufficient_visible_parts")
        if review is None or review.review_status != TemplateReviewStatus.APPROVED:
            reasons.append("manual_template_approval_required")
        elif review.revision < assignment.revision:
            reasons.append("approval_predates_final_assignment")
        # Repeated delivery never creates another template, including evidence
        # connected transitively through a third image or source observation.
        existing_owners = retained_owners[component_by_sample[sample_id]]
        if player_id in existing_owners:
            reasons.append("duplicate_gallery_evidence")
        if any(owner != player_id for owner in existing_owners):
            reasons.append("conflicting_gallery_evidence_owner")
        change = GalleryChange(
            action=GalleryAction.AWAIT_REVIEW if reasons else GalleryAction.ACTIVATE,
            sample_id=sample_id,
            player_id=player_id,
            feature_space_id=policy.feature_space_id,
            template_id=(
                review.template_id
                if review is not None
                and review.template_id is not None
                and review.template_id not in revoked_template_ids
                and templates_by_id[review.template_id].player_id == player_id
                else None
            ),
            review=review,
            reasons=sorted(set(reasons)),
        )
        if reasons:
            changes.append(change)
        else:
            pending[player_id].append((change, item))

    pending_components = defaultdict(list)
    for entries in pending.values():
        for change, item in entries:
            pending_components[component_by_sample[change.sample_id]].append((change, item))
    cross_owner_conflicts = set()
    duplicates = set()
    for entries in pending_components.values():
        if len({change.player_id for change, _ in entries}) > 1:
            cross_owner_conflicts.update(change.sample_id for change, _ in entries)
        else:
            ordered = sorted(entries, key=lambda entry: entry[0].sample_id)
            duplicates.update(change.sample_id for change, _ in ordered[1:])
    for player_id in sorted(pending):
        proposed = pending[player_id]
        distinct = []
        for change, item in proposed:
            if change.sample_id in duplicates or change.sample_id in cross_owner_conflicts:
                changes.append(
                    change.model_copy(
                        deep=True,
                        update={
                            "action": GalleryAction.AWAIT_REVIEW,
                            "reasons": [
                                "conflicting_gallery_evidence_owner"
                                if change.sample_id in cross_owner_conflicts
                                else "duplicate_proposed_template_evidence"
                            ],
                        },
                    )
                )
            else:
                distinct.append((change, item))
        overflow = len(retained[player_id]) + len(distinct) > policy.max_templates_per_player
        for change, _ in distinct:
            if overflow:
                change = change.model_copy(
                    deep=True,
                    update={
                        "action": GalleryAction.REPLACEMENT_REQUIRED,
                        "reasons": ["template_capacity_requires_manual_selection"],
                        "replacement_template_ids": sorted(
                            template.template_id for template in retained[player_id]
                        ),
                    },
                )
            changes.append(change)
    for player_id, templates in sorted(retained.items()):
        if len(templates) > policy.max_templates_per_player:
            for template in templates:
                changes.append(
                    GalleryChange(
                        action=GalleryAction.REPLACEMENT_REQUIRED,
                        sample_id=template.sample_id,
                        player_id=player_id,
                        template_id=template.template_id,
                        feature_space_id=policy.feature_space_id,
                        reasons=["existing_gallery_capacity_requires_manual_selection"],
                        replacement_template_ids=sorted(t.template_id for t in templates),
                    )
                )
    changes.sort(
        key=lambda item: (item.action, item.player_id, item.sample_id, item.template_id or "")
    )
    return GalleryPlan(
        match_id=snapshot.match_id,
        gallery_revision=snapshot.revision,
        feature_space_id=policy.feature_space_id,
        policy_version=policy.policy_version,
        changes=changes,
    ).model_copy(deep=True)
