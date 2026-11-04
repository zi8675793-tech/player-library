"""Score segments against one fixed gallery; never expand it with query results."""

from collections import defaultdict
from collections.abc import Mapping
from statistics import median

from app.contracts.evidence import (
    GallerySnapshot,
    InlineFeature,
    PreparedEvidence,
    PreparedSample,
    PreparedSegment,
)
from app.contracts.identity import (
    AssociationResult,
    Candidate,
    MatchingPolicy,
    SampleDiagnostic,
    SamplePlayerScore,
    Template,
    TemplateComparison,
)
from app.identity._eligibility import eligible_template, prepare_gallery
from app.identity.distance import kpr_distance
from app.identity.prepare import group_duplicate_samples, merge_assets


def _support(scores: list[SamplePlayerScore], policy: MatchingPolicy) -> str | None:
    """Abstentions stay in the caller's denominator; a lone player gets no vote."""
    if policy.tau is None or policy.delta is None:
        return None
    ranked = sorted(
        (score for score in scores if score.distance is not None),
        key=lambda score: (score.distance, score.player_id),
    )
    if len(ranked) < 2:
        return None
    first, second = ranked[:2]
    if (
        first.distance < second.distance
        and first.distance <= policy.tau
        and second.distance - first.distance >= policy.delta
    ):
        return first.player_id
    return None


def _candidate(
    prepared: PreparedSegment,
    player_id: str,
    scores: list[SamplePlayerScore],
    snapshot: GallerySnapshot,
    policy: MatchingPolicy,
    *,
    preview: bool = False,
) -> Candidate:
    distances = [score.preview_distance if preview else score.distance for score in scores]
    return Candidate(
        match_id=snapshot.match_id,
        segment_id=prepared.segment.segment_id,
        player_id=player_id,
        feature_space_id=policy.feature_space_id,
        distance=float(median(distances)),
        common_parts=min(score.common_parts for score in scores),
        gallery_revision=snapshot.revision,
        template_ids=sorted({item for score in scores for item in score.template_ids}),
        query_sample_ids=[score.sample_id for score in scores],
        score_kind="single_template_preview" if preview else "double_template",
    )


def _score_sample(
    query: PreparedSample,
    player_id: str,
    templates: list[Template],
    qualified: dict[str, PreparedSample],
    qualification_reasons: dict[str, list[str]],
    used_template_ids: set[str],
    duplicate_components: Mapping[str, str],
    prepared: PreparedEvidence,
    policy: MatchingPolicy,
) -> SamplePlayerScore:
    comparisons = []
    comparable = []
    for template in templates:
        reference = qualified.get(template.template_id)
        reason = qualification_reasons.get(template.template_id, [])
        if not reason and reference is not None and template.template_id not in used_template_ids:
            reason = ["template_capacity_exceeded"]
        if (
            reference is not None
            and duplicate_components[query.sample.sample_id]
            == duplicate_components[reference.sample.sample_id]
        ):
            reason = ["duplicate_query_evidence"]
        if reason or reference is None:
            comparisons.append(
                TemplateComparison(
                    template_id=template.template_id,
                    sample_id=template.sample_id,
                    common_parts=0,
                    reason=";".join(reason or ["ineligible_template"]),
                )
            )
            continue
        result = kpr_distance(prepared.feature_space, query.feature, reference.feature, policy)
        comparisons.append(
            TemplateComparison(
                template_id=template.template_id,
                sample_id=template.sample_id,
                distance=result.distance,
                common_parts=result.common_parts,
                reason=result.reason,
            )
        )
        if result.comparable:
            comparable.append((result.distance, template.template_id, result.common_parts))
    comparable.sort(key=lambda item: (item[0], item[1]))
    selected = comparable[:2]
    return SamplePlayerScore(
        sample_id=query.sample.sample_id,
        player_id=player_id,
        distance=float(sum(item[0] for item in selected) / 2) if len(selected) == 2 else None,
        preview_distance=selected[0][0] if len(selected) == 1 else None,
        template_ids=[item[1] for item in selected],
        comparisons=comparisons,
        common_parts=min((item[2] for item in selected), default=0),
        reason=None if len(selected) == 2 else "fewer_than_two_comparable_templates",
    )


def _template_groups(templates, qualified, assets):
    """Attach template records to the shared transitive sample components."""
    groups = group_duplicate_samples(
        [qualified[template.template_id] for template in templates], assets
    )
    return [
        [
            template
            for template in templates
            if template.sample_id in {item.sample.sample_id for item in group}
        ]
        for group in groups
    ]


def associate_segments(
    prepared: PreparedEvidence,
    snapshot: GallerySnapshot,
    policy: MatchingPolicy,
    *,
    feature_values: Mapping[tuple[str, int], InlineFeature] | None = None,
) -> list[AssociationResult]:
    """Return local previews, common-query rankings and per-sample diagnostics.

    Candidate/margin/support calculations for automatic decisions all use Q, the
    same representative samples with two comparable references for every player.
    Samples outside Q still participate in the strong-disagreement veto.
    """
    if prepared.match_id != snapshot.match_id:
        raise ValueError("query evidence and gallery must belong to the same match")
    if (
        prepared.feature_space_id != policy.feature_space_id
        or prepared.policy_version != policy.policy_version
        or prepared.policy_fingerprint != policy.fingerprint
    ):
        raise ValueError("prepared evidence does not match the selected space/policy")
    gallery = prepare_gallery(snapshot, policy, feature_values)
    if (
        prepared.feature_space is not None
        and gallery.feature_space is not None
        and prepared.feature_space != gallery.feature_space
    ):
        raise ValueError("same feature_space_id has conflicting semantic contracts")
    player_ids = sorted(player.player_id for player in snapshot.players if player.alias_of is None)
    templates_by_player = defaultdict(list)
    qualified: dict[str, PreparedSample] = {}
    qualification_reasons: dict[str, list[str]] = {}
    used_template_ids: set[str] = set()
    # Query assets are kept by prepare so equal image hashes cross the boundary.
    assets = merge_assets(snapshot.assets, prepared.assets)
    for template in sorted(snapshot.templates, key=lambda template: template.template_id):
        if template.player_id not in player_ids:
            continue
        templates_by_player[template.player_id].append(template)
        sample, reasons = eligible_template(
            template, snapshot, policy, feature_values, prepared_gallery=gallery
        )
        if sample is None:
            qualification_reasons[template.template_id] = reasons
        else:
            qualified[template.template_id] = sample
    duplicate_components = {}
    for group in group_duplicate_samples(
        [
            *(sample for segment in prepared.segments for sample in segment.samples),
            *qualified.values(),
        ],
        assets,
    ):
        component = min(sample.sample.sample_id for sample in group)
        for sample in group:
            duplicate_components[sample.sample.sample_id] = component
    gallery_conflicts = sorted(
        {
            template.template_id
            for group in _template_groups(
                [template for template in snapshot.templates if template.template_id in qualified],
                qualified,
                assets,
            )
            if len({template.player_id for template in group}) > 1
            for template in group
        }
    )
    for player_id in player_ids:
        templates = [
            template
            for template in templates_by_player[player_id]
            if template.template_id in qualified
        ]
        representatives = []
        for group in _template_groups(templates, qualified, assets):
            ordered = sorted(group, key=lambda template: template.template_id)
            representatives.append(ordered[0])
            for duplicate in ordered[1:]:
                qualification_reasons[duplicate.template_id] = ["duplicate_template_evidence"]
        used_template_ids.update(
            template.template_id
            for template in sorted(representatives, key=lambda template: template.template_id)[
                : policy.max_templates_per_player
            ]
        )
    results = []
    for segment in sorted(prepared.segments, key=lambda item: item.segment.segment_id):
        representatives = set(segment.representative_sample_ids)
        diagnostics = []
        for sample in sorted(
            segment.samples, key=lambda item: (item.sample.time_s, item.sample.sample_id)
        ):
            scores = (
                [
                    _score_sample(
                        sample,
                        player_id,
                        templates_by_player[player_id],
                        qualified,
                        qualification_reasons,
                        used_template_ids,
                        duplicate_components,
                        prepared,
                        policy,
                    )
                    for player_id in player_ids
                ]
                if prepared.feature_space is not None
                else []
            )
            diagnostics.append(
                SampleDiagnostic(
                    sample_id=sample.sample.sample_id,
                    bucket_index=sample.bucket_index,
                    region=sample.region,
                    representative=sample.sample.sample_id in representatives,
                    scores=scores,
                    supported_player_id=_support(scores, policy),
                    reason="thresholds_unset"
                    if policy.tau is None or policy.delta is None
                    else None,
                )
            )
        representative_diagnostics = [item for item in diagnostics if item.representative]
        common = [
            item
            for item in representative_diagnostics
            if player_ids
            and len(item.scores) == len(player_ids)
            and all(score.distance is not None for score in item.scores)
        ]
        candidates = []
        local_candidates = []
        for player_id in player_ids:
            common_scores = [
                next(score for score in item.scores if score.player_id == player_id)
                for item in common
            ]
            if common_scores:
                candidates.append(_candidate(segment, player_id, common_scores, snapshot, policy))
            local_scores = [
                score
                for item in representative_diagnostics
                for score in item.scores
                if score.player_id == player_id and score.distance is not None
            ]
            if local_scores:
                local_candidates.append(
                    _candidate(segment, player_id, local_scores, snapshot, policy)
                )
            else:
                previews = [
                    score
                    for item in representative_diagnostics
                    for score in item.scores
                    if score.player_id == player_id and score.preview_distance is not None
                ]
                if previews:
                    local_candidates.append(
                        _candidate(segment, player_id, previews, snapshot, policy, preview=True)
                    )
        candidates.sort(key=lambda item: (item.distance, item.player_id))
        local_candidates.sort(key=lambda item: (item.distance, item.player_id))
        first = candidates[0] if candidates else None
        second = candidates[1] if len(candidates) > 1 else None
        margin = second.distance - first.distance if second is not None else None
        common_coverage = (
            len(common) / len(representative_diagnostics) if representative_diagnostics else 0.0
        )
        buckets = len({item.bucket_index for item in common})
        regions = sorted({item.region for item in common})
        support_rate = (
            sum(item.supported_player_id == first.player_id for item in common) / len(common)
            if first is not None and policy.tau is not None and policy.delta is not None
            else None
        )
        disagreement_reference = first or (local_candidates[0] if local_candidates else None)
        disagreements = [
            item.sample_id
            for item in diagnostics
            if item.supported_player_id is not None
            and disagreement_reference is not None
            and item.supported_player_id != disagreement_reference.player_id
        ]
        blocking = []
        if not player_ids:
            blocking.append("no_gallery_players")
        if prepared.feature_space is None:
            blocking.append("missing_feature_space")
        if not representative_diagnostics:
            blocking.append("no_representative_samples")
        if common_coverage < policy.min_common_query_coverage:
            blocking.append("insufficient_common_query_coverage")
        if buckets < policy.min_query_buckets:
            blocking.append("insufficient_common_query_buckets")
        if regions != [0, 1, 2]:
            blocking.append("incomplete_common_temporal_coverage")
        if len(candidates) < 2:
            blocking.append("fewer_than_two_candidates")
        if first is not None and second is not None and first.distance == second.distance:
            blocking.append("tied_candidates")
        if disagreements:
            blocking.append("strong_support_disagreement")
        if gallery_conflicts:
            blocking.append("gallery_identity_conflict")
        results.append(
            AssociationResult(
                match_id=prepared.match_id,
                segment_id=segment.segment.segment_id,
                feature_space_id=policy.feature_space_id,
                policy_version=policy.policy_version,
                policy_fingerprint=policy.fingerprint,
                gallery_revision=snapshot.revision,
                candidates=candidates,
                local_candidates=local_candidates,
                sample_diagnostics=diagnostics,
                common_query_sample_ids=[item.sample_id for item in common],
                common_query_coverage=common_coverage,
                common_bucket_count=buckets,
                covered_regions=regions,
                first_player_id=first.player_id if first else None,
                first_distance=first.distance if first else None,
                second_distance=second.distance if second else None,
                margin=margin,
                support_rate=support_rate,
                strong_disagreement_sample_ids=disagreements,
                blocking_reasons=blocking,
                template_ids=sorted(used_template_ids),
                gallery_conflict_template_ids=gallery_conflicts,
            )
        )
    return results
