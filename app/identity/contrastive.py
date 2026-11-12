"""Raw multisample support against independently observed different people.

Negative identities come exclusively from persistent separated cooccurrence in
the candidate's own clip. Neither discovered groups nor templates participate.
"""

import hashlib
import json
import math
from collections import defaultdict
from statistics import median

from app.contracts.discovery import DiscoveryDirection, DiscoveryPair


def _pair_id(match_id, space_id, left, right):
    payload = json.dumps(
        (match_id, space_id, left, right),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return "dsp_" + hashlib.sha256(payload).hexdigest()[:32]


def contrastive_pair(
    left,
    right,
    selected,
    all_samples,
    active,
    evidence,
    policy,
    distances,
    components,
    geometric_conflicts,
):
    """Score one unordered pair, using reciprocal independent geometric negatives.

    A finite distance alone supplies no vote. Each representative needs two
    independent candidate observations, two observations of every geometric
    negative, and the configured distance margin against every negative. Missing
    negatives or missing comparisons remain unknown rather than reducing the
    support denominator. Every usable raw query can veto a misleading majority.
    """
    del evidence  # Geometry has already been checked once by the caller.
    if left > right:
        left, right = right, left
    if policy.feature_space_id is None:
        raise ValueError("contrastive support requires the resolved feature_space_id")
    comparisons = {}
    reasons = set()
    conflict_obs = geometric_conflicts.get((left, right), [])
    veto = bool(conflict_obs)
    if veto:
        reasons.add("spatially_separated_cooccurrence")
    directions = []
    has_margin_ambiguity = False

    def compare(query, reference):
        pair = distances.pair(query, reference)
        key = tuple(sorted((query.sample_id, reference.sample_id)))
        comparisons[key] = pair
        return pair.distance

    for sid, candidate in ((left, right), (right, left)):
        negatives = sorted(
            other
            for other in selected
            if other != candidate
            and active[other].clip_id == active[candidate].clip_id
            and geometric_conflicts.get(tuple(sorted((candidate, other))))
        )
        if not negatives:
            reasons.add("missing_independent_geometric_negatives")
        involved = [sid, candidate, *negatives]
        component_segments = defaultdict(set)
        for segment_id in involved:
            for sample in all_samples[segment_id]:
                component_segments[components[sample.sample_id]].add(segment_id)
        shared = {component for component, scopes in component_segments.items() if len(scopes) > 1}
        if shared:
            reasons.add("shared_duplicate_components_excluded")

        def independent(samples):
            unique = {}
            for sample in sorted(samples, key=lambda sample: (sample.time_s, sample.sample_id)):
                component = components[sample.sample_id]
                if component not in shared:
                    unique.setdefault(component, sample)
            return list(unique.values())

        pools = {segment_id: independent(selected[segment_id]) for segment_id in involved}
        raw_queries = independent(all_samples[sid])
        representative_ids = {sample.sample_id for sample in pools[sid]}
        # The score uses representatives, while the diagnostic retains every
        # usable query/reference comparison, including far appearance pairs.
        for query in raw_queries:
            for target in (candidate, *negatives):
                for reference in independent(all_samples[target]):
                    distance = compare(query, reference)
                    if (
                        target == candidate
                        and distance is not None
                        and distance >= policy.candidate_far_distance
                    ):
                        reasons.add("far_sample_distances_present_not_identity_veto")

        def score(query, target):
            options = []
            target_pool = pools[target] if target in pools else independent(selected[target])
            for reference in target_pool:
                if components[query.sample_id] == components[reference.sample_id]:
                    continue
                value = compare(query, reference)
                if value is not None:
                    options.append((value, reference.sample_id))
            options.sort()
            if len(options) < 2:
                return None, []
            nearest = options[:2]
            return sum(value for value, _ in nearest) / 2, [sample_id for _, sample_id in nearest]

        contrasts, supports, scores, supported_buckets = [], [], [], set()
        for query in raw_queries:
            candidate_score, candidate_ids = score(query, candidate)
            competitors = []
            for negative in negatives:
                negative_score, negative_ids = score(query, negative)
                competitors.append(
                    {
                        "segment_id": negative,
                        "distance": negative_score,
                        "sample_ids": negative_ids,
                        "margin": None
                        if candidate_score is None or negative_score is None
                        else negative_score - candidate_score,
                    }
                )
                if (
                    candidate_score is not None
                    and negative_score is not None
                    and negative_score <= policy.negative_evidence_distance
                    and negative_score + policy.counter_evidence_margin <= candidate_score
                ):
                    veto = True
                    reasons.add("minority_raw_sample_supports_geometric_negative")
            complete = bool(negatives) and all(item["distance"] is not None for item in competitors)
            margin_passes = (
                complete
                and candidate_score is not None
                and all(
                    candidate_score + policy.contrastive_margin <= item["distance"]
                    for item in competitors
                )
            )
            # Evaluate independent support compensation when geometric negatives are absent
            compensated = False
            if (
                not negatives
                and getattr(policy, "enable_geometric_negative_compensation", False)
                and candidate_score is not None
            ):
                comp_max_dist = getattr(policy, "compensation_max_distance", 0.22)
                comp_max_raw = getattr(policy, "compensation_max_raw_distance", 0.28)
                comp_min_parts = getattr(policy, "compensation_min_common_parts", 4)
                comp_rival_margin = getattr(
                    policy, "compensation_rival_margin", policy.rival_margin
                )
                # 1. Strict distance anchor
                if candidate_score <= comp_max_dist:
                    # 2. Multi-part robustness on candidate reference pairs
                    ref_pairs_ok = all(
                        (
                            p := distances.pair(
                                query, next(s for s in all_samples[candidate] if s.sample_id == cid)
                            )
                        ).common_parts
                        >= comp_min_parts
                        and p.distance is not None
                        and p.distance <= comp_max_dist
                        for cid in candidate_ids
                    )
                    # 3. Raw sample minority veto: all usable raw queries must not exceed threshold
                    raw_queries_ok = True
                    for rq in raw_queries:
                        rq_score, _ = score(rq, candidate)
                        if rq_score is None or rq_score > comp_max_raw:
                            raw_queries_ok = False
                            reasons.add("minority_raw_sample_disagreement_with_candidate")
                            break
                    # 4. Rival ambiguity check across selected segments
                    has_close_rival = False
                    for other_sid in selected:
                        if (
                            other_sid != candidate
                            and other_sid != sid
                            and active[other_sid].clip_id != active[candidate].clip_id
                        ):
                            other_score, _ = score(query, other_sid)
                            if (
                                other_score is not None
                                and other_score < candidate_score + comp_rival_margin
                            ):
                                has_close_rival = True
                                break
                    if ref_pairs_ok and raw_queries_ok and not has_close_rival:
                        compensated = True

            supported = bool(
                (margin_passes or compensated)
                and candidate_score is not None
                and candidate_score <= policy.candidate_far_distance
            )
            representative = query.sample_id in representative_ids
            if representative and candidate_score is not None:
                scores.append(candidate_score)
            if representative and supported:
                supports.append(query.sample_id)
                supported_buckets.add(
                    math.floor((query.time_s - active[sid].start_s) / policy.bucket_s)
                )
            if representative and complete and candidate_score is not None and not margin_passes:
                has_margin_ambiguity = True
            reason = (
                "compensated_appearance_anchor_support"
                if compensated
                else "contrastive_multisample_support"
                if supported
                else "candidate_two_independent_observations_missing"
                if candidate_score is None
                else "missing_independent_geometric_negatives"
                if not negatives
                else "negative_two_independent_observations_missing"
                if not complete
                else "insufficient_competitor_margin"
                if not margin_passes
                else "candidate_exceeds_experimental_upper_bound"
            )
            contrasts.append(
                {
                    "sample_id": query.sample_id,
                    "representative": representative,
                    "candidate_distance": candidate_score,
                    "candidate_sample_ids": candidate_ids,
                    "competitor_scores": competitors,
                    "supported": supported,
                    "reason": reason,
                }
            )
        coverage = len(supports) / len(selected[sid]) if selected[sid] else 0.0
        if len(supported_buckets) < policy.min_support_buckets:
            reasons.add("insufficient_contrastive_support_buckets")
        if coverage < policy.min_coverage:
            reasons.add("insufficient_contrastive_query_coverage")
        directions.append(
            DiscoveryDirection(
                query_segment_id=sid,
                query_sample_ids=[sample.sample_id for sample in pools[sid]],
                supported_sample_ids=supports,
                coverage=coverage,
                score=float(median(scores)) if scores else None,
                competitor_segment_ids=negatives,
                sample_contrasts=contrasts,
            )
        )
    supported = (
        all(
            len(direction.supported_sample_ids) >= policy.min_support_buckets
            and direction.coverage >= policy.min_coverage
            and direction.score is not None
            and direction.score <= policy.candidate_far_distance
            for direction in directions
        )
        and "insufficient_contrastive_support_buckets" not in reasons
    )
    if supported:
        if any(
            any(
                c.get("reason") == "compensated_appearance_anchor_support"
                for c in direction.sample_contrasts
            )
            for direction in directions
        ):
            reasons.add("compensated_contrastive_support_lacking_geometric_negatives")
        else:
            reasons.add("reciprocal_cooccurrence_contrastive_support")
    status = (
        "conflict"
        if veto
        else "supported"
        if supported
        else "ambiguous"
        if has_margin_ambiguity
        else "insufficient"
    )
    pair_score = (
        max(direction.score for direction in directions)
        if all(direction.score is not None for direction in directions)
        else None
    )
    return DiscoveryPair(
        pair_id=_pair_id(active[left].match_id, policy.feature_space_id, left, right),
        left_segment_id=left,
        right_segment_id=right,
        status=status,
        score=pair_score,
        reasons=sorted(reasons),
        directions=directions,
        comparisons=[comparisons[key] for key in sorted(comparisons)],
        conflict_observation_ids=sorted(conflict_obs),
    )
