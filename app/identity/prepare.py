"""Prepare segment evidence without changing raw evidence or parsing external notes."""

from collections.abc import Mapping, Sequence
from math import floor

from app.contracts.evidence import (
    EvidenceBatch,
    FeatureSpace,
    FileFeatureReference,
    IdentityEvidence,
    InlineFeature,
    MediaAsset,
    PreparedEvidence,
    PreparedSample,
    PreparedSegment,
    Sample,
)
from app.contracts.identity import (
    IdentityRisk,
    MatchingPolicy,
    RiskReview,
    SampleExclusion,
    Segment,
)
from app.identity.distance import FeatureValidationError, validate_feature, validate_feature_space

FeatureValues = Mapping[tuple[str, int], InlineFeature]


def resolve_feature(
    sample: Sample, space: FeatureSpace, feature_values: FeatureValues | None = None
) -> InlineFeature | None:
    """Resolve a declared reference from caller-provided memory; never open files."""
    reference = next(
        (
            feature
            for feature in sample.features
            if feature.feature_space_id == space.feature_space_id
        ),
        None,
    )
    if reference is None:
        return None
    if isinstance(reference, FileFeatureReference):
        feature = (feature_values or {}).get((reference.feature_file_id, reference.row_index))
        if feature is not None and feature.feature_space_id != reference.feature_space_id:
            raise FeatureValidationError(
                "hydrated feature does not match reference feature_space_id"
            )
        return feature
    return reference


def merge_assets(*collections: Sequence[MediaAsset]) -> list[MediaAsset]:
    """Build a read-only comparison view of compatible resource content facts.

    Missing hashes/sizes remain unknown in their original declarations. Only this
    derived view combines known facts; conflicting known facts are input errors.
    """
    indexed: dict[str, MediaAsset] = {}
    for asset in sorted(
        (asset for collection in collections for asset in collection),
        key=lambda asset: (asset.asset_id, asset.file.file_id, asset.file.path),
    ):
        previous = indexed.get(asset.asset_id)
        if previous is None:
            indexed[asset.asset_id] = asset.model_copy(deep=True)
            continue
        if previous.media_type != asset.media_type or any(
            left is not None and right is not None and left != right
            for left, right in (
                (previous.file.sha256, asset.file.sha256),
                (previous.file.size_bytes, asset.file.size_bytes),
            )
        ):
            raise ValueError("query/gallery asset_id has conflicting content declarations")
        indexed[asset.asset_id] = previous.model_copy(
            deep=True,
            update={
                "file": previous.file.model_copy(
                    deep=True,
                    update={
                        "sha256": previous.file.sha256 or asset.file.sha256,
                        "size_bytes": (
                            previous.file.size_bytes
                            if previous.file.size_bytes is not None
                            else asset.file.size_bytes
                        ),
                    },
                )
            },
        )
    return list(indexed.values())


def evidence_keys(sample: Sample, assets: Sequence[MediaAsset] = ()) -> set[tuple]:
    """Equivalent evidence keys exclude source version and import batch identity."""
    keys = {
        ("sample", sample.match_id, sample.sample_id),
        (
            "source",
            sample.match_id,
            sample.source.producer,
            sample.source.record_type,
            sample.source.source_record_id,
        ),
    }
    observation = sample.frame_idx if sample.frame_idx is not None else sample.time_s
    keys.add(
        (
            "frame" if sample.frame_idx is not None else "time",
            sample.match_id,
            sample.clip_id,
            sample.track_id,
            observation,
        )
    )
    # Time still identifies a shared observation if only one record has a frame.
    keys.add(("time", sample.match_id, sample.clip_id, sample.track_id, sample.time_s))
    if sample.image_asset_id is not None:
        keys.add(("asset", sample.match_id, sample.image_asset_id))
        asset = next((item for item in assets if item.asset_id == sample.image_asset_id), None)
        if asset is not None and asset.file.sha256 is not None:
            keys.add(("image", sample.match_id, asset.file.sha256))
    return keys


def are_duplicate_samples(a: Sample, b: Sample, assets: Sequence[MediaAsset] = ()) -> bool:
    return bool(evidence_keys(a, assets) & evidence_keys(b, assets))


def _within(sample: Sample, segment: Segment) -> bool:
    return (
        (sample.match_id, sample.clip_id, sample.track_id)
        == (segment.match_id, segment.clip_id, segment.track_id)
        and segment.start_s <= sample.time_s < segment.end_s
        and (
            sample.frame_idx is None
            or segment.frame_start is None
            or segment.frame_start <= sample.frame_idx < segment.frame_end_exclusive
        )
    )


def _risk_applies(risk: IdentityRisk, segment: Segment, samples: Sequence[Sample]) -> bool:
    if risk.match_id != segment.match_id:
        return False
    if (
        risk.segment_id is not None
        and risk.segment_id != segment.segment_id
        and risk.segment_id != segment.supersedes_segment_id
    ):
        return False
    if risk.clip_id is not None and risk.clip_id != segment.clip_id:
        return False
    if risk.track_id is not None and risk.track_id != segment.track_id:
        return False
    if risk.sample_id is not None and not any(
        sample.sample_id == risk.sample_id and _within(sample, segment) for sample in samples
    ):
        return False
    return risk.start_s is None or (risk.start_s < segment.end_s and risk.end_s > segment.start_s)


def active_risks(
    evidence: IdentityEvidence, segment: Segment, risk_reviews: Sequence[RiskReview] = ()
) -> list[IdentityRisk]:
    """Keep known mixed segments blocked even if their offending images are excluded."""
    scoped_samples = [sample for sample in evidence.samples if _within(sample, segment)]
    risks = list(evidence.risks)
    for sample in scoped_samples:
        quality = sample.tracking_quality
        if quality is None:
            continue
        risks.extend(quality.risks)
        if quality.mixed_identity:
            risks.append(
                IdentityRisk(
                    risk_id=f"mixed_identity:{sample.sample_id}",
                    kind="mixed_identity",
                    match_id=sample.match_id,
                    sample_id=sample.sample_id,
                    source=sample.source.model_copy(deep=True),
                )
            )
    indexed = {}
    for risk in risks:
        if not _risk_applies(risk, segment, scoped_samples):
            continue
        previous = indexed.get(risk.risk_id)
        if previous is not None and previous != risk:
            raise ValueError(f"risk {risk.risk_id!r}: conflicting risk declarations")
        indexed[risk.risk_id] = risk
    reviews = {}
    for review in risk_reviews:
        if review.match_id != segment.match_id or review.segment_id != segment.segment_id:
            continue
        previous = reviews.get(review.risk_id)
        if previous is not None and previous.revision == review.revision and previous != review:
            raise ValueError(f"risk {review.risk_id!r}: conflicting reviews at same revision")
        if previous is None or review.revision > previous.revision:
            reviews[review.risk_id] = review
    return [
        risk.model_copy(deep=True)
        for key, risk in sorted(indexed.items())
        if key not in reviews or reviews[key].resolution != "cleared"
    ]


def _rank(sample: PreparedSample) -> tuple:
    return (
        not sample.clean_approved,
        -sample.visible_parts,
        -sample.visibility_sum,
        sample.sample.time_s,
        sample.sample.sample_id,
    )


def group_duplicate_samples(
    samples: Sequence[PreparedSample], assets: Sequence[MediaAsset] = ()
) -> list[list[PreparedSample]]:
    """Group transitive duplicate observations before choosing any representative.

    An observation can connect records by different keys, such as sharing a frame
    with one record and an image hash with another. Pairwise greedy selection can
    count that connected component twice and is intentionally avoided here.
    """
    parents = list(range(len(samples)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    known = {}
    for index, sample in enumerate(samples):
        for key in evidence_keys(sample.sample, assets):
            if key in known:
                parents[find(index)] = find(known[key])
            known[key] = index
    groups = {}
    for index, sample in enumerate(samples):
        groups.setdefault(find(index), []).append(sample)
    return sorted(
        (sorted(group, key=_rank) for group in groups.values()),
        key=lambda group: _rank(group[0]),
    )


def deduplicate_samples(
    samples: Sequence[PreparedSample], assets: Sequence[MediaAsset] = ()
) -> tuple[list[PreparedSample], list[SampleExclusion]]:
    """Choose one deterministic representative from each transitive duplicate group."""
    unique = []
    exclusions = []
    for ordered in group_duplicate_samples(samples, assets):
        selected = ordered[0]
        unique.append(selected)
        exclusions.extend(
            SampleExclusion(
                sample_id=sample.sample.sample_id,
                reason="duplicate_evidence",
                duplicate_of=selected.sample.sample_id,
            )
            for sample in ordered[1:]
        )
    return sorted(unique, key=_rank), exclusions


def _validate_segments(segments: Sequence[Segment]) -> None:
    grouped = {}
    for segment in segments:
        if segment.active:
            grouped.setdefault((segment.match_id, segment.clip_id, segment.track_id), []).append(
                segment
            )
    for group in grouped.values():
        ordered = sorted(group, key=lambda segment: (segment.start_s, segment.segment_id))
        for left, right in zip(ordered, ordered[1:]):
            if left.end_s > right.start_s or (
                left.frame_start is not None
                and right.frame_start is not None
                and left.frame_end_exclusive > right.frame_start
            ):
                raise ValueError("active identity segments on the same track must not overlap")


def prepare_evidence(
    batch: EvidenceBatch,
    policy: MatchingPolicy,
    *,
    feature_values: FeatureValues | None = None,
    risk_reviews: Sequence[RiskReview] = (),
) -> PreparedEvidence:
    """Validate and select evidence; unknown or missing evidence remains reviewable.

    All usable images are retained for later conflict checks. Only distinct
    observations compete for the single representative in each time bucket.
    """
    evidence = batch.identity_evidence or IdentityEvidence()
    _validate_segments(evidence.segments)
    if any(review.match_id != batch.match_id for review in risk_reviews):
        raise ValueError("risk reviews must belong to the batch match_id")
    space = next(
        (
            item
            for item in evidence.feature_spaces
            if item.feature_space_id == policy.feature_space_id
        ),
        None,
    )
    if space is not None:
        validate_feature_space(space, policy)
    prepared = PreparedEvidence(
        match_id=batch.match_id,
        feature_space_id=policy.feature_space_id,
        policy_version=policy.policy_version,
        policy_fingerprint=policy.fingerprint,
        feature_space=space.model_copy(deep=True) if space is not None else None,
        assets=[
            asset.model_copy(deep=True)
            for asset in sorted(batch.assets, key=lambda item: item.asset_id)
        ],
    )
    by_segment = {segment.segment_id: [] for segment in evidence.segments}
    segments = {segment.segment_id: segment for segment in evidence.segments}
    features = {}
    for sample in sorted(evidence.samples, key=lambda item: item.sample_id):
        feature = resolve_feature(sample, space, feature_values) if space is not None else None
        if feature is not None:
            validate_feature(space, feature, policy)
        features[sample.sample_id] = feature
        if sample.segment_id is not None:
            targets = [segments[sample.segment_id]] if sample.segment_id in segments else []
        else:
            targets = [
                segment
                for segment in evidence.segments
                if segment.active and _within(sample, segment)
            ]
        if len(targets) != 1:
            prepared.exclusions.append(
                SampleExclusion(sample_id=sample.sample_id, reason="unresolved_segment_reference")
            )
            continue
        segment = targets[0]
        if not _within(sample, segment):
            raise ValueError(f"sample {sample.sample_id!r} outside identity segment scope/range")
        by_segment[segment.segment_id].append(sample)
    for segment in sorted(evidence.segments, key=lambda item: item.segment_id):
        result = PreparedSegment(
            segment=segment.model_copy(deep=True),
            risks=active_risks(evidence, segment, risk_reviews),
        )
        if not segment.active:
            result.reasons.append("inactive_segment")
        if space is None:
            result.reasons.append("missing_feature_space")
        for sample in by_segment[segment.segment_id]:
            feature = features[sample.sample_id]
            quality = sample.tracking_quality
            reason = None
            if not segment.active:
                reason = "inactive_segment"
            elif quality is not None and (
                quality.usability == "unusable" or quality.clean_review == "rejected"
            ):
                reason = "unusable_sample"
            elif space is None:
                reason = "missing_feature_space"
            elif feature is None:
                reference = next(
                    (
                        item
                        for item in sample.features
                        if item.feature_space_id == policy.feature_space_id
                    ),
                    None,
                )
                reason = "unhydrated_feature" if reference is not None else "missing_feature"
            else:
                visible_parts = sum(value > 0 for value in feature.visibility)
                if visible_parts < policy.min_common_parts:
                    reason = "insufficient_visible_parts"
            if reason is not None:
                result.exclusions.append(SampleExclusion(sample_id=sample.sample_id, reason=reason))
                continue
            offset = sample.time_s - segment.start_s
            result.samples.append(
                PreparedSample(
                    sample=sample.model_copy(deep=True),
                    feature=feature.model_copy(deep=True),
                    bucket_index=floor(offset / policy.sample_bucket_s),
                    region=min(2, floor(3 * offset / (segment.end_s - segment.start_s))),
                    visible_parts=visible_parts,
                    visibility_sum=sum(feature.visibility),
                    clean_approved=quality is not None and quality.clean_review == "approved",
                )
            )
        unique, duplicates = deduplicate_samples(result.samples, batch.assets)
        result.exclusions.extend(duplicates)
        buckets = {}
        for sample in unique:
            buckets.setdefault(sample.bucket_index, sample)
        representatives = sorted(
            buckets.values(), key=lambda item: (item.bucket_index, item.sample.sample_id)
        )
        result.representative_sample_ids = [item.sample.sample_id for item in representatives]
        representative_ids = set(result.representative_sample_ids)
        duplicate_ids = {item.sample_id for item in duplicates}
        result.exclusions.extend(
            SampleExclusion(sample_id=item.sample.sample_id, reason="nonrepresentative_bucket")
            for item in unique
            if item.sample.sample_id not in representative_ids
            and item.sample.sample_id not in duplicate_ids
        )
        result.samples.sort(key=lambda item: (item.sample.time_s, item.sample.sample_id))
        result.exclusions.sort(key=lambda item: (item.sample_id, item.reason))
        result.covered_regions = sorted({item.region for item in representatives})
        result.bucket_count = len(representatives)
        if representatives:
            result.coverage_start_s = min(item.sample.time_s for item in representatives)
            result.coverage_end_s = max(item.sample.time_s for item in representatives)
        if result.bucket_count < policy.min_query_buckets:
            result.reasons.append("insufficient_observation_buckets")
        if result.covered_regions != [0, 1, 2]:
            result.reasons.append("insufficient_temporal_coverage")
        if result.risks:
            result.reasons.append("unresolved_identity_risk")
        prepared.segments.append(result)
    return prepared
