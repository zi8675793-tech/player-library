"""Synthetic KPR and segment preparation behavior, independent of sample identities."""

import math
from copy import deepcopy

import pytest

from app.contracts.common import Provenance
from app.contracts.evidence import (
    EvidenceBatch,
    FeatureFile,
    FeatureSpace,
    FileFeatureReference,
    IdentityEvidence,
    InlineFeature,
    MediaAsset,
    Sample,
    SourceFile,
    TrackingQuality,
)
from app.contracts.identity import IdentityRisk, MatchingPolicy, RiskReview, Segment
from app.identity.distance import FeatureValidationError, kpr_distance, validate_feature
from app.identity.prepare import are_duplicate_samples, prepare_evidence


def space(parts=3, dimension=2, feature_space_id="fs"):
    return FeatureSpace(
        feature_space_id=feature_space_id,
        backend="kpr",
        model_hash="0" * 64,
        preprocessing={"version": "v1", "color_space": "RGB"},
        layout="parts",
        shape=[parts, dimension],
        normalization="l2",
        distance="euclidean_sqrt_visibility_mean_div2",
        visibility={"shape": [parts], "semantics": "visibility"},
    )


def feature(parts=3, dimension=2, visibility=None, feature_space_id="fs"):
    return InlineFeature(
        feature_space_id=feature_space_id,
        values=[1.0, *([0.0] * (dimension - 1))] * parts,
        visibility=visibility if visibility is not None else [1.0] * parts,
    )


def sample(sample_id, time_s, *, quality=None, feature_value=None, asset_id=None, frame=None):
    return Sample(
        match_id="m",
        sample_id=sample_id,
        clip_id="c",
        track_id="t",
        segment_id="s",
        time_s=time_s,
        frame_idx=frame,
        source=Provenance(
            producer="synthetic",
            record_type="sample",
            source_record_id=sample_id,
            evidence_version="1",
        ),
        image_asset_id=asset_id,
        tracking_quality=quality,
        features=[feature_value if feature_value is not None else feature()],
    )


def batch(samples=(), *, assets=(), risks=(), end_s=3.0, feature_space=None):
    return EvidenceBatch(
        match_id="m",
        batch_id="b",
        producer="synthetic",
        assets=list(assets),
        identity_evidence=IdentityEvidence(
            segments=[
                Segment(
                    segment_id="s",
                    match_id="m",
                    clip_id="c",
                    track_id="t",
                    start_s=0.0,
                    end_s=end_s,
                )
            ],
            samples=list(samples),
            feature_spaces=[feature_space or space()],
            risks=list(risks),
        ),
    )


@pytest.mark.parametrize("parts,dimension", [(2, 2), (4, 7), (8, 1)])
def test_variable_layout_and_equal_distance(parts, dimension):
    policy = MatchingPolicy(feature_space_id="fs")
    f = feature(parts, dimension)
    result = kpr_distance(space(parts, dimension), f, f, policy)
    assert result.comparable and result.distance == 0.0
    assert result.common_parts == parts


def test_visibility_weighted_distance_formula_and_immutability():
    policy = MatchingPolicy(feature_space_id="fs")
    a = feature(visibility=[1.0, 0.25, 0.0])
    b = InlineFeature(
        feature_space_id="fs", values=[1.0, 0.0, 0.0, 1.0, -1.0, 0.0], visibility=[1.0] * 3
    )
    before = deepcopy((a, b))
    assert kpr_distance(space(), a, b, policy).distance == pytest.approx(math.sqrt(2) / 6)
    assert (a, b) == before


def test_incomparable_results_are_distinct_from_invalid_features():
    policy = MatchingPolicy(feature_space_id="fs")
    result = kpr_distance(space(), feature(visibility=[1.0, 0.0, 0.0]), feature(), policy)
    assert not result.comparable and result.distance is None
    assert result.reason == "insufficient_common_parts"
    other = feature(feature_space_id="different")
    result = kpr_distance(space(), feature(), other, policy)
    assert not result.comparable and result.reason == "feature_space_mismatch"
    for invalid in [
        feature().model_copy(update={"values": [2.0, 0.0] * 3}),
        feature().model_copy(update={"values": [float("nan"), 0.0] * 3}),
        feature().model_copy(update={"values": [1.0]}),
        feature().model_copy(update={"visibility": [1.1] * 3}),
        feature().model_copy(update={"visibility": None}),
    ]:
        with pytest.raises(FeatureValidationError):
            kpr_distance(space(), invalid, feature(), policy)


def test_tiny_weights_and_invisible_parts():
    policy = MatchingPolicy(feature_space_id="fs")
    invisible = feature(visibility=[0.0, 1.0, 1.0]).model_copy(
        update={"values": [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]}
    )
    validate_feature(space(), invisible, policy)
    tiny = feature(visibility=[1e-15] * 3)
    result = kpr_distance(space(), tiny, tiny, policy)
    assert result.reason == "insufficient_visibility_weight" and result.distance is None
    with pytest.raises(FeatureValidationError, match="part count"):
        validate_feature(
            space(), feature(), MatchingPolicy(feature_space_id="fs", min_common_parts=4)
        )
    with pytest.raises(FeatureValidationError, match="unsupported"):
        validate_feature(space().model_copy(update={"distance": "other"}), feature(), policy)


def test_bucket_selection_preserves_all_usable_images_and_ignores_detection_score_and_notes():
    samples = [
        sample("late", 0.4, quality=TrackingQuality(score=1.0, score_semantics="detection")),
        sample(
            "early",
            0.1,
            quality=TrackingQuality(
                score=0.01, score_semantics="detection", notes="mixed identity"
            ),
        ),
        sample("approved", 1.2, quality=TrackingQuality(clean_review="approved")),
        sample("unreviewed", 1.1),
        sample("tail", 2.8),
    ]
    prepared = prepare_evidence(batch(samples), MatchingPolicy(feature_space_id="fs"))
    segment = prepared.segments[0]
    assert segment.representative_sample_ids == ["early", "approved", "tail"]
    assert len(segment.samples) == 5
    assert not segment.risks
    assert segment.covered_regions == [0, 1, 2]
    assert segment.bucket_count == 3
    assert segment.coverage_start_s == 0.1 and segment.coverage_end_s == 2.8
    assert {item.sample_id for item in segment.exclusions} == {"late", "unreviewed"}


def test_known_mixed_segment_remains_blocked_after_bad_sample_exclusion():
    samples = [sample("head", 0.1), sample("middle", 1.4), sample("tail", 2.8)]
    samples.append(
        sample("bad", 1.5, quality=TrackingQuality(mixed_identity=True, usability="unusable"))
    )
    prepared = prepare_evidence(batch(samples), MatchingPolicy(feature_space_id="fs"))
    result = prepared.segments[0]
    assert result.bucket_count == 3
    assert [risk.kind for risk in result.risks] == ["mixed_identity"]
    assert "unresolved_identity_risk" in result.reasons
    assert any(
        item.sample_id == "bad" and item.reason == "unusable_sample" for item in result.exclusions
    )


def test_risk_review_is_explicit_scoped_and_does_not_mutate_evidence():
    risk = IdentityRisk(risk_id="r", kind="duplicate_box", match_id="m", segment_id="s")
    source = batch([sample("head", 0.1)], risks=[risk])
    before = source.model_dump_json()
    review = RiskReview(
        risk_id="r",
        match_id="m",
        segment_id="s",
        actor="human",
        reason="reviewed source",
        revision=1,
        resolution="cleared",
    )
    result = prepare_evidence(source, MatchingPolicy(feature_space_id="fs"), risk_reviews=[review])
    assert result.segments[0].risks == []
    assert source.model_dump_json() == before
    confirmed = review.model_copy(update={"resolution": "confirmed"})
    assert (
        prepare_evidence(source, MatchingPolicy(feature_space_id="fs"), risk_reviews=[confirmed])
        .segments[0]
        .risks
    )


def test_duplicate_observations_and_content_do_not_increase_support_and_order_is_stable():
    assets = [
        MediaAsset(
            asset_id=f"a{i}",
            media_type="image",
            file=SourceFile(file_id=f"f{i}", path=f"image{i}", sha256="a" * 64),
        )
        for i in range(2)
    ]
    samples = [
        sample("first", 0.1, asset_id="a0"),
        sample("copy", 1.2, asset_id="a1"),
        sample("tail", 2.8),
    ]
    source = batch(samples, assets=assets)
    policy = MatchingPolicy(feature_space_id="fs")
    assert are_duplicate_samples(samples[0], samples[1], assets)
    result = prepare_evidence(source, policy)
    assert result.segments[0].representative_sample_ids == ["first", "tail"]
    assert result.segments[0].bucket_count == 2
    assert len(result.segments[0].samples) == 3
    reversed_source = source.model_copy(deep=True)
    reversed_source.identity_evidence.samples.reverse()
    assert prepare_evidence(reversed_source, policy) == result
    assert any(item.reason == "duplicate_evidence" for item in result.segments[0].exclusions)


def test_hydration_is_in_memory_and_missing_features_are_reviewable():
    ref = FileFeatureReference(feature_space_id="fs", feature_file_id="ff", row_index=0)
    source = batch([sample("one", 0.1, feature_value=ref)])
    source.identity_evidence.feature_files = [
        FeatureFile(
            feature_file_id="ff",
            feature_space_id="fs",
            sample_count=1,
            values_file=SourceFile(file_id="f", path="unopened.npz"),
            index_file=SourceFile(file_id="i", path="unopened.jsonl"),
            visibility_file=SourceFile(file_id="f", path="unopened.npz"),
        )
    ]
    policy = MatchingPolicy(feature_space_id="fs")
    assert prepare_evidence(source, policy).segments[0].exclusions[0].reason == "unhydrated_feature"
    values = {("ff", 0): feature()}
    before = deepcopy(values)
    assert len(prepare_evidence(source, policy, feature_values=values).segments[0].samples) == 1
    assert values == before
    with pytest.raises(FeatureValidationError, match="feature_space_id"):
        prepare_evidence(
            source, policy, feature_values={("ff", 0): feature(feature_space_id="other")}
        )


def test_sparse_or_short_segments_preserve_coverage_reasons_and_unresolved_samples():
    source = batch([sample("one", 0.1)], end_s=0.4)
    source.identity_evidence.samples.append(
        sample("late-reference", 0.2).model_copy(update={"segment_id": "missing"})
    )
    result = prepare_evidence(source, MatchingPolicy(feature_space_id="fs"))
    assert result.exclusions[0].reason == "unresolved_segment_reference"
    assert set(result.segments[0].reasons) == {
        "insufficient_observation_buckets",
        "insufficient_temporal_coverage",
    }


def test_active_overlaps_and_illegal_excluded_features_raise():
    source = batch([sample("one", 0.1)])
    source.identity_evidence.segments.append(
        source.identity_evidence.segments[0].model_copy(update={"segment_id": "overlap"})
    )
    with pytest.raises(ValueError, match="overlap"):
        prepare_evidence(source, MatchingPolicy(feature_space_id="fs"))
    bad = sample("bad", 0.1, quality=TrackingQuality(usability="unusable"))
    bad.features[0].values = [0.0] * 6
    source = batch([])
    source.identity_evidence.samples.append(bad)
    with pytest.raises(FeatureValidationError, match="L2"):
        prepare_evidence(source, MatchingPolicy(feature_space_id="fs"))


def test_invisible_finite_values_do_not_poison_distance():
    huge = feature(visibility=[0.0, 1.0, 1.0]).model_copy(
        update={"values": [1e308, -1e308, 1.0, 0.0, 1.0, 0.0]}
    )
    opposite = huge.model_copy(update={"values": [-1e308, 1e308, 1.0, 0.0, 1.0, 0.0]})
    assert (
        kpr_distance(space(), huge, opposite, MatchingPolicy(feature_space_id="fs")).distance == 0.0
    )


def test_results_are_detached_and_assets_order_is_irrelevant():
    assets = [
        MediaAsset(
            asset_id=f"a{i}", media_type="image", file=SourceFile(file_id=f"f{i}", path=f"image{i}")
        )
        for i in range(2)
    ]
    source = batch([sample("head", 0.1, asset_id="a0")], assets=assets)
    before = source.model_dump_json()
    policy = MatchingPolicy(feature_space_id="fs")
    result = prepare_evidence(source, policy)
    reversed_source = source.model_copy(deep=True)
    reversed_source.assets.reverse()
    assert prepare_evidence(reversed_source, policy) == result
    result.segments[0].samples[0].sample.source.locations.append({"file_id": "mutation"})
    result.segments[0].samples[0].feature.values[0] = -1.0
    assert source.model_dump_json() == before


def test_structured_risks_require_known_scope_consistency():
    from pydantic import ValidationError

    risk = IdentityRisk(
        risk_id="r", kind="quality_review", match_id="m", segment_id="s", clip_id="wrong"
    )
    with pytest.raises(ValidationError, match="scope"):
        batch([sample("head", 0.1)], risks=[risk])
    risk = IdentityRisk(risk_id="r", kind="quality_review", match_id="other", segment_id="late")
    with pytest.raises(ValidationError, match="match_id"):
        batch([], risks=[risk])
