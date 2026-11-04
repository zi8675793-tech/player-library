"""Synthetic, open-set candidate scoring; no supplied jersey labels are truth."""

import copy
import math

import pytest

from app.contracts.common import Provenance
from app.contracts.evidence import (
    EvidenceBatch,
    FeatureSpace,
    GallerySnapshot,
    IdentityEvidence,
    InlineFeature,
    MediaAsset,
    Sample,
    SourceFile,
    Track,
    TrackingQuality,
)
from app.contracts.identity import Assignment, MatchingPolicy, Player, Segment, Template
from app.identity.associate import associate_segments
from app.identity.prepare import prepare_evidence


def feature(angle=0.0, visibility=None):
    return InlineFeature(
        feature_space_id="space",
        values=[math.cos(angle), math.sin(angle)] * 3,
        visibility=visibility or [1.0] * 3,
    )


def source(identifier):
    return Provenance(
        producer="synthetic",
        record_type="sample",
        source_record_id=identifier,
        evidence_version="1",
    )


def sample(identifier, segment, time_s, angle=0.0, visibility=None):
    return Sample(
        match_id="match",
        sample_id=identifier,
        clip_id=segment.clip_id,
        track_id=segment.track_id,
        segment_id=segment.segment_id,
        time_s=time_s,
        source=source(identifier),
        tracking_quality=TrackingQuality(usability="usable", clean_review="approved"),
        features=[feature(angle, visibility)],
    )


def segment(identifier, *, end_s=1.5):
    return Segment(
        match_id="match",
        clip_id=identifier,
        track_id="track",
        segment_id=identifier,
        start_s=0.0,
        end_s=end_s,
    )


def evidence(segments, samples):
    return IdentityEvidence(
        segments=segments,
        tracks=[
            Track(
                **item.model_dump(exclude={"segment_id", "active", "supersedes_segment_id"}),
                source=source(item.segment_id),
            )
            for item in segments
        ],
        samples=samples,
        feature_spaces=[
            FeatureSpace(
                feature_space_id="space",
                backend="kpr",
                model_hash="a" * 64,
                preprocessing={"version": "1", "color_space": "RGB"},
                layout="parts",
                shape=[3, 2],
                normalization="l2",
                distance="euclidean_sqrt_visibility_mean_div2",
                visibility={"shape": [3], "semantics": "part visibility"},
            )
        ],
    )


def context(angles=(0.0, math.pi), queries=(0.0, 0.0, 0.0)):
    policy = MatchingPolicy(feature_space_id="space", tau=0.2, delta=0.1)
    segments, samples, templates, players, assignments = [], [], [], [], []
    for index, angle in enumerate(angles):
        player_id = f"pl_{index}"
        item = segment(f"seed-{index}")
        segments.append(item)
        players.append(Player(match_id="match", player_id=player_id))
        assignments.append(
            Assignment(
                match_id="match",
                segment_id=item.segment_id,
                player_id=player_id,
                status="confirmed",
                method="manual",
                revision=1,
            )
        )
        for observation in range(2):
            sid = f"seed-{index}-{observation}"
            samples.append(sample(sid, item, 0.1 + observation * 0.5, angle))
            templates.append(
                Template(
                    template_id=f"template-{index}-{observation}",
                    match_id="match",
                    player_id=player_id,
                    feature_space_id="space",
                    sample_id=sid,
                    review_status="approved",
                    revision=1,
                )
            )
    snapshot = GallerySnapshot(
        match_id="match",
        revision=1,
        players=players,
        assignments=assignments,
        templates=templates,
        evidence=evidence(segments, samples),
    )
    query_segment = segment("query", end_s=max(len(queries) * 0.5, 1.5))
    query_samples = [
        sample(f"query-{i}", query_segment, 0.1 + i * 0.5, angle) for i, angle in enumerate(queries)
    ]
    batch = EvidenceBatch(
        match_id="match",
        batch_id="batch",
        producer="synthetic",
        identity_evidence=evidence([query_segment], query_samples),
    )
    return batch, snapshot, policy


def associate(batch, snapshot, policy):
    return associate_segments(prepare_evidence(batch, policy), snapshot, policy)[0]


def test_nearest_two_mean_then_segment_median_with_three_candidates():
    batch, snapshot, policy = context(angles=(0.0, math.pi / 2, math.pi))
    extra = sample("extra", snapshot.evidence.segments[0], 1.1, math.pi)
    snapshot.evidence.samples.append(extra)
    snapshot.templates.append(
        Template(
            template_id="extra-template",
            match_id="match",
            player_id="pl_0",
            feature_space_id="space",
            sample_id="extra",
            review_status="approved",
            revision=1,
        )
    )
    result = associate(batch, snapshot, policy)
    assert [item.player_id for item in result.candidates] == ["pl_0", "pl_1", "pl_2"]
    assert [item.distance for item in result.candidates] == pytest.approx([0.0, 2**-0.5, 1.0])
    assert result.margin == pytest.approx(2**-0.5)
    assert result.support_rate == 1.0
    assert result.common_query_coverage == 1.0
    assert result.blocking_reasons == []
    assert result.candidates[0].template_ids == ["template-0-0", "template-0-1"]
    assert result.candidates[0].common_parts == 3


def test_common_queries_are_identical_for_every_candidate_and_local_scores_remain():
    batch, snapshot, policy = context()
    # Query 2 cannot see the two visible parts on player 1's templates.
    for item in snapshot.evidence.samples[2:]:
        item.features[0].visibility = [1.0, 1.0, 0.0]
    batch.identity_evidence.samples[2].features[0].visibility = [1.0, 0.0, 1.0]
    result = associate(batch, snapshot, policy)
    assert result.common_query_sample_ids == ["query-0", "query-1"]
    assert result.common_query_coverage == pytest.approx(2 / 3)
    assert all(
        item.query_sample_ids == result.common_query_sample_ids for item in result.candidates
    )
    assert result.local_candidates[0].query_sample_ids == ["query-0", "query-1", "query-2"]
    assert "insufficient_common_query_coverage" in result.blocking_reasons
    pairs = result.sample_diagnostics[2].scores[1].comparisons
    assert all(
        pair.distance is None and pair.reason == "insufficient_common_parts" for pair in pairs
    )


def test_minority_under_twenty_percent_vetoes_majority_median():
    batch, snapshot, policy = context(queries=(0.0,) * 5 + (math.pi,))
    result = associate(batch, snapshot, policy)
    assert result.first_player_id == "pl_0"
    assert result.first_distance == 0.0
    assert result.support_rate == pytest.approx(5 / 6)
    assert result.strong_disagreement_sample_ids == ["query-5"]
    assert "strong_support_disagreement" in result.blocking_reasons


def test_nonrepresentative_usable_sample_also_checks_disagreement():
    batch, snapshot, policy = context()
    batch.identity_evidence.samples.append(
        sample("minority", batch.identity_evidence.segments[0], 0.2, math.pi)
    )
    result = associate(batch, snapshot, policy)
    assert "minority" not in result.common_query_sample_ids
    assert "minority" in result.strong_disagreement_sample_ids
    assert not next(
        item for item in result.sample_diagnostics if item.sample_id == "minority"
    ).representative


def test_abstentions_are_in_support_denominator():
    batch, snapshot, policy = context(queries=(0.0,) * 4 + (math.pi / 2,))
    result = associate(batch, snapshot, policy)
    assert result.support_rate == 0.8
    assert result.sample_diagnostics[-1].supported_player_id is None


def test_single_candidate_has_no_margin_or_vote():
    batch, snapshot, policy = context(angles=(0.0,))
    result = associate(batch, snapshot, policy)
    assert len(result.candidates) == 1
    assert result.margin is None
    assert result.second_distance is None
    assert result.support_rate == 0.0
    assert "fewer_than_two_candidates" in result.blocking_reasons


def test_single_template_is_preview_and_missing_player_blocks_common_queries():
    batch, snapshot, policy = context()
    snapshot.templates.pop()
    result = associate(batch, snapshot, policy)
    assert result.candidates == []
    assert [(item.player_id, item.score_kind) for item in result.local_candidates] == [
        ("pl_0", "double_template"),
        ("pl_1", "single_template_preview"),
    ]
    assert result.margin is None
    snapshot.players.append(Player(match_id="match", player_id="pl_unseen"))
    result = associate(batch, snapshot, policy)
    assert result.common_query_coverage == 0.0


def test_same_image_content_is_not_self_support_even_across_sample_and_asset_ids():
    batch, snapshot, policy = context()
    batch.assets.append(
        MediaAsset(
            asset_id="query-image",
            media_type="image",
            file=SourceFile(file_id="query-file", path="query.png", sha256="b" * 64),
        )
    )
    snapshot.assets.append(
        MediaAsset(
            asset_id="seed-image",
            media_type="image",
            file=SourceFile(file_id="seed-file", path="seed.png", sha256="b" * 64),
        )
    )
    batch.identity_evidence.samples[0].image_asset_id = "query-image"
    snapshot.evidence.samples[0].image_asset_id = "seed-image"
    result = associate(batch, snapshot, policy)
    query = result.sample_diagnostics[0]
    assert query.scores[0].distance is None
    assert any(item.reason == "duplicate_query_evidence" for item in query.scores[0].comparisons)
    assert result.common_query_coverage == pytest.approx(2 / 3)


def test_duplicate_source_observation_cannot_supply_two_templates():
    batch, snapshot, policy = context()
    snapshot.evidence.samples[1].source.source_record_id = snapshot.evidence.samples[
        0
    ].source.source_record_id
    result = associate(batch, snapshot, policy)
    assert not result.candidates
    assert result.local_candidates[0].score_kind == "single_template_preview"
    assert any(
        item.reason == "duplicate_template_evidence"
        for item in result.sample_diagnostics[0].scores[0].comparisons
    )


def test_template_current_assignment_and_review_are_both_required():
    batch, snapshot, policy = context()
    snapshot.assignments[0].status = "proposed"
    snapshot.templates[-1].review_status = "unreviewed"
    result = associate(batch, snapshot, policy)
    assert result.candidates == []
    assert all(score.distance is None for score in result.sample_diagnostics[0].scores)
    assert (
        "template_assignment_not_confirmed"
        in result.sample_diagnostics[0].scores[0].comparisons[0].reason
    )


def test_reordering_all_input_lists_is_deterministic_and_does_not_mutate_inputs():
    batch, snapshot, policy = context()
    originals = copy.deepcopy((batch, snapshot, policy))
    first = associate(batch, snapshot, policy)
    assert (batch, snapshot, policy) == originals
    batch.identity_evidence.samples.reverse()
    snapshot.players.reverse()
    snapshot.evidence.samples.reverse()
    snapshot.templates.reverse()
    snapshot.assignments.reverse()
    snapshot.evidence.segments.reverse()
    snapshot.evidence.tracks.reverse()
    reversed_originals = copy.deepcopy((batch, snapshot, policy))
    assert associate(batch, snapshot, policy) == first
    assert (batch, snapshot, policy) == reversed_originals


def test_same_space_id_with_different_contract_is_error_and_other_space_has_no_templates():
    batch, snapshot, policy = context()
    snapshot.evidence.feature_spaces[0].preprocessing.version = "changed"
    with pytest.raises(ValueError, match="semantic contracts"):
        associate(batch, snapshot, policy)
    batch, snapshot, policy = context()
    for template in snapshot.templates:
        template.feature_space_id = "other-space"
    assert not associate(batch, snapshot, policy).candidates


def test_mean_of_two_templates_and_median_of_queries_are_distinct_aggregations():
    batch, snapshot, policy = context(queries=(0.0, 0.0, math.pi))
    snapshot.evidence.samples[1].features = [feature(math.pi / 3)]
    result = associate(batch, snapshot, policy)
    first_scores = [item.scores[0].distance for item in result.sample_diagnostics]
    assert first_scores == pytest.approx([0.25, 0.25, (1 + math.sqrt(3) / 2) / 2])
    assert result.candidates[0].distance == pytest.approx(0.25)
    assert result.candidates[0].distance != pytest.approx(sum(first_scores) / 3)


def test_tied_candidates_keep_zero_margin_and_no_identity_vote():
    batch, snapshot, policy = context(angles=(0.0, 0.0))
    result = associate(batch, snapshot, policy)
    assert result.margin == 0.0
    assert result.support_rate == 0.0
    assert "tied_candidates" in result.blocking_reasons
    assert all(item.supported_player_id is None for item in result.sample_diagnostics)


def test_unconfigured_thresholds_keep_candidate_diagnostics_without_support_votes():
    batch, snapshot, _ = context()
    policy = MatchingPolicy(feature_space_id="space")
    result = associate(batch, snapshot, policy)
    assert result.candidates[0].player_id == "pl_0"
    assert result.support_rate is None
    assert all(item.reason == "thresholds_unset" for item in result.sample_diagnostics)


def test_prepared_sampling_policy_cannot_be_reused_with_different_parameters():
    batch, snapshot, policy = context()
    prepared = prepare_evidence(batch, policy)
    changed = MatchingPolicy(feature_space_id="space", tau=0.2, delta=0.1, sample_bucket_s=0.25)
    with pytest.raises(ValueError, match="selected space/policy"):
        associate_segments(prepared, snapshot, changed)


def test_conflicting_confirmed_gallery_evidence_blocks_automatic_but_keeps_distances():
    batch, snapshot, policy = context()
    # Human-confirmed distinct segments assert two players for one source observation.
    snapshot.evidence.samples[2].source.source_record_id = snapshot.evidence.samples[
        0
    ].source.source_record_id
    before = snapshot.model_dump_json()
    result = associate(batch, snapshot, policy)
    assert "gallery_identity_conflict" in result.blocking_reasons
    assert result.gallery_conflict_template_ids == ["template-0-0", "template-1-0"]
    assert len(result.candidates) == 2
    assert result.candidates[0].distance == 0.0
    assert result.candidates[1].distance == 1.0
    assert snapshot.model_dump_json() == before


def test_same_asset_id_cannot_hide_conflicting_content_across_query_and_gallery():
    batch, snapshot, policy = context()
    for assets, digest in [(batch.assets, "a" * 64), (snapshot.assets, "b" * 64)]:
        assets.append(
            MediaAsset(
                asset_id="shared-image",
                media_type="image",
                file=SourceFile(file_id="shared-file", path="shared.png", sha256=digest),
            )
        )
    with pytest.raises(ValueError, match="conflicting content"):
        associate(batch, snapshot, policy)


def test_transitive_self_evidence_excludes_selected_and_duplicate_reference_members():
    batch, snapshot, policy = context()
    connector = sample("zz-connector", snapshot.evidence.segments[0], 1.1)
    connector.source = snapshot.evidence.samples[0].source.model_copy(deep=True)
    connector.image_asset_id = "connector-image"
    snapshot.evidence.samples.append(connector)
    snapshot.templates.append(
        Template(
            template_id="zz-connector-template",
            match_id="match",
            player_id="pl_0",
            feature_space_id="space",
            sample_id=connector.sample_id,
            review_status="approved",
            revision=1,
        )
    )
    snapshot.assets.append(
        MediaAsset(
            asset_id="connector-image",
            media_type="image",
            file=SourceFile(file_id="connector-file", path="connector.png", sha256="b" * 64),
        )
    )
    batch.identity_evidence.samples[0].image_asset_id = "query-image"
    batch.assets.append(
        MediaAsset(
            asset_id="query-image",
            media_type="image",
            file=SourceFile(file_id="query-file", path="query.png", sha256="b" * 64),
        )
    )
    before = [item.model_dump(mode="json") for item in (batch, snapshot)]
    result = associate(batch, snapshot, policy)
    first_score = result.sample_diagnostics[0].scores[0]
    excluded = {
        pair.template_id
        for pair in first_score.comparisons
        if pair.reason == "duplicate_query_evidence"
    }
    assert excluded == {"template-0-0", "zz-connector-template"}
    assert first_score.distance is None
    assert first_score.preview_distance == 0.0
    assert result.common_query_sample_ids == ["query-1", "query-2"]
    assert result.common_query_coverage == pytest.approx(2 / 3)
    assert before == [item.model_dump(mode="json") for item in (batch, snapshot)]
