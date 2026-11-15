"""T2 synthetic contract/physics tests; no images, labels or model downloads.

The fixtures use real T1 frozen dataclasses and 512-dimensional float32 unit
vectors. Every geometry witness names an actual observed frame; an interval
overlap, repeated crop or second prototype is never an independent witness.
"""

import builtins
import math
from dataclasses import FrozenInstanceError, fields, replace

import numpy as np
import pytest

from app.features.dual_stream import (
    AppearanceEmbedding,
    DualStreamFeature,
    JerseyNumberPrediction,
    TrackletSegment,
    TrackObservation,
)
from app.identity.constraints import CannotLinkConstraint, SpatioTemporalConstraintEngine
from app.identity.matching import (
    DecisionOutcome,
    IncrementalDecision,
    IncrementalMatcher,
    IncrementalPlan,
    PlayerProfile,
    PlayerPrototype,
)


def segment(video="v", track=1, frames=(0, 1, 2), *, x=0.0):
    observations = tuple(
        TrackObservation(video, track, frame, frame / 60, (x, 0.0, x + 20, 50.0), 0.9)
        for frame in frames
    )
    return TrackletSegment(
        f"{video}_{track}_{frames[0]}",
        video,
        track,
        frames[0],
        frames[-1],
        frames[0] / 60,
        frames[-1] / 60,
        len(observations),
        observations,
    )


def unit_vector(distance=0.0, *, axis=0):
    """Cosine distance from e[axis] is the requested value, up to float32."""
    vector = np.zeros(512, dtype=np.float32)
    vector[axis] = 1.0 - distance
    vector[(axis + 1) % 512] = math.sqrt(max(0.0, 1 - (1 - distance) ** 2))
    return vector


def feature(
    part,
    *,
    number=None,
    legible=True,
    confidence=0.95,
    distance=0.0,
    quality=0.95,
    model="synthetic-reid-v1",
    vector=None,
):
    jersey = (
        None
        if number is None
        else JerseyNumberPrediction(number, confidence, legible, {number: confidence}, str(number))
    )
    appearance = AppearanceEmbedding(
        unit_vector(distance) if vector is None else vector, quality, model
    )
    return DualStreamFeature(part.segment_id, jersey, appearance)


def prototype(item, player_id="p", *, name=None, view="generic", weight=0.95, revision=2):
    return PlayerPrototype(
        name or f"proto-{player_id}-{item.segment_id}", player_id, view, item, weight, revision
    )


def profile(*items, player_id="p", number=None, status="active", revision=1):
    return PlayerProfile(player_id, player_id, number, tuple(items), status, revision)


def plan(features, players=(), *, segments, pending=(), matcher=None, assignments=None, revision=4):
    return (matcher or IncrementalMatcher()).plan(
        features,
        players,
        segments=segments,
        batch_id="synthetic-t2",
        from_revision=revision,
        pending_features=pending,
        player_assignments=assignments,
    )


def decisions(result):
    return {item.segment_id: item for item in result.decisions}


def assert_deferred(result, *parts):
    by_id = decisions(result)
    for part in parts:
        item = by_id[part.segment_id]
        assert item.outcome == DecisionOutcome.INSUFFICIENT
        assert item.target_player_id is None
        assert part.segment_id in result.deferred_segments


@pytest.mark.parametrize(
    ("contract", "names"),
    [
        (CannotLinkConstraint, "segment_a segment_b reason"),
        (PlayerPrototype, "prototype_id player_id view_type feature weight created_at_revision"),
        (
            PlayerProfile,
            "player_id display_name confirmed_jersey_number prototypes status created_at_revision",
        ),
        (
            IncrementalDecision,
            "segment_id outcome target_player_id best_distance competitive_margin evidence",
        ),
        (
            IncrementalPlan,
            "batch_id from_revision to_revision decisions new_players "
            "updated_prototypes deferred_segments",
        ),
    ],
)
def test_contract_fields_match_frozen_documentation(contract, names):
    assert [item.name for item in fields(contract)] == names.split()
    assert contract.__dataclass_params__.frozen


def test_three_outcomes_keep_the_documented_wire_values():
    assert {item.value for item in DecisionOutcome} == {
        "matched_existing",
        "created_new_identity",
        "insufficient_evidence",
    }


@pytest.mark.parametrize("x", [0.0, 5.0, 20.0, 500.0])
def test_one_shared_physical_frame_forbids_even_overlapping_boxes(x):
    left, right = segment(track=1, frames=(9,)), segment(track=2, frames=(9,), x=x)
    engine = SpatioTemporalConstraintEngine(segments=[left, right])
    assert engine.cannot_link(left.segment_id, right.segment_id)
    assert engine.cannot_link(right.segment_id, left.segment_id)
    assert engine.pair_relation(left, right) == "conflict"
    assert engine.is_forbidden(0, 1)
    assert engine.is_forbidden(1, 0)
    assert not engine.is_unresolved(0, 1)
    assert math.isinf(engine.constrained_distance(left.segment_id, right.segment_id, 0.0))
    assert not engine.can_merge_clusters({0}, {1})
    assert len(engine.constraints) == 1
    witness = engine.constraints[0]
    assert {witness.segment_a, witness.segment_b} == {left.segment_id, right.segment_id}
    assert witness.reason == "same_frame_spatial_separation"


def test_physical_frame_identity_takes_priority_over_rounded_timestamps():
    left, right = segment(track=1, frames=(9,)), segment(track=2, frames=(9,))
    shifted = replace(right.observations[0], timestamp_s=0.151)
    right = replace(right, observations=(shifted,), start_time_s=0.151, end_time_s=0.151)
    engine = SpatioTemporalConstraintEngine(segments=[left, right])
    assert engine.cannot_link(left.segment_id, right.segment_id)


def test_same_frame_index_in_different_videos_does_not_conflict():
    left, right = segment("clip_a"), segment("clip_b")
    engine = SpatioTemporalConstraintEngine([left, right])
    assert not engine.cannot_link(left.segment_id, right.segment_id)
    assert engine.pair_relation(left, right) == "clear"
    assert engine.constrained_distance(left.segment_id, right.segment_id, 0.12) == 0.12


def test_interval_overlap_without_an_actual_shared_frame_is_not_a_cannot_link():
    left = segment(track=1, frames=(0, 4))
    right = segment(track=2, frames=(1, 3))
    engine = SpatioTemporalConstraintEngine(segments=[left, right])
    assert not engine.cannot_link(left.segment_id, right.segment_id)


def test_same_track_segments_and_self_are_not_different_people():
    left, right = segment(frames=(0, 1)), segment(frames=(1, 2))
    engine = SpatioTemporalConstraintEngine(segments=[left, right])
    assert not engine.cannot_link(left.segment_id, right.segment_id)
    assert not engine.cannot_link(left.segment_id, left.segment_id)


def test_unknown_segment_metadata_is_unresolved():
    known = segment()
    engine = SpatioTemporalConstraintEngine(segments=[known])
    assert engine.pair_relation(known.segment_id, "missing") == "unresolved"


def test_player_blocker_checks_every_assigned_historical_segment():
    left, right, unrelated = segment(track=1), segment(track=2), segment("other")
    engine = SpatioTemporalConstraintEngine(
        segments=[left, right, unrelated],
        player_assignments={right.segment_id: "p", unrelated.segment_id: "other-p"},
    )
    assert engine.is_segment_forbidden_for_player(left.segment_id, "p")
    assert not engine.is_segment_forbidden_for_player(left.segment_id, "other-p")


@pytest.mark.parametrize(
    "change",
    [
        {"observation_count": 9},
        {"start_frame": 9},
        {"end_frame": 99},
        {"video_id": "different"},
        {"observations": ()},
    ],
)
def test_inconsistent_segment_metadata_fails_before_graph_construction(change):
    with pytest.raises(ValueError):
        SpatioTemporalConstraintEngine(segments=[replace(segment(), **change)])


def test_duplicate_segment_ids_are_rejected_by_graph():
    part = segment()
    with pytest.raises(ValueError):
        SpatioTemporalConstraintEngine(segments=[part, part])


def test_cross_video_readable_matching_jersey_matches_existing():
    old, query = segment("old"), segment("new")
    player = profile(prototype(feature(old, number=27), view="dorsal_with_number"), number=27)
    result = plan([feature(query, number=27)], [player], segments=[old, query])
    item = result.decisions[0]
    assert item.outcome == DecisionOutcome.MATCHED_EXISTING
    assert item.target_player_id == "p"
    assert item.best_distance == pytest.approx(0.0, abs=1e-7)
    assert item.competitive_margin >= 0.04
    assert not result.new_players
    assert not result.deferred_segments


@pytest.mark.parametrize("number", [0, 10, 15, 99])
def test_readable_different_number_vetoes_identical_appearance(number):
    old, query = segment("old"), segment("new")
    player = profile(prototype(feature(old, number=27)), number=27)
    result = plan([feature(query, number=number)], [player], segments=[old, query])
    assert result.decisions[0].target_player_id is None
    assert math.isinf(result.decisions[0].best_distance)
    assert len(result.new_players) == 1
    assert result.new_players[0].confirmed_jersey_number == number
    assert result.decisions[0].outcome == DecisionOutcome.CREATED_NEW


def test_unreadable_number_falls_back_to_appearance():
    old, query = segment("old"), segment("new")
    player = profile(prototype(feature(old, number=27)), number=27)
    item = replace(feature(query), jersey=JerseyNumberPrediction(None, 0.1, False, {}))
    result = plan([item], [player], segments=[old, query])
    assert result.decisions[0].outcome == DecisionOutcome.MATCHED_EXISTING


def test_unreadable_nonempty_number_is_rejected_as_a_contradictory_contract():
    query = segment()
    with pytest.raises(ValueError):
        plan([feature(query, number=10, legible=False)], segments=[query])


def test_readable_number_does_not_have_a_second_hidden_confidence_gate():
    old, query = segment("old"), segment("new")
    player = profile(prototype(feature(old, number=27)), number=27)
    result = plan([feature(query, number=10, confidence=0.1)], [player], segments=[old, query])
    assert result.decisions[0].target_player_id is None
    assert math.isinf(result.decisions[0].best_distance)
    assert len(result.new_players) == 1
    assert result.new_players[0].confirmed_jersey_number == 10
    assert result.decisions[0].outcome == DecisionOutcome.CREATED_NEW


def test_matching_number_does_not_override_bad_appearance():
    old, query = segment("old"), segment("new")
    player = profile(prototype(feature(old, number=27)), number=27)
    result = plan([feature(query, number=27, distance=1.0)], [player], segments=[old, query])
    assert_deferred(result, query)


def test_same_frame_blocks_gallery_match_even_with_matching_number():
    old, query = segment(track=1), segment(track=2, x=100.0)
    player = profile(prototype(feature(old, number=27)), number=27)
    result = plan([feature(query, number=27)], [player], segments=[old, query])
    assert_deferred(result, query)


def test_historical_nonprototype_assignment_also_blocks_match():
    proto_part, historic, query = segment("proto"), segment(track=1), segment(track=2)
    player = profile(prototype(feature(proto_part)), number=None)
    result = plan(
        [feature(query)],
        [player],
        segments=[proto_part, historic, query],
        assignments={historic.segment_id: "p"},
    )
    assert_deferred(result, query)


def test_best_of_multiple_views_is_used_instead_of_a_centroid():
    frontal, dorsal, lateral, query = [segment(name) for name in ("front", "back", "side", "q")]
    player = profile(
        prototype(feature(frontal, distance=1.0), view="frontal"),
        prototype(feature(dorsal, number=27), view="dorsal_with_number"),
        prototype(feature(lateral, distance=1.0), view="lateral"),
        number=27,
    )
    result = plan([feature(query, number=27)], [player], segments=[frontal, dorsal, lateral, query])
    assert result.decisions[0].outcome == DecisionOutcome.MATCHED_EXISTING
    assert result.decisions[0].best_distance == pytest.approx(0.0, abs=1e-7)


def test_margin_competes_between_players_not_prototypes_of_one_player():
    old1, old2, query = segment("a"), segment("b"), segment("q")
    player = profile(prototype(feature(old1)), prototype(feature(old2)))
    result = plan([feature(query)], [player], segments=[old1, old2, query])
    assert result.decisions[0].outcome == DecisionOutcome.MATCHED_EXISTING


@pytest.mark.parametrize(
    ("distance", "expected"),
    [
        (0.25, DecisionOutcome.MATCHED_EXISTING),
        (0.251, DecisionOutcome.INSUFFICIENT),
    ],
)
def test_appearance_threshold_is_inclusive(distance, expected):
    old, query = segment("old"), segment("q")
    result = plan(
        [feature(query, distance=distance)],
        [profile(prototype(feature(old)))],
        segments=[old, query],
    )
    assert result.decisions[0].outcome == expected


@pytest.mark.parametrize(
    ("runner_distance", "expected"),
    [
        (0.04, DecisionOutcome.MATCHED_EXISTING),
        (0.039, DecisionOutcome.INSUFFICIENT),
        (0.0, DecisionOutcome.INSUFFICIENT),
    ],
)
def test_competitive_margin_is_inclusive_and_ties_defer(runner_distance, expected):
    old1, old2, query = segment("a"), segment("b"), segment("q")
    players = [
        profile(prototype(feature(old1))),
        profile(prototype(feature(old2, distance=runner_distance), "runner"), player_id="runner"),
    ]
    result = plan([feature(query)], players, segments=[old1, old2, query])
    assert result.decisions[0].outcome == expected


def test_ambiguous_existing_match_never_creates_a_competing_fragment():
    old1, old2, first, second = [segment(video) for video in ("a", "b", "q1", "q2")]
    players = [
        profile(prototype(feature(old1))),
        profile(prototype(feature(old2), "runner"), player_id="runner"),
    ]
    result = plan([feature(first), feature(second)], players, segments=[old1, old2, first, second])
    assert_deferred(result, first, second)
    assert not result.new_players


def test_runner_up_above_distance_threshold_still_limits_margin():
    old1, old2, query = segment("a"), segment("b"), segment("q")
    players = [
        profile(prototype(feature(old1, distance=0.25))),
        profile(prototype(feature(old2, distance=0.28), "runner"), player_id="runner"),
    ]
    result = plan([feature(query)], players, segments=[old1, old2, query])
    assert_deferred(result, query)


@pytest.mark.parametrize("status", ["merged", "archived"])
def test_nonactive_profiles_are_not_matching_targets(status):
    old, query = segment("old"), segment("q")
    player = profile(prototype(feature(old)), status=status)
    result = plan([feature(query)], [player], segments=[old, query])
    assert_deferred(result, query)


def test_same_model_is_required_for_embedding_comparison():
    old, query = segment("old"), segment("q")
    player = profile(prototype(feature(old, model="other-model")))
    result = plan([feature(query)], [player], segments=[old, query])
    assert_deferred(result, query)


def test_zero_weight_prototype_cannot_establish_identity():
    old, query = segment("old"), segment("q")
    player = profile(prototype(feature(old), weight=0.0))
    result = plan([feature(query)], [player], segments=[old, query])
    assert_deferred(result, query)


def test_low_quality_candidate_remains_deferred():
    old, query = segment("old"), segment("q")
    player = profile(prototype(feature(old)))
    result = plan([feature(query, quality=0.1)], [player], segments=[old, query])
    assert_deferred(result, query)
    assert not result.updated_prototypes


def test_low_quality_prototype_cannot_establish_identity():
    old, query = segment("old"), segment("q")
    player = profile(prototype(feature(old, quality=0.1)))
    result = plan([feature(query)], [player], segments=[old, query])
    assert_deferred(result, query)


@pytest.mark.parametrize(
    ("quality", "expected"),
    [
        (0.5, DecisionOutcome.MATCHED_EXISTING),
        (0.499, DecisionOutcome.INSUFFICIENT),
    ],
)
def test_quality_gate_has_an_inclusive_boundary(quality, expected):
    old, query = segment("old"), segment("q")
    result = plan(
        [feature(query, quality=quality)], [profile(prototype(feature(old)))], segments=[old, query]
    )
    assert result.decisions[0].outcome == expected


@pytest.mark.parametrize(
    ("weight", "expected"),
    [
        (0.1, DecisionOutcome.MATCHED_EXISTING),
        (0.099, DecisionOutcome.INSUFFICIENT),
    ],
)
def test_prototype_trust_gate_has_an_inclusive_boundary(weight, expected):
    old, query = segment("old"), segment("q")
    result = plan(
        [feature(query)], [profile(prototype(feature(old), weight=weight))], segments=[old, query]
    )
    assert result.decisions[0].outcome == expected


def test_self_prototype_is_excluded_from_its_own_reassessment():
    query = segment("q")
    item = feature(query)
    result = plan([item], [profile(prototype(item))], segments=[query])
    assert_deferred(result, query)


def test_two_simultaneous_candidates_cannot_match_one_existing_player():
    old, left, right = segment("old"), segment(track=1), segment(track=2, x=100.0)
    player = profile(prototype(feature(old)))
    result = plan([feature(left), feature(right)], [player], segments=[old, left, right])
    matched = [
        item for item in result.decisions if item.outcome == DecisionOutcome.MATCHED_EXISTING
    ]
    assert len(matched) <= 1
    assert all(item.target_player_id == "p" for item in matched)
    assert not result.new_players


def test_conflicting_numbers_in_one_batch_cannot_match_an_unnumbered_profile():
    old, left, right = segment("old"), segment("q1"), segment("q2")
    player = profile(prototype(feature(old)))
    result = plan(
        [feature(left, number=10), feature(right, number=27)], [player], segments=[old, left, right]
    )
    matched = [
        item for item in result.decisions if item.outcome == DecisionOutcome.MATCHED_EXISTING
    ]
    assert len(matched) <= 1


def test_isolated_segment_defers_instead_of_creating_a_fragment():
    query = segment()
    result = plan([feature(query)], segments=[query])
    assert_deferred(result, query)
    assert not result.new_players


def test_isolated_segment_with_confirmed_number_creates_player():
    query = segment()
    result = plan([feature(query, number=27)], segments=[query])
    assert len(result.new_players) == 1
    player = result.new_players[0]
    assert player.confirmed_jersey_number == 27
    assert result.decisions[0].outcome == DecisionOutcome.CREATED_NEW
    assert result.decisions[0].evidence["reason"] == "confirmed_jersey_number_seed"
    assert result.decisions[0].evidence["new_player_id"] == player.player_id


def test_two_cross_video_witnesses_create_one_new_profile_and_link_decisions():
    first, second = segment("a"), segment("b")
    result = plan([feature(first, number=27), feature(second, number=27)], segments=[first, second])
    assert len(result.new_players) == 1
    player = result.new_players[0]
    assert player.confirmed_jersey_number == 27
    assert player.status == "active"
    assert player.created_at_revision == result.to_revision
    assert {item.feature.segment_id for item in player.prototypes} == {
        first.segment_id,
        second.segment_id,
    }
    assert all(item.player_id == player.player_id for item in player.prototypes)
    assert all(item.outcome == DecisionOutcome.CREATED_NEW for item in result.decisions)
    assert all(item.target_player_id is None for item in result.decisions)
    assert all(item.evidence["new_player_id"] == player.player_id for item in result.decisions)
    assert not result.deferred_segments


def test_two_disjoint_segments_of_one_video_do_not_count_as_independent_clips():
    first, second = segment(frames=(0, 1)), segment(track=2, frames=(10, 11))
    result = plan([feature(first), feature(second)], segments=[first, second])
    assert_deferred(result, first, second)
    assert not result.new_players


def test_new_profile_retains_separate_prototypes_within_the_configured_cap():
    parts = [segment(video) for video in ("a", "b", "c", "d")]
    result = plan(
        [feature(item) for item in parts],
        segments=parts,
        matcher=IncrementalMatcher(max_prototypes=2),
    )
    assert len(result.new_players) == 1
    assert len(result.new_players[0].prototypes) == 2
    assert all(item.outcome == DecisionOutcome.CREATED_NEW for item in result.decisions)


def test_newly_added_prototype_cannot_cascade_a_match_within_revision_r():
    old, first, second = segment("old"), segment("q1"), segment("q2")
    vectors = []
    for angle in (0.7, 1.4):
        value = np.zeros(512, dtype=np.float32)
        value[:2] = math.cos(angle), math.sin(angle)
        vectors.append(value)
    result = plan(
        [feature(first, vector=vectors[0]), feature(second, vector=vectors[1])],
        [profile(prototype(feature(old)))],
        segments=[old, first, second],
    )
    by_id = decisions(result)
    assert by_id[first.segment_id].outcome == DecisionOutcome.MATCHED_EXISTING
    assert_deferred(result, second)


def test_pending_pool_supplies_independent_witness_from_an_earlier_batch():
    old, query = segment("old"), segment("new")
    result = plan([feature(query)], segments=[old, query], pending=[feature(old)])
    assert len(result.new_players) == 1
    assert {item.segment_id for item in result.decisions} == {old.segment_id, query.segment_id}
    assert all(item.outcome == DecisionOutcome.CREATED_NEW for item in result.decisions)


def test_different_jersey_numbers_cannot_form_a_new_identity_cluster():
    first, second = segment("a"), segment("b")
    result = plan([feature(first, number=10), feature(second, number=27)], segments=[first, second])
    assert len(result.new_players) == 2
    assert {p.confirmed_jersey_number for p in result.new_players} == {10, 27}
    assert (
        result.decisions[0].evidence["new_player_id"]
        != result.decisions[1].evidence["new_player_id"]
    )


def test_different_models_cannot_form_a_new_identity_cluster():
    first, second = segment("a"), segment("b")
    result = plan([feature(first), feature(second, model="different")], segments=[first, second])
    assert_deferred(result, first, second)
    assert not result.new_players


@pytest.mark.parametrize(
    "gap",
    [
        "no_prototypes",
        "different_model",
        "low_quality",
        "missing_assignment",
        "missing_prototype_geometry",
    ],
)
def test_incomparable_existing_player_is_not_evidence_of_new_identity(gap):
    old, first, second = segment("old"), segment("q1"), segment("q2")
    item = feature(
        old,
        model="other-model" if gap == "different_model" else "synthetic-reid-v1",
        quality=0.1 if gap == "low_quality" else 0.95,
    )
    player = profile() if gap == "no_prototypes" else profile(prototype(item))
    geometry = [first, second] if gap == "missing_prototype_geometry" else [old, first, second]
    assignments = {"missing-historical-segment": "p"} if gap == "missing_assignment" else None
    result = plan(
        [feature(first), feature(second)], [player], segments=geometry, assignments=assignments
    )
    assert_deferred(result, first, second)
    assert not result.new_players


def test_new_identity_cluster_never_contains_a_transitive_cannot_link():
    left, right, bridge = segment(track=1), segment(track=2, x=100.0), segment("bridge")
    parts = [left, right, bridge]
    result = plan([feature(item) for item in parts], segments=parts)
    for player in result.new_players:
        sources = {item.feature.segment_id for item in player.prototypes}
        assert not {left.segment_id, right.segment_id} <= sources
    groups = {}
    for item in result.decisions:
        if item.outcome == DecisionOutcome.CREATED_NEW:
            groups.setdefault(item.evidence["new_player_id"], set()).add(item.segment_id)
    assert all(not {left.segment_id, right.segment_id} <= group for group in groups.values())


def test_complete_link_clustering_prevents_an_appearance_bridge():
    # Adjacent angles are close but endpoint distance is above the threshold.
    parts = [segment(video) for video in ("a", "b", "c")]
    angles = (0.0, 0.6, 1.2)
    vectors = []
    for angle in angles:
        value = np.zeros(512, dtype=np.float32)
        value[:2] = math.cos(angle), math.sin(angle)
        vectors.append(value)
    result = plan(
        [feature(part, vector=value) for part, value in zip(parts, vectors, strict=True)],
        segments=parts,
    )
    assert all(len(player.prototypes) < 3 for player in result.new_players)
    groups = {}
    for item in result.decisions:
        if item.outcome == DecisionOutcome.CREATED_NEW:
            groups.setdefault(item.evidence["new_player_id"], set()).add(item.segment_id)
    assert all(len(group) < 3 for group in groups.values())


def test_empty_batch_returns_a_well_formed_revision_plan():
    result = plan([], segments=[], revision=9)
    assert result.batch_id == "synthetic-t2"
    assert result.from_revision == 9
    assert result.to_revision == 10
    assert not result.decisions
    assert not result.new_players
    assert not result.updated_prototypes
    assert not result.deferred_segments


def test_revision_snapshot_and_input_arrays_are_not_mutated():
    old, query = segment("old"), segment("q")
    old_feature, query_feature = feature(old), feature(query)
    original = old_feature.appearance.vector.copy()
    player = profile(prototype(old_feature))
    old_prototypes = player.prototypes
    result = plan([query_feature], [player], segments=[old, query], revision=4)
    assert player.prototypes is old_prototypes
    np.testing.assert_array_equal(old_feature.appearance.vector, original)
    assert player.created_at_revision == 1
    assert old_prototypes[0].created_at_revision == 2
    assert result.from_revision == 4 and result.to_revision == 5
    assert all(item.created_at_revision == 5 for item in result.updated_prototypes)
    with pytest.raises(FrozenInstanceError):
        result.to_revision = 6


def test_proposed_prototypes_own_their_vectors_and_jersey_votes():
    old, query = segment("old"), segment("q")
    item = feature(query, number=27)
    result = plan(
        [item], [profile(prototype(feature(old, number=27)), number=27)], segments=[old, query]
    )
    proposed = result.updated_prototypes[0].feature
    assert item.appearance.vector.flags.writeable
    assert not np.shares_memory(item.appearance.vector, proposed.appearance.vector)
    original = proposed.appearance.vector.copy()
    item.appearance.vector[:] = 0
    item.jersey.votes[27] = 100.0
    np.testing.assert_array_equal(proposed.appearance.vector, original)
    assert proposed.jersey.votes[27] == 0.95


def test_replay_and_input_order_preserve_ids_and_decisions():
    first, second, third = segment("a"), segment("b"), segment("c")
    parts = [third, first, second]
    features = [feature(item) for item in parts]
    matcher = IncrementalMatcher()
    result1 = plan(features, segments=parts, matcher=matcher)
    result2 = plan(list(reversed(features)), segments=list(reversed(parts)), matcher=matcher)
    assert [item.segment_id for item in result1.decisions] == sorted(
        item.segment_id for item in parts
    )
    assert [
        (item.segment_id, item.outcome, item.target_player_id, item.evidence.get("new_player_id"))
        for item in result1.decisions
    ] == [
        (item.segment_id, item.outcome, item.target_player_id, item.evidence.get("new_player_id"))
        for item in result2.decisions
    ]
    assert [item.player_id for item in result1.new_players] == [
        item.player_id for item in result2.new_players
    ]
    assert [proto.prototype_id for item in result1.new_players for proto in item.prototypes] == [
        proto.prototype_id for item in result2.new_players for proto in item.prototypes
    ]


@pytest.mark.parametrize(
    "bad_vector",
    [
        np.zeros(512, dtype=np.float32),
        np.ones(512, dtype=np.float32),
        np.ones(511, dtype=np.float32),
        np.full(512, np.nan, dtype=np.float32),
        np.full(512, np.inf, dtype=np.float32),
        np.eye(1, 512, dtype=np.float64).reshape(512),
    ],
)
def test_invalid_embedding_contract_is_rejected(bad_vector):
    query = segment()
    with pytest.raises(ValueError):
        plan([feature(query, vector=bad_vector)], segments=[query])


@pytest.mark.parametrize("quality", [-0.1, 1.1, math.nan, math.inf])
def test_invalid_quality_is_rejected(quality):
    query = segment()
    with pytest.raises(ValueError):
        plan([feature(query, quality=quality)], segments=[query])


def test_duplicate_candidate_id_is_rejected():
    query = segment()
    item = feature(query)
    with pytest.raises(ValueError):
        plan([item, item], segments=[query])


def test_missing_candidate_geometry_is_rejected():
    query = segment()
    with pytest.raises(ValueError):
        plan([feature(query)], segments=[])


@pytest.mark.parametrize("revision", [-1, True, 1.5])
def test_invalid_snapshot_revision_is_rejected(revision):
    with pytest.raises(ValueError):
        plan([], segments=[], revision=revision)


@pytest.mark.parametrize("value", [0.0, 0.039, math.nan, math.inf])
def test_competitive_margin_cannot_be_configured_below_the_required_floor(value):
    with pytest.raises(ValueError):
        IncrementalMatcher(competitive_margin=value)


def test_future_gallery_prototype_is_not_a_valid_revision_r_snapshot():
    old, query = segment("old"), segment("q")
    player = profile(prototype(feature(old), revision=5))
    with pytest.raises(ValueError):
        plan([feature(query)], [player], segments=[old, query], revision=4)


def test_identity_planning_has_no_file_reads(monkeypatch):
    first, second = segment("a"), segment("b")

    def refuse_file_read(*args, **kwargs):
        raise AssertionError("Identity planning must consume supplied evidence only")

    monkeypatch.setattr(builtins, "open", refuse_file_read)
    result = plan([feature(first), feature(second)], segments=[first, second])
    assert len(result.new_players) == 1


def test_new_identity_witnesses_must_also_pass_prototype_weight_admission():
    first, second = segment("a"), segment("b")
    result = plan(
        [feature(first, quality=0.95), feature(second, quality=0.95)],
        segments=[first, second],
        matcher=IncrementalMatcher(min_prototype_weight=0.99),
    )
    assert not result.new_players
    assert_deferred(result, first, second)


def test_new_profile_prototypes_at_admission_boundary_are_usable_next_batch():
    first, second, third = segment("a"), segment("b"), segment("c")
    matcher = IncrementalMatcher(min_prototype_weight=0.95)
    created = plan(
        [feature(first, quality=0.95), feature(second, quality=0.95)],
        segments=[first, second],
        matcher=matcher,
    )
    assert len(created.new_players) == 1
    reused = plan(
        [feature(third, quality=0.95)],
        created.new_players,
        segments=[first, second, third],
        matcher=matcher,
        revision=created.to_revision,
    )
    assert reused.decisions[0].outcome == DecisionOutcome.MATCHED_EXISTING
    assert reused.decisions[0].target_player_id == created.new_players[0].player_id


def test_zero_quality_cannot_create_unusable_zero_weight_prototypes():
    first, second = segment("a"), segment("b")
    result = plan(
        [feature(first, quality=0), feature(second, quality=0)],
        segments=[first, second],
        matcher=IncrementalMatcher(min_quality=0, min_prototype_weight=0),
    )
    assert not result.new_players
    assert_deferred(result, first, second)


def test_zero_quality_cannot_add_an_unusable_prototype_to_existing_player():
    old, query = segment("old"), segment("query")
    result = plan(
        [feature(query, quality=0)],
        [profile(prototype(feature(old)))],
        segments=[old, query],
        matcher=IncrementalMatcher(min_quality=0, min_prototype_weight=0),
    )
    assert not result.updated_prototypes


def test_maximal_compatible_clique_extracts_separate_players_with_cannot_link():
    # In v1, track 1 and track 2 co-occur (cannot-link).
    # In v2, track 1 and track 2 co-occur (cannot-link).
    # Transitive appearance connects all 4 into one connected component, but each
    # player must form a separate compatible clique across the 2 videos.
    p1_v1 = segment(video="v1", track=1, frames=(0, 1, 2))
    p2_v1 = segment(video="v1", track=2, frames=(0, 1, 2), x=100.0)
    p1_v2 = segment(video="v2", track=1, frames=(0, 1, 2))
    p2_v2 = segment(video="v2", track=2, frames=(0, 1, 2), x=100.0)

    def vec(angle):
        v = np.zeros(512, dtype=np.float32)
        v[0] = math.cos(angle)
        v[1] = math.sin(angle)
        return v

    parts = [p1_v1, p2_v1, p1_v2, p2_v2]
    features = [
        feature(p1_v1, vector=vec(0.0)),
        feature(p2_v1, vector=vec(0.4)),
        feature(p1_v2, vector=vec(0.1)),
        feature(p2_v2, vector=vec(0.45)),
    ]
    result = plan(features, segments=parts)
    assert len(result.new_players) == 2
    groups = {
        item.evidence["new_player_id"]: set()
        for item in result.decisions
        if item.outcome == DecisionOutcome.CREATED_NEW
    }
    for item in result.decisions:
        assert item.outcome == DecisionOutcome.CREATED_NEW
        groups[item.evidence["new_player_id"]].add(item.segment_id)

    assert {frozenset(g) for g in groups.values()} == {
        frozenset({p1_v1.segment_id, p1_v2.segment_id}),
        frozenset({p2_v1.segment_id, p2_v2.segment_id}),
    }


def test_matching_jersey_number_activates_appearance_discount_for_gallery_match():
    """Bug 4: Identical jersey numbers with compatible appearance (<= 0.45) receive 0.5x distance discount."""
    old, query = segment("v1", track=1), segment("v2", track=1)
    player = profile(prototype(feature(old, number=27, distance=0.0)), number=27)
    # Raw cosine distance is 0.28, which exceeds the default threshold 0.25.
    # With identical jersey number 27 and raw_dist <= 0.45, discounted distance becomes 0.14.
    result = plan([feature(query, number=27, distance=0.28)], [player], segments=[old, query])
    assert result.decisions[0].outcome == DecisionOutcome.MATCHED_EXISTING
    assert result.decisions[0].target_player_id == player.player_id
    assert result.decisions[0].best_distance == pytest.approx(0.14, abs=1e-5)


def test_matching_jersey_number_discounts_lateral_unreadable_prototype_under_player_profile():
    """Bug 4: Lateral/dorsal prototype without readable number also enjoys discount if player confirmed number matches query."""
    old, query = segment("v1", track=1), segment("v2", track=1)
    # Prototype feature has number=None (e.g. side angle view), but player confirmed number is 27.
    proto = prototype(feature(old, number=None, distance=0.0), view="lateral")
    player = profile(proto, number=27)

    # 1. Query has confirmed number 27 and raw distance 0.28 -> discounted to 0.14 -> MATCHED_EXISTING
    result_with_num = plan(
        [feature(query, number=27, distance=0.28)], [player], segments=[old, query]
    )
    assert result_with_num.decisions[0].outcome == DecisionOutcome.MATCHED_EXISTING
    assert result_with_num.decisions[0].target_player_id == player.player_id
    assert result_with_num.decisions[0].best_distance == pytest.approx(0.14, abs=1e-5)

    # 2. Query has no number (number=None) -> pure appearance fallback (no discount) -> 0.28 > 0.25 -> INSUFFICIENT
    result_no_num = plan(
        [feature(query, number=None, distance=0.28)], [player], segments=[old, query]
    )
    assert result_no_num.decisions[0].outcome == DecisionOutcome.INSUFFICIENT
    assert result_no_num.decisions[0].target_player_id is None
    assert result_no_num.decisions[0].best_distance == pytest.approx(0.28, abs=1e-5)


def test_matching_jersey_number_does_not_discount_when_appearance_diverges():
    """Bug 4 safety gate: Raw appearance distance > 0.45 does not get discounted even with matching jersey numbers."""
    old, query = segment("v1", track=1), segment("v2", track=1)
    player = profile(prototype(feature(old, number=27, distance=0.0)), number=27)
    # Raw distance is 0.48 (> 0.45, divergent appearance) -> no discount applied.
    result = plan([feature(query, number=27, distance=0.48)], [player], segments=[old, query])
    assert result.decisions[0].outcome == DecisionOutcome.INSUFFICIENT
    assert result.decisions[0].target_player_id is None
    assert result.decisions[0].best_distance == pytest.approx(0.48, abs=1e-5)


def test_matching_jersey_number_boosts_discovery_clique():
    """Bug 4: Cross-video pair with identical numbers and raw distance 0.28 forms a clique via discounted distance 0.14."""
    s1 = segment("v1", track=1)
    s2 = segment("v2", track=1)
    f1 = feature(s1, number=27, distance=0.0)
    f2 = feature(s2, number=27, distance=0.28)
    result = plan([f1, f2], segments=[s1, s2])
    assert len(result.new_players) == 1
    assert result.new_players[0].confirmed_jersey_number == 27
    assert {d.outcome for d in result.decisions} == {DecisionOutcome.CREATED_NEW}
