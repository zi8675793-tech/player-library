"""Contrastive geometry gates use synthetic facts, never the real acceptance set."""

import math

import numpy as np
import pytest
from test_identity_discovery import angle_vector, synthetic_facts
from test_machine_input import machine_settings

from app.contracts.api import ReviewRequest, SetAssignmentCommand, UpdateRequest
from app.contracts.common import Provenance
from app.contracts.discovery import DiscoveryComparison, DiscoveryPolicy
from app.contracts.evidence import Sample
from app.contracts.identity import Segment
from app.identity.consistency import segment_tracks
from app.identity.contrastive import contrastive_pair
from app.identity.discovery import discover_identities
from app.storage.database import connect
from app.workflows.discover import discover_library
from app.workflows.read import get_discovery_reports, get_identity_state
from app.workflows.review import apply_review
from app.workflows.update import update_library


class Distances:
    def __init__(self):
        self.segment_scores = {
            frozenset(("left", "right")): 0.28,
            frozenset(("left", "negative-b")): 0.52,
            frozenset(("right", "negative-a")): 0.52,
        }
        self.sample_scores = {}

    def pair(self, left, right):
        value = self.sample_scores.get(
            frozenset((left.sample_id, right.sample_id)),
            self.segment_scores.get(frozenset((left.segment_id, right.segment_id)), 0.7),
        )
        a, b = sorted((left.sample_id, right.sample_id))
        return DiscoveryComparison(
            left_sample_id=a,
            right_sample_id=b,
            distance=value,
            common_parts=3 if value is not None else 0,
            reason=None if value is not None else "synthetic_incomparability",
        )


def fixture():
    active, selected, components = {}, {}, {}
    for sid, clip in (
        ("left", "a"),
        ("right", "b"),
        ("negative-a", "a"),
        ("negative-b", "b"),
    ):
        active[sid] = Segment(
            match_id="m", segment_id=sid, clip_id=clip, track_id=sid, start_s=0.0, end_s=1.1
        )
        selected[sid] = [
            make_sample(sid, clip, index, time) for index, time in enumerate((0.0, 0.5, 1.0))
        ]
        components.update({sample.sample_id: sample.sample_id for sample in selected[sid]})
    return {
        "left": "left",
        "right": "right",
        "selected": selected,
        "all_samples": {sid: list(samples) for sid, samples in selected.items()},
        "active": active,
        "evidence": None,
        "policy": DiscoveryPolicy(feature_space_id="space"),
        "distances": Distances(),
        "components": components,
        "geometric_conflicts": {
            ("left", "negative-a"): ["oa-1", "oa-2", "oa-3"],
            ("negative-b", "right"): ["ob-1", "ob-2", "ob-3"],
        },
    }


def make_sample(sid, clip, index, time):
    return Sample(
        match_id="m",
        sample_id=f"{sid}-{index}",
        clip_id=clip,
        track_id=sid,
        segment_id=sid,
        time_s=float(time),
        source=Provenance(
            producer="synthetic",
            record_type="sample",
            source_record_id=f"{sid}-{index}",
            evidence_version="1",
        ),
    )


def add_negative(data, *, value):
    sid = "negative-b2"
    data["active"][sid] = Segment(
        match_id="m", segment_id=sid, clip_id="b", track_id=sid, start_s=0.0, end_s=1.1
    )
    data["selected"][sid] = [
        make_sample(sid, "b", index, time) for index, time in enumerate((0.0, 0.5, 1.0))
    ]
    data["all_samples"][sid] = list(data["selected"][sid])
    data["components"].update({s.sample_id: s.sample_id for s in data["selected"][sid]})
    data["geometric_conflicts"][sid, "right"] = ["oc-1", "oc-2", "oc-3"]
    data["distances"].segment_scores[frozenset(("left", sid))] = value


def test_cross_view_can_pass_by_independent_contrast_above_within_segment_tau():
    data = fixture()
    pair = contrastive_pair(**data)
    assert data["policy"].tau == 0.18
    assert pair.status == "supported" and pair.score == 0.28
    assert len(pair.directions) == 2
    assert all(direction.coverage == 1.0 for direction in pair.directions)
    assert {tuple(direction.competitor_segment_ids) for direction in pair.directions} == {
        ("negative-a",),
        ("negative-b",),
    }
    assert len(pair.comparisons) == 27
    assert all(
        len(contrast["candidate_sample_ids"]) == 2
        for direction in pair.directions
        for contrast in direction.sample_contrasts
    )


def test_near_candidate_without_geometric_negatives_is_unknown():
    data = fixture()
    data["geometric_conflicts"] = {}
    pair = contrastive_pair(**data)
    assert pair.status == "insufficient"
    assert "missing_independent_geometric_negatives" in pair.reasons
    assert all(direction.coverage == 0.0 for direction in pair.directions)


def test_every_geometric_negative_needs_two_comparable_observations():
    data = fixture()
    add_negative(data, value=None)
    pair = contrastive_pair(**data)
    assert pair.status == "insufficient"
    direction = next(
        direction for direction in pair.directions if direction.query_segment_id == "left"
    )
    assert direction.competitor_segment_ids == ["negative-b", "negative-b2"]
    assert direction.coverage == 0.0
    assert all(
        contrast["reason"] == "negative_two_independent_observations_missing"
        for contrast in direction.sample_contrasts
    )


def test_similar_teammate_competitor_keeps_ambiguous_instead_of_forced_match():
    data = fixture()
    add_negative(data, value=0.30)
    pair = contrastive_pair(**data)
    assert pair.status == "ambiguous"
    assert next(d for d in pair.directions if d.query_segment_id == "left").coverage == 0.0


def test_nonrepresentative_minority_supporting_negative_vetoes_majority():
    data = fixture()
    minority = make_sample("left", "a", "minority", 0.25)
    data["all_samples"]["left"].append(minority)
    data["components"][minority.sample_id] = minority.sample_id
    for sid, distance in (("right", 0.42), ("negative-b", 0.1)):
        for sample in data["selected"][sid]:
            data["distances"].sample_scores[frozenset((minority.sample_id, sample.sample_id))] = (
                distance
            )
    pair = contrastive_pair(**data)
    assert pair.status == "conflict"
    assert all(direction.coverage == 1.0 for direction in pair.directions)
    assert "minority_raw_sample_supports_geometric_negative" in pair.reasons
    contrast = next(
        c for c in pair.directions[0].sample_contrasts if c["sample_id"] == minority.sample_id
    )
    assert contrast["representative"] is False


def test_duplicate_replay_components_are_removed_before_nearest_two_support():
    data = fixture()
    for left, right in zip(data["selected"]["left"], data["selected"]["right"], strict=True):
        data["components"][right.sample_id] = left.sample_id
    pair = contrastive_pair(**data)
    assert pair.status == "insufficient"
    assert all(direction.coverage == 0.0 for direction in pair.directions)
    assert "shared_duplicate_components_excluded" in pair.reasons


def test_duplicate_removal_never_shrinks_original_coverage_denominator():
    data = fixture()
    data["components"]["right-0"] = "left-0"
    pair = contrastive_pair(**data)
    assert pair.status == "insufficient"
    assert all(direction.coverage == 2 / 3 for direction in pair.directions)


def test_absolute_far_pair_is_diagnostic_and_geometry_is_a_direct_veto():
    data = fixture()
    data["distances"].sample_scores[frozenset(("left-0", "right-0"))] = 0.60
    pair = contrastive_pair(**data)
    assert pair.status == "supported"
    assert "far_sample_distances_present_not_identity_veto" in pair.reasons
    data["geometric_conflicts"]["left", "right"] = ["separated-1", "separated-2", "separated-3"]
    pair = contrastive_pair(**data)
    assert pair.status == "conflict"
    assert pair.conflict_observation_ids == ["separated-1", "separated-2", "separated-3"]


def test_pair_id_and_diagnostics_do_not_depend_on_argument_direction_or_input_order():
    data = fixture()
    before = contrastive_pair(**data)
    data["left"], data["right"] = data["right"], data["left"]
    for collection in (data["selected"], data["all_samples"]):
        for sid, samples in collection.items():
            collection[sid] = list(reversed(samples))
    assert contrastive_pair(**data) == before


def multiview_people(people, *, copies=False):
    specifications = []
    for clip in ("first", "second"):
        for person in range(people):
            vector = np.eye(people + 1)[person]
            if clip == "second" and not copies:
                vector = vector * math.cos(0.6)
                vector[-1] = math.sin(0.6)
            family = f"person-{person}" if copies else f"{clip}-{person}"
            specifications.append((clip, str(person), vector.tolist(), family))
    return synthetic_facts(specifications)


def full_discovery(batch):
    policy = DiscoveryPolicy(version="complete_support_v2", support_mode="cooccurrence_contrastive")
    segmented = segment_tracks(batch)
    report = discover_identities(
        segmented.batch, segmentation_report=segmented.report, policy=policy
    )
    return segmented, report, policy


@pytest.mark.parametrize("people", [2, 3, 4])
def test_contrastive_full_discovery_recovers_variable_people_across_view_shift(people):
    segmented, report, _ = full_discovery(multiview_people(people))
    segments = {s.segment_id: s for s in segmented.batch.identity_evidence.segments}
    assert len(report.groups) == people
    assert not report.unknown
    assert {
        frozenset(
            (segments[sid].clip_id, segments[sid].track_id) for sid in group.member_segment_ids
        )
        for group in report.groups
    } == {frozenset((("first", str(person)), ("second", str(person)))) for person in range(people)}
    assert all(
        pair.score > report.policy.tau for pair in report.pairs if pair.status == "supported"
    )
    assert all(len(group.templates) >= 2 for group in report.groups)
    assert all(
        direction.competitor_segment_ids
        for pair in report.pairs
        if pair.status == "supported"
        for direction in pair.directions
    )


def test_contrastive_full_discovery_refuses_same_team_indistinguishable_people():
    batch = synthetic_facts(
        [
            (clip, str(person), angle_vector(person * 0.08), f"{clip}-{person}")
            for clip in ("first", "second")
            for person in range(2)
        ]
    )
    _, report, _ = full_discovery(batch)
    assert not report.groups
    assert len(report.unknown) == 4
    assert any(pair.status == "ambiguous" for pair in report.pairs)


def test_contrastive_full_discovery_needs_negatives_and_independent_images():
    batch = synthetic_facts([(clip, "0", [1.0, 0.0], clip) for clip in ("first", "second")])
    _, report, _ = full_discovery(batch)
    assert not report.groups and len(report.unknown) == 2
    _, copied, _ = full_discovery(multiview_people(2, copies=True))
    assert not copied.groups and len(copied.unknown) == 4
    assert any("duplicate" in reason for pair in copied.pairs for reason in pair.reasons)


def test_contrastive_full_discovery_minority_contradiction_cannot_be_outvoted():
    batch = synthetic_facts(
        [
            (clip, str(person), angle_vector(person * 0.3), f"{clip}-{person}")
            for clip in ("first", "second")
            for person in range(2)
        ]
    )
    minority = next(
        s for s in batch.identity_evidence.samples if s.clip_id == "first" and s.track_id == "0"
    )
    minority.features[0].values = angle_vector(0.2) * 3
    segmented, report, _ = full_discovery(batch)
    segments = {s.segment_id: s for s in segmented.batch.identity_evidence.segments}
    assert all(
        segments[sid].track_id == "1" for group in report.groups for sid in group.member_segment_ids
    )
    assert any(
        "minority_raw_sample_supports_geometric_negative" in pair.reasons for pair in report.pairs
    )
    assert len(report.unknown) >= 2


def test_contrastive_full_discovery_bridge_and_templates_cannot_supply_circular_proof():
    specs = []
    for clip, angle in (("a1", 0.0), ("a2", 0.0), ("bridge", 0.6), ("c1", 1.2), ("c2", 1.2)):
        specs.extend(
            [
                (clip, "0", angle_vector(angle), f"{clip}-primary"),
                (clip, "1", angle_vector(math.pi), f"{clip}-other"),
            ]
        )
    segmented, report, _ = full_discovery(synthetic_facts(specs))
    segments = {s.segment_id: s for s in segmented.batch.identity_evidence.segments}
    memberships = [
        {segments[sid].clip_id for sid in group.member_segment_ids if segments[sid].track_id == "0"}
        for group in report.groups
    ]
    assert {"a1", "a2"} in memberships and {"c1", "c2"} in memberships
    assert all(not ({"a1", "c1"} <= clips) for clips in memberships)
    assert any(
        segments[item.segment_id].clip_id == "bridge" and segments[item.segment_id].track_id == "0"
        for item in report.unknown
    )
    pairs = {pair.pair_id: pair for pair in report.pairs}
    for group in report.groups:
        for template in group.templates:
            assert all(
                pairs[pair_id].status == "supported" for pair_id in template.support_pair_ids
            )
            assert template.sample_id not in template.witness_sample_ids


def test_contrastive_workflow_binds_positive_and_negative_raw_context_versions(tmp_path):
    segmented, report, policy = full_discovery(multiview_people(2))
    settings = machine_settings(tmp_path / "unused", tmp_path).model_copy(
        update={"discovery": policy}
    )
    db = connect(":memory:")
    imported = update_library(
        db,
        UpdateRequest(batch=segmented.batch, expected_revision=0),
        settings=settings,
        feature_values={},
    )
    accepted = discover_library(
        db,
        segmented.batch,
        segmentation_report=segmented.report,
        feature_values={},
        settings=settings,
        expected_revision=imported.revision,
    )
    stored = get_discovery_reports(db, segmented.batch.match_id).data[0]
    comparison_ids = {
        sample_id
        for pair in report.pairs
        if pair.status == "supported"
        for comparison in pair.comparisons
        for sample_id in (comparison.left_sample_id, comparison.right_sample_id)
    }
    assert comparison_ids <= stored["sample_sources"].keys()
    assert len(stored["sample_sources"]) == 24
    assert len(stored["player_ids_by_group"]) == 2
    segments = {s.segment_id: s for s in segmented.batch.identity_evidence.segments}
    group = next(
        group
        for group in report.groups
        if all(segments[sid].track_id == "0" for sid in group.member_segment_ids)
    )
    player_id = stored["player_ids_by_group"][group.group_id]
    negative = next(
        s for s in segments.values() if s.active and s.clip_id == "second" and s.track_id == "1"
    )
    assert any(
        t.active and t.player_id == player_id
        for t in get_identity_state(db, segmented.batch.match_id).data.templates
    )
    reviewed = apply_review(
        db,
        ReviewRequest(
            match_id=segmented.batch.match_id,
            expected_revision=accepted.revision,
            actor="human",
            commands=[
                SetAssignmentCommand(
                    segment_id=negative.segment_id,
                    player_id=None,
                    reason="negative context identity disputed",
                )
            ],
        ),
        settings=settings,
    )
    assert reviewed.revision == accepted.revision + 1
    assert all(
        not t.active
        for t in get_identity_state(db, segmented.batch.match_id).data.templates
        if t.player_id == player_id
    )
    db.close()
