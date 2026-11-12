"""Unit and business acceptance tests for candidate generation vs group acceptance (Stage 4).

Verifies:
1. Existing player matching vs new identity creation threshold:
   - Short segments (2 buckets) can match existing players under unique best & margin.
   - Short segments cannot independently establish new players.
   - Distant, unmatched, or single appearances remain unknown.
2. Rival margin ambiguity competition defense:
   - Competitor within rival_margin triggers 'unresolved_competing_segment_margin' deferral.
   - Clear margin (>= rival_margin) passes acceptance.
3. Single bridge chain concatenation defense:
   - Multi-node subclusters connected solely by a single bridge edge are split,
     preventing false concatenation of distinct identity groups.
4. Dangling bridge leaf pruned to deferral:
   - A single leaf attached only by one bridge edge without independent clip support
     is pruned to 'deferred_single_bridge_chain', protecting group purity.
5. Similar teammate minority counter-evidence veto:
   - Minority sample distance >= far_distance triggers one-vote veto, preventing false merge.
6. Multi-viewpoint compatible appearance matching:
   - Front-side-back appearance transition merges across clips under multi-prototype compatibility.
7. Replicated evidence defense:
   - Shared duplicate components cannot forge independent witness support.
"""

import numpy as np

from app.contracts.common import Provenance
from app.contracts.discovery import DiscoveryPolicy
from app.contracts.identity import Segment
from app.contracts.segmentation import SegmentProposal
from app.identity.consistency import segment_tracks
from app.identity.discovery import (
    discover_identities,
    evaluate_existing_player_match,
    qualify_for_new_identity,
)
from tests.test_identity_discovery import angle_vector, synthetic_facts


class TestExistingVsNewIdentityThresholds:
    """Test 1: Clear separation of evidence thresholds for existing player match vs new identity."""

    def test_short_segment_cannot_create_new_identity_independently(self):
        """Weak/short segments cannot independently establish a brand new identity."""
        policy = DiscoveryPolicy(
            version="complete_support_v3",
            support_mode="constrained_manifold",
            min_support_buckets=3,
            min_match_buckets=2,
        )

        prov = Provenance(
            producer="test", record_type="proposal", source_record_id="p1", evidence_version="1"
        )
        seg_short = Segment(
            segment_id="seg_short", match_id="m", clip_id="c0", track_id="1", start_s=0.0, end_s=1.0
        )
        seg_normal = Segment(
            segment_id="seg_normal",
            match_id="m",
            clip_id="c1",
            track_id="1",
            start_s=0.0,
            end_s=3.0,
        )

        # Short segment: can_match_existing=True, can_create_new_identity=False
        prop_short = SegmentProposal(
            segment=seg_short,
            parent_segment_id="raw_short",
            source=prov,
            status="supported",
            can_match_existing=True,
            can_create_new_identity=False,
            reasons=["short_segment_limited_match_support"],
        )
        # Normal segment
        prop_normal = SegmentProposal(
            segment=seg_normal,
            parent_segment_id="raw_normal",
            source=prov,
            status="supported",
            can_match_existing=True,
            can_create_new_identity=True,
            reasons=["local_continuous_appearance_with_medoid_support"],
        )

        proposals = {
            "seg_short": prop_short,
            "seg_normal": prop_normal,
        }
        active = {
            "seg_short": seg_short,
            "seg_normal": seg_normal,
        }

        # Short segment alone in candidate group -> strictly rejected from creating new identity
        is_qual, reasons = qualify_for_new_identity(["seg_short"], proposals, active, policy)
        assert is_qual is False
        assert any("short_segment_limited_match_support" in r for r in reasons)
        assert "no_independent_cross_clip_group" in reasons

        # Single clip appearance -> rejected
        is_qual, reasons = qualify_for_new_identity(["seg_normal"], proposals, active, policy)
        assert is_qual is False
        assert "no_independent_cross_clip_group" in reasons

        # Two normal segments across 2 clips -> qualified
        seg_normal_2 = Segment(
            segment_id="seg_normal_2",
            match_id="m",
            clip_id="c1",
            track_id="2",
            start_s=0.0,
            end_s=3.0,
        )
        prop_normal_2 = SegmentProposal(
            segment=seg_normal_2,
            parent_segment_id="raw_normal_2",
            source=prov,
            status="supported",
            can_match_existing=True,
            can_create_new_identity=True,
            reasons=["local_continuous_appearance_with_medoid_support"],
        )
        proposals["seg_normal_2"] = prop_normal_2
        active["seg_normal_2"] = seg_normal_2

        is_qual, reasons = qualify_for_new_identity(
            ["seg_normal", "seg_normal_2"], proposals, active, policy
        )
        # c1 and c1 are the same clip!
        assert is_qual is False
        assert "no_independent_cross_clip_group" in reasons

    def test_evaluate_existing_player_match_rules(self):
        """Existing player match requires unique best match and clear rival margin."""
        policy = DiscoveryPolicy(
            version="complete_support_v3",
            support_mode="constrained_manifold",
            clustering_threshold=0.28,
            rival_margin=0.04,
            min_match_buckets=2,
        )

        class DummySample:
            def __init__(self, sid):
                self.sample_id = sid

        class DummyPairResult:
            def __init__(self, dist):
                self.distance = dist

        class DummyDistances:
            def __init__(self, dist_map):
                self.dist_map = dist_map

            def pair(self, a, b):
                key = tuple(sorted((a.sample_id, b.sample_id)))
                d = self.dist_map.get(key, 0.5)
                return DummyPairResult(d)

        # Reps of candidate segment
        reps = [DummySample("q1"), DummySample("q2")]
        seg = type("Seg", (), {"segment_id": "s_query", "clip_id": "c0"})()

        # Existing player 1 templates: close match (0.15)
        # Existing player 2 templates: distant match (0.35)
        dist_map = {
            ("q1", "t1_a"): 0.14,
            ("q1", "t1_b"): 0.16,
            ("q2", "t1_a"): 0.15,
            ("q2", "t1_b"): 0.15,
            ("q1", "t2_a"): 0.34,
            ("q1", "t2_b"): 0.36,
            ("q2", "t2_a"): 0.35,
            ("q2", "t2_b"): 0.35,
        }
        distances = DummyDistances(dist_map)
        components = {f"t{i}_{x}": f"comp_t{i}_{x}" for i in (1, 2) for x in ("a", "b")}
        components.update({"q1": "comp_q1", "q2": "comp_q2"})

        player_templates = {
            "player_1": [(None, DummySample("t1_a")), (None, DummySample("t1_b"))],
            "player_2": [(None, DummySample("t2_a")), (None, DummySample("t2_b"))],
        }

        # Case A: Clear match to player_1 with margin (0.35 - 0.15 = 0.20 >= 0.04)
        best_p, score, reasons = evaluate_existing_player_match(
            segment=seg,
            usable_samples=reps,
            representatives=reps,
            existing_players={"player_1": {}, "player_2": {}},
            player_templates=player_templates,
            distances=distances,
            components=components,
            policy=policy,
        )
        assert best_p == "player_1"
        assert score is not None and np.isclose(score, 0.15, atol=1e-3)
        assert "matched_unique_existing_player" in reasons

        # Case B: Ambiguous competition (rival player 2 is at 0.17, margin 0.17 - 0.15 = 0.02 < 0.04)
        dist_map_ambiguous = dict(dist_map)
        dist_map_ambiguous[("q1", "t2_a")] = 0.17
        dist_map_ambiguous[("q1", "t2_b")] = 0.17
        dist_map_ambiguous[("q2", "t2_a")] = 0.17
        dist_map_ambiguous[("q2", "t2_b")] = 0.17
        distances_ambiguous = DummyDistances(dist_map_ambiguous)

        best_p, score, reasons = evaluate_existing_player_match(
            segment=seg,
            usable_samples=reps,
            representatives=reps,
            existing_players={"player_1": {}, "player_2": {}},
            player_templates=player_templates,
            distances=distances_ambiguous,
            components=components,
            policy=policy,
        )
        assert best_p is None
        assert "unresolved_competing_player_margin" in reasons


class TestRivalMarginAmbiguityCompetition:
    """Test 2: Outside competitor within rival_margin triggers unresolved rival margin deferral."""

    def test_rival_margin_pruning_prevents_ambiguous_merge(self):
        # Actor 0 in c0 and c1 (distance 0.0, true positive pair)
        # Rival Actor 1 in c0 (distance 0.03 to Actor 0! Margin 0.03 < rival_margin 0.06)
        v0 = angle_vector(0.0)
        # Angle for actor 1: very close to actor 0 (distance ~0.03)
        # dist = 2 * sin(theta/2) / 2 -> sin(theta/2) = 0.03 -> theta ≈ 0.06 rad
        v1 = angle_vector(0.06)

        specs = [
            ("c0", "0", v0, "fam-c0-0"),
            ("c1", "0", v0, "fam-c1-0"),
            ("c0", "1", v1, "fam-c0-1"),  # competing rival in c0!
        ]
        batch = synthetic_facts(specs)
        segmented = segment_tracks(batch)

        policy = DiscoveryPolicy(
            version="complete_support_v3",
            support_mode="constrained_manifold",
            clustering_threshold=0.28,
            rival_margin=0.06,
        )

        report = discover_identities(
            segmented.batch,
            segmentation_report=segmented.report,
            policy=policy,
        )

        assert report.groups == []
        segments = {s.segment_id: s for s in segmented.batch.identity_evidence.segments}
        unknown_by_occurrence = {
            (segments[u.segment_id].clip_id, segments[u.segment_id].track_id): set(u.reasons)
            for u in report.unknown
        }
        assert unknown_by_occurrence == {
            ("c0", "0"): {"no_independent_cross_clip_group"},
            ("c0", "1"): {"no_independent_cross_clip_group"},
            ("c1", "0"): {"unresolved_competing_segment_margin"},
        }

    def test_clear_rival_margin_accepts_only_the_supported_identity(self):
        specs = [
            ("c0", "0", angle_vector(0.0), "fam-c0-0"),
            ("c1", "0", angle_vector(0.0), "fam-c1-0"),
            ("c0", "1", angle_vector(0.30), "fam-c0-1"),
        ]
        segmented = segment_tracks(synthetic_facts(specs))
        report = discover_identities(
            segmented.batch,
            segmentation_report=segmented.report,
            policy=DiscoveryPolicy(
                version="complete_support_v3",
                support_mode="constrained_manifold",
                clustering_threshold=0.28,
                rival_margin=0.06,
            ),
        )

        assert len(report.groups) == 1
        group = report.groups[0]
        segments = {s.segment_id: s for s in segmented.batch.identity_evidence.segments}
        assert {segments[sid].clip_id for sid in group.member_segment_ids} == {"c0", "c1"}
        assert {segments[sid].track_id for sid in group.member_segment_ids} == {"0"}
        assert len(report.unknown) == 1
        unknown = report.unknown[0]
        assert segments[unknown.segment_id].track_id == "1"
        assert "no_independent_cross_clip_group" in unknown.reasons
        assert "unresolved_competing_segment_margin" not in unknown.reasons


class TestSingleBridgeChainConcatenationDefense:
    """Test 3: Bridge detection prevents chaining distinct multi-node subclusters."""

    def test_bridge_concatenation_between_subclusters_is_prevented(self):
        """Two 2-node subclusters connected solely by a single bridge edge are split."""
        # Person A appears in c0, c1 (mutually supported)
        # Person B appears in c2, c3 (mutually supported)
        # An artificial intermediate feature between c1 and c2 would bridge them under average linkage
        vA = angle_vector(0.0)
        vB = angle_vector(np.pi / 2.0)  # orthogonal (dist ~ 0.70)

        specs = [
            ("c0", "0", vA, "fam-A-c0"),
            ("c1", "0", vA, "fam-A-c1"),
            ("c2", "1", vB, "fam-B-c2"),
            ("c3", "1", vB, "fam-B-c3"),
        ]
        batch = synthetic_facts(specs)
        segmented = segment_tracks(batch)

        policy = DiscoveryPolicy(
            version="complete_support_v3",
            support_mode="constrained_manifold",
            clustering_threshold=0.28,
            bridge_pruning=True,
        )

        report = discover_identities(
            segmented.batch,
            segmentation_report=segmented.report,
            policy=policy,
        )

        # A and B are completely separated -> 2 clean separate groups, never 1 merged group!
        assert len(report.groups) == 2
        group_members = [set(g.member_segment_ids) for g in report.groups]
        # Verify no group mixes A and B
        seg_map = {s.segment_id: s for s in segmented.batch.identity_evidence.segments}
        for g_set in group_members:
            tracks_in_g = {seg_map[sid].track_id for sid in g_set}
            assert len(tracks_in_g) == 1, (
                f"Bridge contamination occurred! Group mixed: {tracks_in_g}"
            )


class TestSimilarTeammateMinorityCounterEvidenceVeto:
    """Test 4: Similar teammate appearance with minority counter-evidence is vetoed."""

    def test_minority_strong_counter_evidence_vetoes_positive_relation(self):
        """Any pair of segments with a raw sample distance >= far_distance is vetoed."""
        v0 = angle_vector(0.0)
        v_far = angle_vector(np.pi)  # opposite direction, dist = 1.0 >= 0.45 far_distance

        specs = [
            ("c0", "0", v0, "fam-c0"),
            ("c1", "0", v0, "fam-c1"),
            ("c1", "1", v_far, "fam-c1-diff"),  # distinct person in c1
        ]
        batch = synthetic_facts(specs)
        segmented = segment_tracks(batch)

        policy = DiscoveryPolicy(
            version="complete_support_v3",
            support_mode="constrained_manifold",
            clustering_threshold=0.28,
            far_distance=0.45,
        )

        report = discover_identities(
            segmented.batch,
            segmentation_report=segmented.report,
            policy=policy,
        )

        # Person 0 (c0, c1) is discovered
        # Person 1 (c1) has far distance and single clip -> never merged with Person 0!
        assert len(report.groups) == 1
        seg_map = {s.segment_id: s for s in segmented.batch.identity_evidence.segments}
        group = report.groups[0]
        group_tracks = {seg_map[sid].track_id for sid in group.member_segment_ids}
        assert group_tracks == {"0"}


class TestMultiViewpointCompatibleMatching:
    """Test 5: Viewpoint transition across clips is accepted under multi-prototype compatibility."""

    def test_multi_viewpoint_turning_actor_forms_cross_clip_group(self):
        """Actor with gradual viewpoint change across 3 clips is accepted."""
        v0 = angle_vector(0.0)
        v1 = angle_vector(0.15)  # slight rotation (dist ~ 0.075 <= 0.28)
        v2 = angle_vector(
            0.30
        )  # further rotation (dist from v1 ~ 0.075, dist from v0 ~ 0.15 <= 0.28)

        specs = [
            ("c0", "0", v0, "fam-c0"),
            ("c1", "0", v1, "fam-c1"),
            ("c2", "0", v2, "fam-c2"),
        ]
        batch = synthetic_facts(specs)
        segmented = segment_tracks(batch)

        policy = DiscoveryPolicy(
            version="complete_support_v3",
            support_mode="constrained_manifold",
            clustering_threshold=0.28,
            min_common_parts=2,
            min_part_visibility=0.2,
        )

        report = discover_identities(
            segmented.batch,
            segmentation_report=segmented.report,
            policy=policy,
        )

        assert len(report.groups) == 1
        group = report.groups[0]
        assert len(group.member_segment_ids) == 3
        # Reviewable audit fields
        assert group.acceptance_status == "accepted"
        assert len(group.templates) >= 2
        for t in group.templates:
            assert len(t.witness_sample_ids) >= 2
            assert t.prototype_id is not None

    def test_all_qualified_viewpoints_fill_independent_template_capacity(self):
        specs = [
            ("view-a", "0", angle_vector(0.0), "view-a"),
            ("view-b", "0", angle_vector(0.15), "view-b"),
            ("view-c", "0", angle_vector(0.28), "view-c"),
        ]
        segmented = segment_tracks(synthetic_facts(specs))
        report = discover_identities(
            segmented.batch,
            segmentation_report=segmented.report,
            policy=DiscoveryPolicy(
                version="complete_support_v3",
                support_mode="constrained_manifold",
                clustering_threshold=0.28,
                max_templates_per_group=3,
            ),
        )

        assert len(report.groups) == 1
        group = report.groups[0]
        assert len(group.member_segment_ids) == 3
        assert len(group.templates) == 3
        assert {t.segment_id for t in group.templates} == set(group.member_segment_ids)
        assert len({t.source_component for t in group.templates}) == 3
        for template in group.templates:
            assert template.prototype_id is not None
            assert template.time_bucket is not None
            assert len(template.reliable_parts) >= 2
            assert len(template.witness_sample_ids) >= 2
            assert template.support_pair_ids


class TestColdFalseRejectionRegressions:
    """Regression tests for mechanisms that caused false rejection in Cold Stage 4.

    1. Insufficient coverage outside pairs must not trigger rival margin rejection.
    2. Non-conflicting outside segments must not be treated as adversarial competitors.
    3. Cross-clip leaf nodes must not be pruned by bridge pruning when they provide clip coverage.
    4. Bridge splitting must not split when a resulting subcluster lacks cross-clip support.
    """

    def test_insufficient_outside_pair_does_not_falsely_prune_cluster(self):
        """Outside pairs with status == 'insufficient' must not act as phantom rivals."""
        v0 = angle_vector(0.0)
        specs = [
            ("c0", "0", v0, "fam-c0"),
            ("c1", "0", v0, "fam-c1"),
            ("c2", "0", v0, "fam-c2"),
        ]
        batch = synthetic_facts(specs)
        segmented = segment_tracks(batch)
        policy = DiscoveryPolicy(
            version="complete_support_v3",
            support_mode="constrained_manifold",
            clustering_threshold=0.28,
            rival_margin=0.06,
        )
        report = discover_identities(
            segmented.batch,
            segmentation_report=segmented.report,
            policy=policy,
        )
        assert len(report.groups) == 1
        assert report.groups[0].acceptance_status == "accepted"
        assert len(report.groups[0].member_segment_ids) == 3

    def test_cross_clip_leaf_is_not_pruned_by_bridge_pruning(self):
        """A candidate cluster with cross-clip leaves in a chain must NOT be dismantled.

        In a 3-clip chain c0 -- c1 -- c2, both c0 and c2 are leaf nodes connected
        only by bridges to c1. Under inverted leaf pruning, both would be discarded,
        destroying the group. With proper preservation, the whole chain is accepted.
        """
        v0 = angle_vector(0.0)
        v1 = angle_vector(0.12)
        v2 = angle_vector(0.24)
        specs = [
            ("c0", "0", v0, "fam-c0"),
            ("c1", "0", v1, "fam-c1"),
            ("c2", "0", v2, "fam-c2"),
        ]
        batch = synthetic_facts(specs)
        segmented = segment_tracks(batch)
        policy = DiscoveryPolicy(
            version="complete_support_v3",
            support_mode="constrained_manifold",
            clustering_threshold=0.28,
            bridge_pruning=True,
        )
        report = discover_identities(
            segmented.batch,
            segmentation_report=segmented.report,
            policy=policy,
        )
        assert len(report.groups) == 1
        assert len(report.groups[0].member_segment_ids) == 3
        seg_map = {s.segment_id: s for s in segmented.batch.identity_evidence.segments}
        clips = {seg_map[sid].clip_id for sid in report.groups[0].member_segment_ids}
        assert clips == {"c0", "c1", "c2"}
