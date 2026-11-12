"""Tests for constrained manifold clustering discovery policy and workflow.

Verifies:
1. DiscoveryPolicy accepts 'constrained_manifold' support mode with clustering_threshold.
2. discover_identities routes to constrained manifold clustering and maps core clusters
   to DiscoveredGroup with AutomaticTemplateSupport templates from distinct member segments.
3. Idempotent replay produces bitwise identical DiscoveryReport.
"""

from types import SimpleNamespace

import numpy as np

from app.contracts.discovery import DiscoveryPolicy
from app.identity.consistency import segment_tracks
from app.identity.discovery import discover_identities, evaluate_existing_player_match
from tests.test_identity_discovery import angle_vector, synthetic_facts


def _match_counterexample(a_scores, b_scores=None):
    queries = [SimpleNamespace(sample_id=f"q{i}") for i in range(len(a_scores))]
    templates = {
        pid: [(None, SimpleNamespace(sample_id=f"{pid}{i}", clip_id=f"clip-{i}")) for i in range(2)]
        for pid in ("A", "B")
        if b_scores is not None or pid == "A"
    }
    all_samples = queries + [sample for values in templates.values() for _, sample in values]

    class Distances:
        features = {sample.sample_id: object() for sample in all_samples}

        def pair(self, query, template):
            index = int(query.sample_id[1:])
            scores = a_scores if template.sample_id.startswith("A") else b_scores
            return SimpleNamespace(distance=scores[index])

    policy = DiscoveryPolicy(
        support_mode="constrained_manifold", min_coverage=0.70, rival_margin=0.04
    )
    return evaluate_existing_player_match(
        segment=SimpleNamespace(segment_id="candidate"),
        usable_samples=queries,
        representatives=queries,
        existing_players={pid: SimpleNamespace() for pid in templates},
        player_templates=templates,
        distances=Distances(),
        components={sample.sample_id: sample.sample_id for sample in all_samples},
        policy=policy,
        match_threshold=0.28,
    )


def test_rival_above_acceptance_threshold_still_participates_in_margin():
    assert _match_counterexample([0.27] * 3, [0.29] * 3) == (
        None,
        0.27,
        ["unresolved_competing_player_margin"],
    )


def test_incompatible_extra_view_does_not_veto_two_compatible_templates():
    assert _match_counterexample([0.21] * 3) == ("A", 0.21, ["matched_unique_existing_player"])


def test_small_sample_tolerance_does_not_swallow_minority_supporting_another_player():
    assert _match_counterexample([0.20, 0.20, 0.20, 0.32], [0.36, 0.36, 0.36, 0.20]) == (
        None,
        0.20,
        ["minority_sample_supports_competing_player"],
    )


def test_discovery_policy_manifold_contract():
    policy = DiscoveryPolicy(
        version="complete_support_v3",
        support_mode="constrained_manifold",
        clustering_threshold=0.28,
        min_common_parts=2,
        min_part_visibility=0.2,
        max_templates_per_group=6,
    )
    assert policy.support_mode == "constrained_manifold"
    assert policy.clustering_threshold == 0.28
    assert policy.min_common_parts == 2
    assert policy.min_part_visibility == 0.2


def test_discover_identities_manifold_branch():
    # 2 people:
    # Person 0 appears in clip 0, clip 1, clip 2 (cross-clip core player)
    # Person 1 appears only in clip 0 (single clip, should remain unknown)
    specs = []
    # Person 0: vector at 0 rad
    v0 = angle_vector(0.0)
    for c in range(3):
        specs.append((f"c{c}", "0", v0, f"c{c}-fam-0"))
    # Person 1: vector at pi/2 rad (orthogonal) in c0 only
    v1 = angle_vector(np.pi / 2.0)
    specs.append(("c0", "1", v1, "fam-1"))

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

    assert report.schema_version == "identity-discovery-v1"
    # Person 0 is cross-clip (3 clips) -> 1 core group
    assert len(report.groups) == 1
    group = report.groups[0]
    # Check that member segments belong to person 0
    seg_map = {s.segment_id: s for s in segmented.batch.identity_evidence.segments}
    group_track_ids = {seg_map[sid].track_id for sid in group.member_segment_ids}
    assert group_track_ids == {"0"}
    # 2-3 templates from distinct member segments
    assert 2 <= len(group.templates) <= 3
    template_segs = [t.segment_id for t in group.templates]
    assert len(template_segs) == len(set(template_segs))

    # Person 1 (track 1) was single-clip -> must be in unknown
    unknown_track_ids = {seg_map[u.segment_id].track_id for u in report.unknown}
    assert "1" in unknown_track_ids

    # Idempotent replay check
    replay = discover_identities(
        segmented.batch,
        segmentation_report=segmented.report,
        policy=policy,
    )
    assert report.report_id == replay.report_id
    assert report.model_dump(mode="json") == replay.model_dump(mode="json")


def test_manifold_acceptance_support_pairs_and_witnesses():
    """Accepted groups and templates must have non-empty, resolvable cross-clip support pairs."""
    specs = []
    v0 = angle_vector(0.0)
    for c in range(3):
        specs.append((f"c{c}", "0", v0, f"c{c}-fam-0"))

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
    seg_map = {s.segment_id: s for s in segmented.batch.identity_evidence.segments}
    report_pairs_map = {p.pair_id: p for p in report.pairs}

    # Group has non-empty support pairs, all existing in report.pairs
    assert len(group.support_pair_ids) > 0
    for pid in group.support_pair_ids:
        assert pid in report_pairs_map
        pair = report_pairs_map[pid]
        assert pair.status == "supported"
        assert seg_map[pair.left_segment_id].clip_id != seg_map[pair.right_segment_id].clip_id

    # Every template has non-empty support pairs and cross-clip witnesses
    assert len(group.templates) >= 2
    for t in group.templates:
        assert len(t.support_pair_ids) > 0
        assert len(t.witness_sample_ids) > 0
        t_clip = seg_map[t.segment_id].clip_id
        for pid in t.support_pair_ids:
            assert pid in report_pairs_map
            pair = report_pairs_map[pid]
            assert pair.status == "supported"
            # Cross-clip support pair connects to another clip
            other_sid = (
                pair.right_segment_id
                if pair.left_segment_id == t.segment_id
                else pair.left_segment_id
            )
            assert seg_map[other_sid].clip_id != t_clip


def test_segment_pooling_does_not_read_other_segments_of_same_track():
    """Segment pooling strictly uses samples of the segment and never averages other segments."""
    from app.contracts.common import Provenance
    from app.contracts.evidence import FeatureSpace, InlineFeature, Sample
    from app.contracts.identity import Segment
    from app.identity.representation import pool_segment_features

    space = FeatureSpace(
        feature_space_id="fs-test",
        backend="kpr",
        model_hash="0" * 64,
        preprocessing={"version": "v1", "color_space": "RGB"},
        layout="parts",
        shape=[2, 4],
        normalization="l2",
        distance="euclidean_sqrt_visibility_mean_div2",
        visibility={"shape": [2], "semantics": "visibility"},
    )

    prov = Provenance(
        producer="test", record_type="obs", source_record_id="1", evidence_version="1"
    )
    # Two distinct segments on the same raw track
    seg_a = Segment(
        match_id="m",
        clip_id="c0",
        track_id="t1",
        segment_id="seg_a",
        start_s=0.0,
        end_s=5.0,
    )
    seg_b = Segment(
        match_id="m",
        clip_id="c0",
        track_id="t1",
        segment_id="seg_b",
        start_s=10.0,
        end_s=15.0,
    )

    sample_a = Sample(
        match_id="m",
        sample_id="samp_a",
        clip_id="c0",
        track_id="t1",
        segment_id="seg_a",
        time_s=2.0,
        source=prov,
        features=[
            InlineFeature(
                feature_space_id="fs-test",
                values=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                visibility=[1.0, 1.0],
            )
        ],
    )

    sample_b = Sample(
        match_id="m",
        sample_id="samp_b",
        clip_id="c0",
        track_id="t1",
        segment_id="seg_b",
        time_s=12.0,
        source=prov,
        features=[
            InlineFeature(
                feature_space_id="fs-test",
                values=[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                visibility=[1.0, 1.0],
            )
        ],
    )

    reps = pool_segment_features(
        segments=[seg_a, seg_b],
        samples=[sample_a, sample_b],
        space=space,
        min_vis=0.2,
    )

    # seg_a pooled representation contains strictly sample_a
    rep_a = reps["seg_a"]
    assert rep_a.sample_ids == ["samp_a"]
    assert np.allclose(rep_a.feature[0], [1.0, 0.0, 0.0, 0.0])
    assert "samp_b" not in rep_a.sample_ids

    # seg_b pooled representation contains strictly sample_b
    rep_b = reps["seg_b"]
    assert rep_b.sample_ids == ["samp_b"]
    assert np.allclose(rep_b.feature[0], [0.0, 1.0, 0.0, 0.0])
    assert "samp_a" not in rep_b.sample_ids


def test_same_track_supported_and_unknown_isolation(tmp_path):
    """When a raw track has both supported and unknown segments, only supported segments can be assigned.

    Unknown segments NEVER enter Player, Assignment, or Template.
    """
    from test_machine_input import machine_settings

    from app.contracts.api import UpdateRequest
    from app.contracts.common import Provenance
    from app.contracts.identity import Segment
    from app.contracts.segmentation import (
        SegmentationPolicy,
        SegmentationReport,
        SegmentProposal,
        TrackConsistencyReport,
    )
    from app.identity.fingerprints import evidence_fingerprint
    from app.storage.database import connect
    from app.workflows.discover import discover_library
    from app.workflows.read import get_identity_state
    from app.workflows.update import update_library

    v0 = angle_vector(0.0)
    specs = [("c0", "0", v0, "fam-c0"), ("c1", "0", v0, "fam-c1"), ("c2", "0", v0, "fam-c2")]
    batch = synthetic_facts(specs)
    evidence = batch.identity_evidence

    # Split c0/track 0 into two active segments:
    # 1. seg_c0_supp: supported [0.0, 1.3]
    # 2. seg_c0_unk: unknown [1.3, 2.6]
    orig_c0_seg = next(s for s in evidence.segments if s.clip_id == "c0" and s.track_id == "0")
    evidence.segments.remove(orig_c0_seg)

    seg_c0_supp = Segment(
        match_id="synthetic-match",
        clip_id="c0",
        track_id="0",
        segment_id="seg_c0_supp",
        start_s=0.0,
        end_s=1.3,
        frame_start=0,
        frame_end_exclusive=13,
    )
    seg_c0_unk = Segment(
        match_id="synthetic-match",
        clip_id="c0",
        track_id="0",
        segment_id="seg_c0_unk",
        start_s=1.3,
        end_s=2.6,
        frame_start=13,
        frame_end_exclusive=26,
    )
    evidence.segments.extend([seg_c0_supp, seg_c0_unk])

    # Re-assign samples in c0 to the two segments
    for sample in evidence.samples:
        if sample.clip_id == "c0" and sample.track_id == "0":
            if sample.time_s < 1.3:
                sample.segment_id = "seg_c0_supp"
            else:
                sample.segment_id = "seg_c0_unk"

    # Other segments
    seg_c1 = next(s for s in evidence.segments if s.clip_id == "c1" and s.track_id == "0")
    seg_c2 = next(s for s in evidence.segments if s.clip_id == "c2" and s.track_id == "0")

    prov = Provenance(
        producer="synthetic-seg",
        record_type="proposal",
        source_record_id="p1",
        evidence_version="1",
    )

    # Build segmentation report with seg_c0_supp=supported, seg_c0_unk=unknown
    prop_c0_supp = SegmentProposal(
        segment=seg_c0_supp,
        parent_segment_id="raw-c0-0",
        source=prov,
        status="supported",
        reasons=[],
    )
    prop_c0_unk = SegmentProposal(
        segment=seg_c0_unk,
        parent_segment_id="raw-c0-0",
        source=prov,
        status="unknown",
        reasons=["appearance_instability"],
    )
    prop_c1 = SegmentProposal(
        segment=seg_c1,
        parent_segment_id="raw-c1-0",
        source=prov,
        status="supported",
        reasons=[],
    )
    prop_c2 = SegmentProposal(
        segment=seg_c2,
        parent_segment_id="raw-c2-0",
        source=prov,
        status="supported",
        reasons=[],
    )

    seg_report = SegmentationReport(
        policy=SegmentationPolicy(),
        policy_fingerprint="0" * 64,
        input_fingerprint="0" * 64,
        derived_evidence_fingerprint=evidence_fingerprint(batch, {}),
        tracks=[
            TrackConsistencyReport(
                parent_segment=orig_c0_seg,
                proposals=[prop_c0_supp, prop_c0_unk],
            ),
            TrackConsistencyReport(
                parent_segment=seg_c1,
                proposals=[prop_c1],
            ),
            TrackConsistencyReport(
                parent_segment=seg_c2,
                proposals=[prop_c2],
            ),
        ],
        summary={},
    )

    policy = DiscoveryPolicy(
        version="complete_support_v3",
        support_mode="constrained_manifold",
        clustering_threshold=0.28,
        min_common_parts=2,
        min_part_visibility=0.2,
    )

    report = discover_identities(
        batch,
        segmentation_report=seg_report,
        policy=policy,
    )

    assert len(report.groups) == 1
    group = report.groups[0]
    # Only supported segments can be in group.member_segment_ids
    assert "seg_c0_supp" in group.member_segment_ids
    assert "seg_c0_unk" not in group.member_segment_ids
    assert seg_c1.segment_id in group.member_segment_ids
    assert seg_c2.segment_id in group.member_segment_ids

    # seg_c0_unk MUST be in unknown
    unknown_ids = {u.segment_id for u in report.unknown}
    assert "seg_c0_unk" in unknown_ids

    # Persistence verification in SQLite: unknown segment NEVER enters Player, Assignment, Template
    settings = machine_settings(tmp_path / "unused", tmp_path)
    settings = settings.model_copy(
        update={
            "discovery": policy,
            "match": settings.match.model_copy(update={"match_id": "synthetic-match"}),
        }
    )
    db = connect(":memory:")
    imported = update_library(
        db,
        UpdateRequest(batch=batch, expected_revision=0),
        feature_values={},
        settings=settings,
    )
    accepted = discover_library(
        db,
        batch,
        segmentation_report=seg_report,
        feature_values={},
        settings=settings,
        expected_revision=imported.revision,
    )
    assert accepted.revision == imported.revision + 1

    state = get_identity_state(db, "synthetic-match")
    # seg_c0_supp is proposed to the discovered player
    assigned_players = {
        a.segment_id: a.player_id for a in state.data.assignments if a.player_id is not None
    }
    assert "seg_c0_supp" in assigned_players
    assert assigned_players["seg_c0_supp"] is not None

    # seg_c0_unk MUST NOT have any assigned player
    assert "seg_c0_unk" not in assigned_players

    # Templates must only contain samples from supported segments
    sample_to_seg = {s.sample_id: s.segment_id for s in evidence.samples}
    template_segment_ids = {sample_to_seg[t.sample_id] for t in state.data.templates}
    assert "seg_c0_unk" not in template_segment_ids
    assert "seg_c0_supp" in template_segment_ids or seg_c1.segment_id in template_segment_ids


def test_manifold_rejects_group_when_minority_strong_counter_evidence_exists():
    """Group must be rejected by all-pair conflict gate if any pair contains contradictory evidence."""
    # 3 clips: c0 and c1 are very close, c2 is far (orthogonal, distance ~0.707 >= 0.45 far_distance)
    v0 = angle_vector(0.0)
    v1 = angle_vector(0.02)
    v2 = angle_vector(np.pi / 2.0)

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

    # All-pair conflict gate must reject forming a cluster containing the contradictory c2
    # c2 vs c0 is contradictory (far_distance >= 0.45)
    for g in report.groups:
        # No group may merge c0 and c2
        clips_in_group = {s[:2] for s in g.member_segment_ids}  # e.g. 'c0', 'c2'
        assert not ({"c0", "c2"}.issubset(clips_in_group))


def test_same_clip_overlapping_tracks_never_merge():
    """Tracks overlapping in time in the same clip can never be merged into the same player."""
    v0 = angle_vector(0.0)
    # Clip 0 has track 0 and track 1 simultaneously observed
    # Clip 1 has track 0
    specs = [
        ("c0", "0", v0, "fam-c0-0"),
        ("c0", "1", v0, "fam-c0-1"),  # Same appearance, same clip, overlapping in time!
        ("c1", "0", v0, "fam-c1-0"),
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

    seg_map = {s.segment_id: s for s in segmented.batch.identity_evidence.segments}
    # In clip c0, track 0 and track 1 can NEVER appear together in any discovered group
    for group in report.groups:
        c0_tracks = {
            seg_map[sid].track_id
            for sid in group.member_segment_ids
            if seg_map[sid].clip_id == "c0"
        }
        assert len(c0_tracks) <= 1


def test_partial_duplicate_cannot_form_identity_group_regression():
    """Adversarial test: 5 out of 6 samples are cross-clip identical copies; 1 sample is independent.

    Asserts:
    1. Cross-clip pair fails independent support after dedup (< min_support_buckets=3 unshared).
    2. Directional coverage fails (< min_coverage=0.8).
    3. Pair is marked status='insufficient' with structured failure reasons.
    4. Algorithm blocks group formation (0 groups formed, segments remain unknown).
    """
    v0 = angle_vector(0.0)
    # Both clips share image family 'shared_fam'
    specs = [("c0", "0", v0, "shared_fam"), ("c1", "0", v0, "shared_fam")]
    batch = synthetic_facts(specs)

    # Make the 6th sample (frame 25) in each clip unique
    for a in batch.assets:
        if "sample-c0-0-25" in a.asset_id:
            a.file.sha256 = "aa" * 32
        elif "sample-c1-0-25" in a.asset_id:
            a.file.sha256 = "bb" * 32

    segmented = segment_tracks(batch)
    policy = DiscoveryPolicy(
        version="complete_support_v3",
        support_mode="constrained_manifold",
        clustering_threshold=0.28,
        min_support_buckets=3,
        min_coverage=0.8,
    )
    report = discover_identities(
        segmented.batch,
        segmentation_report=segmented.report,
        policy=policy,
    )

    # 1. Exactly one cross-clip pair evaluated
    assert len(report.pairs) == 1
    pair = report.pairs[0]

    # 2. Assert status is insufficient, NOT supported
    assert pair.status == "insufficient"
    assert "insufficient_independent_components_after_dedup" in pair.reasons
    assert "insufficient_coverage" in pair.reasons
    assert "shared_duplicate_components_excluded" in pair.reasons

    # 3. Assert NO identity group is forged from partial duplicate Penetration
    assert len(report.groups) == 0

    # 4. Both segments remain safely in unknown
    assert len(report.unknown) == 2
