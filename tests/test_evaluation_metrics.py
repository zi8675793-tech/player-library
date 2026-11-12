"""Tests for evaluation units and dual-track metric computation engine."""

import pytest
from test_identity_discovery import synthetic_facts

from app.contracts.identity import Assignment, AssignmentEvidence, Segment
from app.evaluation.metrics import (
    compute_supervised_metrics,
    compute_unsupervised_metrics,
    measure_execution,
)
from app.evaluation.units import (
    EvaluationUnit,
    extract_evaluation_units,
    project_units_to_segments,
)


def test_evaluation_unit_extraction_and_manifest():
    specs = [
        ("clip-a", "0", [1.0, 0.0], "a0"),
        ("clip-b", "0", [1.0, 0.0], "b0"),
    ]
    batch = synthetic_facts(specs)
    manifest = extract_evaluation_units(batch)

    assert manifest.schema_version == "fixed-evaluation-units-v1"
    assert manifest.match_id == "synthetic-match"
    # Each synthetic track has 6 samples and 26 observations
    assert manifest.total_samples == 12
    assert manifest.total_observations == 52
    assert len(manifest.sample_units) == 12
    assert len(manifest.observation_units) == 52
    assert manifest.total_duration_s == pytest.approx(5.2, rel=1e-3)
    assert manifest.clip_count == 2
    assert manifest.track_count == 2

    # Deterministic unit_id uniqueness
    unit_ids = [u.unit_id for u in manifest.units]
    assert len(unit_ids) == len(set(unit_ids))


def test_projection_to_segments_and_coverage():
    specs = [
        ("clip-a", "0", [1.0, 0.0], "a0"),
        ("clip-b", "0", [1.0, 0.0], "b0"),
    ]
    batch = synthetic_facts(specs)
    manifest = extract_evaluation_units(batch)

    # Segment clip-a as assigned to pl_1, and clip-b as unassigned
    seg_a = Segment(
        segment_id="seg-a",
        match_id="synthetic-match",
        clip_id="clip-a",
        track_id="0",
        start_s=0.0,
        end_s=2.6,
        frame_start=0,
        frame_end_exclusive=26,
    )
    seg_b = Segment(
        segment_id="seg-b",
        match_id="synthetic-match",
        clip_id="clip-b",
        track_id="0",
        start_s=0.0,
        end_s=2.6,
        frame_start=0,
        frame_end_exclusive=26,
    )
    asg_a = Assignment(
        segment_id="seg-a",
        match_id="synthetic-match",
        player_id="pl_1",
        status="proposed",
        method="automatic",
        revision=1,
    )
    asg_b = Assignment(
        segment_id="seg-b",
        match_id="synthetic-match",
        player_id=None,
        status="unassigned",
        method="automatic",
        unassigned_reason="ambiguous",
        revision=1,
    )

    projected = project_units_to_segments(manifest.units, [seg_a, seg_b], [asg_a, asg_b])
    assert len(projected) == len(manifest.units)

    assigned_units = [p for p in projected if p.status == "assigned"]
    unassigned_units = [p for p in projected if p.status == "unassigned"]

    # All clip-a units (6 samples + 26 observations = 32) are assigned
    assert len(assigned_units) == 32
    assert all(p.player_id == "pl_1" for p in assigned_units)
    assert all(p.clip_id == "clip-a" for p in assigned_units)

    # All clip-b units are unassigned with ambiguous
    assert len(unassigned_units) == 32
    assert all(p.unassigned_reason == "ambiguous" for p in unassigned_units)

    unsupervised = compute_unsupervised_metrics(manifest, projected, segments=[seg_a, seg_b])
    # Exactly half the samples and half the duration
    assert unsupervised.sample_coverage == pytest.approx(6 / 12)
    assert unsupervised.observation_coverage == pytest.approx(26 / 52)
    assert unsupervised.duration_coverage == pytest.approx(2.6 / 5.2)
    assert unsupervised.unknown_reason_distribution == {"ambiguous": 32}


def test_supervised_metrics_default_unvalidated():
    specs = [("clip-a", "0", [1.0, 0.0], "a0")]
    batch = synthetic_facts(specs)
    manifest = extract_evaluation_units(batch)
    projected = project_units_to_segments(manifest.units, [], [])

    # None or empty ground truth must return unvalidated
    supervised = compute_supervised_metrics(projected, None)
    assert not supervised.identity_recognition_evaluated
    assert supervised.status == "unvalidated"
    assert "未提供身份真值" in (supervised.message or "")
    assert supervised.pairwise is None


def test_supervised_metrics_strict_boundary_conditions():
    # Case 1: When positive predictions == 0, Precision is strictly NaN (never 100%)
    units = [
        EvaluationUnit(
            unit_id=f"u{i}",
            unit_type="sample",
            clip_id=f"clip-{i}",
            track_id="0",
            time_s=1.0,
        )
        for i in range(3)
    ]
    # Projected: all unassigned
    from app.evaluation.units import ProjectedUnit

    projected = [
        ProjectedUnit(
            unit_id=u.unit_id,
            unit_type="sample",
            clip_id=u.clip_id,
            track_id=u.track_id,
            time_s=u.time_s,
            status="unassigned",
            player_id=None,
        )
        for u in units
    ]
    # Ground truth: u0 and u1 are the same person ("P_A"), u2 is "P_B"
    gt = {"u0": "P_A", "u1": "P_A", "u2": "P_B"}

    supervised = compute_supervised_metrics(projected, gt)
    assert supervised.identity_recognition_evaluated
    assert supervised.pairwise is not None
    pw = supervised.pairwise

    # Precision must be None / NaN because TP + FP == 0!
    assert pw.precision is None
    assert "无法计算" in pw.precision_display
    assert pw.positive_pairs_predicted == 0

    # Recall must be 0.0 because GT has a positive pair (u0, u1) across clips, but predicted positive is 0!
    assert pw.recall == 0.0
    assert pw.recall_display == "0.00%"
    assert pw.positive_pairs_truth == 1  # (u0, u1) cross clip
    assert pw.fn == 1
    assert pw.tp == 0


def test_supervised_metrics_pairwise_counts_and_f1():
    from app.evaluation.units import ProjectedUnit

    # 4 units in 4 different clips:
    # u0 (clip0): GT="Alice", Pred="pl_1"
    # u1 (clip1): GT="Alice", Pred="pl_1"  -> (u0, u1): GT same, Pred same => TP
    # u2 (clip2): GT="Bob",   Pred="pl_1"  -> (u0, u2): GT diff, Pred same => FP
    #                                      -> (u1, u2): GT diff, Pred same => FP
    # u3 (clip3): GT="Bob",   Pred="pl_2"  -> (u2, u3): GT same, Pred diff => FN
    #                                      -> (u0, u3): GT diff, Pred diff => TN
    #                                      -> (u1, u3): GT diff, Pred diff => TN
    projected = [
        ProjectedUnit(
            unit_id="u0",
            unit_type="sample",
            clip_id="c0",
            track_id="0",
            time_s=0.0,
            status="assigned",
            player_id="pl_1",
        ),
        ProjectedUnit(
            unit_id="u1",
            unit_type="sample",
            clip_id="c1",
            track_id="0",
            time_s=0.0,
            status="assigned",
            player_id="pl_1",
        ),
        ProjectedUnit(
            unit_id="u2",
            unit_type="sample",
            clip_id="c2",
            track_id="0",
            time_s=0.0,
            status="assigned",
            player_id="pl_1",
        ),
        ProjectedUnit(
            unit_id="u3",
            unit_type="sample",
            clip_id="c3",
            track_id="0",
            time_s=0.0,
            status="assigned",
            player_id="pl_2",
        ),
    ]
    gt = {"u0": "Alice", "u1": "Alice", "u2": "Bob", "u3": "Bob"}
    supervised = compute_supervised_metrics(projected, gt)
    pw = supervised.pairwise
    assert pw is not None
    assert pw.tp == 1
    assert pw.fp == 2
    assert pw.fn == 1
    assert pw.tn == 2

    # Precision = 1 / (1 + 2) = 1/3
    assert pw.precision == pytest.approx(1 / 3)
    # Recall = 1 / (1 + 1) = 1/2
    assert pw.recall == pytest.approx(0.5)
    # F1 = 2 * (1/3) * (1/2) / (1/3 + 1/2) = 0.4
    assert pw.f1 == pytest.approx(0.4)


def test_supervised_metrics_over_merging_impurity():
    from app.evaluation.units import ProjectedUnit

    # Predicted pl_1 contains both "Alice" and "Bob" -> over-merging
    projected = [
        ProjectedUnit(
            unit_id="u0",
            unit_type="sample",
            clip_id="c0",
            track_id="0",
            time_s=0.0,
            status="assigned",
            player_id="pl_1",
        ),
        ProjectedUnit(
            unit_id="u1",
            unit_type="sample",
            clip_id="c1",
            track_id="0",
            time_s=0.0,
            status="assigned",
            player_id="pl_1",
        ),
    ]
    gt = {"u0": "Alice", "u1": "Bob"}
    supervised = compute_supervised_metrics(projected, gt)

    assert supervised.merge_error_count == 1
    err = supervised.merge_errors[0]
    assert err.group_id == "pl_1"
    assert err.gt_person_ids == ["Alice", "Bob"]
    assert err.unit_count == 2


def test_supervised_metrics_over_splitting_fragmentation():
    from app.evaluation.units import ProjectedUnit

    # "Alice" is split into pl_1 and pl_2
    projected = [
        ProjectedUnit(
            unit_id="u0",
            unit_type="sample",
            clip_id="c0",
            track_id="0",
            time_s=0.0,
            status="assigned",
            player_id="pl_1",
        ),
        ProjectedUnit(
            unit_id="u1",
            unit_type="sample",
            clip_id="c1",
            track_id="0",
            time_s=0.0,
            status="assigned",
            player_id="pl_2",
        ),
    ]
    gt = {"u0": "Alice", "u1": "Alice"}
    supervised = compute_supervised_metrics(projected, gt)

    assert supervised.split_error_count == 1
    err = supervised.split_errors[0]
    assert err.gt_person_id == "Alice"
    assert err.predicted_player_ids == ["pl_1", "pl_2"]


def test_supervised_metrics_unlabelled_and_ambiguous_tolerance():
    from app.evaluation.units import ProjectedUnit

    # u0 is Alice, u1 is unlabelled, u2 is ambiguous
    projected = [
        ProjectedUnit(
            unit_id="u0",
            unit_type="sample",
            clip_id="c0",
            track_id="0",
            time_s=0.0,
            status="assigned",
            player_id="pl_1",
        ),
        ProjectedUnit(
            unit_id="u1",
            unit_type="sample",
            clip_id="c1",
            track_id="0",
            time_s=0.0,
            status="assigned",
            player_id="pl_1",
        ),
        ProjectedUnit(
            unit_id="u2",
            unit_type="sample",
            clip_id="c2",
            track_id="0",
            time_s=0.0,
            status="unassigned",
            player_id=None,
        ),
    ]
    gt = {"u0": "Alice", "u1": "unlabelled", "u2": "ambiguous"}
    supervised = compute_supervised_metrics(projected, gt)

    # u1 and u2 must be excluded, not treated as "different person" or error
    assert supervised.unlabelled_units_excluded == 2
    # Only 1 valid unit (u0), no pairs can be formed
    assert supervised.pairwise is not None
    assert supervised.pairwise.evaluated_pairs_count == 0
    assert supervised.merge_error_count == 0
    assert supervised.split_error_count == 0


def test_performance_measurement_context():
    with measure_execution() as perf:
        _ = sum(i * i for i in range(100_000))
    assert perf["runtime_s"] > 0
    assert perf["peak_memory_mb"] >= 0


def test_strictly_half_open_time_boundary():
    """Sample at segment boundary time must belong to the right segment regardless of list order."""
    point = EvaluationUnit(
        unit_id="boundary",
        unit_type="sample",
        clip_id="a",
        track_id="1",
        time_s=1.0,
    )
    seg_left = Segment(
        match_id="m",
        segment_id="left",
        clip_id="a",
        track_id="1",
        start_s=0.0,
        end_s=1.0,
    )
    seg_right = Segment(
        match_id="m",
        segment_id="right",
        clip_id="a",
        track_id="1",
        start_s=1.0,
        end_s=2.0,
    )
    asg_left = Assignment(
        match_id="m",
        segment_id="left",
        player_id="pl_left",
        status="proposed",
        method="automatic",
        revision=1,
    )
    asg_right = Assignment(
        match_id="m",
        segment_id="right",
        player_id="pl_right",
        status="proposed",
        method="automatic",
        revision=1,
    )

    proj_left_first = project_units_to_segments(
        [point], [seg_left, seg_right], [asg_left, asg_right]
    )[0]
    proj_right_first = project_units_to_segments(
        [point], [seg_right, seg_left], [asg_left, asg_right]
    )[0]

    assert proj_left_first.segment_id == "right"
    assert proj_left_first.player_id == "pl_right"
    assert proj_right_first.segment_id == "right"
    assert proj_right_first.player_id == "pl_right"


def test_supervised_metrics_zero_f1_when_tp_zero_and_f1_denom_positive():
    """When TP=0, but FP>0 or FN>0, F1 must be 0.0, not NaN."""
    from app.evaluation.units import ProjectedUnit

    # 3 units in 3 different clips:
    # u0 (c0): GT="x", Pred="pl_p"
    # u1 (c1): GT="x", Pred="pl_q"
    # u2 (c2): GT="y", Pred="pl_p"
    # Pairs:
    # (u0, u1): GT same (x, x), Pred diff (pl_p, pl_q) => FN=1
    # (u0, u2): GT diff (x, y), Pred same (pl_p, pl_p) => FP=1
    # (u1, u2): GT diff (x, y), Pred diff (pl_q, pl_p) => TN=1
    # TP=0, FP=1, FN=1, TN=1
    # 2*TP + FP + FN = 2 > 0 -> F1 = 0.0!
    projected = [
        ProjectedUnit(
            unit_id="u0",
            unit_type="sample",
            clip_id="c0",
            track_id="1",
            time_s=0.0,
            status="assigned",
            player_id="pl_p",
        ),
        ProjectedUnit(
            unit_id="u1",
            unit_type="sample",
            clip_id="c1",
            track_id="1",
            time_s=0.0,
            status="assigned",
            player_id="pl_q",
        ),
        ProjectedUnit(
            unit_id="u2",
            unit_type="sample",
            clip_id="c2",
            track_id="1",
            time_s=0.0,
            status="assigned",
            player_id="pl_p",
        ),
    ]
    gt = {"u0": "x", "u1": "x", "u2": "y"}
    supervised = compute_supervised_metrics(projected, gt)
    pw = supervised.pairwise
    assert pw is not None
    assert pw.tp == 0
    assert pw.fp == 1
    assert pw.fn == 1
    assert pw.precision == 0.0
    assert pw.recall == 0.0
    assert pw.f1 == 0.0
    assert pw.f1_display == "0.00%"


def test_persistent_player_id_and_group_id_separation_in_merging_and_splitting():
    """Over-merging/splitting must evaluate persistent player_id, and group_id is experimental."""
    from app.evaluation.units import ProjectedUnit

    # Two cross-clip samples with the same persistent player_id='pl_same', but different discovery group_ids
    shared_player = [
        ProjectedUnit(
            unit_id="u0",
            unit_type="sample",
            clip_id="c0",
            track_id="1",
            time_s=0.0,
            status="assigned",
            player_id="pl_same",
            group_id="group1",
        ),
        ProjectedUnit(
            unit_id="u1",
            unit_type="sample",
            clip_id="c1",
            track_id="1",
            time_s=0.0,
            status="assigned",
            player_id="pl_same",
            group_id="group2",
        ),
    ]

    # Case A: Same ground truth 'x'
    same_truth = compute_supervised_metrics(shared_player, {"u0": "x", "u1": "x"})
    assert same_truth.pairwise.tp == 1
    assert same_truth.split_error_count == 0  # player_id is unified: no split error!
    assert same_truth.merge_error_count == 0
    # Group-level experimental stats show group split
    assert same_truth.experimental_group_split_error_count == 1

    # Case B: Different ground truth 'x' and 'y'
    diff_truth = compute_supervised_metrics(shared_player, {"u0": "x", "u1": "y"})
    assert diff_truth.pairwise.fp == 1
    assert diff_truth.merge_error_count == 1  # player_id merged x and y: merge error!
    assert diff_truth.split_error_count == 0
    # Group-level experimental stats show no group merge
    assert diff_truth.experimental_group_merge_error_count == 0


def test_coverage_funnel_and_loss_breakdown():
    """Verify construction of unified coverage funnel and detailed loss breakdown."""
    from app.evaluation.funnel import build_coverage_funnel, build_loss_breakdown

    specs = [
        ("clip-a", "0", [1.0, 0.0], "a0"),
        ("clip-b", "0", [1.0, 0.0], "b0"),
    ]
    batch = synthetic_facts(specs)
    manifest = extract_evaluation_units(batch)

    seg_a = Segment(
        segment_id="seg-a",
        match_id="synthetic-match",
        clip_id="clip-a",
        track_id="0",
        start_s=0.0,
        end_s=2.6,
    )
    seg_b = Segment(
        segment_id="seg-b",
        match_id="synthetic-match",
        clip_id="clip-b",
        track_id="0",
        start_s=0.0,
        end_s=2.6,
    )
    asg_a = Assignment(
        segment_id="seg-a",
        match_id="synthetic-match",
        player_id="pl_1",
        status="proposed",
        method="automatic",
        revision=1,
    )
    asg_b = Assignment(
        segment_id="seg-b",
        match_id="synthetic-match",
        player_id=None,
        status="unassigned",
        method="automatic",
        unassigned_reason="ambiguous",
        evidence=[
            AssignmentEvidence(
                kind="insufficient_evidence",
                description="appearance_instability",
            )
        ],
        revision=1,
    )

    projected = project_units_to_segments(manifest.units, [seg_a, seg_b], [asg_a, asg_b])
    funnel = build_coverage_funnel(
        manifest,
        projected,
        batch,
        segments=[seg_a, seg_b],
        assignments=[asg_a, asg_b],
    )

    assert funnel.schema_version == "coverage-funnel-v1"
    assert funnel.denominators.total_tracks == 2
    assert funnel.denominators.total_samples == 12
    assert len(funnel.stages) == 7

    # Raw tracks stage
    raw_stage = funnel.stages[0]
    assert raw_stage.stage_name == "raw_tracks"
    assert raw_stage.samples_count == 12
    assert raw_stage.sample_coverage == 1.0

    # Final assigned stage
    final_stage = next(s for s in funnel.stages if s.stage_name == "final_assigned_identities")
    assert final_stage.samples_count == 6
    assert final_stage.sample_coverage == 0.5
    assert final_stage.tracks_count == 1

    # Loss breakdown
    loss = build_loss_breakdown(
        manifest,
        projected,
        batch,
        segments=[seg_a, seg_b],
        assignments=[asg_a, asg_b],
    )
    assert loss.schema_version == "coverage-loss-breakdown-v1"
    assert loss.total_unassigned_samples == 6
    assert "clip-a" in loss.by_clip
    assert "clip-b" in loss.by_clip
    assert loss.by_clip["clip-a"].lost_samples == 0
    assert loss.by_clip["clip-b"].lost_samples == 6
    assert "appearance_instability" in loss.by_reason
    assert loss.by_reason["appearance_instability"].hit_samples == 6
    assert "不可简单线性相加" in loss.overlap_warning


def test_funnel_stage_4_cross_clip_comparable_segments_regression():
    """Funnel Stage 4 (cross_clip_comparable_segments) must strictly reflect only qualified segments.

    Regression verification:
    - Pre-qualification disqualified segments (segment_not_supported, short_segment_limited_match_support, etc.)
      must NEVER be added to qualified_seg_ids.
    - Stage 4 samples and duration must be strictly bounded by the qualified segments rather than
      falsely jumping back up to 100% of raw batch.
    """
    from app.contracts.discovery import (
        DiscoveredGroup,
        DiscoveryPolicy,
        DiscoveryReport,
        UnknownSegment,
    )
    from app.evaluation.funnel import build_coverage_funnel

    # 4 tracks with 6 samples each = 24 samples total
    specs = [
        ("clip-a", "0", [1.0, 0.0], "a0"),
        ("clip-b", "0", [1.0, 0.0], "b0"),
        ("clip-c", "0", [1.0, 0.0], "c0"),
        ("clip-d", "0", [1.0, 0.0], "d0"),
    ]
    batch = synthetic_facts(specs)
    manifest = extract_evaluation_units(batch)

    # 4 active segments
    segments = [
        Segment(
            segment_id=f"seg-{cid}",
            match_id="synthetic-match",
            clip_id=cid,
            track_id="0",
            start_s=0.0,
            end_s=2.6,
            frame_start=0,
            frame_end_exclusive=26,
            active=True,
        )
        for cid in ["clip-a", "clip-b", "clip-c", "clip-d"]
    ]
    projected = project_units_to_segments(manifest.units, segments, [])

    # Discovery report:
    # - Group has seg-clip-a and seg-clip-b (2 qualified segments assigned to group)
    # - seg-clip-c is qualified but has no cross-clip partner: reason=['no_independent_cross_clip_group']
    # - seg-clip-d is disqualified during consistency: reason=['segment_not_supported']
    disc_report = DiscoveryReport(
        report_id="dsr-test",
        match_id="synthetic-match",
        policy=DiscoveryPolicy(),
        policy_fingerprint="0" * 64,
        input_fingerprint="0" * 64,
        segmentation_fingerprint="0" * 64,
        feature_space_id="synthetic-space",
        groups=[
            DiscoveredGroup(
                group_id="dsg-1",
                member_segment_ids=["seg-clip-a", "seg-clip-b"],
                anchor_segment_id="seg-clip-a",
                support_pair_ids=[],
                templates=[],
            )
        ],
        pairs=[],
        unknown=[
            UnknownSegment(segment_id="seg-clip-c", reasons=["no_independent_cross_clip_group"]),
            UnknownSegment(
                segment_id="seg-clip-d",
                reasons=["segment_not_supported", "appearance_or_visibility_unresolved"],
            ),
        ],
        summary={"qualified_segments": 3, "groups": 1, "member_segments": 2, "unknown_segments": 2},
    )

    funnel = build_coverage_funnel(
        manifest,
        projected,
        batch,
        segments=segments,
        discovery_report=disc_report,
    )

    stage_4 = next(s for s in funnel.stages if s.stage_name == "cross_clip_comparable_segments")
    assert stage_4.applicable is True
    # Exactly 3 qualified segments (seg-clip-a, seg-clip-b, seg-clip-c), NOT 4!
    assert stage_4.segments_count == 3
    # 3 segments * 6 samples = 18 samples (75% coverage), NOT 24 samples (100%)!
    assert stage_4.samples_count == 18
    assert stage_4.sample_coverage == 0.75
    assert stage_4.duration_s == pytest.approx(7.8, rel=1e-3)
    assert stage_4.duration_coverage == pytest.approx(0.75, rel=1e-3)
    # Monotonicity check: stage 4 must not equal raw batch 100%
    assert stage_4.sample_coverage < 1.0


def test_incremental_funnel_monotonicity_regression():
    """Assert multi-batch incremental funnel avoids single-batch scope inversion and is strictly monotonic non-increasing."""
    from app.contracts.common import Provenance
    from app.contracts.segmentation import (
        SegmentationPolicy,
        SegmentationReport,
        SegmentProposal,
        TrackConsistencyReport,
    )
    from app.evaluation.funnel import build_coverage_funnel

    specs = [
        ("clip-a", "0", [1.0, 0.0], "a0"),
        ("clip-b", "0", [1.0, 0.0], "b0"),
    ]
    batch = synthetic_facts(specs)
    manifest = extract_evaluation_units(batch)

    # 2 segments (one for clip-a, one for clip-b), each covering 6 samples
    seg_a = Segment(
        segment_id="seg-a",
        match_id="synthetic-match",
        clip_id="clip-a",
        track_id="0",
        start_s=0.0,
        end_s=2.6,
        frame_start=0,
        frame_end_exclusive=26,
    )
    seg_b = Segment(
        segment_id="seg-b",
        match_id="synthetic-match",
        clip_id="clip-b",
        track_id="0",
        start_s=0.0,
        end_s=2.6,
        frame_start=0,
        frame_end_exclusive=26,
    )
    segments = [seg_a, seg_b]

    # Assign both seg_a and seg_b to pl_1 (total 12 samples assigned)
    assignments = [
        Assignment(
            segment_id="seg-a",
            match_id="synthetic-match",
            player_id="pl_1",
            status="proposed",
            method="automatic",
            revision=1,
        ),
        Assignment(
            segment_id="seg-b",
            match_id="synthetic-match",
            player_id="pl_1",
            status="proposed",
            method="automatic",
            revision=1,
        ),
    ]
    projected = project_units_to_segments(manifest.units, segments, assignments)

    prov = Provenance(
        producer="test", record_type="proposal", source_record_id="p1", evidence_version="1"
    )
    prop_a = SegmentProposal(
        segment=seg_a,
        parent_segment_id="seg-a",
        source=prov,
        status="supported",
        reasons=["supported"],
    )
    prop_b = SegmentProposal(
        segment=seg_b,
        parent_segment_id="seg-b",
        source=prov,
        status="supported",
        reasons=["supported"],
    )

    report_batch_1 = SegmentationReport(
        schema_version="track-consistency-report-v1",
        policy=SegmentationPolicy(),
        policy_fingerprint="0" * 64,
        input_fingerprint="0" * 64,
        tracks=[TrackConsistencyReport(parent_segment=seg_a, proposals=[prop_a])],
        summary={},
    )
    report_batch_2 = SegmentationReport(
        schema_version="track-consistency-report-v1",
        policy=SegmentationPolicy(),
        policy_fingerprint="0" * 64,
        input_fingerprint="0" * 64,
        tracks=[TrackConsistencyReport(parent_segment=seg_b, proposals=[prop_b])],
        summary={},
    )
    all_seg_reports = [report_batch_1, report_batch_2]

    # 1. Verify that if only report_batch_2 (all_seg_reports[-1]) were passed, the inversion occurs:
    buggy_funnel = build_coverage_funnel(
        manifest,
        projected,
        batch,
        segments=segments,
        assignments=assignments,
        segmentation_report=all_seg_reports[-1],
    )
    stage_3_buggy = next(
        s for s in buggy_funnel.stages if s.stage_name == "locally_consistent_segments"
    )
    stage_6_buggy = next(
        s for s in buggy_funnel.stages if s.stage_name == "final_assigned_identities"
    )
    assert stage_3_buggy.samples_count == 6  # only batch 2
    assert stage_6_buggy.samples_count == 12  # both batches
    assert stage_3_buggy.samples_count < stage_6_buggy.samples_count  # Inverted!

    # 2. Now verify with combined_seg_report (the fix):
    combined_tracks = [t for r in all_seg_reports for t in r.tracks]
    combined_seg_report = SegmentationReport(
        schema_version="track-consistency-report-v1",
        policy=all_seg_reports[0].policy,
        policy_fingerprint=all_seg_reports[0].policy_fingerprint,
        input_fingerprint="0" * 64,
        tracks=combined_tracks,
        summary={},
    )
    fixed_funnel = build_coverage_funnel(
        manifest,
        projected,
        batch,
        segments=segments,
        assignments=assignments,
        segmentation_report=combined_seg_report,
    )
    stage_3_fixed = next(
        s for s in fixed_funnel.stages if s.stage_name == "locally_consistent_segments"
    )
    stage_6_fixed = next(
        s for s in fixed_funnel.stages if s.stage_name == "final_assigned_identities"
    )
    assert stage_3_fixed.samples_count == 12  # full 12 samples from both batches
    assert stage_6_fixed.samples_count == 12
    assert stage_3_fixed.samples_count >= stage_6_fixed.samples_count

    # 3. Assert strict monotonicity across all applicable stages
    applicable_stages = [
        s for s in fixed_funnel.stages if s.applicable and s.samples_count is not None
    ]
    for i in range(len(applicable_stages) - 1):
        s_curr = applicable_stages[i]
        s_next = applicable_stages[i + 1]
        assert s_curr.samples_count >= s_next.samples_count, (
            f"Funnel inversion detected between {s_curr.stage_name} ({s_curr.samples_count}) "
            f"and {s_next.stage_name} ({s_next.samples_count})"
        )
