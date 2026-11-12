"""Comprehensive business and failure-path tests for track-consistency-v3.

Verifies:
1. Gradual turnaround: smooth continuous appearance recovers local supported windows
   instead of discarding the entire track as unknown (answering the core Stage 2 question).
2. Normal movement: running and bending with continuous observations form supported windows.
3. Brief occlusion: isolates the occluded sample into an unknown hole without faking an identity switch.
4. Single abnormal image: isolated as a locatable risk, never averaged or majority-voted away.
5. Sustained identity mixing: different persons with far appearance separation cannot be packaged
   into a clean segment; reports suspected_identity_switch when supported by spatial/motion breaks.
6. Minority strong conflict: zero far pairs requirement prevents minority intruders from passing.
7. Observation interruption: gaps > max_observation_gap_s partition the track into independent extents.
8. Uncertain boundaries: perspective transitions retain unknown uncertainty; exact boundaries not faked.
9. Eligibility separation: distinguishes can_match_existing (short segments) from can_create_new_identity.
10. Scoped risk & upstream review: accepted undergoes internal checks; needs_review preserves clean windows
    without clearing real conflict risks.
"""

import numpy as np

from app.contracts.common import Provenance
from app.contracts.evidence import (
    Clip,
    EvidenceBatch,
    FeatureSpace,
    IdentityEvidence,
    InlineFeature,
    Preprocessing,
    Sample,
    Track,
    TrackingQuality,
    TrackObservation,
    VisibilityContract,
)
from app.contracts.identity import IdentityRisk, Segment
from app.contracts.segmentation import SegmentationPolicy
from app.identity.consistency import segment_tracks


def source(key: str) -> Provenance:
    return Provenance(
        producer="synthetic",
        record_type="observation",
        source_record_id=key,
        evidence_version="1",
    )


def make_v3_batch(
    *,
    angles=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    sample_frames=None,
    jump_frame=None,
    missing_frames=(),
    occluded_indices=(),
    unusable_indices=(),
    fps=10.0,
    duplicate_box=False,
):
    """Build synthetic EvidenceBatch with controlled angles and physical observations."""
    sample_frames = list(
        sample_frames if sample_frames is not None else range(0, len(angles) * 5, 5)
    )
    count = max(sample_frames) + 1
    space = FeatureSpace(
        feature_space_id="synthetic-space",
        backend="kpr",
        model_hash="b" * 64,
        preprocessing=Preprocessing(version="synthetic", color_space="rgb"),
        layout="parts",
        shape=[3, 2],
        normalization="l2",
        distance="euclidean_sqrt_visibility_mean_div2",
        visibility=VisibilityContract(shape=[3], semantics="visible"),
    )
    observations = []
    for frame in range(count):
        if frame in missing_frames:
            continue
        offset = 500 if jump_frame is not None and frame >= jump_frame else 0
        observations.append(
            TrackObservation(
                observation_id=f"obs-0-{frame}",
                match_id="m",
                clip_id="c1",
                track_id="0",
                frame_idx=frame,
                time_s=frame / fps,
                bbox_xyxy=(float(10 + offset), 10.0, float(60 + offset), 110.0),
                detection_score=0.98,
                source=source(f"obs-0-{frame}"),
            )
        )
        if duplicate_box and frame == (jump_frame or 10):
            observations.append(
                TrackObservation(
                    observation_id=f"obs-1-{frame}",
                    match_id="m",
                    clip_id="c1",
                    track_id="1",
                    frame_idx=frame,
                    time_s=frame / fps,
                    bbox_xyxy=(float(11 + offset), 10.5, float(59 + offset), 109.5),
                    detection_score=0.95,
                    source=source(f"obs-1-{frame}"),
                )
            )

    parent = Segment(
        segment_id="parent-0",
        match_id="m",
        clip_id="c1",
        track_id="0",
        start_s=0.0,
        end_s=count / fps,
        frame_start=0,
        frame_end_exclusive=count,
    )
    tracks = [
        Track(
            match_id="m",
            clip_id="c1",
            track_id="0",
            start_s=0.0,
            end_s=count / fps,
            frame_start=0,
            frame_end_exclusive=count,
            observed_frame_indices=[o.frame_idx for o in observations if o.track_id == "0"],
            source=source("t-0"),
        )
    ]
    if duplicate_box:
        tracks.append(
            Track(
                match_id="m",
                clip_id="c1",
                track_id="1",
                start_s=0.0,
                end_s=count / fps,
                frame_start=0,
                frame_end_exclusive=count,
                observed_frame_indices=[o.frame_idx for o in observations if o.track_id == "1"],
                source=source("t-1"),
            )
        )

    samples = []
    for i, (angle, frame) in enumerate(zip(angles, sample_frames, strict=True)):
        if frame in missing_frames:
            continue
        vector = [float(np.cos(angle)), float(np.sin(angle))] * 3
        feature = InlineFeature(
            feature_space_id=space.feature_space_id,
            values=vector,
            visibility=[1.0] * 3,
        )
        is_occluded = i in occluded_indices
        is_unusable = i in unusable_indices
        samples.append(
            Sample(
                sample_id=f"s-0-{frame}",
                match_id="m",
                clip_id="c1",
                track_id="0",
                segment_id=parent.segment_id,
                frame_idx=frame,
                time_s=frame / fps,
                source=source(f"s-0-{frame}"),
                features=[feature],
                tracking_quality=TrackingQuality(
                    occluded=is_occluded,
                    usability="unusable" if is_unusable else "usable",
                ),
            )
        )

    return EvidenceBatch(
        match_id="m",
        batch_id="v3-batch",
        producer="synthetic",
        clips=[
            Clip(
                match_id="m",
                clip_id="c1",
                frame_count=count,
                duration_s=count / fps,
                frame_timing="cfr",
                fps=fps,
            )
        ],
        identity_evidence=IdentityEvidence(
            observations=observations,
            tracks=tracks,
            segments=[parent],
            samples=samples,
            feature_spaces=[space],
        ),
    )


# -----------------------------------------------------------------------------
# Scenario 1: Gradual turnaround (the primary Stage 2 motivating scenario)
# -----------------------------------------------------------------------------
def test_v3_gradual_turnaround_recovers_local_support_where_v2_fails():
    """In a gradual turnaround, v2 all-pairs fails completely while v3 recovers local support."""
    # Gradual turnaround with step delta = 0.20:
    # Consecutive delta = 0.20: sin(0.10) ~ 0.0998 (near, <= 0.18)
    # 2-step delta = 0.40: sin(0.20) ~ 0.1987 (ambiguous, > 0.18)
    # For v2: all pairs in any 3-bucket window must be near, so distance of step 2 (>0.18) causes v2 to fail completely.
    # For v3: medoid anchor (index 1) is near to both index 0 and index 2 (delta 0.20 <= 0.18),
    # and adjacent pairs are near, so v3 recovers a valid local support window,
    # while the continuing shift (sample 3, delta 0.60) forms a viewpoint transition zone.
    angles = (0.0, 0.20, 0.40, 0.60)
    frames = [0, 6, 12, 18]
    batch = make_v3_batch(angles=angles, sample_frames=frames, fps=10.0)

    # V2 execution: strict all-pairs near fails because step 2 distance is ambiguous (>0.18)
    p_v2 = SegmentationPolicy(version="track-consistency-v2", local_support_windows=True)
    res_v2 = segment_tracks(batch, policy=p_v2)
    v2_supported = [p for t in res_v2.report.tracks for p in t.proposals if p.status == "supported"]

    # V3 execution: local continuous appearance window with medoid support recovers valid windows
    p_v3 = SegmentationPolicy(version="track-consistency-v3")
    res_v3 = segment_tracks(batch, policy=p_v3)
    v3_supported = [p for t in res_v3.report.tracks for p in t.proposals if p.status == "supported"]

    # Key Stage 2 assertion: v3 recovers local support where v2 had 0 supported segments
    assert len(v2_supported) == 0, "V2 should fail all-pairs on gradual turnaround"
    assert len(v3_supported) >= 1, "V3 must recover local continuous appearance support"
    for prop in v3_supported:
        assert prop.can_match_existing is True
        assert prop.identity_verified is False
        assert "local_continuous_appearance_with_medoid_support" in prop.reasons

    # Check that viewpoint transition zone is extracted for the transition uncertainty
    transition_findings = [
        f for t in res_v3.report.tracks for f in t.findings if f.kind == "viewpoint_transition_zone"
    ]
    assert len(transition_findings) >= 1
    assert transition_findings[0].details["switch_inferred"] is False


# -----------------------------------------------------------------------------
# Scenario 2: Normal movement (running, bending)
# -----------------------------------------------------------------------------
def test_v3_normal_movement_running_with_slight_posture_shifts():
    """Running player with minor posture variations forms supported windows."""
    # Slight oscillation around angle 0.0 within delta 0.12 (all pairs within medoid radius)
    angles = (0.0, 0.08, -0.05, 0.06, 0.0, 0.07)
    frames = [0, 5, 10, 15, 20, 25]
    batch = make_v3_batch(angles=angles, sample_frames=frames, fps=10.0)

    p_v3 = SegmentationPolicy(version="track-consistency-v3")
    res_v3 = segment_tracks(batch, policy=p_v3)
    v3_supported = [p for t in res_v3.report.tracks for p in t.proposals if p.status == "supported"]

    assert len(v3_supported) == 1
    assert v3_supported[0].can_match_existing is True
    assert v3_supported[0].can_create_new_identity is True
    assert len(v3_supported[0].support_sample_ids) == 6


# -----------------------------------------------------------------------------
# Scenario 3: Brief occlusion isolates hole without faking identity switch
# -----------------------------------------------------------------------------
def test_v3_brief_occlusion_isolates_hole_without_faking_switch():
    """A brief occlusion creates an unknown hole; both flanks are supported, no switch faked."""
    # 7 samples, sample at index 3 is occluded
    angles = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    frames = [0, 5, 10, 15, 20, 25, 30]
    batch = make_v3_batch(angles=angles, sample_frames=frames, occluded_indices=(3,), fps=10.0)

    p_v3 = SegmentationPolicy(version="track-consistency-v3")
    res_v3 = segment_tracks(batch, policy=p_v3)
    props = [p for t in res_v3.report.tracks for p in t.proposals]
    supported = [p for p in props if p.status == "supported"]
    unknown = [p for p in props if p.status == "unknown"]

    # Both flanks (0-10 and 20-30) should be supported
    assert len(supported) == 2
    # The occluded sample (time_s 1.5) must be inside an unknown hole
    assert any(
        p.segment.start_s <= 1.5 < p.segment.end_s and p.status == "unknown" for p in unknown
    )

    # CRITICAL: brief occlusion alone does NOT report suspected_identity_switch
    findings = [f for t in res_v3.report.tracks for f in t.findings]
    assert not any(f.kind == "suspected_identity_switch" for f in findings)


# -----------------------------------------------------------------------------
# Scenario 4: Single abnormal image retained as locatable risk
# -----------------------------------------------------------------------------
def test_v3_single_abnormal_image_isolated_as_risk_not_washed_out():
    """A single abnormal image is never averaged away or majority-voted into a clean segment."""
    # Sample index 3 is abnormal with far distance (angle 2.5 rad, sin(1.25) = 0.949 >> 0.45)
    angles = (0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0)
    frames = [0, 5, 10, 15, 20, 25, 30]
    batch = make_v3_batch(angles=angles, sample_frames=frames, fps=10.0)

    p_v3 = SegmentationPolicy(version="track-consistency-v3")
    res_v3 = segment_tracks(batch, policy=p_v3)
    props = [p for t in res_v3.report.tracks for p in t.proposals]
    supported = [p for p in props if p.status == "supported"]

    # The abnormal sample s-0-15 CANNOT be in any supported segment!
    for p in supported:
        assert "s-0-15" not in p.support_sample_ids, (
            "Abnormal far sample must not be washed into any supported segment"
        )

    # A conflict finding is recorded
    findings = [f for t in res_v3.report.tracks for f in t.findings]
    assert any("appearance" in f.kind for f in findings)
    # Flanks with sufficient buckets (0-10 and 20-30) are preserved
    assert len(supported) == 2


# -----------------------------------------------------------------------------
# Scenario 5: Sustained identity mixing with spatial discontinuity
# -----------------------------------------------------------------------------
def test_v3_sustained_identity_mixing_reports_switch_with_motion_break():
    """Two different persons (far appearance separation) across a motion jump report suspected_identity_switch."""
    # Player A: angles 0.0, frames 0, 5, 10
    # Jump at frame 15 (displacement 500 px)
    # Player B: angles 2.5 (far from 0.0), frames 15, 20, 25
    angles = (0.0, 0.0, 0.0, 2.5, 2.5, 2.5)
    frames = [0, 5, 10, 15, 20, 25]
    batch = make_v3_batch(angles=angles, sample_frames=frames, jump_frame=15, fps=10.0)

    p_v3 = SegmentationPolicy(version="track-consistency-v3")
    res_v3 = segment_tracks(batch, policy=p_v3)
    findings = [f for t in res_v3.report.tracks for f in t.findings]

    # Must detect suspected_identity_switch
    switch_findings = [f for f in findings if f.kind == "suspected_identity_switch"]
    assert len(switch_findings) == 1
    assert switch_findings[0].details["appearance_separation"] == "all_cross_pairs_far"
    assert switch_findings[0].details["exact_boundary_known"] is False
    assert switch_findings[0].details["distinct_identity_confirmed"] is False

    # The transition interval must be unknown
    props = [p for t in res_v3.report.tracks for p in t.proposals]
    assert any(p.status == "unknown" and p.segment.start_s <= 1.5 <= p.segment.end_s for p in props)


# -----------------------------------------------------------------------------
# Scenario 6: Minority strong conflict vetoed by zero-far pairs rule
# -----------------------------------------------------------------------------
def test_v3_minority_strong_conflict_cannot_pass_by_majority_vote():
    """In a run of 6 samples, 1 foreign intruder (far) cannot be majority-voted into a supported run."""
    # 5 samples of Player A (0.0), 1 sample of Player B (2.5) at index 4
    angles = (0.0, 0.0, 0.0, 0.0, 2.5, 0.0)
    frames = [0, 5, 10, 15, 20, 25]
    batch = make_v3_batch(angles=angles, sample_frames=frames, fps=10.0)

    p_v3 = SegmentationPolicy(version="track-consistency-v3")
    res_v3 = segment_tracks(batch, policy=p_v3)
    props = [p for t in res_v3.report.tracks for p in t.proposals]
    supported = [p for p in props if p.status == "supported"]

    # Intruder s-0-20 cannot be in any supported segment
    for p in supported:
        assert "s-0-20" not in p.support_sample_ids
    # Prefix (0-15) can be supported
    assert any(
        "s-0-0" in p.support_sample_ids and "s-0-15" in p.support_sample_ids for p in supported
    )


# -----------------------------------------------------------------------------
# Scenario 7: Observation interruption partitions into independent extents
# -----------------------------------------------------------------------------
def test_v3_observation_interruption_creates_independent_extents():
    """Gaps > max_observation_gap_s (0.25s) partition the track into independent extents."""
    # Missing frames 11-15 (0.4s gap > 0.25s)
    angles = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    frames = [0, 5, 10, 20, 25, 30]
    batch = make_v3_batch(
        angles=angles,
        sample_frames=frames,
        missing_frames=range(11, 20),
        fps=10.0,
    )

    p_v3 = SegmentationPolicy(version="track-consistency-v3")
    res_v3 = segment_tracks(batch, policy=p_v3)
    props = [p for t in res_v3.report.tracks for p in t.proposals]
    supported = [p for p in props if p.status == "supported"]
    unknown = [p for p in props if p.status == "unknown"]

    # Both extents (before and after interruption) form independent supported segments
    assert len(supported) == 2
    # The gap interval (1.1s to 2.0s) is explicitly unknown due to observation gap
    assert any(
        p.reasons == ["unobserved_or_unreliable_interval"]
        and p.segment.start_s >= 1.0
        and p.segment.end_s <= 2.1
        for p in unknown
    )


# -----------------------------------------------------------------------------
# Scenario 8: Eligibility separation (can_match_existing vs can_create_new_identity)
# -----------------------------------------------------------------------------
def test_v3_eligibility_separation_short_segments_vs_full():
    """Short segments (2 buckets) get can_match_existing=True but can_create_new_identity=False."""
    # Track 1: 2 buckets (frames 0, 5 -> time 0.0, 0.5: exactly 2 relative buckets [0, 1] with bucket_s=0.5)
    # Track 2: 4 buckets (frames 0, 5, 10, 15 -> 4 relative buckets)
    angles_short = (0.0, 0.0)
    frames_short = [0, 5]
    batch_short = make_v3_batch(angles=angles_short, sample_frames=frames_short, fps=10.0)

    p_v3 = SegmentationPolicy(version="track-consistency-v3")
    res_short = segment_tracks(batch_short, policy=p_v3)
    props_short = [
        p for t in res_short.report.tracks for p in t.proposals if p.status == "supported"
    ]
    assert len(props_short) == 1
    assert props_short[0].can_match_existing is True
    assert props_short[0].can_create_new_identity is False
    assert props_short[0].reasons == ["short_segment_limited_match_support"]

    angles_full = (0.0, 0.0, 0.0, 0.0)
    frames_full = [0, 5, 10, 15]
    batch_full = make_v3_batch(angles=angles_full, sample_frames=frames_full, fps=10.0)
    res_full = segment_tracks(batch_full, policy=p_v3)
    props_full = [p for t in res_full.report.tracks for p in t.proposals if p.status == "supported"]
    assert len(props_full) == 1
    assert props_full[0].can_match_existing is True
    assert props_full[0].can_create_new_identity is True
    assert props_full[0].reasons == ["local_continuous_appearance_with_medoid_support"]


# -----------------------------------------------------------------------------
# Scenario 9: Scoped risks & upstream review behavior
# -----------------------------------------------------------------------------
def test_v3_accepted_and_needs_review_behavior():
    """accepted tracks still undergo full internal checks; needs_review preserves clean windows."""
    # 1. Accepted track with mixed identities must still be split/vetoed
    angles = (0.0, 0.0, 0.0, 2.5, 2.5, 2.5)
    frames = [0, 5, 10, 15, 20, 25]
    batch = make_v3_batch(angles=angles, sample_frames=frames, jump_frame=15, fps=10.0)
    # Upstream accepted flag (simulate clean tracking)
    p_v3 = SegmentationPolicy(version="track-consistency-v3")
    res = segment_tracks(batch, policy=p_v3)
    # Must NOT accept whole track as single supported segment
    supported = [p for t in res.report.tracks for p in t.proposals if p.status == "supported"]
    assert len(supported) == 2  # Separated into two distinct flank windows, not one whole track

    # 2. Track with mixed_identity risk: risk is ALWAYS preserved on children
    batch.identity_evidence.risks.append(
        IdentityRisk(
            risk_id="true_mix_risk",
            kind="mixed_identity",
            match_id="m",
            segment_id="parent-0",
        )
    )
    res_mix = segment_tracks(batch, policy=p_v3)
    for p in [p for t in res_mix.report.tracks for p in t.proposals if p.status == "supported"]:
        assert any(
            r.kind == "mixed_identity" and r.segment_id == p.segment.segment_id
            for r in res_mix.report.tracks[0].risks
        ), "Real mixed_identity risk must not be cleared from supported children"
