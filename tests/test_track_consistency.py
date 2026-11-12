"""Synthetic machine evidence only; no fixtures from the companion handoff."""

from copy import deepcopy

import numpy as np
import pytest

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


def source(key):
    return Provenance(
        producer="synthetic", record_type="observation", source_record_id=key, evidence_version="1"
    )


def make_batch(
    *,
    angles=(0, 0, 0, 0, 0, 0),
    sample_frames=None,
    jump_frame=None,
    missing_frames=(),
    occluded=(),
    visibility=None,
    features=True,
    extra_tracks=0,
    camera=False,
):
    sample_frames = list(
        sample_frames if sample_frames is not None else range(0, len(angles) * 5, 5)
    )
    count = max(sample_frames) + 1
    space = FeatureSpace(
        feature_space_id="synthetic-space",
        backend="kpr",
        model_hash="a" * 64,
        preprocessing=Preprocessing(version="synthetic", color_space="rgb"),
        layout="parts",
        shape=[3, 2],
        normalization="l2",
        distance="euclidean_sqrt_visibility_mean_div2",
        visibility=VisibilityContract(shape=[3], semantics="visible"),
    )
    observations, tracks, segments, samples = [], [], [], []
    for t in range(extra_tracks + 1):
        tid = str(t)
        obs = []
        for frame in range(count):
            if frame in missing_frames:
                continue
            offset = (
                500 if jump_frame is not None and frame >= jump_frame and (t == 0 or camera) else 0
            )
            obs.append(
                TrackObservation(
                    observation_id=f"o-{tid}-{frame}",
                    match_id="m",
                    clip_id="clip",
                    track_id=tid,
                    frame_idx=frame,
                    time_s=frame / 10,
                    bbox_xyxy=(float(10 + offset), 10.0, float(60 + offset), 110.0),
                    detection_score=0.99,
                    source=source(f"o-{tid}-{frame}"),
                )
            )
        observations.extend(obs)
        parent = Segment(
            segment_id=f"parent-{tid}",
            match_id="m",
            clip_id="clip",
            track_id=tid,
            start_s=0.0,
            end_s=count / 10,
            frame_start=0,
            frame_end_exclusive=count,
        )
        segments.append(parent)
        tracks.append(
            Track(
                match_id="m",
                clip_id="clip",
                track_id=tid,
                start_s=0.0,
                end_s=count / 10,
                frame_start=0,
                frame_end_exclusive=count,
                observed_frame_indices=[o.frame_idx for o in obs],
                source=source(f"t-{tid}"),
            )
        )
        for i, (angle, frame) in enumerate(zip(angles, sample_frames, strict=True)):
            if frame in missing_frames:
                continue
            vector = [float(np.cos(angle)), float(np.sin(angle))] * 3
            feature = InlineFeature(
                feature_space_id=space.feature_space_id,
                values=vector,
                visibility=list(visibility[i]) if visibility else [1.0] * 3,
            )
            samples.append(
                Sample(
                    sample_id=f"s-{tid}-{frame}",
                    match_id="m",
                    clip_id="clip",
                    track_id=tid,
                    segment_id=parent.segment_id,
                    frame_idx=frame,
                    time_s=frame / 10,
                    source=source(f"s-{tid}-{frame}"),
                    features=[feature] if features else [],
                    tracking_quality=TrackingQuality(occluded=i in occluded),
                )
            )
    return EvidenceBatch(
        match_id="m",
        batch_id="synthetic",
        producer="synthetic",
        clips=[
            Clip(
                match_id="m",
                clip_id="clip",
                frame_count=count,
                duration_s=count / 10,
                frame_timing="cfr",
                fps=10.0,
            )
        ],
        identity_evidence=IdentityEvidence(
            observations=observations,
            tracks=tracks,
            segments=segments,
            samples=samples,
            feature_spaces=[space],
        ),
    )


def findings(result):
    return [f for t in result.report.tracks for f in t.findings]


def proposals(result):
    return [p for t in result.report.tracks for p in t.proposals]


def test_stable_same_person_has_one_supported_proposal_and_immutable_raw_input():
    batch = make_batch(angles=(0, 0.05, -0.1, 0.04, 0.08, -0.02))
    before = deepcopy(batch)
    result = segment_tracks(batch)
    assert batch == before
    assert len(proposals(result)) == 1
    assert proposals(result)[0].status == "supported"
    assert proposals(result)[0].source.producer == "track_consistency_v1"
    assert all(not p.identity_verified for p in proposals(result))
    assert not any(f.kind == "suspected_identity_switch" for f in findings(result))
    assert len(result.report.tracks[0].comparisons) == 15


def test_same_person_abrupt_view_change_does_not_become_identity_switch():
    # Synthetic ground truth is same person; the observations alone cannot prove that.
    result = segment_tracks(make_batch(angles=(0, 0, 0, 2, 2, 2)))
    assert not any(f.kind == "suspected_identity_switch" for f in findings(result))
    assert all(p.status == "unknown" for p in proposals(result))
    assert "appearance_conflict_including_minority" in proposals(result)[0].reasons


def test_true_mixed_track_needs_persistent_appearance_and_local_motion_evidence():
    result = segment_tracks(make_batch(angles=(0, 0, 0, 2, 2, 2), jump_frame=13))
    supported = [p for p in proposals(result) if p.status == "supported"]
    assert len(supported) == 2
    switches = [f for f in findings(result) if f.kind == "suspected_identity_switch"]
    assert len(switches) == 1
    assert switches[0].start_s == 1.0
    assert switches[0].end_s == 1.5
    assert switches[0].details["geometric_break_interval"] == [1.2, 1.3]
    assert not switches[0].details["exact_boundary_known"]
    assert any(
        p.status == "unknown" and p.segment.start_s <= 1.1 and p.segment.end_s >= 1.5
        for p in proposals(result)
    )
    assert any(r.kind == "identity_switch" for r in result.report.tracks[0].risks)


def test_a_single_foreign_sample_blocks_majority_even_in_same_temporal_bucket():
    batch = make_batch(angles=(0,) * 12)
    original = batch.identity_evidence.samples[5]
    foreign = original.model_copy(
        update={
            "sample_id": "foreign",
            "time_s": 2.6,
            "frame_idx": 26,
            "source": source("foreign"),
            "features": [
                InlineFeature(
                    feature_space_id="synthetic-space", values=[-1.0, 0.0] * 3, visibility=[1.0] * 3
                )
            ],
        },
        deep=True,
    )
    batch.identity_evidence.samples.append(foreign)
    result = segment_tracks(batch)
    assert all(p.status == "unknown" for p in proposals(result))
    assert any(
        p.distance == pytest.approx(1.0) and "foreign" in [p.left_sample_id, p.right_sample_id]
        for p in result.report.tracks[0].comparisons
    )


def test_occlusion_and_low_visibility_are_excluded_not_switches():
    vis = [[1.0] * 3 for _ in range(9)]
    vis[4] = [0.01] * 3
    result = segment_tracks(make_batch(angles=(0,) * 9, occluded=(3,), visibility=vis))
    assert {e.reason for e in result.report.tracks[0].exclusions} == {
        "occluded_sample",
        "insufficient_reliable_visible_parts",
    }
    assert any(p.status == "unknown" for p in proposals(result))
    assert not any(f.kind == "suspected_identity_switch" for f in findings(result))


def test_observation_interruption_can_isolate_stable_sides_without_asserting_different_people():
    batch = make_batch(
        angles=(0,) * 8, sample_frames=[0, 5, 10, 15, 30, 35, 40, 45], missing_frames=range(16, 30)
    )
    result = segment_tracks(batch)
    assert len([p for p in proposals(result) if p.status == "supported"]) == 2
    assert any(f.kind == "observation_interruption" for f in findings(result))
    assert not any(f.kind == "suspected_identity_switch" for f in findings(result))


def test_sparse_sampling_and_no_features_remain_unknown():
    sparse = segment_tracks(make_batch(sample_frames=[0, 20, 40, 60, 80, 100]))
    missing = segment_tracks(make_batch(features=False))
    assert all(p.status == "unknown" for p in proposals(sparse))
    assert all(p.status == "unknown" for p in proposals(missing))
    assert sparse.report.summary["insufficient_evidence_duration_fraction"] == 1.0
    assert missing.report.summary["exclusion_counts"]["missing_feature"] == 6


def test_duplicate_boxes_are_not_cannot_link_and_carry_downstream_risk():
    result = segment_tracks(make_batch(extra_tracks=1))
    assert result.report.duplicate_candidates
    assert all(not d["cannot_link"] for d in result.report.duplicate_candidates)
    assert all(any(r.kind == "duplicate_box" for r in t.risks) for t in result.report.tracks)
    assert not any(f.kind == "suspected_identity_switch" for f in findings(result))


def test_repeated_crops_do_not_add_votes_and_descriptor_disagreement_is_not_dropped():
    batch = make_batch()
    duplicate = batch.identity_evidence.samples[0].model_copy(
        update={"sample_id": "duplicate"}, deep=True
    )
    batch.identity_evidence.samples.append(duplicate)
    result = segment_tracks(batch)
    assert result.report.tracks[0].exclusions[0].reason == "duplicate_evidence"
    assert len(proposals(result)[0].support_sample_ids) == 6
    duplicate.features[0].values = [-1.0, 0.0] * 3
    result = segment_tracks(batch)
    assert any(f.kind == "duplicate_descriptor_conflict" for f in findings(result))
    assert any(p.status == "unknown" and p.segment.start_s == 0 for p in proposals(result))


def test_camera_motion_with_appearance_change_is_not_a_switch():
    result = segment_tracks(
        make_batch(angles=(0, 0, 0, 2, 2, 2), jump_frame=13, extra_tracks=2, camera=True)
    )
    assert any(f.kind == "possible_camera_motion" for f in findings(result))
    assert not any(f.kind == "suspected_identity_switch" for f in findings(result))


def test_missing_actual_observations_cannot_support_generic_input():
    batch = make_batch()
    batch.identity_evidence.observations = batch.identity_evidence.observations[:1]
    result = segment_tracks(batch)
    assert all(p.status == "unknown" for p in proposals(result))
    assert result.report.summary["exclusion_counts"]["missing_actual_observation"] == 5


def test_proposals_have_total_disjoint_coverage_stable_ids_and_replay_contract():
    batch = make_batch(angles=(0, 0, 0, 2, 2, 2), jump_frame=13)
    result = segment_tracks(batch)
    track = result.report.tracks[0]
    ranges = [p.segment for p in track.proposals]
    assert ranges[0].start_s == track.parent_segment.start_s
    assert ranges[-1].end_s == track.parent_segment.end_s
    assert all(a.end_s == b.start_s for a, b in zip(ranges, ranges[1:]))
    reordered = batch.model_copy(deep=True)
    reordered.batch_id = "another-batch"
    reordered.identity_evidence.samples.reverse()
    reordered.identity_evidence.observations.reverse()
    assert [p.segment.segment_id for p in proposals(segment_tracks(reordered))] == [
        p.segment.segment_id for p in proposals(result)
    ]
    assert EvidenceBatch.model_validate_json(result.batch.model_dump_json()) == result.batch
    with pytest.raises(ValueError, match="raw active segments"):
        segment_tracks(result.batch)


def test_upstream_risks_are_retained_but_never_create_algorithm_boundaries():
    batch = make_batch()
    before = segment_tracks(batch)
    risk = IdentityRisk(
        risk_id="unknown-quality", kind="quality_review", match_id="m", segment_id="parent-0"
    )
    batch.identity_evidence.risks.append(risk)
    result = segment_tracks(batch)
    assert [(p.status, p.segment.start_s, p.segment.end_s) for p in proposals(before)] == [
        (p.status, p.segment.start_s, p.segment.end_s) for p in proposals(result)
    ]
    assert any(r.kind == "quality_review" for r in result.report.tracks[0].risks)


def test_policy_gates_are_validated_and_can_be_frozen_without_identity_calibration():
    with pytest.raises(ValueError, match="near_distance"):
        SegmentationPolicy(near_distance=0.6, far_distance=0.4)
    result = segment_tracks(make_batch(), policy=SegmentationPolicy())
    assert result.report.policy_fingerprint
    assert (
        result.report.summary["threshold_calibration"]
        == "engineering_gates_not_identity_calibrated"
    )


def test_risk_attached_only_to_sample_is_not_lost_on_child_segments():
    from app.contracts.identity import MatchingPolicy
    from app.identity.prepare import prepare_evidence

    batch = make_batch()
    batch.identity_evidence.samples[0].tracking_quality.risks = [
        IdentityRisk(
            risk_id="sample-only-conflict",
            kind="identity_conflict",
            match_id="m",
            segment_id="parent-0",
            sample_id=batch.identity_evidence.samples[0].sample_id,
        )
    ]
    result = segment_tracks(batch)
    prepared = prepare_evidence(result.batch, MatchingPolicy(feature_space_id="synthetic-space"))
    active = [s for s in prepared.segments if s.segment.active]
    assert active and all(any(r.kind == "identity_conflict" for r in s.risks) for s in active)
    assert any(r.kind == "identity_conflict" for r in result.report.tracks[0].risks)


def test_transitive_duplicate_bridge_does_not_inflate_independent_bucket_count():
    batch = make_batch(angles=(0, 0, 0, 0), sample_frames=[0, 5, 10, 11])
    a, b, c, bridge = batch.identity_evidence.samples
    a.image_asset_id = "image-a"
    b.image_asset_id = "image-b"
    c.image_asset_id = "image-c"
    bridge.source = a.source.model_copy(deep=True)
    bridge.image_asset_id = b.image_asset_id
    result = segment_tracks(batch)
    assert all(p.status == "unknown" for p in proposals(result))
    assert result.report.summary["exclusion_counts"]["duplicate_evidence"] == 2


def test_complementary_visibility_without_enough_common_parts_stays_unknown():
    visibility = [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]] * 2
    result = segment_tracks(make_batch(visibility=visibility))
    assert all(p.status == "unknown" for p in proposals(result))
    assert any(
        not pair.comparable and pair.reason == "insufficient_reliable_common_parts"
        for pair in result.report.tracks[0].comparisons
    )


def test_motion_break_without_appearance_separation_never_asserts_switch():
    result = segment_tracks(make_batch(jump_frame=13))
    assert len([p for p in proposals(result) if p.status == "supported"]) == 2
    assert not any(f.kind == "suspected_identity_switch" for f in findings(result))


def local_policy():
    return SegmentationPolicy(version="track-consistency-v2", local_support_windows=True)


def test_local_windows_retain_stable_view_runs_without_inventing_a_switch():
    batch = make_batch(angles=(0, 0, 0, 2, 2, 2))
    legacy = segment_tracks(batch)
    assert all(p.status == "unknown" for p in proposals(legacy))
    result = segment_tracks(batch, policy=local_policy())
    supported = [p for p in proposals(result) if p.status == "supported"]
    assert len(supported) == 2
    assert [p.support_sample_ids for p in supported] == [
        ["s-0-0", "s-0-5", "s-0-10"],
        ["s-0-15", "s-0-20", "s-0-25"],
    ]
    assert any(p.status == "unknown" for p in proposals(result))
    assert not any(f.kind == "suspected_identity_switch" for f in findings(result))
    windows = [f for f in findings(result) if f.kind == "local_appearance_support_window"]
    assert len(windows) == 2
    assert all(not f.details["switch_inferred"] for f in windows)
    assert all(not f.details["identity_boundary_known"] for f in windows)
    assert all(not p.identity_verified for p in supported)
    assert all(p.source.producer == "track_consistency_v2" for p in proposals(result))


def test_local_windows_never_skip_a_foreign_minority_inside_a_supported_span():
    batch = make_batch(angles=(0,) * 12)
    foreign = batch.identity_evidence.samples[5].model_copy(
        update={
            "sample_id": "foreign-local",
            "time_s": 2.6,
            "frame_idx": 26,
            "source": source("foreign-local"),
            "features": [
                InlineFeature(
                    feature_space_id="synthetic-space",
                    values=[-1.0, 0.0] * 3,
                    visibility=[1.0] * 3,
                )
            ],
        },
        deep=True,
    )
    batch.identity_evidence.samples.append(foreign)
    result = segment_tracks(batch, policy=local_policy())
    supported = [p for p in proposals(result) if p.status == "supported"]
    assert len(supported) == 2
    assert all(not (p.segment.start_s <= 2.6 < p.segment.end_s) for p in supported)
    assert any(
        p.status == "unknown" and p.segment.start_s <= 2.6 < p.segment.end_s
        for p in proposals(result)
    )
    assert all("foreign-local" not in p.support_sample_ids for p in supported)
    assert not any(f.kind == "suspected_identity_switch" for f in findings(result))


def test_local_window_slides_an_unsupported_prefix_without_absorbing_it():
    result = segment_tracks(make_batch(angles=(2, 2, 0, 0, 0, 0, 0)), policy=local_policy())
    supported = [p for p in proposals(result) if p.status == "supported"]
    assert len(supported) == 1
    assert supported[0].segment.start_s == 1.0
    assert supported[0].support_sample_ids == [f"s-0-{frame}" for frame in (10, 15, 20, 25, 30)]
    assert proposals(result)[0].status == "unknown"
    assert proposals(result)[0].segment.start_s == 0.0


def test_local_visibility_runs_keep_their_uncomparable_transition_unknown():
    visibility = [[1.0, 1.0, 0.0]] * 3 + [[0.0, 1.0, 1.0]] * 3
    result = segment_tracks(make_batch(visibility=visibility), policy=local_policy())
    assert len([p for p in proposals(result) if p.status == "supported"]) == 2
    assert any(p.status == "unknown" for p in proposals(result))
    assert any(f.kind == "appearance_or_visibility_unresolved" for f in findings(result))
    assert not any(f.kind == "suspected_identity_switch" for f in findings(result))


def test_local_windows_preserve_observation_holes_and_known_risks():
    batch = make_batch(
        angles=(0, 0, 0, 2, 2, 2, 0, 0, 0),
        sample_frames=[0, 5, 10, 15, 20, 25, 40, 45, 50],
        missing_frames=range(26, 40),
    )
    batch.identity_evidence.risks.append(
        IdentityRisk(
            risk_id="retain-mixed",
            kind="mixed_identity",
            match_id="m",
            segment_id="parent-0",
        )
    )
    result = segment_tracks(batch, policy=local_policy())
    supported = [p for p in proposals(result) if p.status == "supported"]
    assert len(supported) == 3
    assert all(not (p.segment.start_s < 4.0 and p.segment.end_s > 2.6) for p in supported)
    assert any(
        p.status == "unknown" and p.segment.start_s <= 2.6 and p.segment.end_s >= 4.0
        for p in proposals(result)
    )
    for p in supported:
        assert any(
            r.kind == "mixed_identity" and r.segment_id == p.segment.segment_id
            for r in result.report.tracks[0].risks
        )


def test_local_windows_partition_all_extents_and_are_stable_under_reordering():
    batch = make_batch(angles=(0, 0, 0, 2, 2, 2, 0, 0, 0))
    result = segment_tracks(batch, policy=local_policy())
    ps = proposals(result)
    assert ps[0].segment.start_s == 0.0
    assert ps[-1].segment.end_s == batch.identity_evidence.segments[0].end_s
    assert all(a.segment.end_s == b.segment.start_s for a, b in zip(ps, ps[1:]))
    assert len([p for p in ps if p.status == "supported"]) == 3
    shuffled = batch.model_copy(deep=True)
    shuffled.batch_id = "another-handoff"
    shuffled.identity_evidence.samples.reverse()
    shuffled.identity_evidence.observations.reverse()
    replay = segment_tracks(shuffled, policy=local_policy())
    assert [p.model_dump(mode="json") for p in ps] == [
        p.model_dump(mode="json") for p in proposals(replay)
    ]


def test_track_consistency_v3_viewpoint_transition_zone():
    policy = SegmentationPolicy(version="track-consistency-v3")
    # Frontal (0,0,0) -> turning (0.38) -> back (0.8, 0.8, 0.8), with >= 3 buckets per window
    frames = [0, 13, 26, 39, 52, 65, 78]
    batch = make_batch(angles=(0, 0, 0, 0.38, 0.8, 0.8, 0.8), sample_frames=frames)
    result = segment_tracks(batch, policy=policy)
    ps = proposals(result)
    supported = [p for p in ps if p.status == "supported"]
    assert len(supported) == 2
    # Verify transition zone proposal
    transition_proposals = [
        p for p in ps if p.reasons and "viewpoint_transition_uncertainty" in p.reasons
    ]
    assert len(transition_proposals) >= 1
    assert "s-0-39" in transition_proposals[0].support_sample_ids
    # Check findings
    findings = result.report.tracks[0].findings
    transition_findings = [f for f in findings if f.kind == "viewpoint_transition_zone"]
    assert len(transition_findings) >= 1
    assert transition_findings[0].details["switch_inferred"] is False
    assert transition_findings[0].details["identity_boundary_known"] is False
    assert transition_findings[0].details["transition_type"] == "continuous_perspective_shift"


def test_track_consistency_v3_scopes_risks_to_breaks():
    policy = SegmentationPolicy(version="track-consistency-v3")
    batch = make_batch(angles=(0, 0, 0, 0, 0, 0))
    # Inject a parent-level quality review
    parent = batch.identity_evidence.segments[0]
    batch.identity_evidence.risks.append(
        IdentityRisk(
            risk_id="parent_motion_risk",
            kind="quality_review",
            match_id=parent.match_id,
            clip_id=parent.clip_id,
            track_id=parent.track_id,
            segment_id=parent.segment_id,
            source=source("risk"),
        )
    )
    result = segment_tracks(batch, policy=policy)
    supported = [p for p in proposals(result) if p.status == "supported"]
    assert len(supported) == 1
    supp_seg_id = supported[0].segment.segment_id
    # Clean supported segment does not inherit parent's coarse quality_review risk
    supp_risks = [r for r in result.report.tracks[0].risks if r.segment_id == supp_seg_id]
    assert not any(
        r.kind == "quality_review" and r.risk_id.startswith("riskc_parent_motion_risk")
        for r in supp_risks
    )
