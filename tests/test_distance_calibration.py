"""Comprehensive business tests for feature-space distance calibration profiles and candidate evidence."""

import pytest

from app.contracts.calibration import (
    DistanceCalibrationProfile,
)
from app.contracts.common import Provenance
from app.contracts.discovery import DiscoveryPolicy
from app.contracts.evidence import (
    EvidenceBatch,
    FeatureSpace,
    IdentityEvidence,
    InlineFeature,
    MediaAsset,
    Sample,
    SourceFile,
    TrackingQuality,
    TrackObservation,
)
from app.contracts.identity import MatchingPolicy
from app.contracts.segmentation import SegmentationPolicy, SegmentationSpacePolicy
from app.identity.calibration import (
    extract_candidate_calibration_evidence,
    load_calibration_profile,
    save_calibration_profile,
)
from app.identity.consistency import _Distances
from app.identity.distance import (
    FeatureValidationError,
    classify_distance_relation,
    evaluate_feature_relation,
    validate_calibration_profile_space,
)
from app.settings import load_settings


def make_space(
    feature_space_id: str = "fs_kpr_test", parts: int = 3, dimension: int = 2
) -> FeatureSpace:
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


def make_feature(
    feature_space_id: str = "fs_kpr_test",
    parts: int = 3,
    dimension: int = 2,
    values: list[float] | None = None,
    visibility: list[float] | None = None,
) -> InlineFeature:
    vals = values if values is not None else [1.0, *([0.0] * (dimension - 1))] * parts
    vis = visibility if visibility is not None else [1.0] * parts
    return InlineFeature(feature_space_id=feature_space_id, values=vals, visibility=vis)


def make_sample(
    sample_id: str,
    clip_id: str = "c1",
    track_id: str = "t1",
    time_s: float = 0.0,
    frame_idx: int = 0,
    asset_id: str = "asset_1",
    quality: TrackingQuality | None = None,
    feature_val: InlineFeature | None = None,
) -> Sample:
    return Sample(
        match_id="m1",
        sample_id=sample_id,
        clip_id=clip_id,
        track_id=track_id,
        segment_id="s1",
        time_s=time_s,
        frame_idx=frame_idx,
        image_asset_id=asset_id,
        source=Provenance(
            producer="test",
            record_type="sample",
            source_record_id=sample_id,
            evidence_version="1",
        ),
        tracking_quality=quality,
        features=[feature_val or make_feature()],
    )


def make_obs(
    clip_id: str = "c1",
    track_id: str = "t1",
    frame_idx: int = 0,
    time_s: float = 0.0,
    bbox_xyxy: tuple[float, float, float, float] = (100.0, 100.0, 200.0, 300.0),
    det_score: float = 0.95,
) -> TrackObservation:
    return TrackObservation(
        observation_id=f"obs_{clip_id}_{track_id}_{frame_idx}",
        match_id="m1",
        clip_id=clip_id,
        track_id=track_id,
        time_s=time_s,
        frame_idx=frame_idx,
        bbox_xyxy=bbox_xyxy,
        detection_score=det_score,
        source=Provenance(
            producer="test",
            record_type="obs",
            source_record_id=f"{clip_id}:{track_id}:{frame_idx}",
            evidence_version="1",
        ),
    )


def test_classify_distance_relation_semantics():
    near, far = 0.18, 0.45
    assert classify_distance_relation(None, near, far) == "uncomparable"
    assert classify_distance_relation(0.10, near, far) == "near"
    assert classify_distance_relation(0.18, near, far) == "near"
    assert classify_distance_relation(0.25, near, far) == "ambiguous"
    assert classify_distance_relation(0.35, near, far) == "ambiguous"
    assert classify_distance_relation(0.45, near, far) == "far"
    assert classify_distance_relation(0.60, near, far) == "far"


def test_evaluate_feature_relation_with_profile():
    space = make_space("fs_test")
    policy = MatchingPolicy(feature_space_id="fs_test")
    fa = make_feature("fs_test")
    fb = make_feature("fs_test")

    profile = DistanceCalibrationProfile(
        profile_id="prof_1",
        feature_space_id="fs_test",
        calibration_version="v1",
        near_distance=0.15,
        far_distance=0.40,
    )

    rel, res = evaluate_feature_relation(space, fa, fb, policy, profile)
    assert rel == "near"
    assert res.comparable is True
    assert res.distance == 0.0

    # Incompatible feature space with profile
    other_space = make_space("fs_other")
    other_policy = MatchingPolicy(feature_space_id="fs_other")
    fo = make_feature("fs_other")
    with pytest.raises(FeatureValidationError, match="does not match space"):
        evaluate_feature_relation(other_space, fo, fo, other_policy, profile)


def test_profile_bounds_and_calibrated_validation():
    # near must be strictly less than far
    with pytest.raises(ValueError, match="must be strictly less than"):
        DistanceCalibrationProfile(
            profile_id="p1",
            feature_space_id="fs1",
            near_distance=0.45,
            far_distance=0.45,
        )

    with pytest.raises(ValueError, match="must be strictly less than"):
        DistanceCalibrationProfile(
            profile_id="p1",
            feature_space_id="fs1",
            near_distance=0.50,
            far_distance=0.30,
        )

    # Status 'calibrated' requires real evidence fingerprint and notes
    with pytest.raises(ValueError, match="calibrated status requires"):
        DistanceCalibrationProfile(
            profile_id="p1",
            feature_space_id="fs1",
            status="calibrated",
            evidence_fingerprint="0" * 64,
            notes="",
        )

    valid_calibrated = DistanceCalibrationProfile(
        profile_id="p1",
        feature_space_id="fs1",
        status="calibrated",
        evidence_fingerprint="a" * 64,
        notes="Empirically validated on holdout set",
    )
    assert valid_calibrated.status == "calibrated"


def test_space_switching_and_silent_fallback_prevention():
    space_b = make_space("fs_beta")

    prof_a = DistanceCalibrationProfile(
        profile_id="prof_alpha",
        feature_space_id="fs_alpha",
        near_distance=0.15,
        far_distance=0.40,
    )

    # Validate space mismatch against profile
    with pytest.raises(FeatureValidationError, match="feature_space_id 'fs_alpha' does not match"):
        validate_calibration_profile_space(space_b, prof_a)

    # Consistency _Distances with configured spaces rejects uncalibrated spaces without silent fallback
    seg_policy = SegmentationPolicy(
        spaces=[SegmentationSpacePolicy(profile=prof_a)],
        strict_space_calibration=True,
    )
    assert len(seg_policy.spaces) == 1

    sample_b1 = make_sample("s_b1", feature_val=make_feature("fs_beta"))
    sample_b2 = make_sample("s_b2", feature_val=make_feature("fs_beta"))
    distances = _Distances([sample_b1, sample_b2], {"fs_beta": space_b}, {}, seg_policy)

    with pytest.raises(
        FeatureValidationError, match="Missing calibration profile for feature space 'fs_beta'"
    ):
        distances.relation(sample_b1, sample_b2)


def test_discovery_policy_profile_binding_and_fingerprint_check():
    prof = DistanceCalibrationProfile(
        profile_id="prof_m",
        feature_space_id="fs_main",
        near_distance=0.20,
        far_distance=0.42,
    )

    # DiscoveryPolicy automatically syncs thresholds and space id from profile
    disc_policy = DiscoveryPolicy(calibration_profile=prof)
    assert disc_policy.feature_space_id == "fs_main"
    assert disc_policy.intra_segment_near_distance == 0.20
    assert disc_policy.far_distance == 0.42
    assert disc_policy.calibration_fingerprint == prof.fingerprint

    # Mismatched feature_space_id raises ValueError
    with pytest.raises(ValueError, match="policy feature_space_id fs_other does not match"):
        DiscoveryPolicy(
            feature_space_id="fs_other",
            calibration_profile=prof,
        )

    # Stale/incompatible fingerprint raises ValueError
    with pytest.raises(ValueError, match="calibration_fingerprint mismatch"):
        DiscoveryPolicy(
            calibration_profile=prof,
            calibration_fingerprint="f" * 64,
        )


def test_candidate_positive_and_exclusion_mechanisms(tmp_path):
    space = make_space("fs_synth", parts=2, dimension=2)
    asset_1 = MediaAsset(
        asset_id="asset_1",
        media_type="image",
        file=SourceFile(file_id="f1", path="img1.png", sha256="11" * 32, size_bytes=100),
    )
    asset_2 = MediaAsset(
        asset_id="asset_2",
        media_type="image",
        file=SourceFile(file_id="f2", path="img2.png", sha256="22" * 32, size_bytes=100),
    )
    asset_dup = MediaAsset(
        asset_id="asset_dup",
        media_type="image",
        file=SourceFile(file_id="f3", path="img3.png", sha256="11" * 32, size_bytes=100),
    )

    # Two valid continuous samples (dt = 0.2s, no jump, high quality)
    f1 = make_feature("fs_synth", parts=2, dimension=2, values=[1.0, 0.0, 1.0, 0.0])
    f2 = make_feature("fs_synth", parts=2, dimension=2, values=[0.8, 0.6, 0.8, 0.6])
    s1 = make_sample("s1", time_s=0.0, frame_idx=0, asset_id="asset_1", feature_val=f1)
    s2 = make_sample("s2", time_s=0.2, frame_idx=5, asset_id="asset_2", feature_val=f2)
    obs1 = make_obs(frame_idx=0, time_s=0.0, bbox_xyxy=(100.0, 100.0, 200.0, 300.0))
    obs2 = make_obs(frame_idx=5, time_s=0.2, bbox_xyxy=(105.0, 105.0, 205.0, 305.0))

    # Sample s3: temporal gap too large (> 1.0s) from s2
    s3 = make_sample("s3", time_s=1.8, frame_idx=45, asset_id="asset_1", feature_val=f1)
    obs3 = make_obs(frame_idx=45, time_s=1.8, bbox_xyxy=(110.0, 110.0, 210.0, 310.0))

    # Sample s4: duplicate image with s3 (asset_dup has same sha256 as asset_1)
    s4 = make_sample("s4", time_s=2.0, frame_idx=50, asset_id="asset_dup", feature_val=f2)
    obs4 = make_obs(frame_idx=50, time_s=2.0, bbox_xyxy=(112.0, 112.0, 212.0, 312.0))

    # Sample s5: spatial jump
    s5 = make_sample("s5", time_s=2.1, frame_idx=52, asset_id="asset_2", feature_val=f1)
    obs5 = make_obs(frame_idx=52, time_s=2.1, bbox_xyxy=(900.0, 900.0, 1000.0, 1100.0))

    # Sample s6: occluded tracking quality
    s6 = make_sample(
        "s6",
        time_s=2.2,
        frame_idx=55,
        asset_id="asset_1",
        quality=TrackingQuality(occluded=True, score=0.9),
        feature_val=f2,
    )
    obs6 = make_obs(frame_idx=55, time_s=2.2, bbox_xyxy=(905.0, 905.0, 1005.0, 1105.0))

    batch = EvidenceBatch(
        match_id="m1",
        batch_id="b1",
        producer="test",
        assets=[asset_1, asset_2, asset_dup],
        identity_evidence=IdentityEvidence(
            feature_spaces=[space],
            samples=[s1, s2, s3, s4, s5, s6],
            observations=[obs1, obs2, obs3, obs4, obs5, obs6],
        ),
    )

    _, pos_pairs, neg_pairs, exclusions, counts = extract_candidate_calibration_evidence(
        batch,
        {},
        max_positive_gap_s=1.0,
    )

    # Exactly 1 valid candidate positive (s1 -> s2)
    assert len(pos_pairs) == 1
    assert pos_pairs[0]["left_sample_id"] == "s1"
    assert pos_pairs[0]["right_sample_id"] == "s2"
    assert pos_pairs[0]["distance"] is not None

    # Exclusions properly caught
    assert counts.get("temporal_gap_too_large") == 1  # s2 -> s3
    assert counts.get("duplicate_image") == 1  # s3 -> s4
    assert counts.get("spatial_jump") == 1  # s4 -> s5
    assert counts.get("low_quality_or_occluded") == 1  # s5 -> s6


def test_physical_negatives_and_overlapping_box_exclusion():
    space = make_space("fs_neg", parts=2, dimension=2)
    asset_a = MediaAsset(
        asset_id="asset_a",
        media_type="image",
        file=SourceFile(file_id="fa", path="imga.png", sha256="aa" * 32, size_bytes=100),
    )
    asset_b = MediaAsset(
        asset_id="asset_b",
        media_type="image",
        file=SourceFile(file_id="fb", path="imgb.png", sha256="bb" * 32, size_bytes=100),
    )
    asset_c = MediaAsset(
        asset_id="asset_c",
        media_type="image",
        file=SourceFile(file_id="fc", path="imgc.png", sha256="cc" * 32, size_bytes=100),
    )

    f_a = make_feature("fs_neg", parts=2, dimension=2, values=[1.0, 0.0, 1.0, 0.0])
    f_b = make_feature("fs_neg", parts=2, dimension=2, values=[0.0, 1.0, 0.0, 1.0])
    f_c = make_feature("fs_neg", parts=2, dimension=2, values=[-1.0, 0.0, -1.0, 0.0])

    # Same clip, same frame (frame_idx=10, time=0.4s) across different tracks
    sa = make_sample(
        "sa",
        clip_id="c1",
        track_id="t1",
        frame_idx=10,
        time_s=0.4,
        asset_id="asset_a",
        feature_val=f_a,
    )
    sb = make_sample(
        "sb",
        clip_id="c1",
        track_id="t2",
        frame_idx=10,
        time_s=0.4,
        asset_id="asset_b",
        feature_val=f_b,
    )
    sc = make_sample(
        "sc",
        clip_id="c1",
        track_id="t3",
        frame_idx=10,
        time_s=0.4,
        asset_id="asset_c",
        feature_val=f_c,
    )

    # sa and sb are physically separated
    obsa = make_obs(
        clip_id="c1",
        track_id="t1",
        frame_idx=10,
        time_s=0.4,
        bbox_xyxy=(100.0, 100.0, 200.0, 300.0),
    )
    obsb = make_obs(
        clip_id="c1",
        track_id="t2",
        frame_idx=10,
        time_s=0.4,
        bbox_xyxy=(500.0, 100.0, 600.0, 300.0),
    )
    # sc overlaps heavily with sa (IoU > 0.8: duplicate detection box on the same person!)
    obsc = make_obs(
        clip_id="c1",
        track_id="t3",
        frame_idx=10,
        time_s=0.4,
        bbox_xyxy=(105.0, 102.0, 203.0, 298.0),
    )

    batch = EvidenceBatch(
        match_id="m1",
        batch_id="b1",
        producer="test",
        assets=[asset_a, asset_b, asset_c],
        identity_evidence=IdentityEvidence(
            feature_spaces=[space],
            samples=[sa, sb, sc],
            observations=[obsa, obsb, obsc],
        ),
    )

    _, pos_pairs, neg_pairs, exclusions, counts = extract_candidate_calibration_evidence(
        batch,
        {},
        max_negative_iou=0.2,
    )

    # sa and sc must NOT be a negative (excluded as duplicate_or_overlapping_box)
    assert counts.get("duplicate_or_overlapping_box") >= 1
    # sa and sb are a valid physical negative pair
    valid_pairs = [(p["left_sample_id"], p["right_sample_id"]) for p in neg_pairs]
    assert ("sa", "sb") in valid_pairs or ("sb", "sa") in valid_pairs
    assert ("sa", "sc") not in valid_pairs


def test_profile_serialization_and_roundtrip(tmp_path):
    profile = DistanceCalibrationProfile(
        profile_id="prof_serial_001",
        feature_space_id="fs_serial",
        calibration_version="stage1-diagnostic-v1",
        near_distance=0.18,
        far_distance=0.45,
        status="uncalibrated",
        evidence_source="test_evidence",
        evidence_fingerprint="1" * 64,
        config_fingerprint="2" * 64,
        notes="Uncalibrated engineering baseline",
    )

    out_file = tmp_path / "calibration-profile.json"
    save_calibration_profile(profile, out_file)
    assert out_file.is_file()

    loaded = load_calibration_profile(out_file)
    assert loaded.profile_id == profile.profile_id
    assert loaded.feature_space_id == profile.feature_space_id
    assert loaded.near_distance == profile.near_distance
    assert loaded.far_distance == profile.far_distance
    assert loaded.status == profile.status
    assert loaded.fingerprint == profile.fingerprint


def test_batch_policy_consistency_with_settings(tmp_path):
    prof = DistanceCalibrationProfile(
        profile_id="prof_bound",
        feature_space_id="fs_bound",
        calibration_version="stage1-v1",
        near_distance=0.17,
        far_distance=0.43,
    )
    prof_path = tmp_path / "calibration-profile.json"
    save_calibration_profile(prof, prof_path)

    toml_content = f"""
[paths]
input_root = "{tmp_path}"
runtime_root = "{tmp_path}/var"
calibration_profile = "{prof_path}"

[input]
adapter = "machine"

[match]
match_id = "test-match"

[matching]
min_common_parts = 2
max_templates_per_player = 6
auto_assign_enabled = false

[playback]
before_event_s = 2.0
after_event_s = 2.0

[server]
host = "127.0.0.1"
port = 8000
"""
    config_path = tmp_path / "test_config.toml"
    config_path.write_text(toml_content)

    settings = load_settings(config_path)
    # Verify settings automatically binds the profile consistently
    assert settings.segmentation.strict_space_calibration is True
    assert len(settings.segmentation.calibration_profiles) == 1
    assert settings.segmentation.calibration_profiles[0].feature_space_id == "fs_bound"
    assert settings.discovery.feature_space_id == "fs_bound"
    assert settings.discovery.intra_segment_near_distance == 0.17
    assert settings.discovery.far_distance == 0.43
    assert settings.discovery.calibration_fingerprint == prof.fingerprint


def test_calibration_profile_tamper_bypass_regression():
    """Assert that any conflict between policy thresholds and bound calibration profile raises ValueError."""
    prof = DistanceCalibrationProfile(
        profile_id="calib_prof_tamper",
        feature_space_id="fs_kpr_test",
        calibration_version="stage1-v1",
        near_distance=0.18,
        far_distance=0.45,
    )
    # 1. Consistent policy construction succeeds
    pol = DiscoveryPolicy(
        calibration_profile=prof,
        feature_space_id="fs_kpr_test",
    )
    assert pol.tau == 0.18
    assert pol.intra_segment_near_distance == 0.18
    assert pol.far_distance == 0.45

    # 2. Tampered near thresholds trigger ValueError
    with pytest.raises(
        ValueError, match="conflicts with attached calibration_profile.near_distance"
    ):
        DiscoveryPolicy(
            calibration_profile=prof,
            feature_space_id="fs_kpr_test",
            tau=0.40,
        )

    with pytest.raises(
        ValueError, match="conflicts with attached calibration_profile.near_distance"
    ):
        DiscoveryPolicy(
            calibration_profile=prof,
            feature_space_id="fs_kpr_test",
            intra_segment_near_distance=0.25,
        )

    with pytest.raises(
        ValueError, match="conflicts with attached calibration_profile.near_distance"
    ):
        DiscoveryPolicy(
            calibration_profile=prof,
            feature_space_id="fs_kpr_test",
            inter_segment_tau=0.25,
        )

    # 3. Tampered far thresholds trigger ValueError
    with pytest.raises(
        ValueError, match="conflicts with attached calibration_profile.far_distance"
    ):
        DiscoveryPolicy(
            calibration_profile=prof,
            feature_space_id="fs_kpr_test",
            far_distance=0.55,
        )

    with pytest.raises(
        ValueError, match="conflicts with attached calibration_profile.far_distance"
    ):
        DiscoveryPolicy(
            calibration_profile=prof,
            feature_space_id="fs_kpr_test",
            candidate_far_distance=0.50,
        )

    # 4. SegmentationSpacePolicy tampering triggers ValueError
    with pytest.raises(ValueError, match="conflicts with attached profile.near_distance"):
        SegmentationSpacePolicy(
            profile=prof,
            feature_space_id="fs_kpr_test",
            near_distance=0.25,
        )

    with pytest.raises(ValueError, match="conflicts with attached profile.far_distance"):
        SegmentationSpacePolicy(
            profile=prof,
            feature_space_id="fs_kpr_test",
            far_distance=0.55,
        )

    # 5. SegmentationPolicy spaces conflict triggers ValueError
    space_pol = SegmentationSpacePolicy(
        feature_space_id="fs_kpr_test",
        near_distance=0.28,
        far_distance=0.45,
    )
    with pytest.raises(ValueError, match="conflicts with calibration profile near_distance"):
        SegmentationPolicy(
            spaces=[space_pol],
            calibration_profiles=[prof],
        )


def test_calibration_evidence_strict_admission_regression():
    """Assert candidate positives and physical negatives 100% exclude missing-bbox, low-vis, and risk samples."""
    from app.contracts.identity import IdentityRisk, RiskKind

    space = make_space(feature_space_id="fs_strict", parts=2, dimension=2)
    prov = Provenance(
        producer="test", record_type="observation", source_record_id="rec1", evidence_version="1"
    )

    feat_norm1 = make_feature(
        "fs_strict", parts=2, dimension=2, values=[1.0, 0.0, 1.0, 0.0], visibility=[1.0, 1.0]
    )
    feat_norm2 = make_feature(
        "fs_strict", parts=2, dimension=2, values=[0.8, 0.6, 0.8, 0.6], visibility=[1.0, 1.0]
    )
    feat_low_vis = make_feature(
        "fs_strict", parts=2, dimension=2, values=[1.0, 0.0, 1.0, 0.0], visibility=[0.05, 0.05]
    )
    feat_neg_norm = make_feature(
        "fs_strict", parts=2, dimension=2, values=[0.0, 1.0, 0.0, 1.0], visibility=[1.0, 1.0]
    )

    # Normal samples on track 1 (adjacent, dt=0.5s)
    s_norm1 = make_sample(
        "s_norm1", clip_id="c1", track_id="t1", time_s=0.0, frame_idx=0, feature_val=feat_norm1
    )
    s_norm2 = make_sample(
        "s_norm2", clip_id="c1", track_id="t1", time_s=0.5, frame_idx=1, feature_val=feat_norm2
    )

    # Offending samples on track 1
    s_no_bbox = make_sample(
        "s_no_bbox", clip_id="c1", track_id="t1", time_s=1.0, frame_idx=2, feature_val=feat_norm1
    )
    s_low_vis = make_sample(
        "s_low_vis", clip_id="c1", track_id="t1", time_s=1.5, frame_idx=3, feature_val=feat_low_vis
    )
    s_mixed = make_sample(
        "s_mixed",
        clip_id="c1",
        track_id="t1",
        time_s=2.0,
        frame_idx=4,
        quality=TrackingQuality(mixed_identity=True),
        feature_val=feat_norm1,
    )
    s_conflict = make_sample(
        "s_conflict",
        clip_id="c1",
        track_id="t1",
        time_s=2.5,
        frame_idx=5,
        quality=TrackingQuality(
            risks=[
                IdentityRisk(
                    risk_id="risk_conf",
                    kind=RiskKind.IDENTITY_CONFLICT,
                    match_id="m1",
                    sample_id="s_conflict",
                )
            ]
        ),
        feature_val=feat_norm1,
    )

    # Frame 0 co-occurring tracks for physical negative evaluation
    s_neg_norm = make_sample(
        "s_neg_norm",
        clip_id="c1",
        track_id="t2",
        time_s=0.0,
        frame_idx=0,
        feature_val=feat_neg_norm,
    )
    s_neg_no_bbox = make_sample(
        "s_neg_no_bbox",
        clip_id="c1",
        track_id="t3",
        time_s=0.0,
        frame_idx=0,
        feature_val=feat_norm1,
    )
    s_neg_low_vis = make_sample(
        "s_neg_low_vis",
        clip_id="c1",
        track_id="t4",
        time_s=0.0,
        frame_idx=0,
        feature_val=feat_low_vis,
    )
    s_neg_switch = make_sample(
        "s_neg_switch",
        clip_id="c1",
        track_id="t5",
        time_s=0.0,
        frame_idx=0,
        quality=TrackingQuality(
            risks=[
                IdentityRisk(
                    risk_id="risk_sw",
                    kind=RiskKind.IDENTITY_SWITCH,
                    match_id="m1",
                    sample_id="s_neg_switch",
                )
            ]
        ),
        feature_val=feat_norm1,
    )

    all_samples = [
        s_norm1,
        s_norm2,
        s_no_bbox,
        s_low_vis,
        s_mixed,
        s_conflict,
        s_neg_norm,
        s_neg_no_bbox,
        s_neg_low_vis,
        s_neg_switch,
    ]

    # Observations: note s_no_bbox and s_neg_no_bbox DO NOT have observations!
    observations = [
        TrackObservation(
            observation_id="o1",
            match_id="m1",
            clip_id="c1",
            track_id="t1",
            frame_idx=0,
            time_s=0.0,
            bbox_xyxy=[10.0, 10.0, 20.0, 20.0],
            source=prov,
        ),
        TrackObservation(
            observation_id="o2",
            match_id="m1",
            clip_id="c1",
            track_id="t1",
            frame_idx=1,
            time_s=0.5,
            bbox_xyxy=[11.0, 11.0, 21.0, 21.0],
            source=prov,
        ),
        # frame 2 missing for s_no_bbox
        TrackObservation(
            observation_id="o4",
            match_id="m1",
            clip_id="c1",
            track_id="t1",
            frame_idx=3,
            time_s=1.5,
            bbox_xyxy=[12.0, 12.0, 22.0, 22.0],
            source=prov,
        ),
        TrackObservation(
            observation_id="o5",
            match_id="m1",
            clip_id="c1",
            track_id="t1",
            frame_idx=4,
            time_s=2.0,
            bbox_xyxy=[13.0, 13.0, 23.0, 23.0],
            source=prov,
        ),
        TrackObservation(
            observation_id="o6",
            match_id="m1",
            clip_id="c1",
            track_id="t1",
            frame_idx=5,
            time_s=2.5,
            bbox_xyxy=[14.0, 14.0, 24.0, 24.0],
            source=prov,
        ),
        # Frame 0 negative observations
        TrackObservation(
            observation_id="o_neg1",
            match_id="m1",
            clip_id="c1",
            track_id="t2",
            frame_idx=0,
            time_s=0.0,
            bbox_xyxy=[100.0, 100.0, 120.0, 120.0],
            source=prov,
        ),
        # t3 has no observation for s_neg_no_bbox
        TrackObservation(
            observation_id="o_neg3",
            match_id="m1",
            clip_id="c1",
            track_id="t4",
            frame_idx=0,
            time_s=0.0,
            bbox_xyxy=[200.0, 200.0, 220.0, 220.0],
            source=prov,
        ),
        TrackObservation(
            observation_id="o_neg4",
            match_id="m1",
            clip_id="c1",
            track_id="t5",
            frame_idx=0,
            time_s=0.0,
            bbox_xyxy=[300.0, 300.0, 320.0, 320.0],
            source=prov,
        ),
    ]

    features = {}
    for s in all_samples:
        if s.sample_id in ("s_low_vis", "s_neg_low_vis"):
            # visibility < min_part_visibility (0.2)
            features[(s.sample_id, "fs_strict")] = make_feature(
                "fs_strict", parts=2, dimension=2, visibility=[0.05, 0.05]
            )
        elif s.sample_id == "s_norm2":
            features[(s.sample_id, "fs_strict")] = make_feature(
                "fs_strict",
                parts=2,
                dimension=2,
                values=[0.8, 0.6, 0.8, 0.6],
                visibility=[1.0, 1.0],
            )
        elif s.sample_id == "s_neg_norm":
            features[(s.sample_id, "fs_strict")] = make_feature(
                "fs_strict",
                parts=2,
                dimension=2,
                values=[0.0, 1.0, 0.0, 1.0],
                visibility=[1.0, 1.0],
            )
        else:
            features[(s.sample_id, "fs_strict")] = make_feature(
                "fs_strict",
                parts=2,
                dimension=2,
                values=[1.0, 0.0, 1.0, 0.0],
                visibility=[1.0, 1.0],
            )

    assets = [
        MediaAsset(
            asset_id=f"ast_{s.sample_id}",
            media_type="image",
            file=SourceFile(
                file_id=f"f_{s.sample_id}",
                path=f"img_{s.sample_id}.png",
                sha256=f"{i:02x}" * 32,
                size_bytes=100,
            ),
        )
        for i, s in enumerate(all_samples, 1)
    ]
    for s in all_samples:
        s.image_asset_id = f"ast_{s.sample_id}"

    batch = EvidenceBatch(
        match_id="m1",
        batch_id="b1",
        producer="test",
        sources=[SourceFile(file_id="src1", path="p1", sha256="0" * 64, size_bytes=100)],
        assets=assets,
        identity_evidence=IdentityEvidence(
            tracks=[],
            observations=observations,
            segments=[],
            samples=all_samples,
            feature_spaces=[space],
        ),
    )

    _, pos, neg, exclusions, counts = extract_candidate_calibration_evidence(
        batch,
        features,
        feature_space_id="fs_strict",
        min_common_parts=2,
        min_part_visibility=0.2,
    )

    # 1. Exactly only the clean pair enters positive pairs
    assert len(pos) == 1
    assert pos[0]["left_sample_id"] == "s_norm1"
    assert pos[0]["right_sample_id"] == "s_norm2"

    # 2. Exactly only the clean pair enters physical negatives
    assert len(neg) == 1
    assert {neg[0]["left_sample_id"], neg[0]["right_sample_id"]} == {"s_norm1", "s_neg_norm"}

    # 3. Offending samples are 100% excluded with specific structured reasons
    excl_reasons = {e.reason for e in exclusions}
    assert "missing_observation_or_bbox" in excl_reasons
    assert "insufficient_visible_parts" in excl_reasons
    assert "explicit_identity_risk" in excl_reasons

    # Check that s_no_bbox, s_low_vis, s_mixed, s_conflict never appear in any positive or negative pair
    admitted_sample_ids = {p["left_sample_id"] for p in pos} | {p["right_sample_id"] for p in pos}
    admitted_sample_ids |= {n["left_sample_id"] for n in neg} | {n["right_sample_id"] for n in neg}
    for bad_id in (
        "s_no_bbox",
        "s_low_vis",
        "s_mixed",
        "s_conflict",
        "s_neg_no_bbox",
        "s_neg_low_vis",
        "s_neg_switch",
    ):
        assert bad_id not in admitted_sample_ids
