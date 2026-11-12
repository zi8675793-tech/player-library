"""Identity facts generated here, never supplied to discovery or read from the handoff."""

import hashlib
import math
from copy import deepcopy

import numpy as np
import pytest

from app.contracts.common import Provenance
from app.contracts.discovery import DiscoveryPolicy
from app.contracts.evidence import (
    AppearanceEvidence,
    Clip,
    EvidenceBatch,
    FeatureSpace,
    IdentityEvidence,
    InlineFeature,
    MediaAsset,
    Preprocessing,
    Sample,
    SourceFile,
    Track,
    TrackObservation,
    VisibilityContract,
)
from app.contracts.identity import Segment
from app.identity.consistency import segment_tracks
from app.identity.discovery import discover_identities
from app.identity.prepare import evidence_keys


def provenance(key):
    return Provenance(
        producer="synthetic-discovery",
        record_type="observation",
        source_record_id=key,
        evidence_version="1",
    )


def synthetic_facts(specifications):
    """Each (clip, track, vector, image_family) has six samples, thirty observations.

    image_family controls exact content duplicates independently of appearance.
    Truth lives in each test's assertions, never in jerseys, source keys or features.
    """
    space = FeatureSpace(
        feature_space_id="synthetic-discovery-space",
        backend="kpr",
        model_hash="a" * 64,
        preprocessing=Preprocessing(version="synthetic", color_space="rgb"),
        layout="parts",
        shape=[3, len(specifications[0][2])],
        normalization="l2",
        distance="euclidean_sqrt_visibility_mean_div2",
        visibility=VisibilityContract(shape=[3], semantics="visible"),
    )
    evidence = IdentityEvidence(feature_spaces=[space])
    assets = []
    clip_ids = sorted({item[0] for item in specifications})
    for clip_id, track_id, vector, family in specifications:
        segment_id = f"raw-{clip_id}-{track_id}"
        end = 2.6
        evidence.segments.append(
            Segment(
                segment_id=segment_id,
                match_id="synthetic-match",
                clip_id=clip_id,
                track_id=track_id,
                start_s=0.0,
                end_s=end,
                frame_start=0,
                frame_end_exclusive=26,
            )
        )
        evidence.tracks.append(
            Track(
                match_id="synthetic-match",
                clip_id=clip_id,
                track_id=track_id,
                start_s=0.0,
                end_s=end,
                frame_start=0,
                frame_end_exclusive=26,
                observed_frame_indices=list(range(26)),
                source=provenance(segment_id),
            )
        )
        # Tracks in a shared clip are geometrically separated, never duplicate boxes.
        offset = 200.0 * int(track_id)
        for frame in range(26):
            key = f"obs-{clip_id}-{track_id}-{frame}"
            evidence.observations.append(
                TrackObservation(
                    observation_id=key,
                    match_id="synthetic-match",
                    clip_id=clip_id,
                    track_id=track_id,
                    time_s=frame / 10.0,
                    frame_idx=frame,
                    bbox_xyxy=(10.0 + offset, 10.0, 60.0 + offset, 110.0),
                    source=provenance(key),
                )
            )
        for frame in range(0, 26, 5):
            key = f"sample-{clip_id}-{track_id}-{frame}"
            image_hash = hashlib.sha256(f"{family}-{frame}".encode()).hexdigest()
            asset_id = f"asset-{key}"
            assets.append(
                MediaAsset(
                    asset_id=asset_id,
                    media_type="image",
                    file=SourceFile(
                        file_id=f"file-{key}", path=f"synthetic/{key}.png", sha256=image_hash
                    ),
                )
            )
            evidence.samples.append(
                Sample(
                    sample_id=key,
                    match_id="synthetic-match",
                    clip_id=clip_id,
                    track_id=track_id,
                    segment_id=segment_id,
                    time_s=frame / 10.0,
                    frame_idx=frame,
                    source=provenance(key),
                    image_asset_id=asset_id,
                    # Shared team cues and absent number answers cannot separate identity.
                    appearance=AppearanceEvidence(team_id="same-team"),
                    features=[
                        InlineFeature(
                            feature_space_id=space.feature_space_id,
                            values=list(vector) * 3,
                            visibility=[1.0] * 3,
                        )
                    ],
                )
            )
    return EvidenceBatch(
        match_id="synthetic-match",
        batch_id="first",
        producer="synthetic-discovery",
        clips=[
            Clip(
                match_id="synthetic-match",
                clip_id=c,
                frame_count=26,
                duration_s=2.6,
                frame_timing="cfr",
                fps=10.0,
            )
            for c in clip_ids
        ],
        assets=assets,
        identity_evidence=evidence,
    )


def discover(batch):
    segmented = segment_tracks(batch)
    return segmented, discover_identities(segmented.batch, segmentation_report=segmented.report)


def clip_groups(segmented, report):
    segments = {s.segment_id: s for s in segmented.batch.identity_evidence.segments}
    return {frozenset(segments[sid].clip_id for sid in g.member_segment_ids) for g in report.groups}


def angle_vector(angle):
    return [float(math.cos(angle)), float(math.sin(angle))]


@pytest.mark.parametrize("people", [1, 2, 4])
def test_unknown_count_is_determined_by_cross_clip_evidence(people):
    specs = []
    for person in range(people):
        vector = np.eye(people + 1)[person].tolist()
        for occurrence in range(3):
            clip = f"clip-{person}-{occurrence}"
            specs.append((clip, "0", vector, clip))
    segmented, report = discover(synthetic_facts(specs))
    assert len(report.groups) == people
    assert clip_groups(segmented, report) == {
        frozenset(f"clip-{person}-{occurrence}" for occurrence in range(3))
        for person in range(people)
    }
    assert not report.unknown
    assert all(len(group.support_pair_ids) == 3 for group in report.groups)
    assert all(len(group.templates) >= 2 for group in report.groups)


def test_a_single_appearance_is_unknown_and_is_not_a_new_player_per_track():
    batch = synthetic_facts(
        [
            ("one-a", "0", [1.0, 0.0, 0.0], "one-a"),
            ("one-b", "0", [1.0, 0.0, 0.0], "one-b"),
            ("singleton", "0", [0.0, 1.0, 0.0], "singleton"),
        ]
    )
    segmented, report = discover(batch)
    assert clip_groups(segmented, report) == {frozenset({"one-a", "one-b"})}
    singleton = next(
        s
        for s in segmented.batch.identity_evidence.segments
        if s.active and s.clip_id == "singleton"
    )
    assert singleton.segment_id in {u.segment_id for u in report.unknown}
    assert len(report.groups) == 1


def test_same_team_similar_people_and_contradictory_links_cannot_be_merged():
    specs = [
        (clip, str(person), angle_vector(person * 0.08), f"{clip}-{person}")
        for clip in ("first", "second")
        for person in range(2)
    ]
    _, report = discover(synthetic_facts(specs))
    # Appearance alone cannot identify the two co-occurring people across clips.
    assert not report.groups
    assert len(report.unknown) == 4
    assert any(pair.status == "conflict" for pair in report.pairs)


def test_one_bridge_cannot_chain_two_incompatible_identities():
    specs = [
        (clip, "0", angle_vector(angle), clip)
        for clip, angle in [("a1", 0.0), ("a2", 0.0), ("bridge", 0.32), ("c1", 0.64), ("c2", 0.64)]
    ]
    segmented, report = discover(synthetic_facts(specs))
    groups = clip_groups(segmented, report)
    assert all(
        not (any(c.startswith("a") for c in group) and any(c.startswith("c") for c in group))
        for group in groups
    )
    bridge = next(
        s for s in segmented.batch.identity_evidence.segments if s.active and s.clip_id == "bridge"
    )
    assert bridge.segment_id in {u.segment_id for u in report.unknown}
    assert groups == {frozenset({"a1", "a2"}), frozenset({"c1", "c2"})}


def test_repeated_cross_clip_image_sequence_is_not_independent_identity_support():
    _, report = discover(
        synthetic_facts(
            [
                ("copy-a", "0", [1.0, 0.0], "same-images"),
                ("copy-b", "0", [1.0, 0.0], "same-images"),
            ]
        )
    )
    assert not report.groups
    assert len(report.unknown) == 2


def test_all_unknown_is_reported_without_invented_groups_or_templates():
    _, report = discover(
        synthetic_facts(
            [
                ("only-a", "0", [1.0, 0.0], "only-a"),
                ("only-b", "0", [0.0, 1.0], "only-b"),
            ]
        )
    )
    assert not report.groups
    assert len(report.unknown) == 2
    assert all(u.reasons for u in report.unknown)


def test_reordering_and_rebatching_preserve_groups_and_support_ids():
    batch = synthetic_facts(
        [
            ("a", "0", [1.0, 0.0], "a"),
            ("b", "0", [1.0, 0.0], "b"),
            ("c", "0", [0.0, 1.0], "c"),
            ("d", "0", [0.0, 1.0], "d"),
        ]
    )
    _, report = discover(batch)
    reordered = deepcopy(batch)
    reordered.batch_id = "unrelated-handoff"
    for field in ("segments", "tracks", "samples", "observations"):
        getattr(reordered.identity_evidence, field).reverse()
    reordered.assets.reverse()
    reordered.clips.reverse()
    _, replay = discover(reordered)
    assert [g.model_dump(mode="json") for g in report.groups] == [
        g.model_dump(mode="json") for g in replay.groups
    ]
    assert [p.pair_id for p in report.pairs] == [p.pair_id for p in replay.pairs]


def test_templates_use_disjoint_other_segment_witnesses_and_preserve_machine_semantics():
    segmented, report = discover(
        synthetic_facts(
            [
                ("a", "0", [1.0, 0.0], "a"),
                ("b", "0", [1.0, 0.0], "b"),
            ]
        )
    )
    assert len(report.groups) == 1
    samples = {s.sample_id: s for s in segmented.batch.identity_evidence.samples}
    for template in report.groups[0].templates:
        origin = samples[template.sample_id]
        assert template.witness_sample_ids
        for witness_id in template.witness_sample_ids:
            witness = samples[witness_id]
            assert witness.segment_id != origin.segment_id
            assert not evidence_keys(origin, segmented.batch.assets) & evidence_keys(
                witness, segmented.batch.assets
            )


def test_known_mixed_risk_cannot_be_overridden_by_consistent_features():
    from app.contracts.identity import IdentityRisk

    batch = synthetic_facts(
        [
            ("a", "0", [1.0, 0.0], "a"),
            ("b", "0", [1.0, 0.0], "b"),
        ]
    )
    batch.identity_evidence.risks.append(
        IdentityRisk(
            risk_id="known-mixed",
            kind="mixed_identity",
            match_id=batch.match_id,
            segment_id="raw-a-0",
        )
    )
    _, report = discover(batch)
    assert not report.groups
    assert len(report.unknown) == 2


def test_one_reliable_minority_sample_blocks_a_whole_segment_assignment():
    batch = synthetic_facts(
        [
            ("a", "0", [1.0, 0.0], "a"),
            ("b", "0", [1.0, 0.0], "b"),
        ]
    )
    foreign = next(s for s in batch.identity_evidence.samples if s.clip_id == "b")
    foreign.features[0].values = [-1.0, 0.0] * 3
    segmented, report = discover(batch)
    assert not report.groups
    assert any(
        p.status == "unknown"
        for track in segmented.report.tracks
        for p in track.proposals
        if p.segment.clip_id == "b"
    )


def test_uncalibrated_discovery_is_explicitly_experimental():
    from pydantic import ValidationError

    from app.contracts.discovery import DiscoveryPolicy

    _, report = discover(
        synthetic_facts(
            [
                ("a", "0", [1.0, 0.0], "a"),
                ("b", "0", [1.0, 0.0], "b"),
            ]
        )
    )
    assert report.status == "experimental"
    assert report.policy.calibration == "unvalidated"
    assert report.groups
    assert report.policy_fingerprint and report.input_fingerprint
    with pytest.raises(ValidationError):
        DiscoveryPolicy(calibration="validated")


def test_stale_segmentation_report_cannot_prove_missing_observations():
    batch = synthetic_facts(
        [
            ("a", "0", [1.0, 0.0], "a"),
            ("b", "0", [1.0, 0.0], "b"),
        ]
    )
    segmented = segment_tracks(batch)
    stale = segmented.batch.model_copy(deep=True)
    stale.identity_evidence.observations = []
    with pytest.raises(ValueError, match="[Ss]egmentation|observation|report"):
        discover_identities(stale, segmentation_report=segmented.report)


def test_discovery_policy_decoupled_parameters():
    # Test decoupled fields can be independently set
    policy = DiscoveryPolicy(
        intra_segment_near_distance=0.15,
        inter_segment_tau=0.20,
        candidate_far_distance=0.40,
        negative_evidence_distance=0.50,
        contrastive_margin=0.05,
        counter_evidence_margin=0.03,
        rival_margin=0.06,
    )
    assert policy.intra_segment_near_distance == 0.15
    assert policy.inter_segment_tau == 0.20
    assert policy.candidate_far_distance == 0.40
    assert policy.negative_evidence_distance == 0.50
    assert policy.contrastive_margin == 0.05
    assert policy.counter_evidence_margin == 0.03
    assert policy.rival_margin == 0.06

    # Test legacy fields sync to decoupled fields
    legacy = DiscoveryPolicy(tau=0.22, far_distance=0.42, margin=0.08)
    assert legacy.intra_segment_near_distance == 0.22
    assert legacy.inter_segment_tau == 0.22
    assert legacy.candidate_far_distance == 0.42
    assert legacy.negative_evidence_distance == 0.42
    assert legacy.contrastive_margin == 0.08
    assert legacy.counter_evidence_margin == 0.08
    assert legacy.rival_margin == 0.08


def test_discovery_v3_geometric_negative_compensation():
    # Two clips, each with only 1 player (no co-occurring negative in either clip)
    specs = [
        ("clip_a", "0", [1.0, 0.0], "a0"),
        ("clip_b", "0", [1.0, 0.0], "b0"),
    ]
    batch = synthetic_facts(specs)
    segmented = segment_tracks(batch)
    policy_v3 = DiscoveryPolicy(
        version="complete_support_v3",
        compensation_min_common_parts=2,
    )
    report = discover_identities(
        segmented.batch,
        segmentation_report=segmented.report,
        policy=policy_v3,
    )
    # Without compensation, v2 would give 0 groups because negatives are missing.
    # With v3 compensation, the ultra-tight matching pairs form 1 group.
    assert len(report.groups) == 1
    assert len(report.unknown) == 0
    pair = report.pairs[0]
    assert pair.status == "supported"
    assert "compensated_contrastive_support_lacking_geometric_negatives" in pair.reasons


def test_discovery_v3_compensation_rejected_when_distance_exceeds_threshold():
    # Distance ~ 0.25 > 0.22 (angle 0.52 rad -> distance ~ 0.25)
    def angle_vec(theta):
        return [math.cos(theta), math.sin(theta)]

    specs = [
        ("clip_a", "0", angle_vec(0.0), "a0"),
        ("clip_b", "0", angle_vec(0.52), "b0"),
    ]
    batch = synthetic_facts(specs)
    segmented = segment_tracks(batch)
    policy_v3 = DiscoveryPolicy(
        version="complete_support_v3",
        compensation_min_common_parts=2,
        compensation_max_distance=0.22,
    )
    report = discover_identities(
        segmented.batch,
        segmentation_report=segmented.report,
        policy=policy_v3,
    )
    # Exceeds compensation anchor threshold -> rejected
    assert len(report.groups) == 0
    assert len(report.unknown) == 2
