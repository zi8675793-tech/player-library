"""Comprehensive verification of open-set incremental identity discovery and gallery maintenance.

Verifies:
1. Unsupported modes raise explicit error (never silently falling back).
2. Empty library first batch with cross-clip evidence creates new identity.
3. Single appearance evidence deferred; subsequent clip supplements proof to create identity.
4. Returning player after absence matches existing player without duplicate player creation.
5. New player appearing later creates new player while preserving existing players.
6. Ambiguous candidate competing between existing players remains unassigned.
7. Historical deferred segments reprocessed when new evidence arrives.
8. Existing players are never automatically merged.
9. Human review and manual constraints take precedence and are not overwritten.
10. Variable batch size does not turn batch_id into an identity boundary.
11. Historical segment without supported proposal cannot be restored and safely defers.
12. Duplicate batch idempotency (changed=False, revision unchanged).
13. Mixed-identity segment blocked (remains unassigned).
14. Model space incompatibility blocked from matching.
15. Erroneous template revocation on assignment rejection.
16. Atomic transaction failure rollback.
17. Dynamic event resolution tracking player mapping.
"""

import json
from collections import defaultdict
from unittest.mock import patch

import pytest
from test_identity_discovery import angle_vector, synthetic_facts
from test_machine_input import machine_settings
from test_track_consistency import make_batch

from app.contracts.api import (
    ReviewRequest,
    SetAssignmentCommand,
    UpdateRequest,
)
from app.contracts.common import Provenance
from app.contracts.discovery import DiscoveryPolicy
from app.contracts.events import Event, PointEventTime, Subject, TrackReference
from app.contracts.evidence import (
    EvidenceBatch,
    FeatureSpace,
    IdentityEvidence,
    Preprocessing,
    VisibilityContract,
)
from app.contracts.identity import IdentityRisk, ManualConstraint, RiskKind
from app.identity.consistency import segment_tracks
from app.storage.database import connect, transaction
from app.storage.repositories import Repository
from app.workflows.incremental import update_incremental
from app.workflows.read import get_events, get_identity_state
from app.workflows.review import apply_review


def make_incremental_settings(tmp_path, **discovery_overrides):
    settings = machine_settings(tmp_path / "machine", tmp_path)
    disc = {
        "version": "complete_support_v3",
        "support_mode": "constrained_manifold",
        "clustering_threshold": 0.28,
        "min_common_parts": 2,
        "min_part_visibility": 0.2,
        "max_templates_per_group": 6,
        "intra_segment_near_distance": 0.18,
        "far_distance": 0.45,
        "rival_margin": 0.04,
    }
    disc.update(discovery_overrides)
    return settings.model_copy(update={"discovery": DiscoveryPolicy(**disc)})


def make_clip_batch(clip_name: str, angle: float = 0.0, extra_track: bool = False):
    angles = (angle, angle + 0.01, angle - 0.01, angle, angle + 0.01, angle)
    raw = make_batch(angles=angles, extra_tracks=1 if extra_track else 0)
    text = raw.model_dump_json()
    for old, new in (
        ('"clip"', json.dumps(clip_name)),
        ("parent-0", f"parent-{clip_name}-0"),
        ("parent-1", f"parent-{clip_name}-1"),
        ("s-0-", f"s-{clip_name}-0-"),
        ("s-1-", f"s-{clip_name}-1-"),
        ("o-0-", f"o-{clip_name}-0-"),
        ("o-1-", f"o-{clip_name}-1-"),
        ('"t-0"', json.dumps(f"t-{clip_name}-0")),
        ('"t-1"', json.dumps(f"t-{clip_name}-1")),
    ):
        text = text.replace(old, new)
    b = EvidenceBatch.model_validate_json(text)
    return b.model_copy(update={"batch_id": f"batch-{clip_name}"})


def identity_partition(assignments):
    """Return a frozenset of frozensets of segment IDs belonging to the same player."""
    groups = defaultdict(set)
    for a in assignments:
        if a.player_id is not None:
            groups[a.player_id].add(a.segment_id)
    return frozenset(frozenset(sids) for sids in groups.values())


def test_unsupported_support_mode_raises_explicit_error(tmp_path):
    """Incremental pipeline strictly forbids unsupported modes and silent fallbacks."""
    b = make_clip_batch("alpha", angle=0.0)
    seg = segment_tracks(b)
    db = connect(":memory:")

    # Attempting 'absolute' mode
    settings_abs = make_incremental_settings(tmp_path, support_mode="absolute")
    with pytest.raises(ValueError, match="unsupported support_mode for incremental identity"):
        update_incremental(
            db,
            UpdateRequest(batch=seg.batch, expected_revision=0),
            segmentation_report=seg.report,
            feature_values={},
            settings=settings_abs,
        )

    # Attempting 'cooccurrence_contrastive' mode
    settings_contra = make_incremental_settings(tmp_path, support_mode="cooccurrence_contrastive")
    with pytest.raises(ValueError, match="unsupported support_mode for incremental identity"):
        update_incremental(
            db,
            UpdateRequest(batch=seg.batch, expected_revision=0),
            segmentation_report=seg.report,
            feature_values={},
            settings=settings_contra,
        )


def test_empty_library_first_batch_creates_identity(tmp_path):
    """Empty library receives cross-clip evidence and creates new identity (Requirement A/B/C counts)."""
    settings = make_incremental_settings(tmp_path)
    b1 = make_clip_batch("alpha", angle=0.0)
    b2 = make_clip_batch("beta", angle=0.0)
    comb = EvidenceBatch(
        match_id="m",
        batch_id="b-1",
        producer="synthetic",
        clips=[*b1.clips, *b2.clips],
        identity_evidence=IdentityEvidence(
            tracks=[*b1.identity_evidence.tracks, *b2.identity_evidence.tracks],
            observations=[*b1.identity_evidence.observations, *b2.identity_evidence.observations],
            segments=[*b1.identity_evidence.segments, *b2.identity_evidence.segments],
            samples=[*b1.identity_evidence.samples, *b2.identity_evidence.samples],
            feature_spaces=b1.identity_evidence.feature_spaces,
        ),
    )
    seg_res = segment_tracks(comb)
    db = connect(":memory:")

    res = update_incremental(
        db,
        UpdateRequest(batch=seg_res.batch, expected_revision=0),
        segmentation_report=seg_res.report,
        feature_values={},
        settings=settings,
    )
    assert res.revision == 1
    summary = res.data["summary"]
    # In empty library with 2 clips: 2 segments get created_new_identity, 0 matched_existing, 0 insufficient
    assert summary["created_new_identity_count"] == 2
    assert summary["matched_existing_count"] == 0
    assert summary["insufficient_evidence_count"] == 0
    assert summary["new_players_count"] == 1

    state = get_identity_state(db, "m")
    assert len(state.data.players) == 1
    player_id = state.data.players[0].player_id
    assigned = [a for a in state.data.assignments if a.player_id is not None]
    assert len(assigned) == 2
    assert all(a.player_id == player_id for a in assigned)
    assert all(a.status == "proposed" for a in assigned)
    assert all(a.method == "automatic" for a in assigned)
    # Automatic templates created
    assert len(state.data.templates) >= 2
    assert all(t.admission == "automatic_experimental" for t in state.data.templates)


def test_single_clip_deferred_then_cross_clip_creates_identity(tmp_path):
    """Single clip evidence stays deferred as insufficient_evidence; subsequent clip creates identity."""
    settings = make_incremental_settings(tmp_path)
    db = connect(":memory:")

    # Batch 1: clip alpha only (single appearance)
    b_alpha = make_clip_batch("alpha", angle=0.0)
    seg_alpha = segment_tracks(b_alpha)
    res1 = update_incremental(
        db,
        UpdateRequest(batch=seg_alpha.batch, expected_revision=0),
        segmentation_report=seg_alpha.report,
        feature_values={},
        settings=settings,
    )
    assert res1.revision == 1
    sum1 = res1.data["summary"]
    assert sum1["created_new_identity_count"] == 0
    assert sum1["matched_existing_count"] == 0
    assert sum1["insufficient_evidence_count"] == 1
    assert sum1["new_players_count"] == 0

    state1 = get_identity_state(db, "m")
    assert len(state1.data.players) == 0
    assert all(a.status == "unassigned" for a in state1.data.assignments)

    # Batch 2: clip beta arrives with matching appearance (angle=0.0)
    b_beta = make_clip_batch("beta", angle=0.0)
    seg_beta = segment_tracks(b_beta)
    res2 = update_incremental(
        db,
        UpdateRequest(batch=seg_beta.batch, expected_revision=1),
        segmentation_report=seg_beta.report,
        feature_values={},
        settings=settings,
    )
    assert res2.revision == 2
    sum2 = res2.data["summary"]
    # Historical alpha + incoming beta form cross-clip group!
    assert sum2["created_new_identity_count"] == 2
    assert sum2["new_players_count"] == 1

    state2 = get_identity_state(db, "m")
    assert len(state2.data.players) == 1
    player_id = state2.data.players[0].player_id
    assigned = [a for a in state2.data.assignments if a.player_id is not None]
    assert len(assigned) == 2
    assert assigned[0].player_id == assigned[1].player_id == player_id


def test_returning_player_matches_existing_without_duplicate_player(tmp_path):
    """Player A in Batch 1. Batch 2 unrelated person. Batch 3 Player A returns -> matched_existing."""
    settings = make_incremental_settings(tmp_path)
    db = connect(":memory:")

    # Batch 1: Player A appears across alpha and beta (angle=0.0)
    b1_a = make_clip_batch("alpha", angle=0.0)
    b1_b = make_clip_batch("beta", angle=0.0)
    comb1 = EvidenceBatch(
        match_id="m",
        batch_id="b-1",
        producer="synthetic",
        clips=[*b1_a.clips, *b1_b.clips],
        identity_evidence=IdentityEvidence(
            tracks=[*b1_a.identity_evidence.tracks, *b1_b.identity_evidence.tracks],
            observations=[
                *b1_a.identity_evidence.observations,
                *b1_b.identity_evidence.observations,
            ],
            segments=[*b1_a.identity_evidence.segments, *b1_b.identity_evidence.segments],
            samples=[*b1_a.identity_evidence.samples, *b1_b.identity_evidence.samples],
            feature_spaces=b1_a.identity_evidence.feature_spaces,
        ),
    )
    seg1 = segment_tracks(comb1)
    res1 = update_incremental(
        db,
        UpdateRequest(batch=seg1.batch, expected_revision=0),
        segmentation_report=seg1.report,
        feature_values={},
        settings=settings,
    )
    assert res1.revision == 1
    state1 = get_identity_state(db, "m")
    assert len(state1.data.players) == 1
    player_a_id = state1.data.players[0].player_id

    # Batch 2: Player A absent; single appearance Person Y (angle=2.0)
    b2 = make_clip_batch("gamma", angle=2.0)
    seg2 = segment_tracks(b2)
    res2 = update_incremental(
        db,
        UpdateRequest(batch=seg2.batch, expected_revision=1),
        segmentation_report=seg2.report,
        feature_values={},
        settings=settings,
    )
    assert res2.revision == 2
    assert res2.data["summary"]["insufficient_evidence_count"] == 1
    assert res2.data["summary"]["new_players_count"] == 0

    # Batch 3: Player A returns in clip delta (angle=0.0)
    b3 = make_clip_batch("delta", angle=0.0)
    seg3 = segment_tracks(b3)
    res3 = update_incremental(
        db,
        UpdateRequest(batch=seg3.batch, expected_revision=2),
        segmentation_report=seg3.report,
        feature_values={},
        settings=settings,
    )
    assert res3.revision == 3
    sum3 = res3.data["summary"]
    # Matches existing Player A!
    assert sum3["matched_existing_count"] == 1
    assert sum3["new_players_count"] == 0

    state3 = get_identity_state(db, "m")
    assert len(state3.data.players) == 1
    delta_seg = next(s for s in state3.data.evidence.segments if s.clip_id == "delta" and s.active)
    delta_assign = next(a for a in state3.data.assignments if a.segment_id == delta_seg.segment_id)
    assert delta_assign.player_id == player_a_id
    assert delta_assign.status == "proposed"


def test_independent_view_templates_prevent_duplicate_returning_identity(tmp_path):
    settings = make_incremental_settings(tmp_path, max_templates_per_group=3)
    target_clips = {"view-a", "view-b", "view-c"}
    rival_clips = {"rival-a", "rival-b"}
    initial = synthetic_facts(
        [
            ("view-a", "0", angle_vector(0.0), "view-a"),
            ("view-b", "0", angle_vector(0.15), "view-b"),
            ("view-c", "0", angle_vector(0.28), "view-c"),
            ("rival-a", "0", angle_vector(2.0), "rival-a"),
            ("rival-b", "0", angle_vector(2.0), "rival-b"),
        ]
    )
    segmented = segment_tracks(initial)
    db = connect(":memory:")
    first = update_incremental(
        db,
        UpdateRequest(batch=segmented.batch, expected_revision=0),
        segmentation_report=segmented.report,
        feature_values={},
        settings=settings,
    )
    assert first.data["summary"]["created_new_identity_count"] == 5
    assert first.data["summary"]["new_players_count"] == 2

    initial_state = get_identity_state(db, "synthetic-match").data
    initial_segments = {s.segment_id: s for s in initial_state.evidence.segments}
    initial_owners = {
        initial_segments[a.segment_id].clip_id: a.player_id
        for a in initial_state.assignments
        if a.player_id is not None
    }
    target_player = initial_owners["view-a"]
    rival_player = initial_owners["rival-a"]
    assert target_player != rival_player
    assert {initial_owners[clip] for clip in target_clips} == {target_player}
    assert {initial_owners[clip] for clip in rival_clips} == {rival_player}
    target_templates = [t for t in initial_state.templates if t.player_id == target_player]
    assert len(target_templates) == 3
    samples = {s.sample_id: s for s in initial_state.evidence.samples}
    assert {samples[t.sample_id].clip_id for t in target_templates} == target_clips

    returning = synthetic_facts(
        [
            ("view-d", "0", angle_vector(0.7), "view-d"),
            ("view-e", "0", angle_vector(0.7), "view-e"),
        ]
    ).model_copy(update={"batch_id": "returning-views"})
    returning_segmented = segment_tracks(returning)
    second = update_incremental(
        db,
        UpdateRequest(batch=returning_segmented.batch, expected_revision=first.revision),
        segmentation_report=returning_segmented.report,
        feature_values={},
        settings=settings,
    )
    assert second.data["summary"]["matched_existing_count"] == 2
    assert second.data["summary"]["created_new_identity_count"] == 0
    assert second.data["summary"]["new_players_count"] == 0
    assert second.data["summary"]["insufficient_evidence_count"] == 0

    final_state = get_identity_state(db, "synthetic-match").data
    assert len(final_state.players) == 2
    final_segments = {s.segment_id: s for s in final_state.evidence.segments}
    final_owners = {
        final_segments[a.segment_id].clip_id: a
        for a in final_state.assignments
        if a.player_id is not None
    }
    assert {final_owners[clip].player_id for clip in target_clips | {"view-d", "view-e"}} == {
        target_player
    }
    assert {final_owners[clip].player_id for clip in rival_clips} == {rival_player}
    assert all(a.status == "proposed" and a.method == "automatic" for a in final_owners.values())


def test_new_player_appearing_later_creates_new_player(tmp_path):
    """Batch 1 has Player A. Batch 2 introduces Person B across 2 clips -> Player B discovered."""
    settings = make_incremental_settings(tmp_path)
    db = connect(":memory:")

    # Batch 1: Player A (angle=0.0) in clips alpha & beta
    b1_a = make_clip_batch("alpha", angle=0.0)
    b1_b = make_clip_batch("beta", angle=0.0)
    comb1 = EvidenceBatch(
        match_id="m",
        batch_id="b-1",
        producer="synthetic",
        clips=[*b1_a.clips, *b1_b.clips],
        identity_evidence=IdentityEvidence(
            tracks=[*b1_a.identity_evidence.tracks, *b1_b.identity_evidence.tracks],
            observations=[
                *b1_a.identity_evidence.observations,
                *b1_b.identity_evidence.observations,
            ],
            segments=[*b1_a.identity_evidence.segments, *b1_b.identity_evidence.segments],
            samples=[*b1_a.identity_evidence.samples, *b1_b.identity_evidence.samples],
            feature_spaces=b1_a.identity_evidence.feature_spaces,
        ),
    )
    seg1 = segment_tracks(comb1)
    update_incremental(
        db,
        UpdateRequest(batch=seg1.batch, expected_revision=0),
        segmentation_report=seg1.report,
        feature_values={},
        settings=settings,
    )
    state1 = get_identity_state(db, "m")
    assert len(state1.data.players) == 1
    player_a_id = state1.data.players[0].player_id

    # Batch 2: Player B (angle=2.5) in clips gamma & delta (well separated from Player A)
    b2_g = make_clip_batch("gamma", angle=2.5)
    b2_d = make_clip_batch("delta", angle=2.5)
    comb2 = EvidenceBatch(
        match_id="m",
        batch_id="b-2",
        producer="synthetic",
        clips=[*b2_g.clips, *b2_d.clips],
        identity_evidence=IdentityEvidence(
            tracks=[*b2_g.identity_evidence.tracks, *b2_d.identity_evidence.tracks],
            observations=[
                *b2_g.identity_evidence.observations,
                *b2_d.identity_evidence.observations,
            ],
            segments=[*b2_g.identity_evidence.segments, *b2_d.identity_evidence.segments],
            samples=[*b2_g.identity_evidence.samples, *b2_d.identity_evidence.samples],
            feature_spaces=b2_g.identity_evidence.feature_spaces,
        ),
    )
    seg2 = segment_tracks(comb2)
    res2 = update_incremental(
        db,
        UpdateRequest(batch=seg2.batch, expected_revision=1),
        segmentation_report=seg2.report,
        feature_values={},
        settings=settings,
    )
    assert res2.revision == 2
    sum2 = res2.data["summary"]
    assert sum2["created_new_identity_count"] == 2
    assert sum2["new_players_count"] == 1

    state2 = get_identity_state(db, "m")
    assert len(state2.data.players) == 2
    pids = {p.player_id for p in state2.data.players}
    assert player_a_id in pids


def test_ambiguous_candidate_competing_existing_players_remains_unassigned(tmp_path):
    """Segment with distance ambiguity between competing existing players remains unassigned."""
    settings = make_incremental_settings(tmp_path, rival_margin=0.06)
    db = connect(":memory:")

    # Batch 1: Player A (angle=0.0) across alpha & beta
    b1_a = make_clip_batch("alpha", angle=0.0)
    b1_b = make_clip_batch("beta", angle=0.0)
    comb1 = EvidenceBatch(
        match_id="m",
        batch_id="b-1",
        producer="synthetic",
        clips=[*b1_a.clips, *b1_b.clips],
        identity_evidence=IdentityEvidence(
            tracks=[*b1_a.identity_evidence.tracks, *b1_b.identity_evidence.tracks],
            observations=[
                *b1_a.identity_evidence.observations,
                *b1_b.identity_evidence.observations,
            ],
            segments=[*b1_a.identity_evidence.segments, *b1_b.identity_evidence.segments],
            samples=[*b1_a.identity_evidence.samples, *b1_b.identity_evidence.samples],
            feature_spaces=b1_a.identity_evidence.feature_spaces,
        ),
    )
    update_incremental(
        db,
        UpdateRequest(batch=segment_tracks(comb1).batch, expected_revision=0),
        segmentation_report=segment_tracks(comb1).report,
        feature_values={},
        settings=settings,
    )

    # Batch 2: Player B (angle=0.6) across gamma & delta (separated from A: distance = 0.6/2 = 0.30 > 0.28)
    b2_g = make_clip_batch("gamma", angle=0.6)
    b2_d = make_clip_batch("delta", angle=0.6)
    comb2 = EvidenceBatch(
        match_id="m",
        batch_id="b-2",
        producer="synthetic",
        clips=[*b2_g.clips, *b2_d.clips],
        identity_evidence=IdentityEvidence(
            tracks=[*b2_g.identity_evidence.tracks, *b2_d.identity_evidence.tracks],
            observations=[
                *b2_g.identity_evidence.observations,
                *b2_d.identity_evidence.observations,
            ],
            segments=[*b2_g.identity_evidence.segments, *b2_d.identity_evidence.segments],
            samples=[*b2_g.identity_evidence.samples, *b2_d.identity_evidence.samples],
            feature_spaces=b2_g.identity_evidence.feature_spaces,
        ),
    )
    update_incremental(
        db,
        UpdateRequest(batch=segment_tracks(comb2).batch, expected_revision=1),
        segmentation_report=segment_tracks(comb2).report,
        feature_values={},
        settings=settings,
    )

    # Batch 3: Ambiguous segment in clip epsilon at angle=0.30 (equidistant to Player A [0.0] and B [0.6], within rival_margin=0.06)
    b3 = make_clip_batch("epsilon", angle=0.30)
    seg3 = segment_tracks(b3)
    res3 = update_incremental(
        db,
        UpdateRequest(batch=seg3.batch, expected_revision=2),
        segmentation_report=seg3.report,
        feature_values={},
        settings=settings,
    )
    assert res3.revision == 3
    sum3 = res3.data["summary"]
    # Must NOT assign first place by force; remains insufficient_evidence!
    assert sum3["insufficient_evidence_count"] == 1
    assert sum3["matched_existing_count"] == 0

    state3 = get_identity_state(db, "m")
    eps_seg = next(s for s in state3.data.evidence.segments if s.clip_id == "epsilon" and s.active)
    eps_assign = next(a for a in state3.data.assignments if a.segment_id == eps_seg.segment_id)
    assert eps_assign.status == "unassigned"
    assert eps_assign.player_id is None
    assert eps_assign.unassigned_reason == "ambiguous"


def test_historical_deferred_segments_reprocessed_with_new_evidence(tmp_path):
    """Historical deferred segment is re-evaluated when new evidence arrives in subsequent batch."""
    settings = make_incremental_settings(tmp_path)
    db = connect(":memory:")

    # Batch 1: Person X in clip alpha (single appearance -> deferred)
    b1 = make_clip_batch("alpha", angle=0.0)
    seg1 = segment_tracks(b1)
    res1 = update_incremental(
        db,
        UpdateRequest(batch=seg1.batch, expected_revision=0),
        segmentation_report=seg1.report,
        feature_values={},
        settings=settings,
    )
    assert res1.data["summary"]["insufficient_evidence_count"] == 1

    # Batch 2: Person Y in clip gamma (single appearance -> deferred)
    b2 = make_clip_batch("gamma", angle=2.5)
    seg2 = segment_tracks(b2)
    res2 = update_incremental(
        db,
        UpdateRequest(batch=seg2.batch, expected_revision=1),
        segmentation_report=seg2.report,
        feature_values={},
        settings=settings,
    )
    assert res2.data["summary"]["insufficient_evidence_count"] == 1

    # Batch 3: Person X reappears in clip delta (angle=0.0)!
    # Historical alpha is pulled from historical unassigned pool and paired with delta!
    b3 = make_clip_batch("delta", angle=0.0)
    seg3 = segment_tracks(b3)
    res3 = update_incremental(
        db,
        UpdateRequest(batch=seg3.batch, expected_revision=2),
        segmentation_report=seg3.report,
        feature_values={},
        settings=settings,
    )
    assert res3.revision == 3
    sum3 = res3.data["summary"]
    assert sum3["created_new_identity_count"] == 2
    assert sum3["new_players_count"] == 1

    state3 = get_identity_state(db, "m")
    assert len(state3.data.players) == 1
    assigned = [a for a in state3.data.assignments if a.player_id is not None]
    assert len(assigned) == 2
    assigned_sids = {a.segment_id for a in assigned}
    alpha_sid = next(
        s.segment_id for s in state3.data.evidence.segments if s.clip_id == "alpha" and s.active
    )
    delta_sid = next(
        s.segment_id for s in state3.data.evidence.segments if s.clip_id == "delta" and s.active
    )
    assert alpha_sid in assigned_sids
    assert delta_sid in assigned_sids


def test_existing_players_never_automatically_merged(tmp_path):
    """Existing players are read-only and never automatically merged with each other."""
    settings = make_incremental_settings(tmp_path)
    db = connect(":memory:")

    # Create Player A (clips alpha & beta, angle 0.0)
    b1_a = make_clip_batch("alpha", angle=0.0)
    b1_b = make_clip_batch("beta", angle=0.0)
    comb1 = EvidenceBatch(
        match_id="m",
        batch_id="b-1",
        producer="synthetic",
        clips=[*b1_a.clips, *b1_b.clips],
        identity_evidence=IdentityEvidence(
            tracks=[*b1_a.identity_evidence.tracks, *b1_b.identity_evidence.tracks],
            observations=[
                *b1_a.identity_evidence.observations,
                *b1_b.identity_evidence.observations,
            ],
            segments=[*b1_a.identity_evidence.segments, *b1_b.identity_evidence.segments],
            samples=[*b1_a.identity_evidence.samples, *b1_b.identity_evidence.samples],
            feature_spaces=b1_a.identity_evidence.feature_spaces,
        ),
    )
    update_incremental(
        db,
        UpdateRequest(batch=segment_tracks(comb1).batch, expected_revision=0),
        segmentation_report=segment_tracks(comb1).report,
        feature_values={},
        settings=settings,
    )

    # Create Player B (clips gamma & delta, angle 2.5)
    b2_g = make_clip_batch("gamma", angle=2.5)
    b2_d = make_clip_batch("delta", angle=2.5)
    comb2 = EvidenceBatch(
        match_id="m",
        batch_id="b-2",
        producer="synthetic",
        clips=[*b2_g.clips, *b2_d.clips],
        identity_evidence=IdentityEvidence(
            tracks=[*b2_g.identity_evidence.tracks, *b2_d.identity_evidence.tracks],
            observations=[
                *b2_g.identity_evidence.observations,
                *b2_d.identity_evidence.observations,
            ],
            segments=[*b2_g.identity_evidence.segments, *b2_d.identity_evidence.segments],
            samples=[*b2_g.identity_evidence.samples, *b2_d.identity_evidence.samples],
            feature_spaces=b2_g.identity_evidence.feature_spaces,
        ),
    )
    update_incremental(
        db,
        UpdateRequest(batch=segment_tracks(comb2).batch, expected_revision=1),
        segmentation_report=segment_tracks(comb2).report,
        feature_values={},
        settings=settings,
    )

    state = get_identity_state(db, "m")
    assert len(state.data.players) == 2
    pids = [p.player_id for p in state.data.players]

    # In Batch 3, import a clip bridging or neutral
    b3 = make_clip_batch("theta", angle=1.25)
    update_incremental(
        db,
        UpdateRequest(batch=segment_tracks(b3).batch, expected_revision=2),
        segmentation_report=segment_tracks(b3).report,
        feature_values={},
        settings=settings,
    )
    post_state = get_identity_state(db, "m")
    # Existing players remain untouched and never merged
    post_pids = [p.player_id for p in post_state.data.players]
    assert set(pids).issubset(set(post_pids))
    assert all(p.alias_of is None for p in post_state.data.players)


def test_variable_batch_size_does_not_make_batch_id_identity_boundary(tmp_path):
    """Importing 4 clips as 1 batch vs 2 batches yields identical identity groupings."""
    settings = make_incremental_settings(tmp_path)

    b_a = make_clip_batch("alpha", angle=0.0)
    b_b = make_clip_batch("beta", angle=0.0)
    b_g = make_clip_batch("gamma", angle=2.5)
    b_d = make_clip_batch("delta", angle=2.5)

    all_batch = EvidenceBatch(
        match_id="m",
        batch_id="b-all",
        producer="synthetic",
        clips=[*b_a.clips, *b_b.clips, *b_g.clips, *b_d.clips],
        identity_evidence=IdentityEvidence(
            tracks=[
                *b_a.identity_evidence.tracks,
                *b_b.identity_evidence.tracks,
                *b_g.identity_evidence.tracks,
                *b_d.identity_evidence.tracks,
            ],
            observations=[
                *b_a.identity_evidence.observations,
                *b_b.identity_evidence.observations,
                *b_g.identity_evidence.observations,
                *b_d.identity_evidence.observations,
            ],
            segments=[
                *b_a.identity_evidence.segments,
                *b_b.identity_evidence.segments,
                *b_g.identity_evidence.segments,
                *b_d.identity_evidence.segments,
            ],
            samples=[
                *b_a.identity_evidence.samples,
                *b_b.identity_evidence.samples,
                *b_g.identity_evidence.samples,
                *b_d.identity_evidence.samples,
            ],
            feature_spaces=b_a.identity_evidence.feature_spaces,
        ),
    )
    seg_all = segment_tracks(all_batch)

    # Condition 1: 1 batch of 4 clips
    db1 = connect(":memory:")
    update_incremental(
        db1,
        UpdateRequest(batch=seg_all.batch, expected_revision=0),
        segmentation_report=seg_all.report,
        feature_values={},
        settings=settings,
    )
    state1 = get_identity_state(db1, "m")
    part1 = identity_partition(state1.data.assignments)

    # Condition 2: 2 batches of 2 clips each
    comb1 = EvidenceBatch(
        match_id="m",
        batch_id="b-1",
        producer="synthetic",
        clips=[*b_a.clips, *b_b.clips],
        identity_evidence=IdentityEvidence(
            tracks=[*b_a.identity_evidence.tracks, *b_b.identity_evidence.tracks],
            observations=[*b_a.identity_evidence.observations, *b_b.identity_evidence.observations],
            segments=[*b_a.identity_evidence.segments, *b_b.identity_evidence.segments],
            samples=[*b_a.identity_evidence.samples, *b_b.identity_evidence.samples],
            feature_spaces=b_a.identity_evidence.feature_spaces,
        ),
    )
    comb2 = EvidenceBatch(
        match_id="m",
        batch_id="b-2",
        producer="synthetic",
        clips=[*b_g.clips, *b_d.clips],
        identity_evidence=IdentityEvidence(
            tracks=[*b_g.identity_evidence.tracks, *b_d.identity_evidence.tracks],
            observations=[*b_g.identity_evidence.observations, *b_d.identity_evidence.observations],
            segments=[*b_g.identity_evidence.segments, *b_d.identity_evidence.segments],
            samples=[*b_g.identity_evidence.samples, *b_d.identity_evidence.samples],
            feature_spaces=b_g.identity_evidence.feature_spaces,
        ),
    )
    db2 = connect(":memory:")
    update_incremental(
        db2,
        UpdateRequest(batch=segment_tracks(comb1).batch, expected_revision=0),
        segmentation_report=segment_tracks(comb1).report,
        feature_values={},
        settings=settings,
    )
    update_incremental(
        db2,
        UpdateRequest(batch=segment_tracks(comb2).batch, expected_revision=1),
        segmentation_report=segment_tracks(comb2).report,
        feature_values={},
        settings=settings,
    )
    state2 = get_identity_state(db2, "m")
    part2 = identity_partition(state2.data.assignments)

    assert part1 == part2
    assert len(part1) == 2


def test_historical_segment_unrestorable_without_supported_proposal(tmp_path):
    """Historical segment without persisted supported proposal safely defers as insufficient_evidence."""
    settings = make_incremental_settings(tmp_path)
    db = connect(":memory:")

    b_a = make_clip_batch("alpha", angle=0.0)
    seg_a = segment_tracks(b_a)
    update_incremental(
        db,
        UpdateRequest(batch=seg_a.batch, expected_revision=0),
        segmentation_report=seg_a.report,
        feature_values={},
        settings=settings,
    )

    # Simulate an old database or corrupted proposal by deleting the proposal for alpha
    alpha_seg_id = next(s.segment_id for s in seg_a.batch.identity_evidence.segments if s.active)
    with transaction(db):
        db.execute(
            "DELETE FROM segmentation_proposals WHERE match_id='m' AND segment_id=?",
            (alpha_seg_id,),
        )

    # Batch 2: beta arrives with matching appearance (angle=0.0)
    b_b = make_clip_batch("beta", angle=0.0)
    seg_b = segment_tracks(b_b)
    res2 = update_incremental(
        db,
        UpdateRequest(batch=seg_b.batch, expected_revision=1),
        segmentation_report=seg_b.report,
        feature_values={},
        settings=settings,
    )
    # Because alpha's proposal is missing, alpha CANNOT be restored to supported qualification!
    # Alpha remains unassigned (insufficient_evidence) and beta is single-appearance in this batch -> also deferred!
    sum2 = res2.data["summary"]
    assert sum2["new_players_count"] == 0
    state2 = get_identity_state(db, "m")
    assert len(state2.data.players) == 0


def test_duplicate_batch_idempotency(tmp_path):
    settings = make_incremental_settings(tmp_path)
    b1 = make_clip_batch("alpha", angle=0.0)
    b2 = make_clip_batch("beta", angle=0.0)
    combined = EvidenceBatch(
        match_id="m",
        batch_id="b-1",
        producer="synthetic",
        clips=[*b1.clips, *b2.clips],
        identity_evidence=IdentityEvidence(
            tracks=[*b1.identity_evidence.tracks, *b2.identity_evidence.tracks],
            observations=[*b1.identity_evidence.observations, *b2.identity_evidence.observations],
            segments=[*b1.identity_evidence.segments, *b2.identity_evidence.segments],
            samples=[*b1.identity_evidence.samples, *b2.identity_evidence.samples],
            feature_spaces=b1.identity_evidence.feature_spaces,
        ),
    )
    seg_res = segment_tracks(combined)
    db = connect(":memory:")

    first = update_incremental(
        db,
        UpdateRequest(batch=seg_res.batch, expected_revision=0),
        segmentation_report=seg_res.report,
        feature_values={},
        settings=settings,
    )
    assert first.revision == 1
    state1 = get_identity_state(db, "m")

    # Re-submitting the exact same batch
    replay = update_incremental(
        db,
        UpdateRequest(
            batch=seg_res.batch.model_copy(update={"batch_id": "b-1-replay"}),
            expected_revision=1,
        ),
        segmentation_report=seg_res.report,
        feature_values={},
        settings=settings,
    )
    assert replay.revision == 1
    assert replay.data["changed"] is False
    state2 = get_identity_state(db, "m")
    assert state1 == state2


def test_mixed_identity_segment_blocked(tmp_path):
    """Segment with mixed_identity risk is blocked from matching and template admission."""
    settings = make_incremental_settings(tmp_path)
    db = connect(":memory:")

    b_a = make_clip_batch("alpha", angle=0.0)
    b_b = make_clip_batch("beta", angle=0.0)

    # Attach a mixed_identity risk to beta's segment
    beta_seg = b_b.identity_evidence.segments[0]
    risk = IdentityRisk(
        risk_id="risk-mixed-beta",
        kind=RiskKind.MIXED_IDENTITY,
        match_id="m",
        segment_id=beta_seg.segment_id,
        clip_id=beta_seg.clip_id,
        track_id=beta_seg.track_id,
    )
    b_b.identity_evidence.risks.append(risk)

    comb = EvidenceBatch(
        match_id="m",
        batch_id="b-mixed",
        producer="synthetic",
        clips=[*b_a.clips, *b_b.clips],
        identity_evidence=IdentityEvidence(
            tracks=[*b_a.identity_evidence.tracks, *b_b.identity_evidence.tracks],
            observations=[*b_a.identity_evidence.observations, *b_b.identity_evidence.observations],
            segments=[*b_a.identity_evidence.segments, *b_b.identity_evidence.segments],
            samples=[*b_a.identity_evidence.samples, *b_b.identity_evidence.samples],
            feature_spaces=b_a.identity_evidence.feature_spaces,
            risks=[risk],
        ),
    )
    seg_res = segment_tracks(comb)
    update_incremental(
        db,
        UpdateRequest(batch=seg_res.batch, expected_revision=0),
        segmentation_report=seg_res.report,
        feature_values={},
        settings=settings,
    )
    state = get_identity_state(db, "m")
    # Mixed segment cannot form a group -> no player created, all unassigned
    assert len(state.data.players) == 0
    assert all(a.status == "unassigned" for a in state.data.assignments)


def test_incompatible_feature_space_blocked(tmp_path):
    """Segment declared in an incompatible feature space cannot match."""
    settings = make_incremental_settings(tmp_path)
    db = connect(":memory:")

    b1 = make_clip_batch("alpha", angle=0.0)
    seg1 = segment_tracks(b1)
    update_incremental(
        db,
        UpdateRequest(batch=seg1.batch, expected_revision=0),
        segmentation_report=seg1.report,
        feature_values={},
        settings=settings,
    )

    # Create a batch with a different feature space
    incompatible_space = FeatureSpace(
        feature_space_id="other-feature-space",
        backend="kpr",
        model_hash="b" * 64,
        preprocessing=Preprocessing(version="other", color_space="rgb"),
        layout="parts",
        shape=[3, 2],
        normalization="l2",
        distance="euclidean_sqrt_visibility_mean_div2",
        visibility=VisibilityContract(shape=[3], semantics="visible"),
    )
    b2 = make_clip_batch("beta", angle=0.0)
    b2.identity_evidence.feature_spaces = [incompatible_space]
    for s in b2.identity_evidence.samples:
        for f in s.features:
            f.feature_space_id = "other-feature-space"
    seg2 = segment_tracks(b2)

    update_incremental(
        db,
        UpdateRequest(batch=seg2.batch, expected_revision=1),
        segmentation_report=seg2.report,
        feature_values={},
        settings=settings,
    )
    state = get_identity_state(db, "m")
    beta_seg = next(s for s in state.data.evidence.segments if s.clip_id == "beta" and s.active)
    beta_assign = next(a for a in state.data.assignments if a.segment_id == beta_seg.segment_id)
    assert beta_assign.status == "unassigned"


def test_template_revocation_on_rejection(tmp_path):
    """When human review rejects a template or reassigns a segment, template is deactivated."""
    settings = make_incremental_settings(tmp_path)
    db = connect(":memory:")

    b1_a = make_clip_batch("alpha", angle=0.0)
    b1_b = make_clip_batch("beta", angle=0.0)
    comb = EvidenceBatch(
        match_id="m",
        batch_id="b-1",
        producer="synthetic",
        clips=[*b1_a.clips, *b1_b.clips],
        identity_evidence=IdentityEvidence(
            tracks=[*b1_a.identity_evidence.tracks, *b1_b.identity_evidence.tracks],
            observations=[
                *b1_a.identity_evidence.observations,
                *b1_b.identity_evidence.observations,
            ],
            segments=[*b1_a.identity_evidence.segments, *b1_b.identity_evidence.segments],
            samples=[*b1_a.identity_evidence.samples, *b1_b.identity_evidence.samples],
            feature_spaces=b1_a.identity_evidence.feature_spaces,
        ),
    )
    seg_res = segment_tracks(comb)
    res = update_incremental(
        db,
        UpdateRequest(batch=seg_res.batch, expected_revision=0),
        segmentation_report=seg_res.report,
        feature_values={},
        settings=settings,
    )
    state = get_identity_state(db, "m")
    assert len(state.data.templates) >= 2
    assert all(t.active for t in state.data.templates)
    target_tpl = state.data.templates[0]

    sample = next(s for s in state.data.evidence.samples if s.sample_id == target_tpl.sample_id)
    active_seg_id = sample.segment_id
    apply_review(
        db,
        ReviewRequest(
            match_id="m",
            expected_revision=res.revision,
            actor="supervisor",
            commands=[
                SetAssignmentCommand(
                    segment_id=active_seg_id,
                    player_id=None,
                    unassigned_reason="other_person",
                    reason="human_reassigned",
                )
            ],
        ),
    )
    post_state = get_identity_state(db, "m")
    deactivated = next(
        t for t in post_state.data.templates if t.template_id == target_tpl.template_id
    )
    assert deactivated.active is False


def test_failure_rollback(tmp_path):
    """Transaction failure rolls back completely leaving revision and state untouched."""
    settings = make_incremental_settings(tmp_path)
    db = connect(":memory:")

    b_a = make_clip_batch("alpha", angle=0.0)
    seg_a = segment_tracks(b_a)
    res = update_incremental(
        db,
        UpdateRequest(batch=seg_a.batch, expected_revision=0),
        segmentation_report=seg_a.report,
        feature_values={},
        settings=settings,
    )
    assert res.revision == 1
    state_before = get_identity_state(db, "m")

    # Force failure during commit by patching put_discovery_report
    b_b = make_clip_batch("beta", angle=0.0)
    seg_b = segment_tracks(b_b)
    with patch(
        "app.workflows.incremental.put_discovery_report",
        side_effect=RuntimeError("simulated database failure"),
    ):
        with pytest.raises(RuntimeError, match="simulated database failure"):
            update_incremental(
                db,
                UpdateRequest(batch=seg_b.batch, expected_revision=1),
                segmentation_report=seg_b.report,
                feature_values={},
                settings=settings,
            )

    # Verify rollback: revision is still 1, state identical
    assert Repository(db).revision("m") == 1
    state_after = get_identity_state(db, "m")
    assert state_before == state_after


def test_human_priority_and_manual_constraints(tmp_path):
    """Manual confirmed assignments and constraints take precedence over automatic updates."""
    settings = make_incremental_settings(tmp_path)
    db = connect(":memory:")

    b_a = make_clip_batch("alpha", angle=0.0)
    seg_a = segment_tracks(b_a)
    update_incremental(
        db,
        UpdateRequest(batch=seg_a.batch, expected_revision=0),
        segmentation_report=seg_a.report,
        feature_values={},
        settings=settings,
    )

    alpha_seg_id = next(s.segment_id for s in seg_a.batch.identity_evidence.segments if s.active)
    with transaction(db):
        Repository(db).put_constraint(
            ManualConstraint(
                constraint_id="c-block-alpha",
                match_id="m",
                segment_id=alpha_seg_id,
                kind="block_automatic",
                actor="expert",
                reason="keep_unassigned",
                revision=2,
            )
        )

    # Batch 2: clip beta arrives. alpha is blocked from automatic assignment
    b_b = make_clip_batch("beta", angle=0.0)
    seg_b = segment_tracks(b_b)
    update_incremental(
        db,
        UpdateRequest(batch=seg_b.batch, expected_revision=1),
        segmentation_report=seg_b.report,
        feature_values={},
        settings=settings,
    )
    state = get_identity_state(db, "m")
    alpha_assign = next(a for a in state.data.assignments if a.segment_id == alpha_seg_id)
    assert alpha_assign.status == "unassigned"


def test_event_dynamic_resolution(tmp_path):
    """Events resolve to current player mappings without inventing events from annotations."""
    settings = make_incremental_settings(tmp_path)
    db = connect(":memory:")

    b_a = make_clip_batch("alpha", angle=0.0)
    b_b = make_clip_batch("beta", angle=0.0)
    comb = EvidenceBatch(
        match_id="m",
        batch_id="b-1",
        producer="synthetic",
        clips=[*b_a.clips, *b_b.clips],
        identity_evidence=IdentityEvidence(
            tracks=[*b_a.identity_evidence.tracks, *b_b.identity_evidence.tracks],
            observations=[*b_a.identity_evidence.observations, *b_b.identity_evidence.observations],
            segments=[*b_a.identity_evidence.segments, *b_b.identity_evidence.segments],
            samples=[*b_a.identity_evidence.samples, *b_b.identity_evidence.samples],
            feature_spaces=b_a.identity_evidence.feature_spaces,
        ),
        events=[
            Event(
                match_id="m",
                event_id="ev-shot-1",
                source_event_key="shot-key-1",
                clip_id="alpha",
                event_type="shot",
                time=PointEventTime(time_s=0.2),
                outcome="made",
                subjects=[
                    Subject(
                        subject_id="subj-1",
                        role="shooter",
                        references=[TrackReference(clip_id="alpha", track_id="0")],
                    )
                ],
                source=Provenance(
                    producer="synthetic",
                    record_type="event",
                    source_record_id="ev-1",
                    evidence_version="1",
                ),
            )
        ],
    )
    seg_res = segment_tracks(comb)
    res = update_incremental(
        db,
        UpdateRequest(batch=seg_res.batch, expected_revision=0),
        segmentation_report=seg_res.report,
        feature_values={},
        settings=settings,
    )
    state = get_identity_state(db, "m")
    player_id = state.data.players[0].player_id

    # Proposed automatic assignment does not enter confirmed events
    from app.contracts.api import EventQuery

    events1 = get_events(db, EventQuery(match_id="m")).data
    assert len(events1) == 1
    assert events1[0].subjects[0].status == "unmapped"
    assert events1[0].subjects[0].reason == "identity_not_confirmed"

    # Confirm the assignment via review
    alpha_seg = next(
        s for s in seg_res.batch.identity_evidence.segments if s.clip_id == "alpha" and s.active
    )
    apply_review(
        db,
        ReviewRequest(
            match_id="m",
            expected_revision=res.revision,
            actor="supervisor",
            commands=[
                SetAssignmentCommand(
                    segment_id=alpha_seg.segment_id,
                    player_id=player_id,
                    reason="manual_confirm",
                )
            ],
        ),
    )

    # Event dynamically resolves to player_id upon confirmation!
    events2 = get_events(db, EventQuery(match_id="m")).data
    assert len(events2) == 1
    assert events2[0].subjects[0].status == "mapped"
    assert events2[0].subjects[0].player_id == player_id


def test_replay_and_change_batch_id_idempotency(tmp_path):
    """Submitting identical content under different batch_ids never duplicates entities or advances revision."""
    settings = make_incremental_settings(tmp_path)
    b1 = make_clip_batch("alpha", angle=0.0)
    b2 = make_clip_batch("beta", angle=0.0)
    comb = EvidenceBatch(
        match_id="m",
        batch_id="b-orig",
        producer="synthetic",
        clips=[*b1.clips, *b2.clips],
        identity_evidence=IdentityEvidence(
            tracks=[*b1.identity_evidence.tracks, *b2.identity_evidence.tracks],
            observations=[*b1.identity_evidence.observations, *b2.identity_evidence.observations],
            segments=[*b1.identity_evidence.segments, *b2.identity_evidence.segments],
            samples=[*b1.identity_evidence.samples, *b2.identity_evidence.samples],
            feature_spaces=b1.identity_evidence.feature_spaces,
        ),
    )
    seg_res = segment_tracks(comb)
    db = connect(":memory:")

    first = update_incremental(
        db,
        UpdateRequest(batch=seg_res.batch, expected_revision=0),
        segmentation_report=seg_res.report,
        feature_values={},
        settings=settings,
    )
    assert first.revision == 1
    state1 = get_identity_state(db, "m")

    # Re-submit with different batch_id
    replay1 = update_incremental(
        db,
        UpdateRequest(
            batch=seg_res.batch.model_copy(update={"batch_id": "b-different-1"}),
            expected_revision=1,
        ),
        segmentation_report=seg_res.report,
        feature_values={},
        settings=settings,
    )
    assert replay1.revision == 1
    assert replay1.data["changed"] is False
    assert replay1.data["summary"]["idempotent_replay"] is True
    state2 = get_identity_state(db, "m")
    assert state1 == state2

    # Re-submit with yet another batch_id
    replay2 = update_incremental(
        db,
        UpdateRequest(
            batch=seg_res.batch.model_copy(update={"batch_id": "b-arbitrary-uuid-999"}),
            expected_revision=1,
        ),
        segmentation_report=seg_res.report,
        feature_values={},
        settings=settings,
    )
    assert replay2.revision == 1
    assert replay2.data["changed"] is False
    state3 = get_identity_state(db, "m")
    assert state1 == state3


def test_shuffled_list_order_recognized_as_identical_input(tmp_path):
    """Reordering lists in EvidenceBatch produces identical fingerprint and is recognized as idempotent."""
    from app.identity.fingerprints import evidence_fingerprint

    settings = make_incremental_settings(tmp_path)
    b1 = make_clip_batch("alpha", angle=0.0)
    b2 = make_clip_batch("beta", angle=0.0)
    comb = EvidenceBatch(
        match_id="m",
        batch_id="b-ordered",
        producer="synthetic",
        clips=[*b1.clips, *b2.clips],
        identity_evidence=IdentityEvidence(
            tracks=[*b1.identity_evidence.tracks, *b2.identity_evidence.tracks],
            observations=[*b1.identity_evidence.observations, *b2.identity_evidence.observations],
            segments=[*b1.identity_evidence.segments, *b2.identity_evidence.segments],
            samples=[*b1.identity_evidence.samples, *b2.identity_evidence.samples],
            feature_spaces=b1.identity_evidence.feature_spaces,
        ),
    )
    seg_res = segment_tracks(comb)
    db = connect(":memory:")

    first = update_incremental(
        db,
        UpdateRequest(batch=seg_res.batch, expected_revision=0),
        segmentation_report=seg_res.report,
        feature_values={},
        settings=settings,
    )
    assert first.revision == 1
    state1 = get_identity_state(db, "m")

    # Construct reversed / shuffled version of the batch
    b_shuffled = EvidenceBatch(
        match_id="m",
        batch_id="b-shuffled",
        producer="synthetic",
        clips=list(reversed(seg_res.batch.clips)),
        identity_evidence=IdentityEvidence(
            tracks=list(reversed(seg_res.batch.identity_evidence.tracks)),
            observations=list(reversed(seg_res.batch.identity_evidence.observations)),
            segments=list(reversed(seg_res.batch.identity_evidence.segments)),
            samples=list(reversed(seg_res.batch.identity_evidence.samples)),
            feature_spaces=seg_res.batch.identity_evidence.feature_spaces,
        ),
    )
    fp_orig = evidence_fingerprint(seg_res.batch, {})
    fp_shuffled = evidence_fingerprint(b_shuffled, {})
    assert fp_orig == fp_shuffled

    # Submit shuffled batch
    replay_shuffled = update_incremental(
        db,
        UpdateRequest(
            batch=b_shuffled,
            expected_revision=1,
        ),
        segmentation_report=seg_res.report,
        feature_values={},
        settings=settings,
    )
    assert replay_shuffled.revision == 1
    assert replay_shuffled.data["changed"] is False
    assert replay_shuffled.data["summary"]["idempotent_replay"] is True
    state_after = get_identity_state(db, "m")
    assert state1 == state_after


def test_noop_does_not_advance_revision(tmp_path):
    """A batch that introduces no new evidence and no state change returns changed=False and keeps revision."""
    settings = make_incremental_settings(tmp_path)
    b1 = make_clip_batch("alpha", angle=0.0)
    b2 = make_clip_batch("beta", angle=0.0)
    comb = EvidenceBatch(
        match_id="m",
        batch_id="b-1",
        producer="synthetic",
        clips=[*b1.clips, *b2.clips],
        identity_evidence=IdentityEvidence(
            tracks=[*b1.identity_evidence.tracks, *b2.identity_evidence.tracks],
            observations=[*b1.identity_evidence.observations, *b2.identity_evidence.observations],
            segments=[*b1.identity_evidence.segments, *b2.identity_evidence.segments],
            samples=[*b1.identity_evidence.samples, *b2.identity_evidence.samples],
            feature_spaces=b1.identity_evidence.feature_spaces,
        ),
    )
    seg_res = segment_tracks(comb)
    db = connect(":memory:")

    first = update_incremental(
        db,
        UpdateRequest(batch=seg_res.batch, expected_revision=0),
        segmentation_report=seg_res.report,
        feature_values={},
        settings=settings,
    )
    assert first.revision == 1
    assert Repository(db).revision("m") == 1

    # Empty evidence batch on expected revision 1
    empty_batch = EvidenceBatch(
        match_id="m",
        batch_id="b-empty",
        producer="synthetic",
        clips=[],
        identity_evidence=IdentityEvidence(
            tracks=[],
            observations=[],
            segments=[],
            samples=[],
            feature_spaces=b1.identity_evidence.feature_spaces,
        ),
    )
    empty_seg = segment_tracks(empty_batch)
    noop_res = update_incremental(
        db,
        UpdateRequest(batch=empty_seg.batch, expected_revision=1),
        segmentation_report=empty_seg.report,
        feature_values={},
        settings=settings,
    )
    assert noop_res.revision == 1
    assert noop_res.data["changed"] is False
    assert Repository(db).revision("m") == 1


def test_template_evidence_chain_and_minimum_profile_requirements(tmp_path):
    """Verifies template admission: automatic_experimental, approval=None, cross-clip multi-witnesses,
    resolved support_pair_ids, deduplication within capacity, and no empty player."""
    from app.storage.discovery import read_discovery_reports

    settings = make_incremental_settings(tmp_path, max_templates_per_group=3)
    b1 = make_clip_batch("alpha", angle=0.0)
    b2 = make_clip_batch("beta", angle=0.0)
    comb = EvidenceBatch(
        match_id="m",
        batch_id="b-1",
        producer="synthetic",
        clips=[*b1.clips, *b2.clips],
        identity_evidence=IdentityEvidence(
            tracks=[*b1.identity_evidence.tracks, *b2.identity_evidence.tracks],
            observations=[*b1.identity_evidence.observations, *b2.identity_evidence.observations],
            segments=[*b1.identity_evidence.segments, *b2.identity_evidence.segments],
            samples=[*b1.identity_evidence.samples, *b2.identity_evidence.samples],
            feature_spaces=b1.identity_evidence.feature_spaces,
        ),
    )
    seg_res = segment_tracks(comb)
    db = connect(":memory:")

    update_incremental(
        db,
        UpdateRequest(batch=seg_res.batch, expected_revision=0),
        segmentation_report=seg_res.report,
        feature_values={},
        settings=settings,
    )
    state = get_identity_state(db, "m")
    assert len(state.data.players) == 1
    created_player = state.data.players[0]

    # Minimum profile evidence requirement: player MUST NOT be empty!
    player_templates = [t for t in state.data.templates if t.player_id == created_player.player_id]
    assert len(player_templates) >= 2

    # Template fields verification
    for tpl in player_templates:
        assert tpl.admission == "automatic_experimental"
        assert tpl.review_status == "unreviewed"
        assert tpl.approval is None
        assert tpl.active is True
        assert tpl.revision == 1

    # Template evidence chain verification
    reports = read_discovery_reports(db, "m")
    assert len(reports) == 1
    report = reports[0]["report"]
    report_pairs_by_id = {p["pair_id"]: p for p in report["pairs"]}
    group = report["groups"][0]

    for tpl_support in group["templates"]:
        # support_pair_ids must point to real supported DiscoveryPairs
        assert len(tpl_support["support_pair_ids"]) >= 1
        for pair_id in tpl_support["support_pair_ids"]:
            assert pair_id in report_pairs_by_id
            assert report_pairs_by_id[pair_id]["status"] == "supported"

        # witnesses must be >= 2 raw witnesses from a different clip
        witness_ids = tpl_support["witness_sample_ids"]
        assert len(witness_ids) >= 2
        tpl_sample = next(
            s for s in state.data.evidence.samples if s.sample_id == tpl_support["sample_id"]
        )
        witness_samples = [s for s in state.data.evidence.samples if s.sample_id in witness_ids]
        for ws in witness_samples:
            assert ws.clip_id != tpl_sample.clip_id


def test_stale_segmentation_proposal_rejected(tmp_path):
    """Mismatched derived_evidence_fingerprint or tampered proposal is rejected as stale."""
    settings = make_incremental_settings(tmp_path)
    b = make_clip_batch("alpha", angle=0.0)
    seg = segment_tracks(b)
    db = connect(":memory:")

    # Case 1: Tampered derived_evidence_fingerprint
    stale_report = seg.report.model_copy(update={"derived_evidence_fingerprint": "0" * 64})
    with pytest.raises(ValueError, match="stale segmentation report: derived_evidence_fingerprint"):
        update_incremental(
            db,
            UpdateRequest(batch=seg.batch, expected_revision=0),
            segmentation_report=stale_report,
            feature_values={},
            settings=settings,
        )

    # Case 2: Tampered proposal segment definition
    tampered_tracks = []
    for t in seg.report.tracks:
        tampered_props = []
        for p in t.proposals:
            tampered_seg = p.segment.model_copy(update={"start_s": p.segment.start_s + 1.0})
            tampered_props.append(p.model_copy(update={"segment": tampered_seg}))
        tampered_tracks.append(t.model_copy(update={"proposals": tampered_props}))
    tampered_report = seg.report.model_copy(
        update={"tracks": tampered_tracks, "derived_evidence_fingerprint": None}
    )
    with pytest.raises(ValueError, match="stale segmentation proposal for segment"):
        update_incremental(
            db,
            UpdateRequest(batch=seg.batch, expected_revision=0),
            segmentation_report=tampered_report,
            feature_values={},
            settings=settings,
        )


def test_complete_summary_and_coverage_separation(tmp_path, monkeypatch):
    """Baseline summary aggregates all DB assignments, separates supported vs all-segment coverage,
    preserves unvalidated calibration and proposed automatic semantics."""
    from test_machine_input import make_machine_input

    from app.presentation import baseline
    from app.presentation.baseline import run_baseline

    root = make_machine_input(
        tmp_path / "input", clips=("alpha", "beta"), track_count=2, parts=2, dimension=4
    )
    settings = machine_settings(root, tmp_path)
    monkeypatch.setattr(baseline, "PROJECT_VAR", settings.paths.runtime_root)
    disc = {
        "version": "complete_support_v3",
        "support_mode": "constrained_manifold",
        "clustering_threshold": 0.28,
        "min_common_parts": 2,
        "min_part_visibility": 0.2,
        "max_templates_per_group": 6,
        "intra_segment_near_distance": 0.18,
        "far_distance": 0.45,
        "rival_margin": 0.04,
    }
    settings = settings.model_copy(update={"discovery": DiscoveryPolicy(**disc)})

    # Run baseline with incremental batches
    run_dir = run_baseline(
        settings,
        run_id="run-summary-test",
        incremental_batches=True,
        batch_size=1,
    )
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "predictions.json").exists()

    with (run_dir / "summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)

    # Requirement 4 metrics
    assert "matched_existing" in summary
    assert "created_new_identity" in summary
    assert "insufficient_evidence" in summary
    assert "players" in summary
    assert "active_templates" in summary
    assert "assigned_segments" in summary
    assert "unassigned_segments" in summary
    assert "deferred_reasons" in summary
    assert isinstance(summary["deferred_reasons"], dict)

    # Coverage metrics separated from accuracy
    assert "supported_segment_coverage" in summary
    assert "all_segment_coverage" in summary
    assert 0.0 <= summary["supported_segment_coverage"] <= 1.0
    assert 0.0 <= summary["all_segment_coverage"] <= 1.0
    assert "coverage_note" in summary
    assert summary["identity_recognition_evaluated"] is False
    assert summary["calibration"] == "unvalidated"

    # Predictions check
    with (run_dir / "predictions.json").open("r", encoding="utf-8") as f:
        pred = json.load(f)
    assert pred["calibration"] == "unvalidated"
    assert pred["identity_status"] == "experimental"
    # All assignments in DB summarized
    assert len(pred["assignments"]) > 0
    for a in pred["assignments"]:
        if a["method"] == "automatic":
            assert a["status"] in {"proposed", "unassigned"}


def test_short_segment_incremental_matching_and_gating_regression(tmp_path):
    """Regression test for short segment incremental matching (2 buckets, can_match_existing=True).

    Verifies:
    1. A short segment with 2 buckets matching an existing gallery player within threshold is accepted
       as matched_existing (not blocked by min_support_buckets=3).
    2. A matched short segment never creates new players or spurious templates.
    3. A short segment with far distance from existing players is deferred as insufficient_evidence
       with short_segment_limited_match_support (never creates a new player).
    4. An explicitly configured short_segment_match_threshold gates the short segment match.
    """
    from app.contracts.segmentation import SegmentationPolicy

    settings = make_incremental_settings(tmp_path)
    db = connect(":memory:")

    # 1. Seed gallery with a cross-clip player from full segments (clips alpha & beta)
    b1 = make_clip_batch("alpha", angle=0.0)
    b2 = make_clip_batch("beta", angle=0.0)
    comb = EvidenceBatch(
        match_id="m",
        batch_id="b-1",
        producer="synthetic",
        clips=[*b1.clips, *b2.clips],
        identity_evidence=IdentityEvidence(
            observations=[*b1.identity_evidence.observations, *b2.identity_evidence.observations],
            tracks=[*b1.identity_evidence.tracks, *b2.identity_evidence.tracks],
            segments=[*b1.identity_evidence.segments, *b2.identity_evidence.segments],
            samples=[*b1.identity_evidence.samples, *b2.identity_evidence.samples],
            feature_spaces=b1.identity_evidence.feature_spaces,
        ),
    )
    seg1 = segment_tracks(comb)
    res1 = update_incremental(
        db,
        UpdateRequest(batch=seg1.batch, expected_revision=0),
        segmentation_report=seg1.report,
        feature_values={},
        settings=settings,
    )
    p1_id = res1.data["plan"]["new_players"][0]["player_id"]

    # 2. Incoming short segment (gamma, 2 samples, 2 buckets, angle=0.0 near p1)
    b_short = make_clip_batch("gamma", angle=0.0)
    b_short.identity_evidence.samples = b_short.identity_evidence.samples[:2]
    b_short.identity_evidence.observations = b_short.identity_evidence.observations[:7]
    end_t = 0.7
    b_short.identity_evidence.segments[0].end_s = end_t
    b_short.identity_evidence.segments[0].frame_end_exclusive = 7
    b_short.identity_evidence.tracks[0].end_s = end_t
    b_short.identity_evidence.tracks[0].frame_end_exclusive = 7
    b_short.identity_evidence.tracks[0].observed_frame_indices = [
        o.frame_idx for o in b_short.identity_evidence.observations
    ]
    b_short.clips[0].duration_s = end_t
    b_short.clips[0].frame_count = 7

    p_v3 = SegmentationPolicy(version="track-consistency-v3")
    seg_short = segment_tracks(b_short, policy=p_v3)
    p0 = seg_short.report.tracks[0].proposals[0]
    assert p0.status == "supported"
    assert p0.can_match_existing is True
    assert p0.can_create_new_identity is False

    res2 = update_incremental(
        db,
        UpdateRequest(batch=seg_short.batch, expected_revision=res1.revision),
        segmentation_report=seg_short.report,
        feature_values={},
        settings=settings,
    )
    plan2 = res2.data["plan"]
    # Must match existing player p1_id!
    matched_decisions = [d for d in plan2["decisions"] if d["category"] == "matched_existing"]
    assert len(matched_decisions) >= 1
    assert matched_decisions[0]["player_id"] == p1_id
    assert matched_decisions[0]["matched_existing_player_id"] == p1_id
    # Must NOT create new player!
    assert len(plan2["new_players"]) == 0

    # 3. Incoming short segment with far angle (delta, angle=2.5 far from p1)
    b_far = make_clip_batch("delta", angle=2.5)
    b_far.identity_evidence.samples = b_far.identity_evidence.samples[:2]
    b_far.identity_evidence.observations = b_far.identity_evidence.observations[:7]
    b_far.identity_evidence.segments[0].end_s = end_t
    b_far.identity_evidence.segments[0].frame_end_exclusive = 7
    b_far.identity_evidence.tracks[0].end_s = end_t
    b_far.identity_evidence.tracks[0].frame_end_exclusive = 7
    b_far.identity_evidence.tracks[0].observed_frame_indices = [
        o.frame_idx for o in b_far.identity_evidence.observations
    ]
    b_far.clips[0].duration_s = end_t
    b_far.clips[0].frame_count = 7

    seg_far = segment_tracks(b_far, policy=p_v3)
    res3 = update_incremental(
        db,
        UpdateRequest(batch=seg_far.batch, expected_revision=res2.revision),
        segmentation_report=seg_far.report,
        feature_values={},
        settings=settings,
    )
    plan3 = res3.data["plan"]
    # Far short segment must be deferred as insufficient_evidence, never create a new player!
    assert all(d["category"] == "insufficient_evidence" for d in plan3["decisions"])
    assert len(plan3["new_players"]) == 0

    # 4. Gating threshold: setting short_segment_match_threshold=0.001 gates the match
    settings_strict = make_incremental_settings(tmp_path, short_segment_match_threshold=0.001)
    b_slight = make_clip_batch("epsilon", angle=0.05)
    b_slight.identity_evidence.samples = b_slight.identity_evidence.samples[:2]
    b_slight.identity_evidence.observations = b_slight.identity_evidence.observations[:7]
    b_slight.identity_evidence.segments[0].end_s = end_t
    b_slight.identity_evidence.segments[0].frame_end_exclusive = 7
    b_slight.identity_evidence.tracks[0].end_s = end_t
    b_slight.identity_evidence.tracks[0].frame_end_exclusive = 7
    b_slight.identity_evidence.tracks[0].observed_frame_indices = [
        o.frame_idx for o in b_slight.identity_evidence.observations
    ]
    b_slight.clips[0].duration_s = end_t
    b_slight.clips[0].frame_count = 7

    seg_slight = segment_tracks(b_slight, policy=p_v3)
    res4 = update_incremental(
        db,
        UpdateRequest(batch=seg_slight.batch, expected_revision=res3.revision),
        segmentation_report=seg_slight.report,
        feature_values={},
        settings=settings_strict,
    )
    plan4 = res4.data["plan"]
    assert all(d["category"] == "insufficient_evidence" for d in plan4["decisions"])
    assert len(plan4["new_players"]) == 0


def test_stage1_interrupted_recovery_and_idempotency(tmp_path, monkeypatch):
    """Stage 1: Verify interrupted reassessment cleanly resumes on replay to identical state."""
    from app.storage.incremental_operations import read_incremental_operations
    from app.workflows import incremental as inc_module

    settings = make_incremental_settings(tmp_path)

    # 1. Prepare 3 batches:
    # Batch 1: Player A appears in clip alpha and beta (angle=0.0) -> forms Player 1
    b1_a = make_clip_batch("alpha", angle=0.0)
    b1_b = make_clip_batch("beta", angle=0.0)
    comb1 = EvidenceBatch(
        match_id="m",
        batch_id="b-1",
        producer="synthetic",
        clips=[*b1_a.clips, *b1_b.clips],
        identity_evidence=IdentityEvidence(
            tracks=[*b1_a.identity_evidence.tracks, *b1_b.identity_evidence.tracks],
            observations=[
                *b1_a.identity_evidence.observations,
                *b1_b.identity_evidence.observations,
            ],
            segments=[*b1_a.identity_evidence.segments, *b1_b.identity_evidence.segments],
            samples=[*b1_a.identity_evidence.samples, *b1_b.identity_evidence.samples],
            feature_spaces=b1_a.identity_evidence.feature_spaces,
        ),
    )
    seg1 = segment_tracks(comb1)

    # Batch 2: Player A absent; single clip gamma with Person Y (angle=2.0) -> deferred as unassigned
    b2 = make_clip_batch("gamma", angle=2.0)
    seg2 = segment_tracks(b2)

    # Batch 3: Person Y returns in clip delta (angle=2.0) -> matches Person Y in Step B, creates Player 2
    b3 = make_clip_batch("delta", angle=2.0)
    seg3 = segment_tracks(b3)

    # --- Run A: Normal uninterrupted execution ---
    db_normal = connect(":memory:")
    r1_norm = update_incremental(
        db_normal,
        UpdateRequest(batch=seg1.batch, expected_revision=0),
        segmentation_report=seg1.report,
        feature_values={},
        settings=settings,
    )
    r2_norm = update_incremental(
        db_normal,
        UpdateRequest(batch=seg2.batch, expected_revision=r1_norm.revision),
        segmentation_report=seg2.report,
        feature_values={},
        settings=settings,
    )
    r3_norm = update_incremental(
        db_normal,
        UpdateRequest(batch=seg3.batch, expected_revision=r2_norm.revision),
        segmentation_report=seg3.report,
        feature_values={},
        settings=settings,
    )
    assert r3_norm.data["changed"] is True
    state_norm = get_identity_state(db_normal, "m").data

    # --- Run B: Interrupted execution on Batch 3 during reassessment ---
    db_interrupted = connect(":memory:")
    r1_int = update_incremental(
        db_interrupted,
        UpdateRequest(batch=seg1.batch, expected_revision=0),
        segmentation_report=seg1.report,
        feature_values={},
        settings=settings,
    )
    r2_int = update_incremental(
        db_interrupted,
        UpdateRequest(batch=seg2.batch, expected_revision=r1_int.revision),
        segmentation_report=seg2.report,
        feature_values={},
        settings=settings,
    )

    # Simulate interruption inside reassess_unassigned
    real_reassess = inc_module.reassess_unassigned
    interrupted_once = False

    def fault_reassess(*args, **kwargs):
        nonlocal interrupted_once
        if not interrupted_once:
            interrupted_once = True
            raise RuntimeError("simulated network loss / process crash during reassessment pass")
        return real_reassess(*args, **kwargs)

    monkeypatch.setattr(inc_module, "reassess_unassigned", fault_reassess)

    with pytest.raises(RuntimeError, match="simulated network loss"):
        update_incremental(
            db_interrupted,
            UpdateRequest(batch=seg3.batch, expected_revision=r2_int.revision),
            segmentation_report=seg3.report,
            feature_values={},
            settings=settings,
        )

    # Verify primary commit succeeded and recorded incremental_primary operation
    ops = read_incremental_operations(db_interrupted, "m")
    primary_ops = [op for op in ops if op["phase"] == "incremental_primary"]
    assert len(primary_ops) == 3  # b-1, b-2, b-3 primary passes all recorded
    complete_ops = [op for op in ops if op["phase"] == "complete"]
    assert len(complete_ops) == 2  # b-1 and b-2 completed, b-3 not yet complete!

    # Replay Batch 3 (at current committed revision): must cleanly resume and complete!
    repo_int = Repository(db_interrupted)
    current_rev_int = repo_int.revision("m")
    r3_resumed = update_incremental(
        db_interrupted,
        UpdateRequest(batch=seg3.batch, expected_revision=current_rev_int),
        segmentation_report=seg3.report,
        feature_values={},
        settings=settings,
    )

    state_resumed = get_identity_state(db_interrupted, "m").data

    # Both runs reach 100% IDENTICAL gallery state!
    assert len(state_resumed.players) == len(state_norm.players) == 2
    assert len(state_resumed.assignments) == len(state_norm.assignments) == 4
    assert {a.segment_id: a.status for a in state_resumed.assignments} == {
        a.segment_id: a.status for a in state_norm.assignments
    }
    assert {a.segment_id: a.player_id for a in state_resumed.assignments} == {
        a.segment_id: a.player_id for a in state_norm.assignments
    }
    assert len(state_resumed.templates) == len(state_norm.templates)

    # Idempotent re-submission after completion returns cached response with changed=False
    r3_idempotent = update_incremental(
        db_interrupted,
        UpdateRequest(batch=seg3.batch, expected_revision=r3_resumed.revision),
        segmentation_report=seg3.report,
        feature_values={},
        settings=settings,
    )
    assert r3_idempotent.data["changed"] is False
    assert r3_idempotent.data["summary"]["idempotent_replay"] is True
    assert r3_idempotent.revision == r3_resumed.revision


def _short_clip_recovery_batch():
    """Two full clips create a player in the primary pass; a short third clip can
    only match in the association-only reassessment, so reassessment must commit."""
    from test_stage5_unified_incremental import _combine_clip_batches

    from app.contracts.segmentation import SegmentationPolicy

    full = [make_clip_batch("c0"), make_clip_batch("c1")]
    short = make_clip_batch("c2")
    short = short.model_copy(
        update={
            "identity_evidence": short.identity_evidence.model_copy(
                update={"samples": short.identity_evidence.samples[:2]}
            )
        }
    )
    batch = _combine_clip_batches([*full, short], "synthetic-recovery-probe")
    return segment_tracks(batch, policy=SegmentationPolicy(version="track-consistency-v3"))


def test_stage1_reassessment_committed_then_completion_fails_resumes(tmp_path, monkeypatch):
    """Scenario c: reassessment commits (revision advances) but the completion record
    write fails. Ordinary replay must reuse the committed reassessment, not re-run it
    at the stale primary revision (which raised RevisionConflict)."""
    from app.storage.incremental_operations import read_incremental_operations
    from app.workflows import incremental as inc_module

    settings = make_incremental_settings(tmp_path)
    seg = _short_clip_recovery_batch()
    request = UpdateRequest(batch=seg.batch, expected_revision=0)

    healthy = connect(":memory:")
    h = update_incremental(
        healthy, request, segmentation_report=seg.report, feature_values={}, settings=settings
    )
    healthy_assign = {
        a.segment_id: a.player_id for a in get_identity_state(healthy, "m").data.assignments
    }
    assert h.revision == 2  # primary rev 1, reassessment rev 2

    db = connect(":memory:")
    real_put = inc_module.put_incremental_operation
    failed_once = False

    def fault_put(connection, match_id, operation_key, phase, *args, **kwargs):
        nonlocal failed_once
        if phase == "complete" and not failed_once:
            failed_once = True
            raise RuntimeError("crash after reassessment commit, before completion record")
        return real_put(connection, match_id, operation_key, phase, *args, **kwargs)

    monkeypatch.setattr(inc_module, "put_incremental_operation", fault_put)
    with pytest.raises(RuntimeError, match="crash after reassessment commit"):
        update_incremental(
            db, request, segmentation_report=seg.report, feature_values={}, settings=settings
        )

    ops = read_incremental_operations(db, "m")
    assert [o["phase"] for o in ops].count("incremental_reassessment") == 1
    assert not [o for o in ops if o["phase"] == "complete"]
    current = Repository(db).revision("m")
    assert current == 2  # reassessment already advanced the revision

    monkeypatch.undo()
    resumed = update_incremental(
        db,
        UpdateRequest(batch=seg.batch, expected_revision=current),
        segmentation_report=seg.report,
        feature_values={},
        settings=settings,
    )
    resumed_assign = {
        a.segment_id: a.player_id for a in get_identity_state(db, "m").data.assignments
    }
    assert resumed_assign == healthy_assign
    assert Repository(db).revision("m") == 2

    # A further duplicate replay is idempotent and returns the same final revision.
    again = update_incremental(
        db,
        UpdateRequest(batch=seg.batch, expected_revision=resumed.revision),
        segmentation_report=seg.report,
        feature_values={},
        settings=settings,
    )
    assert again.data["changed"] is False
    assert again.revision == 2


def test_stage1_resume_completion_after_later_batch_committed(tmp_path, monkeypatch):
    """A stranded completion record can be written even after a later batch advanced
    the match revision; the operation row is keyed by evidence, not by current revision."""
    from app.workflows import incremental as inc_module

    settings = make_incremental_settings(tmp_path)
    seg = _short_clip_recovery_batch()
    db = connect(":memory:")
    real_put = inc_module.put_incremental_operation
    failed_once = False

    def fault_put(connection, match_id, operation_key, phase, *args, **kwargs):
        nonlocal failed_once
        if phase == "complete" and not failed_once:
            failed_once = True
            raise RuntimeError("crash before completion record")
        return real_put(connection, match_id, operation_key, phase, *args, **kwargs)

    monkeypatch.setattr(inc_module, "put_incremental_operation", fault_put)
    with pytest.raises(RuntimeError, match="crash before completion record"):
        update_incremental(
            db,
            UpdateRequest(batch=seg.batch, expected_revision=0),
            segmentation_report=seg.report,
            feature_values={},
            settings=settings,
        )
    monkeypatch.undo()
    stranded_rev = Repository(db).revision("m")

    # A later, independent batch commits and advances the revision further.
    later = segment_tracks(make_clip_batch("c9", angle=2.5))
    update_incremental(
        db,
        UpdateRequest(batch=later.batch, expected_revision=stranded_rev),
        segmentation_report=later.report,
        feature_values={},
        settings=settings,
    )
    assert Repository(db).revision("m") > stranded_rev

    # Replaying the stranded batch recovers its own definite result without conflict.
    resumed = update_incremental(
        db,
        UpdateRequest(batch=seg.batch, expected_revision=Repository(db).revision("m")),
        segmentation_report=seg.report,
        feature_values={},
        settings=settings,
    )
    assert resumed.revision == stranded_rev
    ops = inc_module.get_incremental_operation(
        db, "m", inc_module.evidence_fingerprint(seg.batch, {}), "complete"
    )
    assert ops is not None and ops["committed_revision"] == stranded_rev


def test_stage1_lost_response_and_duplicate_request_are_idempotent(tmp_path):
    """A committed request whose response was lost, and an outright duplicate request,
    both return the cached definite result with changed=False and no revision advance."""
    settings = make_incremental_settings(tmp_path)
    seg = _short_clip_recovery_batch()
    db = connect(":memory:")
    first = update_incremental(
        db,
        UpdateRequest(batch=seg.batch, expected_revision=0),
        segmentation_report=seg.report,
        feature_values={},
        settings=settings,
    )
    state_first = get_identity_state(db, "m").data
    # Lost response: caller resubmits at the original expected revision 0.
    lost = update_incremental(
        db,
        UpdateRequest(batch=seg.batch, expected_revision=0),
        segmentation_report=seg.report,
        feature_values={},
        settings=settings,
    )
    assert lost.data["changed"] is False
    assert lost.data["summary"]["idempotent_replay"] is True
    assert lost.revision == first.revision
    # Duplicate request at the current revision.
    dup = update_incremental(
        db,
        UpdateRequest(batch=seg.batch, expected_revision=first.revision),
        segmentation_report=seg.report,
        feature_values={},
        settings=settings,
    )
    assert dup.data["changed"] is False
    assert dup.revision == first.revision
    state_after = get_identity_state(db, "m").data
    assert {a.segment_id: a.player_id for a in state_after.assignments} == {
        a.segment_id: a.player_id for a in state_first.assignments
    }


def _v3_segment(batch):
    from app.contracts.segmentation import SegmentationPolicy

    return segment_tracks(batch, policy=SegmentationPolicy(version="track-consistency-v3"))


def _two_clip_player(clip_a, clip_b, angle, name):
    from test_stage5_unified_incremental import _combine_clip_batches

    comb = _combine_clip_batches(
        [make_clip_batch(clip_a, angle), make_clip_batch(clip_b, angle)], name
    )
    return _v3_segment(comb)


def test_stage1_machine_tentative_is_reevaluated_not_frozen(tmp_path):
    """Req 3: a historical machine-tentative (proposed/automatic) assignment is
    re-evaluated when new evidence arrives, instead of being frozen forever."""
    settings = make_incremental_settings(tmp_path)
    db = connect(":memory:")
    seg1 = _two_clip_player("alpha", "beta", 0.0, "b1")
    res1 = update_incremental(
        db,
        UpdateRequest(batch=seg1.batch, expected_revision=0),
        segmentation_report=seg1.report,
        feature_values={},
        settings=settings,
    )
    state1 = get_identity_state(db, "m").data
    tentative = {
        a.segment_id: a.player_id
        for a in state1.assignments
        if a.status == "proposed" and a.method == "automatic" and a.player_id
    }
    assert tentative, "first cross-clip batch must create tentative assignments"

    # New evidence (gamma, same appearance) arrives; historical tentative is re-assessed.
    seg2 = _v3_segment(make_clip_batch("gamma", 0.0))
    res2 = update_incremental(
        db,
        UpdateRequest(batch=seg2.batch, expected_revision=res1.revision),
        segmentation_report=seg2.report,
        feature_values={},
        settings=settings,
    )
    diag = res2.data["plan"]["matching_diagnostics"]
    reassessed = {sid for sid in tentative if sid in diag and "reassessment" in diag[sid]}
    assert reassessed, "historical tentative segments must be re-evaluated, not frozen"
    for sid in reassessed:
        assert diag[sid]["reassessment"]["leave_one_out"] is True
        assert diag[sid]["reassessment"]["outcome"] in {"kept", "reassigned"}
    # Same appearance: no better player, so every tentative keeps its player (never revoked).
    state2 = get_identity_state(db, "m").data
    after = {a.segment_id: a.player_id for a in state2.assignments}
    for sid, pid in tentative.items():
        assert after[sid] == pid


def test_stage1_manual_confirmed_assignment_never_auto_changed(tmp_path):
    """Req 2: a human-confirmed assignment outranks automatic re-evaluation and is
    preserved verbatim (player, status, method and revision unchanged)."""
    settings = make_incremental_settings(tmp_path)
    db = connect(":memory:")
    seg1 = _two_clip_player("alpha", "beta", 0.0, "b1")
    res1 = update_incremental(
        db,
        UpdateRequest(batch=seg1.batch, expected_revision=0),
        segmentation_report=seg1.report,
        feature_values={},
        settings=settings,
    )
    state1 = get_identity_state(db, "m").data
    target = next(a for a in state1.assignments if a.status == "proposed" and a.player_id)
    apply_review(
        db,
        ReviewRequest(
            match_id="m",
            expected_revision=res1.revision,
            actor="supervisor",
            commands=[
                SetAssignmentCommand(
                    segment_id=target.segment_id,
                    player_id=target.player_id,
                    reason="manual_confirm",
                )
            ],
        ),
    )
    rev_after_review = Repository(db).revision("m")
    before = next(
        a for a in get_identity_state(db, "m").data.assignments if a.segment_id == target.segment_id
    )
    assert before.status == "confirmed" and before.method == "manual"

    seg2 = _v3_segment(make_clip_batch("gamma", 0.0))
    update_incremental(
        db,
        UpdateRequest(batch=seg2.batch, expected_revision=rev_after_review),
        segmentation_report=seg2.report,
        feature_values={},
        settings=settings,
    )
    after = next(
        a for a in get_identity_state(db, "m").data.assignments if a.segment_id == target.segment_id
    )
    assert after.player_id == before.player_id
    assert after.status == "confirmed"
    assert after.method == "manual"
    assert after.revision == before.revision


def test_stage1_tentative_never_auto_revoked_when_unmatched(tmp_path):
    """Req 3: re-evaluation never revokes a tentative assignment to unassigned, even
    when new evidence forms a separate far player and offers no better match."""
    settings = make_incremental_settings(tmp_path)
    db = connect(":memory:")
    seg1 = _two_clip_player("alpha", "beta", 0.0, "b1")
    res1 = update_incremental(
        db,
        UpdateRequest(batch=seg1.batch, expected_revision=0),
        segmentation_report=seg1.report,
        feature_values={},
        settings=settings,
    )
    before = {
        a.segment_id: a.player_id
        for a in get_identity_state(db, "m").data.assignments
        if a.status == "proposed" and a.player_id
    }
    assert before
    # A far person forms a distinct player; the original tentative segments must survive.
    seg2 = _two_clip_player("gamma", "delta", 2.0, "b2")
    update_incremental(
        db,
        UpdateRequest(batch=seg2.batch, expected_revision=res1.revision),
        segmentation_report=seg2.report,
        feature_values={},
        settings=settings,
    )
    state2 = get_identity_state(db, "m").data
    after = {a.segment_id: a for a in state2.assignments}
    for sid, pid in before.items():
        assert after[sid].player_id == pid
        assert after[sid].status == "proposed"


def test_stage1_reevaluation_is_stable_across_batches(tmp_path):
    """Req 3: repeated re-evaluation converges; the identity partition does not
    oscillate as more same-appearance evidence arrives."""
    settings = make_incremental_settings(tmp_path)
    db = connect(":memory:")
    seg1 = _two_clip_player("alpha", "beta", 0.0, "b1")
    res = update_incremental(
        db,
        UpdateRequest(batch=seg1.batch, expected_revision=0),
        segmentation_report=seg1.report,
        feature_values={},
        settings=settings,
    )
    partitions = [identity_partition(get_identity_state(db, "m").data.assignments)]
    for i, clip in enumerate(["gamma", "delta", "epsilon"], start=2):
        seg = _v3_segment(make_clip_batch(clip, 0.0))
        res = update_incremental(
            db,
            UpdateRequest(batch=seg.batch, expected_revision=res.revision),
            segmentation_report=seg.report,
            feature_values={},
            settings=settings,
        )
        partitions.append(identity_partition(get_identity_state(db, "m").data.assignments))
    # All same-appearance clips belong to one stable player; no fragmentation/oscillation.
    assert all(len(p) == 1 for p in partitions)
    final = get_identity_state(db, "m").data
    assigned = [a for a in final.assignments if a.player_id]
    assert len({a.player_id for a in assigned}) == 1


def test_stage1_tentative_reassignment_is_versioned_and_deactivates_stale_templates(tmp_path):
    """Req 2/3: when leave-one-out matching finds a better existing player, a machine
    tentative assignment is reassigned with a versioned decision (revision+1, superseded
    player evidence), the old player's stale templates from that segment are deactivated,
    and the persistent player ID is reused (no new identity is created).

    The gallery is split into A (beta-only templates) and B (gamma+delta templates) so
    that re-evaluating the beta segment empties A under leave-one-out while B's independent
    samples still match it. This isolates the re-evaluation mechanism from discovery gates.
    """
    from dataclasses import replace

    from test_stage5_unified_incremental import _combine_clip_batches

    from app.contracts.identity import Player
    from app.identity.incremental import plan_incremental_identities
    from app.storage.snapshot import read_identity_snapshot

    settings = make_incremental_settings(tmp_path)
    db = connect(":memory:")
    comb = _combine_clip_batches(
        [make_clip_batch(c, 0.0) for c in ("beta", "gamma", "delta")], "b1"
    )
    seg1 = _v3_segment(comb)
    update_incremental(
        db,
        UpdateRequest(batch=seg1.batch, expected_revision=0),
        segmentation_report=seg1.report,
        feature_values={},
        settings=settings,
    )
    snap = read_identity_snapshot(db, "m")
    g = snap.gallery
    sid_by_sample = {s.sample_id: s.segment_id for s in g.evidence.samples}
    s_beta = next(s.segment_id for s in g.evidence.segments if s.clip_id == "beta" and s.active)

    a, b = "pl_aaa", "pl_bbb"
    split_templates = [
        t.model_copy(update={"player_id": a if sid_by_sample.get(t.sample_id) == s_beta else b})
        for t in g.templates
    ]
    split_assignments = [
        asn.model_copy(update={"player_id": a if asn.segment_id == s_beta else b})
        for asn in g.assignments
    ]
    crafted = g.model_copy(
        update={
            "players": [Player(player_id=a, match_id="m"), Player(player_id=b, match_id="m")],
            "templates": split_templates,
            "assignments": split_assignments,
        }
    )
    snap2 = replace(snap, gallery=crafted)
    stale_a_templates = [
        t.template_id
        for t in split_templates
        if t.player_id == a and sid_by_sample.get(t.sample_id) == s_beta
    ]
    assert stale_a_templates, "A must own the beta segment's templates before re-evaluation"

    seg_beta = _v3_segment(make_clip_batch("beta", 0.0))
    plan = plan_incremental_identities(
        snap2,
        seg_beta.batch,
        segmentation_report=seg_beta.report,
        feature_values=snap.feature_values,
        policy=settings.discovery,
        constraints=snap.constraints,
    )

    # Persistent ID reuse: the better player B already exists; no new identity is created.
    assert not plan.new_players
    decision = next(d for d in plan.decisions if d.segment_id == s_beta)
    assert decision.category == "matched_existing"
    assert decision.player_id == b
    assert decision.matched_existing_player_id == b
    reassigned = decision.assignment
    assert reassigned.player_id == b
    assert reassigned.status == "proposed"
    assert reassigned.method == "automatic"
    assert reassigned.revision == g.revision + 1
    kinds = {e.kind: e for e in reassigned.evidence}
    assert "automatic_reassessment_reassignment" in kinds
    assert kinds["automatic_reassessment_superseded_player"].reference_id == a

    # The old player's stale templates contributed by this segment are deactivated.
    assert set(stale_a_templates).issubset(set(plan.deactivated_template_ids))

    diag = plan.matching_diagnostics[s_beta]["reassessment"]
    assert diag["outcome"] == "reassigned"
    assert diag["leave_one_out"] is True
    assert diag["current_player_id"] == a
    assert diag["new_player_id"] == b
    assert plan.summary["reassigned_tentative_count"] >= 1
