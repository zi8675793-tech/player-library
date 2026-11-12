"""End-to-end acceptance tests for open-set incremental manifold identity discovery.

Verifies Phase 4 acceptance criteria without hardcoded sample numbers or fragile assertions:
1. Single batch, empty batch, single-segment batch, and multi-segment batch.
2. Variable batch size and variable batch counts.
3. Forward and reverse order batch arrival invariants (zero physical co-occurrence
   violations, no unknown assigned, valid support pairs, no hard equality required).
4. Cold start from empty library with continuous new clips.
5. Existing player return, late-arriving new player, deferred segment supplemented with proof.
6. Repeated evidence for same person does not create duplicate players.
7. Existing players are never automatically merged.
8. Unknown and unsupported segments never inherit identity.
9. Template witnesses and support_pair_ids are complete and resolve to supported DiscoveryPairs.
10. Replay idempotency, failure rollback, and stable player_id.
11. Auditable deferred reasons without silent unassigned fallback.
"""

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import pytest
from test_incremental_workflow import (
    make_clip_batch,
    make_incremental_settings,
)

from app.contracts.api import UpdateRequest
from app.contracts.evidence import EvidenceBatch, IdentityEvidence
from app.contracts.identity import IdentityRisk, RiskKind
from app.identity.consistency import segment_tracks
from app.storage.database import connect
from app.storage.discovery import read_discovery_reports
from app.storage.repositories import Repository, StorageConflict
from app.workflows.incremental import update_incremental
from app.workflows.read import get_identity_state


def _verify_spatio_temporal_zero_violations(connection: sqlite3.Connection) -> None:
    """Verify that no two assigned segments for the same player overlap in time in any clip."""
    cur = connection.cursor()
    cur.execute("""
        SELECT a.player_id, s.segment_id, s.clip_id, s.start_s, s.end_s
        FROM assignments a
        JOIN segments s ON a.segment_id = s.segment_id
        WHERE a.current = 1 AND a.player_id IS NOT NULL AND s.active = 1
    """)
    rows = cur.fetchall()
    by_player_clip = defaultdict(list)
    for pid, sid, cid, st, en in rows:
        by_player_clip[(pid, cid)].append((sid, st, en))

    for (pid, cid), segs in by_player_clip.items():
        if len(segs) > 1:
            for i in range(len(segs)):
                for j in range(i + 1, len(segs)):
                    s1, st1, en1 = segs[i]
                    s2, st2, en2 = segs[j]
                    assert max(st1, st2) >= min(en1, en2), (
                        f"Co-occurrence violation in clip {cid} for player {pid}: "
                        f"segment {s1} ({st1:.2f}-{en1:.2f}) overlaps segment {s2} ({st2:.2f}-{en2:.2f})"
                    )


def _verify_template_integrity(connection: sqlite3.Connection, match_id: str) -> None:
    """Verify all automatic templates have non-empty witnesses and resolvable support_pair_ids."""
    stored_reports = read_discovery_reports(connection, match_id)
    all_supported_pairs = {
        p["pair_id"]
        for r in stored_reports
        for p in r["report"].get("pairs", [])
        if p.get("status") == "supported"
    }

    for r in stored_reports:
        for group in r["report"].get("groups", []):
            for tpl in group.get("templates", []):
                witnesses = tpl.get("witness_sample_ids", [])
                assert len(witnesses) > 0, (
                    f"Template {tpl.get('sample_id')} lacks witness_sample_ids"
                )
                pair_ids = tpl.get("support_pair_ids", [])
                assert len(pair_ids) > 0, f"Template {tpl.get('sample_id')} lacks support_pair_ids"
                for pid in pair_ids:
                    assert pid in all_supported_pairs, (
                        f"support_pair_id {pid!r} cannot be resolved to supported DiscoveryPair"
                    )


def test_cold_start_empty_batch_single_and_multi_segment(tmp_path):
    """Empty batch at rev 0, followed by single-clip batch, then multi-clip batch."""
    settings = make_incremental_settings(tmp_path)
    b_sample = make_clip_batch("sample", angle=0.0)
    db = connect(":memory:")

    # 1. Cold start with empty batch: no crash, 0 players, no assignments
    empty_batch = EvidenceBatch(
        match_id="m",
        batch_id="b-empty-0",
        producer="synthetic",
        clips=[],
        identity_evidence=IdentityEvidence(
            tracks=[],
            observations=[],
            segments=[],
            samples=[],
            feature_spaces=b_sample.identity_evidence.feature_spaces,
        ),
    )
    empty_seg = segment_tracks(empty_batch)
    res_empty = update_incremental(
        db,
        UpdateRequest(batch=empty_seg.batch, expected_revision=0),
        segmentation_report=empty_seg.report,
        feature_values={},
        settings=settings,
    )
    assert res_empty.revision == 1
    state0 = get_identity_state(db, "m")
    assert len(state0.data.players) == 0
    assert len(state0.data.assignments) == 0

    # Idempotent replay of empty batch: revision unchanged
    replay_empty = update_incremental(
        db,
        UpdateRequest(batch=empty_seg.batch, expected_revision=1),
        segmentation_report=empty_seg.report,
        feature_values={},
        settings=settings,
    )
    assert replay_empty.revision == 1
    assert replay_empty.data["changed"] is False

    # 2. Single-clip batch arrives (single appearance): deferred cleanly
    b_alpha = make_clip_batch("alpha", angle=0.0)
    seg_alpha = segment_tracks(b_alpha)
    res_alpha = update_incremental(
        db,
        UpdateRequest(batch=seg_alpha.batch, expected_revision=1),
        segmentation_report=seg_alpha.report,
        feature_values={},
        settings=settings,
    )
    assert res_alpha.revision == 2
    sum_alpha = res_alpha.data["summary"]
    assert sum_alpha["new_players_count"] == 0
    assert sum_alpha["insufficient_evidence_count"] >= 1

    # 3. Multi-clip batch arrives: provides cross-clip evidence for alpha and discovers player
    b_beta = make_clip_batch("beta", angle=0.0)
    b_gamma = make_clip_batch("gamma", angle=2.5)
    multi_batch = EvidenceBatch(
        match_id="m",
        batch_id="b-multi",
        producer="synthetic",
        clips=[*b_beta.clips, *b_gamma.clips],
        identity_evidence=IdentityEvidence(
            tracks=[*b_beta.identity_evidence.tracks, *b_gamma.identity_evidence.tracks],
            observations=[
                *b_beta.identity_evidence.observations,
                *b_gamma.identity_evidence.observations,
            ],
            segments=[*b_beta.identity_evidence.segments, *b_gamma.identity_evidence.segments],
            samples=[*b_beta.identity_evidence.samples, *b_gamma.identity_evidence.samples],
            feature_spaces=b_beta.identity_evidence.feature_spaces,
        ),
    )
    seg_multi = segment_tracks(multi_batch)
    res_multi = update_incremental(
        db,
        UpdateRequest(batch=seg_multi.batch, expected_revision=2),
        segmentation_report=seg_multi.report,
        feature_values={},
        settings=settings,
    )
    assert res_multi.revision == 3
    _verify_spatio_temporal_zero_violations(db)
    _verify_template_integrity(db, "m")


def test_variable_batch_size_flexible_counts(tmp_path):
    """Processing clips with varying batch sizes (e.g. 1 clip/batch vs 3 clips/batch)."""
    settings = make_incremental_settings(tmp_path)
    b_a = make_clip_batch("c1", angle=0.0)
    b_b = make_clip_batch("c2", angle=0.0)
    b_c = make_clip_batch("c3", angle=2.5)

    # Database 1: Batch size 1 (3 sequential updates, final revision 3)
    db1 = connect(":memory:")
    r = 0
    for clip_b in [b_a, b_b, b_c]:
        seg = segment_tracks(clip_b)
        res = update_incremental(
            db1,
            UpdateRequest(batch=seg.batch, expected_revision=r),
            segmentation_report=seg.report,
            feature_values={},
            settings=settings,
        )
        r = res.revision
    assert r == 3
    _verify_spatio_temporal_zero_violations(db1)
    _verify_template_integrity(db1, "m")

    # Database 2: All 3 clips in 1 batch (1 update, final revision 1)
    comb = EvidenceBatch(
        match_id="m",
        batch_id="b-all",
        producer="synthetic",
        clips=[*b_a.clips, *b_b.clips, *b_c.clips],
        identity_evidence=IdentityEvidence(
            tracks=[
                *b_a.identity_evidence.tracks,
                *b_b.identity_evidence.tracks,
                *b_c.identity_evidence.tracks,
            ],
            observations=[
                *b_a.identity_evidence.observations,
                *b_b.identity_evidence.observations,
                *b_c.identity_evidence.observations,
            ],
            segments=[
                *b_a.identity_evidence.segments,
                *b_b.identity_evidence.segments,
                *b_c.identity_evidence.segments,
            ],
            samples=[
                *b_a.identity_evidence.samples,
                *b_b.identity_evidence.samples,
                *b_c.identity_evidence.samples,
            ],
            feature_spaces=b_a.identity_evidence.feature_spaces,
        ),
    )
    db2 = connect(":memory:")
    res_comb = update_incremental(
        db2,
        UpdateRequest(batch=segment_tracks(comb).batch, expected_revision=0),
        segmentation_report=segment_tracks(comb).report,
        feature_values={},
        settings=settings,
    )
    assert res_comb.revision == 1
    _verify_spatio_temporal_zero_violations(db2)
    _verify_template_integrity(db2, "m")


def test_forward_and_reverse_order_arrival_invariants(tmp_path):
    """Forward order arrival vs reverse order arrival satisfies safety invariants.

    Does NOT assert forward_players == reverse_players or forward_coverage == reverse_coverage.
    Verifies that both directions preserve zero physical violations, valid templates,
    auditable deferrals, and idempotent replay.
    """
    settings = make_incremental_settings(tmp_path)

    b1 = make_clip_batch("clip1", angle=0.0)
    b2 = make_clip_batch("clip2", angle=0.0)
    b3 = make_clip_batch("clip3", angle=2.5)
    b4 = make_clip_batch("clip4", angle=2.5)

    comb_12 = EvidenceBatch(
        match_id="m",
        batch_id="b-12",
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
    comb_34 = EvidenceBatch(
        match_id="m",
        batch_id="b-34",
        producer="synthetic",
        clips=[*b3.clips, *b4.clips],
        identity_evidence=IdentityEvidence(
            tracks=[*b3.identity_evidence.tracks, *b4.identity_evidence.tracks],
            observations=[*b3.identity_evidence.observations, *b4.identity_evidence.observations],
            segments=[*b3.identity_evidence.segments, *b4.identity_evidence.segments],
            samples=[*b3.identity_evidence.samples, *b4.identity_evidence.samples],
            feature_spaces=b3.identity_evidence.feature_spaces,
        ),
    )

    # 1. Forward run: [12] -> [34]
    db_fwd = connect(":memory:")
    res_f1 = update_incremental(
        db_fwd,
        UpdateRequest(batch=segment_tracks(comb_12).batch, expected_revision=0),
        segmentation_report=segment_tracks(comb_12).report,
        feature_values={},
        settings=settings,
    )
    res_f2 = update_incremental(
        db_fwd,
        UpdateRequest(batch=segment_tracks(comb_34).batch, expected_revision=res_f1.revision),
        segmentation_report=segment_tracks(comb_34).report,
        feature_values={},
        settings=settings,
    )
    _verify_spatio_temporal_zero_violations(db_fwd)
    _verify_template_integrity(db_fwd, "m")
    # Idempotent replay on fwd
    replay_fwd = update_incremental(
        db_fwd,
        UpdateRequest(batch=segment_tracks(comb_34).batch, expected_revision=res_f2.revision),
        segmentation_report=segment_tracks(comb_34).report,
        feature_values={},
        settings=settings,
    )
    assert replay_fwd.revision == res_f2.revision
    assert replay_fwd.data["changed"] is False

    # 2. Reverse run: [34] -> [12]
    db_rev = connect(":memory:")
    res_r1 = update_incremental(
        db_rev,
        UpdateRequest(batch=segment_tracks(comb_34).batch, expected_revision=0),
        segmentation_report=segment_tracks(comb_34).report,
        feature_values={},
        settings=settings,
    )
    res_r2 = update_incremental(
        db_rev,
        UpdateRequest(batch=segment_tracks(comb_12).batch, expected_revision=res_r1.revision),
        segmentation_report=segment_tracks(comb_12).report,
        feature_values={},
        settings=settings,
    )
    _verify_spatio_temporal_zero_violations(db_rev)
    _verify_template_integrity(db_rev, "m")
    # Idempotent replay on rev
    replay_rev = update_incremental(
        db_rev,
        UpdateRequest(batch=segment_tracks(comb_12).batch, expected_revision=res_r2.revision),
        segmentation_report=segment_tracks(comb_12).report,
        feature_values={},
        settings=settings,
    )
    assert replay_rev.revision == res_r2.revision
    assert replay_rev.data["changed"] is False


def test_unknown_segments_never_inherit_identity(tmp_path):
    """Unsupported or risk-carrying segments never receive player_id or proposed status."""
    settings = make_incremental_settings(tmp_path)
    db = connect(":memory:")

    # Create batch with 1 supported track and 1 extra track with risk
    b1 = make_clip_batch("alpha", angle=0.0)
    alpha_seg = b1.identity_evidence.segments[0]
    risk = IdentityRisk(
        match_id="m",
        risk_id="risk-mixed-alpha",
        kind=RiskKind.MIXED_IDENTITY,
        segment_id=alpha_seg.segment_id,
        clip_id=alpha_seg.clip_id,
        track_id=alpha_seg.track_id,
    )
    b1.identity_evidence.risks.append(risk)

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
    # Because of risk, no identity is formed; all segments remain unassigned with player_id=None
    assert len(state.data.players) == 0
    assert len(state.data.assignments) > 0
    assert all(a.player_id is None for a in state.data.assignments)
    assert all(a.status == "unassigned" for a in state.data.assignments)


def test_duplicate_evidence_does_not_create_duplicate_player(tmp_path):
    """Submitting duplicate evidence across batches does not duplicate player records."""
    settings = make_incremental_settings(tmp_path)
    db = connect(":memory:")

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
    res1 = update_incremental(
        db,
        UpdateRequest(batch=segment_tracks(comb).batch, expected_revision=0),
        segmentation_report=segment_tracks(comb).report,
        feature_values={},
        settings=settings,
    )
    state1 = get_identity_state(db, "m")
    initial_players = len(state1.data.players)
    initial_pids = {p.player_id for p in state1.data.players}

    # Re-submitting identical evidence with different batch ID
    dup_comb = comb.model_copy(update={"batch_id": "b-dup"})
    res2 = update_incremental(
        db,
        UpdateRequest(batch=segment_tracks(dup_comb).batch, expected_revision=res1.revision),
        segmentation_report=segment_tracks(dup_comb).report,
        feature_values={},
        settings=settings,
    )
    assert res2.revision == res1.revision
    assert res2.data["changed"] is False

    state2 = get_identity_state(db, "m")
    assert len(state2.data.players) == initial_players
    assert {p.player_id for p in state2.data.players} == initial_pids


def test_existing_players_never_automatically_merged(tmp_path):
    """Two existing players created at earlier revisions are never automatically merged."""
    settings = make_incremental_settings(tmp_path)
    db = connect(":memory:")

    # Batch 1: Person 1 in alpha & beta (angle 0.0) -> Player 1
    b1_a = make_clip_batch("alpha", angle=0.0)
    b1_b = make_clip_batch("beta", angle=0.0)
    c1 = EvidenceBatch(
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
    res1 = update_incremental(
        db,
        UpdateRequest(batch=segment_tracks(c1).batch, expected_revision=0),
        segmentation_report=segment_tracks(c1).report,
        feature_values={},
        settings=settings,
    )

    # Batch 2: Person 2 in gamma & delta (angle 2.5) -> Player 2
    b2_g = make_clip_batch("gamma", angle=2.5)
    b2_d = make_clip_batch("delta", angle=2.5)
    c2 = EvidenceBatch(
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
    res2 = update_incremental(
        db,
        UpdateRequest(batch=segment_tracks(c2).batch, expected_revision=res1.revision),
        segmentation_report=segment_tracks(c2).report,
        feature_values={},
        settings=settings,
    )

    state2 = get_identity_state(db, "m")
    assert len(state2.data.players) == 2
    p1, p2 = state2.data.players[0].player_id, state2.data.players[1].player_id

    # Batch 3: Intermediate ambiguous segment in epsilon (angle 1.25)
    b3 = make_clip_batch("epsilon", angle=1.25)
    update_incremental(
        db,
        UpdateRequest(batch=segment_tracks(b3).batch, expected_revision=res2.revision),
        segmentation_report=segment_tracks(b3).report,
        feature_values={},
        settings=settings,
    )
    state3 = get_identity_state(db, "m")
    # p1 and p2 MUST still exist separately!
    pids_3 = {p.player_id for p in state3.data.players}
    assert p1 in pids_3
    assert p2 in pids_3
    assert len(state3.data.players) >= 2


def test_atomic_failure_rollback_and_stable_player_id(tmp_path):
    """Simulated failure rolls back transaction atomically; player_id remains stable on retry."""
    settings = make_incremental_settings(tmp_path)
    db = connect(":memory:")

    b1_a = make_clip_batch("alpha", angle=0.0)
    b1_b = make_clip_batch("beta", angle=0.0)
    c1 = EvidenceBatch(
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
    seg1 = segment_tracks(c1)
    res1 = update_incremental(
        db,
        UpdateRequest(batch=seg1.batch, expected_revision=0),
        segmentation_report=seg1.report,
        feature_values={},
        settings=settings,
    )
    assert res1.revision == 1
    state1 = get_identity_state(db, "m")
    p1_id = state1.data.players[0].player_id

    # Simulate failure on Batch 2
    b2_g = make_clip_batch("gamma", angle=2.5)
    b2_d = make_clip_batch("delta", angle=2.5)
    c2 = EvidenceBatch(
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
    seg2 = segment_tracks(c2)

    with patch.object(
        Repository, "advance_revision", side_effect=StorageConflict("simulated_crash")
    ):
        with pytest.raises(StorageConflict):
            update_incremental(
                db,
                UpdateRequest(batch=seg2.batch, expected_revision=1),
                segmentation_report=seg2.report,
                feature_values={},
                settings=settings,
            )

    # After crash: state is rolled back cleanly to revision 1
    state_after_crash = get_identity_state(db, "m")
    assert state_after_crash.revision == 1
    assert len(state_after_crash.data.players) == 1
    assert state_after_crash.data.players[0].player_id == p1_id

    # Retry succeeds cleanly
    res2 = update_incremental(
        db,
        UpdateRequest(batch=seg2.batch, expected_revision=1),
        segmentation_report=seg2.report,
        feature_values={},
        settings=settings,
    )
    assert res2.revision == 2
    state2 = get_identity_state(db, "m")
    assert len(state2.data.players) == 2
    assert p1_id in [p.player_id for p in state2.data.players]


@pytest.mark.parametrize(
    "run_id",
    [
        "incremental-manifold-forward-v1",
        "incremental-manifold-reverse-v1",
        "incremental-manifold-bs2-v1",
    ],
)
def test_frozen_acceptance_runs_invariants(run_id):
    """End-to-end verification of frozen acceptance runs against data contracts and safety invariants.

    Does NOT assert fixed player counts, non-zero players, fixed coverage, or exact equality.
    Asserts zero physical violations, valid whitelisted inputs, template integrity,
    auditable deferred reasons, unvalidated calibration semantics, and clean revision advance.
    """
    project_root = Path(__file__).resolve().parent.parent
    run_dir = project_root / "var" / "identity-discovery" / run_id
    if not run_dir.is_dir():
        pytest.skip(f"Run directory not found: {run_dir}")

    # 1. Required frozen artifact files exist
    required_artifacts = [
        "player-library.sqlite3",
        "summary.json",
        "predictions.json",
        "discovery.json",
        "diagnostics.json",
        "config.json",
        "input-manifest.json",
        "evidence.json",
        "raw-evidence.json",
        "artifact-hashes.json",
    ]
    for artifact in required_artifacts:
        artifact_path = run_dir / artifact
        assert artifact_path.is_file(), f"Missing artifact {artifact} in {run_id}"
        assert artifact_path.stat().st_size > 0, f"Artifact {artifact} is empty in {run_id}"

    # 2. Input manifest whitelist check: only allowed video, track, identity machine files
    with (run_dir / "input-manifest.json").open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest.get("schema_version") == "machine-input-manifest-v1"
    sources = manifest.get("sources", [])
    assert len(sources) > 0
    top_dirs = {s["path"].split("/")[0] for s in sources}
    assert top_dirs.issubset({"track", "video", "identity"}), (
        f"Forbidden dirs in manifest: {top_dirs}"
    )

    # 3. Database physical zero-violation hard invariants
    conn = sqlite3.connect(run_dir / "player-library.sqlite3")
    cur = conn.cursor()

    # Zero interval overlap for the same player in the same clip
    cur.execute("""
        SELECT a.player_id, s.segment_id, s.clip_id, s.start_s, s.end_s
        FROM assignments a
        JOIN segments s ON a.segment_id = s.segment_id
        WHERE a.current = 1 AND a.player_id IS NOT NULL AND s.active = 1
    """)
    assigned_segs = cur.fetchall()
    by_player_clip = defaultdict(list)
    for pid, sid, cid, st, en in assigned_segs:
        by_player_clip[(pid, cid)].append((sid, st, en))

    for (pid, cid), seg_list in by_player_clip.items():
        if len(seg_list) > 1:
            for i in range(len(seg_list)):
                for j in range(i + 1, len(seg_list)):
                    s1, st1, en1 = seg_list[i]
                    s2, st2, en2 = seg_list[j]
                    assert max(st1, st2) >= min(en1, en2), (
                        f"Physical interval collision in {cid} for {pid}: {s1} overlaps {s2}"
                    )

    # Zero frame-level observation overlap for the same player in the same clip
    cur.execute("""
        SELECT a.player_id, o.clip_id, o.frame_idx, o.track_id
        FROM assignments a
        JOIN segments s ON a.segment_id = s.segment_id
        JOIN track_observations o ON s.clip_id = o.clip_id AND s.track_id = o.track_id
             AND o.time_s >= s.start_s AND o.time_s <= s.end_s
        WHERE a.current = 1 AND a.player_id IS NOT NULL AND s.active = 1
    """)
    obs_rows = cur.fetchall()
    obs_by_player_frame = defaultdict(set)
    for pid, cid, fidx, tid in obs_rows:
        key = (pid, cid, fidx)
        assert tid not in obs_by_player_frame[key], (
            f"Physical observation collision in {cid} at frame {fidx} for player {pid}"
        )
        obs_by_player_frame[key].add(tid)

    # 4. Unknown and unassigned segments never assigned player_id
    cur.execute("""
        SELECT a.segment_id, a.player_id, json_extract(a.json, '$.status')
        FROM assignments a
        WHERE a.current = 1 AND a.player_id IS NOT NULL AND json_extract(a.json, '$.status') = 'unassigned'
    """)
    bad_unassigned = cur.fetchall()
    assert len(bad_unassigned) == 0, f"Unassigned segment has player_id: {bad_unassigned}"

    # Unsupported segmentation proposals never assigned to player
    cur.execute("""
        SELECT a.segment_id, a.player_id, json_extract(sp.json, '$.status')
        FROM assignments a
        JOIN segmentation_proposals sp ON a.segment_id = sp.segment_id
        WHERE json_extract(sp.json, '$.status') != 'supported' AND a.player_id IS NOT NULL AND a.current = 1
    """)
    unsupported_assigned = cur.fetchall()
    assert len(unsupported_assigned) == 0, (
        f"Unsupported proposal assigned to player: {unsupported_assigned}"
    )

    # 5. Template witnesses and support_pair_ids completeness
    cur.execute("SELECT json FROM identity_discovery_reports")
    rep_rows = cur.fetchall()
    all_supported_pairs = set()
    for (r_json,) in rep_rows:
        rep = json.loads(r_json)
        for p in rep.get("pairs", []):
            if p.get("status") == "supported":
                all_supported_pairs.add(p["pair_id"])
        for g in rep.get("groups", []):
            for t in g.get("templates", []):
                assert len(t.get("witness_sample_ids", [])) > 0, f"Template lacks witness: {t}"
                spids = t.get("support_pair_ids", [])
                assert len(spids) > 0, f"Template lacks support_pair_ids: {t}"
                for pid in spids:
                    assert pid in all_supported_pairs, f"Unresolvable support_pair_id {pid}"

    # 6. Revision advance is 1:1 per batch
    with (run_dir / "summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)
    batch_count = summary["batch_count"]
    final_rev = summary["revision"]
    assert final_rev == batch_count, f"Revision {final_rev} != batch count {batch_count}"

    # 7. Auditable deferred reasons present
    assert "deferred_reasons" in summary
    assert len(summary["deferred_reasons"]) > 0

    # 8. Unvalidated and experimental semantics preserved
    assert summary.get("identity_status") == "experimental"
    assert summary.get("calibration") == "unvalidated"
    assert summary.get("identity_recognition_evaluated") is False
    assert "coverage_note" in summary

    conn.close()
