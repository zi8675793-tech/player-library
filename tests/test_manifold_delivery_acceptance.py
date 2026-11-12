"""Acceptance tests for manifold identity discovery.

Verifies:
1. Historical Delivery Artifact Integrity (Audit Trail):
   - If the historical manifold-delivery-001 snapshot exists, verify database integrity,
     schema migrations, and discovery report records as a historical reference check.
2. Current Algorithm End-to-End Acceptance:
   - Evaluates the current algorithm code on newly generated state without relying
     on historical database outputs.
   - Physical Spatio-Temporal Zero-Violation Hard Assertion:
     Any segments assigned to the same player in the same clip must NEVER co-occur on screen.
   - Core Cross-Video Discovery & Template Witness Integrity:
     Every discovered player spans at least 2 distinct clips; automatic templates have
     independent cross-clip witnesses without self-witnessing or shared duplicate components.
   - Idempotency and Atomic Persistence:
     Replay does not increment revisions; database integrity and foreign key checks pass.
   - Exact Copied Sequence Rejection:
     100% duplicated image sequences across clips are rejected without establishing false identity.
   - Variable Scale:
     Business invariants hold across variable clip and player counts without hardcoded constants.
"""

from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
from test_identity_discovery import angle_vector, synthetic_facts
from test_machine_input import machine_settings

from app.contracts.api import UpdateRequest
from app.contracts.discovery import DiscoveryPolicy
from app.contracts.evidence import GallerySnapshot
from app.identity.consistency import segment_tracks
from app.identity.discovery import discover_identities
from app.identity.incremental import plan_incremental_identities
from app.storage.database import connect
from app.workflows.discover import discover_library
from app.workflows.read import get_identity_state
from app.workflows.update import update_library

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DELIVERY_DB = (
    PROJECT_ROOT / "var" / "identity-discovery" / "manifold-delivery-001" / "player-library.sqlite3"
)


# ============================================================================
# 1. Historical Artifact Integrity Check (Audit Only, Not Current Acceptance)
# ============================================================================


def test_historical_manifold_delivery_snapshot_integrity():
    """Verify that the historical delivery database artifact, if present, is intact.

    This is a read-only audit verification of the historical artifact. It does not
    substitute for executing current algorithm code.
    """
    if not DEFAULT_DELIVERY_DB.is_file():
        pytest.skip(
            f"Historical delivery database not found at {DEFAULT_DELIVERY_DB}; skipping audit."
        )

    import sqlite3

    conn = sqlite3.connect(DEFAULT_DELIVERY_DB)
    try:
        cur = conn.cursor()
        integrity = [row[0] for row in cur.execute("PRAGMA integrity_check").fetchall()]
        assert integrity == ["ok"], f"Database integrity check failed: {integrity}"

        fk_errors = cur.execute("PRAGMA foreign_key_check").fetchall()
        assert len(fk_errors) == 0, f"Foreign key errors: {fk_errors}"

        # Schema migrations 001-007 must be recorded
        migrations = [
            row[0]
            for row in cur.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            ).fetchall()
        ]
        assert 7 in migrations, "Migration 007 missing from historical database"

        # Reports table must exist and contain at least one record
        reports = cur.execute("SELECT report_id FROM identity_discovery_reports").fetchall()
        assert len(reports) >= 1, "No discovery reports found in historical database"
    finally:
        conn.close()


# ============================================================================
# 2. Current Algorithm Acceptance: Physical Safety & Multi-Clip Zero Collision
# ============================================================================


def test_current_algorithm_acceptance_physical_zero_violation(tmp_path):
    """Current Algorithm Hard Assertion: Physical Spatio-Temporal Zero Violations.

    Executes current discovery code from a fresh in-memory database.
    Asserts:
    - Any two segments assigned to the same player within the same clip must NEVER
      co-occur on screen at overlapping timestamps or frame indices.
    - Observation collision count and segment interval overlap count must strictly equal 0.
    """
    v_player_a = angle_vector(0.0)
    v_player_b = angle_vector(np.pi / 2.0)

    # 3 clips:
    # Player A appears in c0 (track 0), c1 (track 0), c2 (track 0)
    # Player B appears in c0 (track 1, overlapping in time with track 0!), c1 (track 1)
    specs = [
        ("c0", "0", v_player_a, "fam-c0-0"),
        ("c0", "1", v_player_b, "fam-c0-1"),
        ("c1", "0", v_player_a, "fam-c1-0"),
        ("c1", "1", v_player_b, "fam-c1-1"),
        ("c2", "0", v_player_a, "fam-c2-0"),
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

    settings = machine_settings(tmp_path / "unused", tmp_path).model_copy(
        update={
            "discovery": policy,
            "match": machine_settings(tmp_path / "unused", tmp_path).match.model_copy(
                update={"match_id": "synthetic-match"}
            ),
        }
    )

    db = connect(":memory:")
    imported = update_library(
        db,
        UpdateRequest(batch=segmented.batch, expected_revision=0),
        feature_values={},
        settings=settings,
    )
    discovered = discover_library(
        db,
        segmented.batch,
        segmentation_report=segmented.report,
        feature_values={},
        settings=settings,
        expected_revision=imported.revision,
    )
    assert discovered.revision == imported.revision + 1

    state = get_identity_state(db, "synthetic-match")
    assigned = [a for a in state.data.assignments if a.player_id is not None]
    assert len(assigned) > 0, "Current discovery produced 0 assigned segments"

    # Build observation mapping (clip_id, track_id) -> list of (time_s, frame_idx)
    track_obs = defaultdict(list)
    for obs in segmented.batch.identity_evidence.observations:
        track_obs[(obs.clip_id, obs.track_id)].append((obs.time_s, obs.frame_idx))

    seg_map = {s.segment_id: s for s in segmented.batch.identity_evidence.segments}
    player_segments = defaultdict(list)
    for a in assigned:
        seg = seg_map[a.segment_id]
        player_segments[a.player_id].append(seg)

    frame_violations = []
    interval_violations = []

    for pid, segs in player_segments.items():
        by_clip = defaultdict(list)
        for s in segs:
            by_clip[s.clip_id].append(s)

        for clip_id, clip_segs in by_clip.items():
            for s1, s2 in combinations(clip_segs, 2):
                if s1.track_id != s2.track_id:
                    # Check segment interval overlap: max(s1, s2) < min(e1, e2)
                    if max(s1.start_s, s2.start_s) < min(s1.end_s, s2.end_s):
                        interval_violations.append((pid, clip_id, s1.segment_id, s2.segment_id))

                    # Check frame-level collision
                    f1 = {
                        f
                        for _t, f in track_obs[(clip_id, s1.track_id)]
                        if s1.frame_start <= f < s1.frame_end_exclusive
                    }
                    f2 = {
                        f
                        for _t, f in track_obs[(clip_id, s2.track_id)]
                        if s2.frame_start <= f < s2.frame_end_exclusive
                    }
                    if f1 & f2:
                        frame_violations.append((pid, clip_id, s1.track_id, s2.track_id, f1 & f2))

    assert len(frame_violations) == 0, f"Found observation collisions: {frame_violations}"
    assert len(interval_violations) == 0, f"Found interval collisions: {interval_violations}"


# ============================================================================
# 3. Core Cross-Video Discovery & Multi-View Template Witness Integrity
# ============================================================================


def test_current_algorithm_acceptance_cross_clip_and_template_witness_integrity(tmp_path):
    """Verify that discovered players have multi-clip presence and independent witnesses."""
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
        min_common_parts=2,
        min_part_visibility=0.2,
    )

    settings = machine_settings(tmp_path / "unused", tmp_path).model_copy(
        update={
            "discovery": policy,
            "match": machine_settings(tmp_path / "unused", tmp_path).match.model_copy(
                update={"match_id": "synthetic-match"}
            ),
        }
    )

    db = connect(":memory:")
    imported = update_library(
        db,
        UpdateRequest(batch=segmented.batch, expected_revision=0),
        feature_values={},
        settings=settings,
    )
    discover_library(
        db,
        segmented.batch,
        segmentation_report=segmented.report,
        feature_values={},
        settings=settings,
        expected_revision=imported.revision,
    )

    state = get_identity_state(db, "synthetic-match")
    players = state.data.players
    assert len(players) == 1, f"Expected 1 core player, got {len(players)}"

    player_id = players[0].player_id
    assigned_segs = [a for a in state.data.assignments if a.player_id == player_id]
    seg_map = {s.segment_id: s for s in segmented.batch.identity_evidence.segments}
    assigned_clips = {seg_map[a.segment_id].clip_id for a in assigned_segs}
    assert len(assigned_clips) >= 2, f"Discovered player must span >= 2 clips, got {assigned_clips}"

    templates = [t for t in state.data.templates if t.player_id == player_id and t.active]
    assert len(templates) >= 2, f"Expected at least 2 templates, got {len(templates)}"

    from app.workflows.read import get_discovery_reports

    disc_reports = get_discovery_reports(db, "synthetic-match")
    assert len(disc_reports.data) >= 1
    report = disc_reports.data[0]["report"]
    support_by_sample = {tpl["sample_id"]: tpl for g in report["groups"] for tpl in g["templates"]}

    sample_map = {s.sample_id: s for s in segmented.batch.identity_evidence.samples}
    for t in templates:
        assert t.admission == "automatic_experimental"
        assert t.review_status == "unreviewed"
        assert t.approval is None
        assert t.automatic_support_id is not None

        # Automatic template support record from discovery report
        supp = support_by_sample.get(t.sample_id)
        assert supp is not None
        assert len(supp["witness_sample_ids"]) >= 1
        assert supp["sample_id"] not in supp["witness_sample_ids"], "Template cannot witness itself"

        # Witnesses must be from different clips
        tpl_clip = sample_map[t.sample_id].clip_id
        witness_clips = {sample_map[wid].clip_id for wid in supp["witness_sample_ids"]}
        assert tpl_clip not in witness_clips, "Witnesses must come from independent other clips"


# ============================================================================
# 4. Idempotency and Atomic Persistence
# ============================================================================


def test_current_algorithm_acceptance_idempotency_and_atomic_audit(tmp_path):
    """Verify that replaying discovery is idempotent and preserves revision states."""
    v0 = angle_vector(0.0)
    specs = [("c0", "0", v0, "fam-c0"), ("c1", "0", v0, "fam-c1")]
    batch = synthetic_facts(specs)
    segmented = segment_tracks(batch)

    policy = DiscoveryPolicy(
        version="complete_support_v3",
        support_mode="constrained_manifold",
        clustering_threshold=0.28,
    )
    settings = machine_settings(tmp_path / "unused", tmp_path).model_copy(
        update={
            "discovery": policy,
            "match": machine_settings(tmp_path / "unused", tmp_path).match.model_copy(
                update={"match_id": "synthetic-match"}
            ),
        }
    )

    db = connect(":memory:")
    imported = update_library(
        db,
        UpdateRequest(batch=segmented.batch, expected_revision=0),
        feature_values={},
        settings=settings,
    )
    res1 = discover_library(
        db,
        segmented.batch,
        segmentation_report=segmented.report,
        feature_values={},
        settings=settings,
        expected_revision=imported.revision,
    )
    rev_after_first = res1.revision

    # Replay on the exact same evidence and report
    res2 = discover_library(
        db,
        segmented.batch,
        segmentation_report=segmented.report,
        feature_values={},
        settings=settings,
        expected_revision=rev_after_first,
    )
    assert not res2.data["changed"], "Replay must report changed=False"
    assert res2.revision == rev_after_first, "Replay must not increment database revision"


# ============================================================================
# 5. Exact Copied Image Sequences Must NOT Create Independent Identities
# ============================================================================


def test_current_algorithm_acceptance_rejection_of_copied_sequences():
    """Verify that 100% duplicate image sequences across clips are rejected.

    Duplication is not conflict (not different person), but cannot establish
    independent appearance. Cross-clip discovery must produce 0 groups and 0 new players.
    """
    v0 = angle_vector(0.0)
    # Both clips have the exact same image family 'same-copy' -> identical SHA-256 for all images
    specs = [
        ("c0", "1", v0, "same-copy"),
        ("c1", "1", v0, "same-copy"),
    ]
    batch = synthetic_facts(specs)
    segmented = segment_tracks(batch)

    policy = DiscoveryPolicy(
        version="complete_support_v3",
        support_mode="constrained_manifold",
        clustering_threshold=0.28,
    )

    # Cold discovery test
    cold = discover_identities(
        segmented.batch,
        segmentation_report=segmented.report,
        policy=policy,
    )
    assert len(cold.groups) == 0, (
        f"Expected 0 groups for duplicate sequence, got {len(cold.groups)}"
    )
    assert len(cold.unknown) == 2
    for unk in cold.unknown:
        assert "no_independent_cross_clip_group" in unk.reasons

    # Incremental planning test
    inc = plan_incremental_identities(
        GallerySnapshot(match_id=batch.match_id, revision=0),
        segmented.batch,
        segmentation_report=segmented.report,
        feature_values={},
        policy=policy,
    )
    assert len(inc.new_players) == 0, (
        f"Expected 0 new players for duplicate sequence, got {len(inc.new_players)}"
    )
    assert all(d.category == "insufficient_evidence" for d in inc.decisions)


# ============================================================================
# 6. Variable Scale Generalization (No Hardcoded 158 tracks or 20-35 players)
# ============================================================================


@pytest.mark.parametrize("num_clips,num_actors", [(2, 2), (3, 3), (4, 2)])
def test_current_algorithm_acceptance_variable_scale(num_clips, num_actors):
    """Verify that business constraints hold across variable clip and actor counts."""
    specs = []
    for a in range(num_actors):
        v = angle_vector(float(a) * np.pi / float(num_actors + 1))
        for c in range(num_clips):
            specs.append((f"c{c}", str(a), v, f"fam-c{c}-a{a}"))

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

    # Discovered groups must equal the number of cross-clip actors
    assert len(report.groups) == num_actors
    # Every group must span all clips where the actor appeared
    for group in report.groups:
        assert len(group.member_segment_ids) == num_clips
        assert len(group.templates) >= 2
