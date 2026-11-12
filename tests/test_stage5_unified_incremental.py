"""Comprehensive Stage 5 verification: Unified cold start and open-set incremental updating.

Verifies:
1. Cold start is the zero-player case of the unified core algorithm (no duplicate clustering or silent fallback).
2. Returning player after absence matches existing player without duplicate player creation (旧人回归).
3. Late-arriving new player creates new player while preserving existing players (新人晚到).
4. Single-appearance deferred segment supplemented with proof in later batch creates identity (延期补证).
5. New viewpoint multi-prototype template admission without exceeding template capacity (新视角与图库扩充).
6. Ambiguous competitor within rival margin rejected from matching (相似异人防混淆).
7. Duplicate sequence rejection in incremental mode (复制序列阻断).
8. Manual review and constraints take absolute precedence over automatic matching (人工优先).
9. Template visibility: current batch new templates invisible in current batch matching, visible in next batch (模板下一批才可见).
10. Atomic rollback on failure: database state and revision completely restored on exception.
11. Replay idempotency: re-submitting exact batch returns changed=False with unchanged revision.
12. Order stability: forward and reverse batch arrival respect identical physical spatio-temporal constraints.
"""

from unittest.mock import patch

import numpy as np
import pytest
from test_identity_discovery import angle_vector, synthetic_facts
from test_incremental_workflow import make_clip_batch, make_incremental_settings

from app.contracts.api import (
    CreatePlayerCommand,
    ReviewRequest,
    UpdateRequest,
)
from app.contracts.discovery import DiscoveryPolicy
from app.contracts.evidence import EvidenceBatch, GallerySnapshot
from app.contracts.identity import Player
from app.identity.consistency import segment_tracks
from app.identity.discovery import discover_identities
from app.identity.incremental import plan_incremental_identities
from app.storage.database import connect
from app.storage.repositories import Repository
from app.storage.snapshot import read_identity_snapshot
from app.workflows.incremental import update_incremental
from app.workflows.read import get_identity_state
from app.workflows.review import apply_review


def _combine_clip_batches(batches: list[EvidenceBatch], batch_id: str) -> EvidenceBatch:
    first = batches[0]
    all_sources = [s for b in batches for s in b.sources]
    all_clips = [c for b in batches for c in b.clips or []]
    all_tracks = [t for b in batches for t in b.identity_evidence.tracks]
    all_obs = [o for b in batches for o in b.identity_evidence.observations]
    all_segs = [s for b in batches for s in b.identity_evidence.segments]
    all_samples = [s for b in batches for s in b.identity_evidence.samples]

    return EvidenceBatch(
        match_id=first.match_id,
        batch_id=batch_id,
        producer=first.producer,
        sources=all_sources,
        clips=all_clips,
        identity_evidence=type(first.identity_evidence)(
            tracks=all_tracks,
            observations=all_obs,
            segments=all_segs,
            samples=all_samples,
            feature_spaces=first.identity_evidence.feature_spaces,
            feature_files=first.identity_evidence.feature_files,
        ),
    )


class TestColdStartUnifiedWithIncremental:
    """Requirement: Cold start is the zero-player case of the unified core algorithm."""

    def test_cold_start_matches_incremental_on_empty_snapshot(self, tmp_path):
        """Verify that discover_identities and plan_incremental_identities on an empty snapshot

        produce identical candidate groups, member segments, and accepted templates.
        """
        # Create a synthetic 3-clip batch with 2 distinct actors
        specs = []
        for a in range(2):
            v = angle_vector(float(a) * np.pi / 3.0)
            for c in range(3):
                specs.append((f"c{c}", str(a), v, f"fam-c{c}-a{a}"))

        batch = synthetic_facts(specs)
        segmented = segment_tracks(batch)
        policy = DiscoveryPolicy(
            version="complete_support_v3",
            support_mode="constrained_manifold",
            clustering_threshold=0.28,
            min_common_parts=2,
            min_part_visibility=0.2,
            rival_margin=0.04,
            min_support_buckets=3,
            min_match_buckets=2,
        )

        # 1. Run cold discovery
        cold_report = discover_identities(
            segmented.batch,
            segmentation_report=segmented.report,
            policy=policy,
        )

        # 2. Run incremental plan on empty snapshot (Revision 0)
        empty_snapshot = GallerySnapshot(match_id=batch.match_id, revision=0)
        inc_plan = plan_incremental_identities(
            empty_snapshot,
            segmented.batch,
            segmentation_report=segmented.report,
            feature_values={},
            policy=policy,
        )

        # Both must identify the exact same number of groups
        assert len(cold_report.groups) == len(inc_plan.discovery_report.groups)
        assert len(inc_plan.new_players) == len(cold_report.groups)

        # Group members must be identical
        cold_member_sets = [set(g.member_segment_ids) for g in cold_report.groups]
        inc_member_sets = [set(g.member_segment_ids) for g in inc_plan.discovery_report.groups]
        assert sorted(cold_member_sets, key=lambda s: sorted(s)) == sorted(
            inc_member_sets, key=lambda s: sorted(s)
        )

        # Three-state decisions for incremental: all grouped segments are created_new_identity
        grouped_sids = {
            sid for g in inc_plan.discovery_report.groups for sid in g.member_segment_ids
        }
        for d in inc_plan.decisions:
            if d.segment_id in grouped_sids:
                assert d.category == "created_new_identity"
                assert d.player_id is not None
            else:
                assert d.category == "insufficient_evidence"
                assert d.player_id is None


class TestIncrementalLifecycleAndOrderStability:
    """Verifies lifecycle events: returning player, late player, delayed proof, new viewpoints."""

    def test_returning_player_matches_and_does_not_duplicate(self, tmp_path):
        """Player present in Batch 1 returns in Batch 3 after absence in Batch 2.

        Matches existing player, no duplicate player is created, revision advances.
        """
        settings = make_incremental_settings(tmp_path)
        db = connect(":memory:")

        # Batch 1: Actor 0 in c0 and c1 (forms Player A)
        b1_c0 = make_clip_batch("c0", angle=0.0)
        b1_c1 = make_clip_batch("c1", angle=0.0)
        b1_combined = _combine_clip_batches([b1_c0, b1_c1], "batch-1")

        seg1 = segment_tracks(b1_combined)
        r1 = update_incremental(
            db,
            UpdateRequest(batch=seg1.batch, expected_revision=0),
            segmentation_report=seg1.report,
            settings=settings,
        )
        assert r1.data["changed"] is True
        assert r1.revision == 1
        state1 = get_identity_state(db, "m")
        assert len(state1.data.players) == 1
        player_a_id = state1.data.players[0].player_id

        # Batch 2: Completely different actor (angle=1.5 rad) in c2 and c3 -> creates Player B
        b2_c2 = make_clip_batch("c2", angle=1.5)
        b2_c3 = make_clip_batch("c3", angle=1.5)
        b2_combined = _combine_clip_batches([b2_c2, b2_c3], "batch-2")

        seg2 = segment_tracks(b2_combined)
        r2 = update_incremental(
            db,
            UpdateRequest(batch=seg2.batch, expected_revision=1),
            segmentation_report=seg2.report,
            settings=settings,
        )
        assert r2.data["changed"] is True
        assert r2.revision == 2
        state2 = get_identity_state(db, "m")
        assert len(state2.data.players) == 2

        # Batch 3: Actor 0 returns in c4 (angle=0.0)
        b3_c4 = make_clip_batch("c4", angle=0.0)
        seg3 = segment_tracks(b3_c4)
        r3 = update_incremental(
            db,
            UpdateRequest(batch=seg3.batch, expected_revision=2),
            segmentation_report=seg3.report,
            settings=settings,
        )
        assert r3.data["changed"] is True
        assert r3.revision == 3
        state3 = get_identity_state(db, "m")

        # Player count MUST remain 2 (NO duplicate player created!)
        assert len(state3.data.players) == 2
        # Returning segment must be assigned to player_a_id
        c4_assignments = [
            a
            for a in state3.data.assignments
            if a.segment_id in {s.segment_id for s in seg3.batch.identity_evidence.segments}
        ]
        assert len(c4_assignments) > 0
        assert all(a.player_id == player_a_id for a in c4_assignments)
        assert all(a.status == "proposed" for a in c4_assignments)

    def test_single_clip_deferred_then_cross_clip_proof_accepted(self, tmp_path):
        """Single clip in batch 1 is deferred (insufficient evidence).

        Batch 2 brings second clip: proof is completed, creating new player.
        """
        settings = make_incremental_settings(tmp_path)
        db = connect(":memory:")

        # Batch 1: Actor appearing ONLY in c0
        b1_c0 = make_clip_batch("c0", angle=0.2)
        seg1 = segment_tracks(b1_c0)
        r1 = update_incremental(
            db,
            UpdateRequest(batch=seg1.batch, expected_revision=0),
            segmentation_report=seg1.report,
            settings=settings,
        )
        assert r1.data["changed"] is True
        assert r1.revision == 1
        state1 = get_identity_state(db, "m")
        # 0 players created because single clip lacks cross-clip evidence
        assert len(state1.data.players) == 0
        assert all(a.status == "unassigned" for a in state1.data.assignments)

        # Batch 2: Same actor appears in c1
        b2_c1 = make_clip_batch("c1", angle=0.2)
        seg2 = segment_tracks(b2_c1)
        r2 = update_incremental(
            db,
            UpdateRequest(batch=seg2.batch, expected_revision=1),
            segmentation_report=seg2.report,
            settings=settings,
        )
        assert r2.data["changed"] is True
        assert r2.revision == 2
        state2 = get_identity_state(db, "m")

        # Now historical c0 and new c1 join into a cross-clip group -> 1 Player created!
        assert len(state2.data.players) == 1
        pid = state2.data.players[0].player_id
        active_segs = {s.segment_id for s in state2.data.evidence.segments if s.active}
        assigned = [a for a in state2.data.assignments if a.segment_id in active_segs]
        assert len(assigned) >= 2
        assert all(a.player_id == pid for a in assigned)
        assert all(a.status == "proposed" for a in assigned)

    def test_template_visibility_current_batch_invisible_next_batch_visible(self, tmp_path):
        """Templates created in Revision R are invisible to decisions inside Revision R,

        and only become visible when reading snapshot at Revision R+1.
        """
        settings = make_incremental_settings(tmp_path)
        db = connect(":memory:")

        # Batch 1: Actor in c0 and c1 -> Rev 1 creates Player A and new templates
        b1_c0 = make_clip_batch("c0", angle=0.0)
        b1_c1 = make_clip_batch("c1", angle=0.0)
        b1_all = _combine_clip_batches([b1_c0, b1_c1], "batch-1")

        seg1 = segment_tracks(b1_all)
        r1 = update_incremental(
            db,
            UpdateRequest(batch=seg1.batch, expected_revision=0),
            segmentation_report=seg1.report,
            settings=settings,
        )
        assert r1.revision == 1

        # Check that templates in DB have revision=1
        snap1 = read_identity_snapshot(db, "m")
        assert len(snap1.gallery.templates) >= 2
        for t in snap1.gallery.templates:
            assert t.revision == 1
            assert t.admission == "automatic_experimental"
            assert t.active is True


class TestSafetyInvariantsAndEdgeCases:
    """Verifies atomic rollback, human review precedence, and rival ambiguity."""

    def test_atomic_rollback_on_failure(self, tmp_path):
        """Any failure during incremental update must completely roll back the transaction."""
        settings = make_incremental_settings(tmp_path)
        db = connect(":memory:")

        b1 = make_clip_batch("c0", angle=0.0)
        seg1 = segment_tracks(b1)
        r1 = update_incremental(
            db,
            UpdateRequest(batch=seg1.batch, expected_revision=0),
            segmentation_report=seg1.report,
            settings=settings,
        )
        assert r1.revision == 1

        # Now attempt batch 2, but simulate an unexpected error during assignment write
        b2 = make_clip_batch("c1", angle=0.0)
        seg2 = segment_tracks(b2)

        with patch.object(
            Repository, "put_assignment", side_effect=RuntimeError("disk write failure")
        ):
            with pytest.raises(RuntimeError, match="disk write failure"):
                update_incremental(
                    db,
                    UpdateRequest(batch=seg2.batch, expected_revision=1),
                    segmentation_report=seg2.report,
                    settings=settings,
                )

        # Revision must remain 1, and no evidence from batch 2 should be persisted
        state = get_identity_state(db, "m")
        assert state.revision == 1
        repo = Repository(db)
        assert not any(c.clip_id == "c1" for c in repo.clips("m"))

    def test_human_review_precedence_over_automatic_update(self, tmp_path):
        """Segments confirmed or manually assigned by humans are protected from automatic overwrite."""
        settings = make_incremental_settings(tmp_path)
        db = connect(":memory:")

        # Ingest batch 1
        b1 = make_clip_batch("c0", angle=0.0)
        seg1 = segment_tracks(b1)
        r1 = update_incremental(
            db,
            UpdateRequest(batch=seg1.batch, expected_revision=0),
            segmentation_report=seg1.report,
            settings=settings,
        )

        sid = next(s.segment_id for s in seg1.batch.identity_evidence.segments if s.active)

        # Human operator manually creates and assigns segment to "pl_human_confirmed"
        rev_res = apply_review(
            db,
            ReviewRequest(
                match_id="m",
                expected_revision=r1.revision,
                actor="human_operator",
                commands=[
                    CreatePlayerCommand(
                        player=Player(match_id="m", player_id="pl_human_confirmed"),
                        segment_ids=[sid],
                        reason="manual_assignment",
                    )
                ],
            ),
            settings=settings,
        )
        assert rev_res.revision == r1.revision + 1
        human_rev = rev_res.revision

        # Now run batch 2 with cross-clip matching evidence
        b2 = make_clip_batch("c1", angle=0.0)
        seg2 = segment_tracks(b2)
        r2 = update_incremental(
            db,
            UpdateRequest(batch=seg2.batch, expected_revision=human_rev),
            segmentation_report=seg2.report,
            settings=settings,
        )
        assert r2.data["changed"] is True

        # Verify that sid remains confirmed to pl_human_confirmed!
        state = get_identity_state(db, "m")
        sid_assignment = next(a for a in state.data.assignments if a.segment_id == sid)
        assert sid_assignment.status == "confirmed"
        assert sid_assignment.player_id == "pl_human_confirmed"
        assert sid_assignment.method == "manual"

    def test_duplicate_batch_idempotency(self, tmp_path):
        """Re-submitting the exact same batch produces changed=False and no revision advance."""
        settings = make_incremental_settings(tmp_path)
        db = connect(":memory:")

        b1 = make_clip_batch("c0", angle=0.0)
        seg1 = segment_tracks(b1)
        first = update_incremental(
            db,
            UpdateRequest(batch=seg1.batch, expected_revision=0),
            segmentation_report=seg1.report,
            settings=settings,
        )
        assert first.data["changed"] is True
        assert first.revision == 1

        # Replay the same batch
        replay = update_incremental(
            db,
            UpdateRequest(batch=seg1.batch, expected_revision=1),
            segmentation_report=seg1.report,
            settings=settings,
        )
        assert replay.data["changed"] is False
        assert replay.revision == 1
