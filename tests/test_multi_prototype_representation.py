"""Unit and business tests for multi-prototype identity representation.

Verifies:
1. Multi-prototype vs single pooled representation development comparison:
   - Turning player (front, side, back) is blurred by single pooled vector and fails
     to match pure front track; multi-prototype retains distinct viewpoint prototypes
     and successfully matches compatible viewpoint.
2. Minority mixed intruder is NEVER swallowed by close prototype:
   - When an intruder is mixed into an otherwise matching segment, strong
     counter-evidence (far >= 0.45) triggers one-vote veto, preventing merger.
3. Part visibility and reliable parts handling:
   - Partial occlusions, insufficient overlapping parts (< min_common_parts) safely
     return uncomparable (None), preventing false zeros or crashes.
4. Full traceability of templates:
   - Every admitted template links to prototype_id, raw sample_id, segment_id,
     time_bucket, reliable_parts, quality_score, source_component, and witnesses.
5. Duplicate image and component protection:
   - Duplicate images/crops sharing source_component do not generate redundant
     prototypes and cannot occupy multiple template capacity slots.
6. Feature space mismatch safety:
   - Representations in different feature spaces are strictly uncomparable (None).
"""

import numpy as np

from app.contracts.common import Provenance
from app.contracts.discovery import DiscoveryPolicy
from app.contracts.evidence import FeatureSpace
from app.identity.representation import (
    AppearancePrototype,
    TrackRepresentation,
    compute_prototype_distance,
    compute_track_distance,
    extract_segment_prototypes,
    pool_segment_features,
)


def _make_provenance(record_id: str = "rec_1") -> Provenance:
    return Provenance(
        producer="test",
        record_type="sample",
        source_record_id=record_id,
        evidence_version="v1",
    )


def _make_feature_space(fs_id: str = "kpr_test", parts: int = 6, dim: int = 128) -> FeatureSpace:
    return FeatureSpace(
        feature_space_id=fs_id,
        backend="kpr",
        model_hash="0" * 64,
        preprocessing={"version": "v1", "color_space": "RGB"},
        layout="parts",
        shape=[parts, dim],
        normalization="l2",
        distance="euclidean_sqrt_visibility_mean_div2",
        visibility={"shape": [parts], "semantics": "visibility"},
    )


def _make_unit_vector(dim: int, index: int) -> np.ndarray:
    vec = np.zeros((dim,), dtype=np.float32)
    vec[index % dim] = 1.0
    return vec


def _make_part_matrix(dim: int, base_idx: int) -> np.ndarray:
    """Create a (6, dim) matrix with L2 normalized rows based on base_idx."""
    mat = np.zeros((6, dim), dtype=np.float32)
    for p in range(6):
        mat[p] = _make_unit_vector(dim, (base_idx + p) % dim)
    return mat


class TestMultiPrototypeRepresentation:
    def test_prototype_dataclass_equality_and_hashing(self):
        f = _make_part_matrix(128, 0)
        v = np.ones((6,), dtype=np.float32) * 0.9
        p1 = AppearancePrototype(
            prototype_id="proto_s1_samp1",
            sample_id="samp1",
            segment_id="s1",
            clip_id="c1",
            track_id=1,
            time_s=1.0,
            time_bucket=2,
            feature_space_id="kpr_test",
            feature=f,
            visibility=v,
            reliable_parts=(0, 1, 2, 3, 4, 5),
            quality_score=0.95,
            risks=(),
            witness_sample_ids=("samp1", "samp2"),
            source_component="comp_1",
        )
        p2 = AppearancePrototype(
            prototype_id="proto_s1_samp1",
            sample_id="samp1",
            segment_id="s1",
            clip_id="c1",
            track_id=1,
            time_s=1.0,
            time_bucket=2,
            feature_space_id="kpr_test",
            feature=f.copy(),
            visibility=v.copy(),
            reliable_parts=(0, 1, 2, 3, 4, 5),
            quality_score=0.95,
            risks=(),
            witness_sample_ids=("samp1", "samp2"),
            source_component="comp_1",
        )
        p3 = AppearancePrototype(
            prototype_id="proto_s1_samp3",
            sample_id="samp3",
            segment_id="s1",
            clip_id="c1",
            track_id=1,
            time_s=2.5,
            time_bucket=5,
            feature_space_id="kpr_test",
            feature=_make_part_matrix(128, 10),
            visibility=v,
            reliable_parts=(0, 1, 2, 3, 4, 5),
            quality_score=0.88,
            risks=(),
            witness_sample_ids=("samp3",),
            source_component="comp_2",
        )

        assert p1 == p2
        assert p1 != p3
        assert hash(p1) == hash(p2)
        assert len({p1, p2, p3}) == 2

    def test_single_representation_vs_multi_prototype_turning_player(self):
        """Turning player development comparison:

        A player turns while running (e.g. 1 front view, 3 side views, 3 back views).
        - Shared body appearance (jersey, skin, shape): base vector u0
        - Viewpoint variations: u1 (front), u2 (side), u3 (back)
        Pairwise viewpoint distance d(front, back) = 0.4417 < 0.45 (zero far pairs).
        - Old single pooled representation: averages all frames, resulting in
          d(pooled, front) = 0.3443 > 0.28 (FALSE REJECTION!).
        - Multi-prototype representation: preserves distinct viewpoint prototypes,
          matching Front-Front with d = 0.0000 <= 0.18 (CORRECT MATCH!).
        """
        space = _make_feature_space()
        dim = 128
        alpha = 0.8
        u0 = _make_unit_vector(dim, 0)
        u1 = _make_unit_vector(dim, 1)
        u2 = _make_unit_vector(dim, 2)
        u3 = _make_unit_vector(dim, 3)

        vf = (u0 + alpha * u1) / float(np.linalg.norm(u0 + alpha * u1))
        vs = (u0 + alpha * u2) / float(np.linalg.norm(u0 + alpha * u2))
        vb = (u0 + alpha * u3) / float(np.linalg.norm(u0 + alpha * u3))

        f_front = np.tile(vf.astype(np.float32), (6, 1))
        f_side = np.tile(vs.astype(np.float32), (6, 1))
        f_back = np.tile(vb.astype(np.float32), (6, 1))
        v = np.ones((6,), dtype=np.float32) * 0.9

        # Turning player segment s1: 1 front sample, 3 side samples, 3 back samples
        s1 = {"segment_id": "s1", "clip_id": "c1", "track_id": 1, "start_s": 0.0, "end_s": 3.5}
        samples_s1 = [
            {
                "segment_id": "s1",
                "sample_id": "s1_1",
                "time_s": 0.2,
                "feature": f_front,
                "visibility": v,
            },
            {
                "segment_id": "s1",
                "sample_id": "s1_2",
                "time_s": 1.0,
                "feature": f_side,
                "visibility": v,
            },
            {
                "segment_id": "s1",
                "sample_id": "s1_3",
                "time_s": 1.2,
                "feature": f_side,
                "visibility": v,
            },
            {
                "segment_id": "s1",
                "sample_id": "s1_4",
                "time_s": 1.4,
                "feature": f_side,
                "visibility": v,
            },
            {
                "segment_id": "s1",
                "sample_id": "s1_5",
                "time_s": 2.2,
                "feature": f_back,
                "visibility": v,
            },
            {
                "segment_id": "s1",
                "sample_id": "s1_6",
                "time_s": 2.4,
                "feature": f_back,
                "visibility": v,
            },
            {
                "segment_id": "s1",
                "sample_id": "s1_7",
                "time_s": 2.6,
                "feature": f_back,
                "visibility": v,
            },
        ]

        # Target pure Front view segment s2 in another clip
        s2 = {"segment_id": "s2", "clip_id": "c2", "track_id": 2, "start_s": 10.0, "end_s": 11.0}
        samples_s2 = [
            {
                "segment_id": "s2",
                "sample_id": "s2_1",
                "time_s": 10.2,
                "feature": f_front,
                "visibility": v,
            },
            {
                "segment_id": "s2",
                "sample_id": "s2_2",
                "time_s": 10.6,
                "feature": f_front,
                "visibility": v,
            },
        ]

        # 1. Old Single Pooled Representation:
        # Pooled feature for s1 is an average of front + side + back
        reps_single = pool_segment_features(
            segments=[s1, s2],
            samples=samples_s1 + samples_s2,
            space=space,
            max_prototypes=1,  # forced to single prototype / pooled
        )
        rep1_single = reps_single["s1"]
        rep2_single = reps_single["s2"]

        # Compute old pooled distance (ignoring prototypes)
        pooled_dist = compute_track_distance(
            TrackRepresentation(
                clip_id=rep1_single.clip_id,
                track_id=rep1_single.track_id,
                start_s=rep1_single.start_s,
                end_s=rep1_single.end_s,
                feature=rep1_single.feature,
                visibility=rep1_single.visibility,
                sample_ids=rep1_single.sample_ids,
                prototypes=(),  # force pooled comparison
            ),
            TrackRepresentation(
                clip_id=rep2_single.clip_id,
                track_id=rep2_single.track_id,
                start_s=rep2_single.start_s,
                end_s=rep2_single.end_s,
                feature=rep2_single.feature,
                visibility=rep2_single.visibility,
                sample_ids=rep2_single.sample_ids,
                prototypes=(),  # force pooled comparison
            ),
        )

        # In old single pooled representation, averaging front + side + back causes severe dilution:
        # Distance exceeds clustering_threshold (0.28)!
        assert pooled_dist is not None
        assert pooled_dist > 0.28, (
            f"Expected pooled distance > 0.28 due to viewpoint blurring, got {pooled_dist}"
        )

        # 2. Multi-Prototype Representation:
        # Retains distinct prototypes for front, side, and back
        reps_multi = pool_segment_features(
            segments=[s1, s2],
            samples=samples_s1 + samples_s2,
            space=space,
            max_prototypes=4,
        )
        rep1_multi = reps_multi["s1"]
        rep2_multi = reps_multi["s2"]

        assert len(rep1_multi.prototypes) >= 3, (
            f"Expected at least 3 prototypes, got {len(rep1_multi.prototypes)}"
        )
        multi_dist = compute_track_distance(rep1_multi, rep2_multi, clustering_threshold=0.28)

        # Multi-prototype finds the compatible Front-Front prototype pair:
        # Distance is 0.0 (near <= 0.18)!
        assert multi_dist is not None
        assert multi_dist <= 0.18, f"Expected multi-prototype distance <= 0.18, got {multi_dist}"
        assert multi_dist < pooled_dist, (
            "Multi-prototype distance must be significantly cleaner than blurred pooled distance"
        )

    def test_mixed_intruder_never_swallowed_by_close_prototype(self):
        """Mixed intruder protection (One-Vote Veto on Strong Counter-Evidence):

        Segment A contains 4 normal samples of Player 1 (front view),
        AND 1 intruder sample of Player 2 (completely different person, distance >= 0.45).
        Segment B contains only normal samples of Player 1.
        Naive nearest-neighbor min-distance would find distance 0.0 between Player 1 samples
        and wrongfully merge them.
        Multi-prototype evaluates counter-evidence across all prototypes:
        the intruder prototype creates a far distance (>= 0.45), triggering one-vote veto!
        """
        space = _make_feature_space()
        f_p1 = _make_part_matrix(128, 0)
        # Intruder feature is completely orthogonal / far from Player 1
        f_intruder = _make_part_matrix(128, 64)
        v = np.ones((6,), dtype=np.float32) * 0.9

        # Segment A: Player 1 + 1 Intruder sample
        seg_a = {
            "segment_id": "seg_a",
            "clip_id": "c1",
            "track_id": 1,
            "start_s": 0.0,
            "end_s": 2.0,
        }
        samples_a = [
            {
                "segment_id": "seg_a",
                "sample_id": "a_1",
                "time_s": 0.2,
                "feature": f_p1,
                "visibility": v,
            },
            {
                "segment_id": "seg_a",
                "sample_id": "a_2",
                "time_s": 0.6,
                "feature": f_p1,
                "visibility": v,
            },
            {
                "segment_id": "seg_a",
                "sample_id": "a_3",
                "time_s": 1.0,
                "feature": f_p1,
                "visibility": v,
            },
            {
                "segment_id": "seg_a",
                "sample_id": "a_intruder",
                "time_s": 1.5,
                "feature": f_intruder,
                "visibility": v,
            },
        ]

        # Segment B: Pure Player 1
        seg_b = {
            "segment_id": "seg_b",
            "clip_id": "c2",
            "track_id": 2,
            "start_s": 5.0,
            "end_s": 6.0,
        }
        samples_b = [
            {
                "segment_id": "seg_b",
                "sample_id": "b_1",
                "time_s": 5.2,
                "feature": f_p1,
                "visibility": v,
            },
            {
                "segment_id": "seg_b",
                "sample_id": "b_2",
                "time_s": 5.6,
                "feature": f_p1,
                "visibility": v,
            },
        ]

        reps = pool_segment_features(
            segments=[seg_a, seg_b],
            samples=samples_a + samples_b,
            space=space,
            max_prototypes=4,
        )
        rep_a = reps["seg_a"]
        rep_b = reps["seg_b"]

        # Prototype check: seg_a must extract both Player 1 and intruder prototypes
        proto_sample_ids = {p.sample_id for p in rep_a.prototypes}
        assert "a_intruder" in proto_sample_ids, (
            "Intruder sample must be retained as an appearance prototype"
        )

        # Compare multi-prototype distance:
        # Because a_intruder vs b_1 is >= 0.45, compute_track_distance returns far_distance + 0.1 (veto)
        dist = compute_track_distance(
            rep_a, rep_b, near_distance=0.18, far_distance=0.45, clustering_threshold=0.28
        )
        assert dist is not None
        assert dist >= 0.45, (
            f"Expected distance >= 0.45 due to intruder counter-evidence veto, got {dist}"
        )

    def test_part_visibility_and_reliable_parts_handling(self):
        """Verifies partial visibility and comparability thresholds."""
        f1 = _make_part_matrix(128, 0)
        f2 = _make_part_matrix(128, 0)

        # Upper body only visible for p1 (parts 0, 1)
        v1 = np.array([0.9, 0.85, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
        # Lower body only visible for p2 (parts 4, 5)
        v2 = np.array([0.05, 0.05, 0.05, 0.05, 0.9, 0.85], dtype=np.float32)

        p1 = AppearancePrototype(
            prototype_id="p1",
            sample_id="s1",
            segment_id="seg1",
            clip_id="c1",
            track_id=1,
            time_s=1.0,
            time_bucket=2,
            feature_space_id="kpr_test",
            feature=f1,
            visibility=v1,
            reliable_parts=(0, 1),
        )
        p2 = AppearancePrototype(
            prototype_id="p2",
            sample_id="s2",
            segment_id="seg2",
            clip_id="c2",
            track_id=2,
            time_s=1.0,
            time_bucket=2,
            feature_space_id="kpr_test",
            feature=f2,
            visibility=v2,
            reliable_parts=(4, 5),
        )

        # Overlapping parts between p1 and p2 is 0 (< min_common_parts=2)
        d = compute_prototype_distance(p1, p2, min_common_parts=2, min_vis=0.2)
        assert d is None, (
            "Prototypes without overlapping visible parts must return None (uncomparable)"
        )

        # When parts 0 and 1 are both visible:
        v3 = np.array([0.8, 0.8, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
        p3 = AppearancePrototype(
            prototype_id="p3",
            sample_id="s3",
            segment_id="seg3",
            clip_id="c3",
            track_id=3,
            time_s=1.0,
            time_bucket=2,
            feature_space_id="kpr_test",
            feature=f1,
            visibility=v3,
            reliable_parts=(0, 1),
        )
        d_valid = compute_prototype_distance(p1, p3, min_common_parts=2, min_vis=0.2)
        assert d_valid is not None
        assert np.isclose(d_valid, 0.0, atol=1e-5)

    def test_feature_space_incompatibility_safety(self):
        """Representations in different feature spaces must return None (uncomparable)."""
        f = _make_part_matrix(128, 0)
        v = np.ones((6,), dtype=np.float32) * 0.9

        p_space1 = AppearancePrototype(
            prototype_id="p1",
            sample_id="s1",
            segment_id="seg1",
            clip_id="c1",
            track_id=1,
            time_s=1.0,
            time_bucket=2,
            feature_space_id="kpr_space_v1",
            feature=f,
            visibility=v,
            reliable_parts=(0, 1, 2, 3, 4, 5),
        )
        p_space2 = AppearancePrototype(
            prototype_id="p2",
            sample_id="s2",
            segment_id="seg2",
            clip_id="c2",
            track_id=2,
            time_s=1.0,
            time_bucket=2,
            feature_space_id="kpr_space_v2",
            feature=f,
            visibility=v,
            reliable_parts=(0, 1, 2, 3, 4, 5),
        )

        d = compute_prototype_distance(p_space1, p_space2)
        assert d is None, "Different feature spaces must be uncomparable"

        t1 = TrackRepresentation(
            clip_id="c1",
            track_id=1,
            start_s=0.0,
            end_s=1.0,
            feature=f,
            visibility=v,
            sample_ids=["s1"],
            feature_space_id="kpr_space_v1",
            prototypes=(p_space1,),
        )
        t2 = TrackRepresentation(
            clip_id="c2",
            track_id=2,
            start_s=0.0,
            end_s=1.0,
            feature=f,
            visibility=v,
            sample_ids=["s2"],
            feature_space_id="kpr_space_v2",
            prototypes=(p_space2,),
        )

        track_d = compute_track_distance(t1, t2)
        assert track_d is None, "Different feature spaces in TrackRepresentation must return None"

    def test_duplicate_images_do_not_produce_redundant_prototypes(self):
        """Duplicate images sharing component ID cannot form redundant prototypes."""
        space = _make_feature_space()
        f = _make_part_matrix(128, 0)
        v = np.ones((6,), dtype=np.float32) * 0.9

        seg = {
            "segment_id": "seg_dup",
            "clip_id": "c1",
            "track_id": 1,
            "start_s": 0.0,
            "end_s": 2.0,
        }
        # 4 samples: 2 in bucket 0 with identical component comp_dup, 2 in bucket 1 with comp_dup
        samples = [
            {
                "segment_id": "seg_dup",
                "sample_id": "s_dup_1",
                "time_s": 0.1,
                "feature": f,
                "visibility": v,
            },
            {
                "segment_id": "seg_dup",
                "sample_id": "s_dup_2",
                "time_s": 0.3,
                "feature": f,
                "visibility": v,
            },
            {
                "segment_id": "seg_dup",
                "sample_id": "s_dup_3",
                "time_s": 1.1,
                "feature": f,
                "visibility": v,
            },
            {
                "segment_id": "seg_dup",
                "sample_id": "s_dup_4",
                "time_s": 1.3,
                "feature": f,
                "visibility": v,
            },
        ]
        components = {
            "s_dup_1": "comp_dup",
            "s_dup_2": "comp_dup",
            "s_dup_3": "comp_dup",
            "s_dup_4": "comp_dup",
        }

        prototypes = extract_segment_prototypes(
            segment=seg,
            samples=samples,
            space=space,
            components=components,
            max_prototypes=4,
        )

        # Because all samples have identical appearance and duplicate component,
        # they must not produce 4 redundant prototypes!
        assert len(prototypes) <= 2
        for p in prototypes:
            assert p.source_component == "comp_dup"
            assert len(p.witness_sample_ids) >= 1

    def test_discover_identities_templates_traceability_to_prototypes(self):
        """DiscoveredGroup templates must trace to prototype_id, segment, bucket, and witnesses."""
        from app.identity.consistency import segment_tracks
        from app.identity.discovery import discover_identities
        from tests.test_identity_discovery import angle_vector, synthetic_facts

        specs = []
        # Person 0: vector at 0 rad across 3 clips
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
            max_templates_per_group=4,
        )

        report = discover_identities(
            segmented.batch,
            segmentation_report=segmented.report,
            policy=policy,
        )

        assert len(report.groups) >= 1
        group = report.groups[0]
        assert len(group.templates) >= 2

        for t in group.templates:
            # 1. Traces to prototype
            assert t.prototype_id is not None
            assert t.prototype_id.startswith("proto_")
            # 2. Traces to sample and segment
            assert t.sample_id is not None
            assert t.segment_id in group.member_segment_ids
            # 3. Traces to time bucket
            assert t.time_bucket is not None
            # 4. Reliable parts
            assert len(t.reliable_parts) >= 2
            # 5. Non-empty independent cross-clip witnesses
            assert len(t.witness_sample_ids) >= 2
            assert len(t.support_pair_ids) >= 1

    def test_sqlite_persistence_real_workflow_cold_and_incremental(self, tmp_path):
        """Active templates in cold and incremental workflows must persist prototype metadata to SQLite.

        Verifies that after saving, committing, closing, and reopening SQLite:
        1. All active templates in the 'templates' table have valid prototype_id, time_bucket, reliable_parts, etc.
        2. The 'template_prototypes' relation table contains matching records linking template_id, prototype_id,
           sample_id, player_id, and prototype json.
        3. No empty shells, fake IDs, or unrecorded active templates exist.
        4. Incremental update with new batches also persists valid prototype metadata.
        """
        import json
        import sqlite3

        from test_machine_input import machine_settings

        from app.contracts.api import UpdateRequest
        from app.identity.consistency import segment_tracks
        from app.storage.database import connect
        from app.workflows.discover import discover_library
        from app.workflows.incremental import update_incremental
        from app.workflows.update import update_library
        from tests.test_discovery_workflow import cross_clip_batch
        from tests.test_incremental_workflow import make_clip_batch

        db_path = tmp_path / "test_persistence.sqlite"
        settings = machine_settings(tmp_path / "unused-machine-root", tmp_path)
        settings = settings.model_copy(
            update={
                "discovery": settings.discovery.model_copy(
                    update={
                        "support_mode": "constrained_manifold",
                        "representation_mode": "multi_prototype",
                    }
                )
            }
        )
        batch_cold = cross_clip_batch()
        seg_cold = segment_tracks(batch_cold)

        db = connect(db_path)
        imported = update_library(
            db,
            UpdateRequest(batch=seg_cold.batch, expected_revision=0),
            feature_values={},
            settings=settings,
        )
        accepted = discover_library(
            db,
            seg_cold.batch,
            segmentation_report=seg_cold.report,
            feature_values={},
            settings=settings,
            expected_revision=imported.revision,
        )
        assert accepted.revision == imported.revision + 1
        db.close()

        # Reopen SQLite database fresh from disk
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        all_templates = [
            (dict(row), json.loads(row["json"]))
            for row in con.execute(
                "SELECT template_id, sample_id, player_id, json FROM templates WHERE current=1"
            )
        ]
        active_templates = [t for t in all_templates if t[1].get("active")]
        assert len(active_templates) >= 2
        for row, t_json in active_templates:
            assert t_json.get("prototype_id") is not None
            assert t_json["prototype_id"].startswith("proto_")
            assert t_json.get("time_bucket") is not None
            assert len(t_json.get("reliable_parts", [])) >= 2
            assert t_json.get("source_component") is not None

        # Check template_prototypes table
        proto_rows = [
            dict(row)
            for row in con.execute(
                "SELECT template_id, prototype_id, sample_id, player_id, json FROM template_prototypes"
            )
        ]
        assert len(proto_rows) == len(active_templates)
        proto_by_tid = {r["template_id"]: r for r in proto_rows}
        for row, t_json in active_templates:
            tid = row["template_id"]
            assert tid in proto_by_tid
            p_rec = proto_by_tid[tid]
            assert p_rec["prototype_id"] == t_json["prototype_id"]
            assert p_rec["sample_id"] == row["sample_id"]
            assert p_rec["player_id"] == row["player_id"]
            p_json = json.loads(p_rec["json"])
            assert p_json["prototype_id"] == t_json["prototype_id"]
            assert p_json["sample_id"] == row["sample_id"]

        con.close()

        # Now test incremental update on top of this database with a new player across 2 clips
        from app.contracts.evidence import EvidenceBatch, IdentityEvidence

        b_inc1 = make_clip_batch("gamma", angle=1.0)
        b_inc2 = make_clip_batch("delta", angle=1.0)
        comb_inc = EvidenceBatch(
            match_id=b_inc1.match_id,
            batch_id="b_inc",
            producer="test",
            clips=[*b_inc1.clips, *b_inc2.clips],
            identity_evidence=IdentityEvidence(
                tracks=[*b_inc1.identity_evidence.tracks, *b_inc2.identity_evidence.tracks],
                observations=[
                    *b_inc1.identity_evidence.observations,
                    *b_inc2.identity_evidence.observations,
                ],
                segments=[*b_inc1.identity_evidence.segments, *b_inc2.identity_evidence.segments],
                samples=[*b_inc1.identity_evidence.samples, *b_inc2.identity_evidence.samples],
                feature_spaces=b_inc1.identity_evidence.feature_spaces,
            ),
        )
        seg_inc = segment_tracks(comb_inc)

        db = connect(db_path)
        inc_res = update_incremental(
            db,
            UpdateRequest(batch=seg_inc.batch, expected_revision=accepted.revision),
            segmentation_report=seg_inc.report,
            feature_values={},
            settings=settings,
        )
        assert inc_res.revision == accepted.revision + 1
        db.close()

        # Reopen SQLite database again to verify incremental persistence
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        all_templates_after = [
            (dict(row), json.loads(row["json"]))
            for row in con.execute(
                "SELECT template_id, sample_id, player_id, json FROM templates WHERE current=1"
            )
        ]
        active_templates_after = [t for t in all_templates_after if t[1].get("active")]
        assert len(active_templates_after) > len(active_templates)
        proto_rows_after = [
            dict(row)
            for row in con.execute(
                "SELECT template_id, prototype_id, sample_id, player_id, json FROM template_prototypes"
            )
        ]
        assert len(proto_rows_after) == len(active_templates_after)
        for row, t_json in active_templates_after:
            assert t_json.get("prototype_id") is not None
            assert t_json["prototype_id"].startswith("proto_")
        con.close()
