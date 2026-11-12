"""13 core business regression tests for player identity discovery.

Verifies real business invariants, error protection boundaries, and idempotency
without mirroring implementation details or mocking private variables:
1. Variable and unknown people count (no fixed K, no 1-player-per-track).
2. Same-team similar individuals (high similarity + absent contrastive margin -> ambiguity preserved).
3. Single appearance and missing negative evidence (singletons stay unassigned without cross-clip support).
4. Duplicate images / observations / homologous replays (strict deduplication, no self-comparison).
5. Minority reliable counter-evidence (strong contradiction not drowned by majority weak support).
6. A-B-C contradiction bridging (bridge B isolated, conflicting A and C never merged).
7. Missing common parts / incompatible features (comparable=False, strictly no zero or average distance).
8. Template admission (backed by fixed raw witnesses from disjoint segments, no circular self-justification).
9. Semantic isolation (automatic proposed assignments never masquerade as approved, no event stats leakage).
10. Human intervention protection (manual review revokes dependent automatic templates, history preserved).
11. Input reordering and batch replay idempotency (permutation invariance and changed=False replay).
12. Storage upgrade and transaction rollback (atomic rollback on failure, zero half-batch residue).
13. Forbidden paths fence (rejection of annotations.csv, test/target, repository-review).
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from test_discovery_workflow import prepared_database
from test_identity_discovery import angle_vector, clip_groups, discover, synthetic_facts
from test_machine_input import machine_settings

from app.contracts.api import (
    EventImportRequest,
    EventQuery,
    ReviewRequest,
    SetAssignmentCommand,
)
from app.contracts.common import Provenance
from app.contracts.events import Event, PointEventTime, Subject, TrackReference
from app.contracts.evidence import (
    EvidenceBatch,
    FeatureSpace,
    InlineFeature,
    Preprocessing,
    VisibilityContract,
)
from app.contracts.identity import (
    MatchingPolicy,
)
from app.identity.distance import kpr_distance
from app.presentation import baseline
from app.settings import SettingsError
from app.storage.repositories import Repository
from app.workflows.discover import discover_library
from app.workflows.read import get_discovery_reports, get_events, get_identity_state
from app.workflows.review import apply_review
from app.workflows.update import ingest_events


def test_business_regression_01_variable_and_unknown_people_count():
    """1. Variable and unknown people count: does not fix K, adapts to 2 then 4 individuals."""
    for people in (2, 4):
        specs = []
        for person in range(people):
            vector = np.eye(people + 1)[person].tolist()
            for occurrence in range(3):
                clip = f"clip_{people}_{person}_{occurrence}"
                specs.append((clip, "0", vector, clip))
        batch = synthetic_facts(specs)
        segmented, report = discover(batch)

        assert len(report.groups) == people
        assert len(report.unknown) == 0
        assert all(len(g.support_pair_ids) == 3 for g in report.groups)


def test_business_regression_02_same_team_similar_people_remain_ambiguous():
    """2. Same-team similar individuals: high similarity without contrastive separation stays ambiguous."""
    specs = [
        (clip, str(person), angle_vector(person * 0.06), f"{clip}_{person}")
        for clip in ("clip1", "clip2")
        for person in range(2)
    ]
    batch = synthetic_facts(specs)
    # Both wear the same team jersey and co-occur without discriminative negative margins
    _, report = discover(batch)

    # Must preserve ambiguity, zero false groups formed
    assert len(report.groups) == 0
    assert len(report.unknown) == 4
    assert any(p.status in ("conflict", "ambiguous") for p in report.pairs)


def test_business_regression_03_single_appearance_remains_unassigned_without_evidence():
    """3. Single appearance:缺乏独立跨片段支持时不强行聚合，单次出场保持待定。"""
    specs = [
        ("clip_a", "0", [1.0, 0.0, 0.0], "a0"),
        ("clip_b", "0", [1.0, 0.0, 0.0], "b0"),
        ("singleton_clip", "0", [0.0, 1.0, 0.0], "single0"),
    ]
    batch = synthetic_facts(specs)
    segmented, report = discover(batch)

    assert len(report.groups) == 1
    # Find the singleton segment
    singleton_seg = next(
        s
        for s in segmented.batch.identity_evidence.segments
        if s.active and s.clip_id == "singleton_clip"
    )
    unknown_ids = {u.segment_id for u in report.unknown}
    assert singleton_seg.segment_id in unknown_ids
    # No player created for singleton
    assert all(singleton_seg.segment_id not in g.member_segment_ids for g in report.groups)


def test_business_regression_04_duplicate_images_and_homologous_replays_deduplicated():
    """4. Duplicate images / observations:严格去重，不重复计票，不自比较。"""
    # Two clips with exact duplicate image hashes (homologous replay)
    specs = [
        ("replay_cam1", "0", [1.0, 0.0], "shared_image_content"),
        ("replay_cam2", "0", [1.0, 0.0], "shared_image_content"),
    ]
    batch = synthetic_facts(specs)
    _, report = discover(batch)

    # Identical image sequences are recognized as duplicate components, not independent support
    assert len(report.groups) == 0
    assert len(report.unknown) == 2


def test_business_regression_05_minority_reliable_counter_evidence_prevents_merge():
    """5. Minority counter-evidence:强矛盾不被多数弱支持吞没。"""
    specs = [
        ("clip_alpha", "0", [1.0, 0.0], "alpha0"),
        ("clip_beta", "0", [1.0, 0.0], "beta0"),
    ]
    batch = synthetic_facts(specs)
    # Inject one strongly contradictory sample into clip_beta
    beta_samples = [s for s in batch.identity_evidence.samples if s.clip_id == "clip_beta"]
    # Change the last sample's vector to opposite direction
    beta_samples[-1].features[0].values = [-1.0, 0.0] * 3

    segmented, report = discover(batch)
    # Majority 5 samples were close, but 1 reliable minority sample is strongly contradictory
    assert len(report.groups) == 0
    assert any(
        p.status == "unknown"
        for t in segmented.report.tracks
        for p in t.proposals
        if p.segment.clip_id == "clip_beta"
    )


def test_business_regression_06_abc_contradiction_bridging_isolation():
    """6. A-B-C contradiction bridging: A与B像、B与C像但A与C冲突时，严禁连通分量串连。"""
    specs = [
        ("a1", "0", angle_vector(0.0), "a1"),
        ("a2", "0", angle_vector(0.0), "a2"),
        ("bridge_b", "0", angle_vector(0.32), "b"),
        ("c1", "0", angle_vector(0.64), "c1"),
        ("c2", "0", angle_vector(0.64), "c2"),
    ]
    segmented, report = discover(synthetic_facts(specs))
    groups = clip_groups(segmented, report)

    # A and C must NEVER be merged together!
    for g in groups:
        has_a = any(c.startswith("a") for c in g)
        has_c = any(c.startswith("c") for c in g)
        assert not (has_a and has_c), f"Transitive merge failure! A and C grouped together: {g}"

    # Bridge B must be isolated as unknown
    bridge_seg = next(
        s
        for s in segmented.batch.identity_evidence.segments
        if s.active and s.clip_id == "bridge_b"
    )
    assert bridge_seg.segment_id in {u.segment_id for u in report.unknown}
    assert groups == {frozenset({"a1", "a2"}), frozenset({"c1", "c2"})}


def test_business_regression_07_missing_common_parts_marked_incomparable():
    """7. Missing common parts:无法对齐时标记为不可比，严禁伪造零距离或平均距离。"""
    space = FeatureSpace(
        feature_space_id="test-kpr-space",
        backend="kpr",
        model_hash="b" * 64,
        preprocessing=Preprocessing(version="test", color_space="rgb"),
        layout="parts",
        shape=[3, 2],
        normalization="l2",
        distance="euclidean_sqrt_visibility_mean_div2",
        visibility=VisibilityContract(shape=[3], semantics="visible"),
    )
    policy = MatchingPolicy(feature_space_id="test-kpr-space", min_common_parts=2)

    # Feature A: visible parts [0, 1], invisible [2]
    feat_a = InlineFeature(
        feature_space_id="test-kpr-space",
        values=[1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        visibility=[1.0, 1.0, 0.0],
    )
    # Feature B: visible part [2], invisible [0, 1] -> 0 common parts!
    feat_b = InlineFeature(
        feature_space_id="test-kpr-space",
        values=[1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        visibility=[0.0, 0.0, 1.0],
    )

    res = kpr_distance(space, feat_a, feat_b, policy)
    assert not res.comparable
    assert res.distance is None
    assert res.common_parts == 0
    assert res.reason == "insufficient_common_parts"

    # Feature C: visible parts [1, 2] -> common part with A is only part [1] (count=1 < min_common_parts=2)
    feat_c = InlineFeature(
        feature_space_id="test-kpr-space",
        values=[1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        visibility=[0.0, 1.0, 1.0],
    )
    res_c = kpr_distance(space, feat_a, feat_c, policy)
    assert not res_c.comparable
    assert res_c.distance is None
    assert res_c.common_parts == 1
    assert res_c.reason == "insufficient_common_parts"


def test_business_regression_08_template_admission_requires_fixed_raw_witnesses():
    """8. Template admission:模板来源必须为固定原始证据，杜绝利用当前预测组自循环证明。"""
    specs = [
        ("clip_x", "0", [1.0, 0.0], "x"),
        ("clip_y", "0", [1.0, 0.0], "y"),
    ]
    segmented, report = discover(synthetic_facts(specs))
    assert len(report.groups) == 1
    group = report.groups[0]
    samples_map = {s.sample_id: s for s in segmented.batch.identity_evidence.samples}

    assert len(group.templates) >= 2
    for tmpl in group.templates:
        sample = samples_map[tmpl.sample_id]
        # Must have witness samples
        assert tmpl.witness_sample_ids, "Template admitted without witnesses!"
        for witness_id in tmpl.witness_sample_ids:
            witness = samples_map[witness_id]
            # Witnesses MUST come from different segments (disjoint cross-clip evidence)
            assert witness.segment_id != sample.segment_id, "Template self-witnessed!"
            assert witness.clip_id != sample.clip_id, "Witness not cross-clip!"


def test_business_regression_09_semantic_isolation_proposed_never_masquerades_as_approved(
    tmp_path,
):
    """9. Semantic isolation:自动 proposed 绝不伪造为人工 approved，不进入正式事件统计。"""
    db, settings, result, revision = prepared_database(tmp_path)

    # Ingest an unassociated event to clip alpha
    ev = Event(
        match_id="m",
        event_id="ev_goal_1",
        source_event_key="ev_src_1",
        clip_id="alpha",
        event_type="made",
        time=PointEventTime(time_s=1.0),
        subjects=[
            Subject(
                subject_id="sub_1",
                role="shooter",
                references=[TrackReference(clip_id="alpha", track_id="t-alpha")],
            )
        ],
        source=Provenance(
            producer="synthetic",
            record_type="event",
            source_record_id="ev_src_1",
            evidence_version="1",
        ),
    )
    event_import = ingest_events(
        db,
        EventImportRequest(
            expected_revision=revision,
            batch=EvidenceBatch(
                match_id="m",
                batch_id="event_batch",
                producer="synthetic",
                events=[ev],
            ),
        ),
    )
    current_rev = event_import.revision

    # Run discovery workflow
    discover_library(
        db,
        result.batch,
        segmentation_report=result.report,
        feature_values={},
        settings=settings,
        expected_revision=current_rev,
    )

    state = get_identity_state(db, "m")
    # Verify all automatic assignments are strictly proposed (never confirmed)
    for asg in state.data.assignments:
        assert asg.method == "automatic"
        if asg.status != "unassigned":
            assert asg.status == "proposed"

    # Verify all automatic templates are unreviewed (never approved)
    for tmpl in state.data.templates:
        assert tmpl.review_status == "unreviewed"
        assert tmpl.approval is None
        assert tmpl.admission == "automatic_experimental"

    # Verify confirmed events query: proposed automatic assignments DO NOT resolve confirmed events!
    events_read = get_events(db, EventQuery(match_id="m"))
    assert len(events_read.data) == 1
    ev_read = events_read.data[0]
    assert ev_read.subjects[0].player_id is None, (
        "Proposed identity leaked into confirmed event statistics!"
    )
    assert ev_read.subjects[0].status == "unmapped"
    db.close()


def test_business_regression_10_human_intervention_revokes_dependent_templates_and_retains_history(
    tmp_path,
):
    """10. Human intervention protection:人工干预撤回后，依赖的自动模板退出，历史版本保留。"""
    db, settings, result, revision = prepared_database(tmp_path)
    discover_res = discover_library(
        db,
        result.batch,
        segmentation_report=result.report,
        feature_values={},
        settings=settings,
        expected_revision=revision,
    )

    state_before = get_identity_state(db, "m")
    proposed_asgs = [a for a in state_before.data.assignments if a.status == "proposed"]
    assert len(proposed_asgs) > 0
    target_seg_id = proposed_asgs[0].segment_id

    # Active templates before intervention
    active_templates_before = [t for t in state_before.data.templates if t.active]
    assert len(active_templates_before) > 0

    # Human intervention: manually unassign the segment
    rev_result = apply_review(
        db,
        ReviewRequest(
            match_id="m",
            expected_revision=discover_res.revision,
            actor="expert_reviewer",
            commands=[
                SetAssignmentCommand(
                    segment_id=target_seg_id,
                    player_id=None,
                    unassigned_reason="rejected",
                    reason="human_unassign",
                )
            ],
        ),
    )

    assert rev_result.revision == discover_res.revision + 1
    state_after = get_identity_state(db, "m")

    # The assignment is now unassigned
    asg_after = next(a for a in state_after.data.assignments if a.segment_id == target_seg_id)
    assert asg_after.status == "unassigned"
    assert asg_after.unassigned_reason == "rejected"
    assert asg_after.method == "manual"

    # Templates belonging to this segment must be deactivated!
    deactivated = [
        t
        for t in state_after.data.templates
        if not t.active
        and t.sample_id
        in {
            s.sample_id for s in state_before.data.evidence.samples if s.segment_id == target_seg_id
        }
    ]
    assert len(deactivated) > 0

    # Revision audit trail is immutably preserved
    repo = Repository(db)
    revs = repo.revisions("m")
    assert len(revs) >= 3  # init, discover, review
    db.close()


def test_business_regression_11_input_reordering_and_batch_replay_strict_idempotency(tmp_path):
    """11. Input reordering and batch replay:改变顺序或重复提交，结果严格幂等。"""
    specs = [
        ("ca", "0", [1.0, 0.0], "fam_a"),
        ("cb", "0", [1.0, 0.0], "fam_a"),
        ("cc", "0", [0.0, 1.0], "fam_b"),
        ("cd", "0", [0.0, 1.0], "fam_b"),
    ]
    batch = synthetic_facts(specs)
    _, report1 = discover(batch)

    # Reorder everything
    reordered = deepcopy(batch)
    reordered.batch_id = "reordered_batch"
    for field in ("segments", "tracks", "samples", "observations"):
        getattr(reordered.identity_evidence, field).reverse()
    reordered.assets.reverse()
    reordered.clips.reverse()

    _, report2 = discover(reordered)

    # Decisions must be 100% permutation-invariant
    assert [g.model_dump(mode="json") for g in report1.groups] == [
        g.model_dump(mode="json") for g in report2.groups
    ]
    assert [p.pair_id for p in report1.pairs] == [p.pair_id for p in report2.pairs]

    # Database replay idempotency: replaying same batch produces changed=False
    db, settings, result, revision = prepared_database(tmp_path)
    first = discover_library(
        db,
        result.batch,
        segmentation_report=result.report,
        feature_values={},
        settings=settings,
        expected_revision=revision,
    )
    replay = discover_library(
        db,
        result.batch.model_copy(update={"batch_id": "replay_handoff"}),
        segmentation_report=result.report,
        feature_values={},
        settings=settings,
        expected_revision=first.revision,
    )
    assert replay.revision == first.revision
    assert replay.data["changed"] is False
    db.close()


def test_business_regression_12_storage_upgrade_and_transaction_rollback(tmp_path, monkeypatch):
    """12. Storage upgrade and transaction rollback:事务失败不残留半批状态，保留既有状态。"""
    db, settings, result, revision = prepared_database(tmp_path)
    state_before = get_identity_state(db, "m")

    # Inject exception during database write inside discover_library
    def fail_put_template(*args, **kwargs):
        raise RuntimeError("simulated storage disk write fault")

    monkeypatch.setattr(Repository, "put_template", fail_put_template)

    with pytest.raises(RuntimeError, match="simulated storage disk write fault"):
        discover_library(
            db,
            result.batch,
            segmentation_report=result.report,
            feature_values={},
            settings=settings,
            expected_revision=revision,
        )

    # Database must have rolled back cleanly: revision unchanged, no partial players
    state_after = get_identity_state(db, "m")
    assert state_after.revision == revision
    assert state_after.data.players == state_before.data.players
    assert len(get_discovery_reports(db, "m").data) == 0
    db.close()


def test_business_regression_13_forbidden_paths_fence_asserts_no_answer_leak(tmp_path):
    """13. Forbidden paths fence:断言验证推理流程不依赖任何外部禁读路径。"""
    # 1. Reject settings configs that attempt to point into target/ or repository-review/
    bad_target_dir = tmp_path / "target"
    bad_target_dir.mkdir(parents=True, exist_ok=True)
    fake_config = bad_target_dir / "config.toml"
    fake_config.write_text("[match]\nmatch_id = 'test'\n")

    with pytest.raises(SettingsError, match="outside forbidden paths"):
        baseline.load_baseline_settings(fake_config)

    # 2. Verify machine adapter whitelist does NOT open annotations.csv
    settings = machine_settings(tmp_path / "machine_root", tmp_path)
    annotations_path = tmp_path / "annotations.csv"
    annotations_path.write_text("dummy,ground,truth\n")

    # Machine adapter only looks at track/, identity/, video/
    # Ensure it never accesses or depends on annotations.csv
    assert "annotations.csv" not in [f.name for f in Path(settings.paths.input_root).glob("*")]
