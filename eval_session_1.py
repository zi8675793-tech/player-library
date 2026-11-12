"""Evaluation script for Session 1: Real test data ingestion & zero-seed cold start baseline.

Usage:
    cd player-library
    uv run python eval_session_1.py
"""

import json
import math
from pathlib import Path

import numpy as np

from app.adapters.sample_test import load_sample_test_input
from app.contracts.api import EventQuery, InitializeRequest, PlayerQuery
from app.settings import load_settings
from app.storage.database import connect
from app.storage.features import load_feature_values
from app.storage.repositories import Repository, RevisionConflict, StorageConflict
from app.storage.snapshot import read_identity_snapshot
from app.workflows.initialize import initialize_library
from app.workflows.read import get_events, get_players, get_review_queue

PROJECT_ROOT = Path(__file__).resolve().parent


def run_session_1_evaluation():
    print("=" * 80)
    print("【对话 1：真实数据全量接入与冷启动基线测试】评估脚本开始运行")
    print("=" * 80)

    # 1. 配置加载与真实数据读取
    config_path = PROJECT_ROOT / "examples" / "sample-test.toml"
    assert config_path.exists(), f"Config file not found: {config_path}"
    settings = load_settings(config_path)
    print(f"[OK] 已加载配置: {config_path}")
    print(f"     输入根目录: {settings.paths.input_root}")
    print(f"     运行时目录: {settings.paths.runtime_root}")
    print(f"     比赛 ID: {settings.match.match_id}")

    batch_id = "eval-session-1-batch"
    print("\n[INFO] 使用 load_sample_test_input 读取并装配真实数据...")
    batch, values = load_sample_test_input(settings, batch_id=batch_id)

    print("[OK] 真实数据装配完成:")
    print(f"     Clips: {len(batch.clips)}")
    print(f"     Segments: {len(batch.identity_evidence.segments)}")
    print(f"     Samples: {len(batch.identity_evidence.samples)}")
    print(f"     Feature values (NPZ 行装配): {len(values)}")
    print(f"     Events: {len(batch.events)}")
    print(f"     Warnings: {len(batch.warnings)}")

    # 准备数据库目标路径
    db_path = PROJECT_ROOT / "var" / "test_session_1.sqlite3"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
        print(f"[INFO] 已清理已有测试数据库: {db_path}")

    conn = connect(db_path)
    repo = Repository(conn)

    # =========================================================================
    # TC-1.1: 零种子冷启动初始化
    # =========================================================================
    print("\n" + "=" * 80)
    print("【TC-1.1】零种子冷启动初始化执行...")
    print("=" * 80)

    try:
        current_rev = repo.revision(settings.match.match_id)
    except KeyError:
        current_rev = 0
    assert current_rev == 0, f"Initial revision must be 0, got {current_rev}"

    req_id = "session-1-zero-seed-init"
    actor = "session-1-evaluator"
    init_request = InitializeRequest(
        batch=batch,
        expected_revision=0,
        request_id=req_id,
        actor=actor,
        reason="dialogue 1 zero-seed cold start baseline",
        seeds=[],
    )

    response = initialize_library(conn, init_request, feature_values=values, settings=settings)
    report = response.data

    print("[OK] initialize_library 调用成功返回:")
    print(f"     revision: {response.revision} (预期 1)")
    print(f"     report.initialized: {report.initialized} (预期 False)")
    print(f"     report.changed: {report.changed} (预期 True)")
    print(f"     report.review_count: {report.review_count} (预期 11)")
    print(f"     report.created_player_ids: {report.created_player_ids} (预期 [])")
    print(f"     report.based_revision: {report.based_revision} (预期 0)")

    # 断言 TC-1.1
    assert response.revision == 1, f"Expected revision 1, got {response.revision}"
    assert report.initialized is False, "report.initialized must be False for zero seeds"
    assert report.changed is True, "report.changed must be True on first ingestion"
    assert report.review_count == 11, f"Expected 11 review items, got {report.review_count}"
    assert report.created_player_ids == [], "No player profiles should be created"
    assert repo.revision(settings.match.match_id) == 1, "DB revision must be 1"

    # =========================================================================
    # TC-1.2: 证据与特征物理存储核查
    # =========================================================================
    print("\n" + "=" * 80)
    print("【TC-1.2】证据与特征物理存储核查...")
    print("=" * 80)

    # 1. 验证 10 个 clips
    stored_clips = repo.clips(settings.match.match_id)
    assert len(stored_clips) == 10, f"Expected 10 clips, got {len(stored_clips)}"
    clip_ids = {c.clip_id for c in stored_clips}
    expected_clip_ids = {"3", "6", "11", "20", "30", "37", "43", "50", "63", "68"}
    assert clip_ids == expected_clip_ids, f"Clip IDs mismatch: {clip_ids} != {expected_clip_ids}"
    print(f"[OK] 10 个 Clips 完整落库: {sorted(clip_ids)}")

    # 2. 验证 11 个 tracks 与 segments
    snapshot = read_identity_snapshot(conn, settings.match.match_id)
    stored_tracks = snapshot.gallery.evidence.tracks
    assert len(stored_tracks) == 11, f"Expected 11 tracks, got {len(stored_tracks)}"
    stored_segments = snapshot.gallery.evidence.segments
    assert len(stored_segments) == 11, f"Expected 11 segments, got {len(stored_segments)}"
    for seg in stored_segments:
        assert seg.active is True
        assert seg.end_s > seg.start_s
        assert seg.frame_start is not None and seg.frame_end_exclusive is not None
        assert seg.frame_end_exclusive > seg.frame_start
    print("[OK] 11 个 Tracks 与 11 个 Segments 完整落库，且时间与帧界限均准确有效")

    # 3. 验证 118 个 samples
    stored_samples = snapshot.gallery.evidence.samples
    assert len(stored_samples) == 118, f"Expected 118 samples, got {len(stored_samples)}"
    print("[OK] 118 个 Samples 记录完整存储在 samples 表中")

    # 4. 验证 118 个特征向量及可见性存储完整性 (feature_values 表)
    loaded_features = load_feature_values(conn, settings.match.match_id)
    assert len(loaded_features) == 118, (
        f"Expected 118 loaded feature rows, got {len(loaded_features)}"
    )
    assert set(loaded_features.keys()) == set(values.keys()), "Feature keys mismatch!"

    # 抽样和全量核对特征数值
    for key, expected_feature in values.items():
        actual = loaded_features[key]
        assert actual.feature_space_id == expected_feature.feature_space_id
        # 验证数值为 768 维 float32，可见性为 6 维
        assert len(actual.values) == 6 * 128 == 768
        assert len(actual.visibility) == 6
        # 检查无 NaN、无 Inf
        assert all(math.isfinite(v) for v in actual.values), (
            f"NaN/Inf found in feature values: {key}"
        )
        assert all(math.isfinite(v) for v in actual.visibility), (
            f"NaN/Inf found in visibility: {key}"
        )
        # 检查可见性在 [0, 1] 范围内
        assert all(0.0 <= v <= 1.0 for v in actual.visibility), f"Visibility out of range: {key}"
        # 精确比对浮点数
        np.testing.assert_allclose(
            actual.values,
            expected_feature.values,
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"Values mismatch for {key}",
        )
        np.testing.assert_allclose(
            actual.visibility,
            expected_feature.visibility,
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"Visibility mismatch for {key}",
        )
    print(
        "[OK] 118 个样本特征 (6x128=768 维 float32) 与可见性 (6 维) 精度核验完全一致，无截断或 NaN/Inf"
    )

    # 5. 验证 11 个 Segment 的 Pending 状态与 Workflow Plans 诊断
    assignments = {
        row[0]: json.loads(row[1])
        for row in conn.execute(
            "SELECT segment_id, json FROM assignments WHERE match_id=? AND current=1",
            (settings.match.match_id,),
        ).fetchall()
    }
    assert len(assignments) == 11, f"Expected 11 current assignments, got {len(assignments)}"
    for seg_id, assign in assignments.items():
        assert assign["status"] == "unassigned", f"Segment {seg_id} status is {assign['status']}"
        assert assign["player_id"] is None, f"Segment {seg_id} has unexpected player_id"
        assert assign["method"] == "automatic"
        assert assign["unassigned_reason"] == "unknown"
        assert assign["evidence"][0]["kind"] == "pending_identity"
    print("[OK] 11 个 Segments 的 assignments 状态均严格为 status='unassigned', reason='unknown'")

    # 验证 workflow_plans 表中的算法诊断
    plans = {
        row[0]: json.loads(row[1])
        for row in conn.execute(
            "SELECT segment_id, json FROM workflow_plans WHERE match_id=?",
            (settings.match.match_id,),
        ).fetchall()
    }
    assert len(plans) == 11, f"Expected 11 workflow plans, got {len(plans)}"
    for seg_id, plan in plans.items():
        assert plan["resolution"]["decision"] == "pending"
        assert plan["resolution"]["player_id"] is None
        assert plan["resolution"]["requires_review"] is True
        assert "no_gallery_players" in plan["resolution"]["reasons"]

    # 特殊风险段检查 (Clip 3 track 6, Clip 30 track 5, Clip 68 track 13)
    risk_rows = conn.execute(
        "SELECT risk_id, json FROM risks WHERE match_id=?", (settings.match.match_id,)
    ).fetchall()
    risk_segments = {
        json.loads(row[1])["segment_id"]: json.loads(row[1])["kind"] for row in risk_rows
    }
    print(
        f"[OK] 检测到真实数据中的 4 条质检风险记录，关联 Segment ID: {list(risk_segments.keys())}"
    )
    for seg_id in risk_segments:
        assert "unresolved_identity_risk" in plans[seg_id]["resolution"]["reasons"]
    print(
        "[OK] 11 个 Segments 的 workflow_plans 均包含完整 pending 诊断，风险段准确附带 'unresolved_identity_risk'"
    )

    # =========================================================================
    # TC-1.3: 事件先到模式（Unassociated Events）一致性
    # =========================================================================
    print("\n" + "=" * 80)
    print("【TC-1.3】事件先到模式（Unassociated Events）一致性核查...")
    print("=" * 80)

    events_resp = get_events(conn, EventQuery(match_id=settings.match.match_id))
    resolved_events = events_resp.data
    assert len(resolved_events) == 10, f"Expected 10 resolved events, got {len(resolved_events)}"

    print("[OK] 成功查询到 10 条原始进球事件:")
    for item in resolved_events:
        evt = item.event
        assert evt.outcome == "made", f"Event {evt.event_id} outcome is {evt.outcome}"
        assert evt.event_type == "goal", f"Event {evt.event_id} type is {evt.event_type}"
        for subject in item.subjects:
            assert subject.role in {"shooter", "scorer"}
            assert subject.status == "unmapped", (
                f"Subject {subject.subject_id} status is {subject.status}"
            )
            assert subject.player_id is None, (
                f"Subject {subject.subject_id} unexpectedly has player_id"
            )
            assert subject.reason == "identity_not_confirmed", f"Subject reason: {subject.reason}"
        print(
            f"     Clip {evt.clip_id:>2}: Event {evt.event_id[:16]}... outcome={evt.outcome} | "
            f"Subject status={item.subjects[0].status} (reason={item.subjects[0].reason})"
        )

    # 查询当前球员列表及进球统计
    players_resp = get_players(conn, PlayerQuery(match_id=settings.match.match_id))
    assert len(players_resp.data) == 0, f"Expected 0 players, got {len(players_resp.data)}"
    print("[OK] 当前球员库查询为空 (0 个球员档案)，未发生任何未经人工确认的进球统计伪造")

    # 查询待复核队列
    review_queue_resp = get_review_queue(conn, settings.match.match_id)
    review_items = review_queue_resp.data
    # get_review_queue 汇聚了三类待复核项：11 个 Segment 身份、10 个未关联事件、4 个质检风险 = 25 项
    assert len(review_items) == 25, (
        f"Expected 25 review queue items (11 identity + 10 event + 4 risk), got {len(review_items)}"
    )
    identity_reviews = [item for item in review_items if item.kind == "identity"]
    event_reviews = [item for item in review_items if item.kind == "event"]
    risk_reviews = [item for item in review_items if item.kind == "risk"]

    assert len(identity_reviews) == 11, (
        f"Expected 11 identity review items, got {len(identity_reviews)}"
    )
    assert len(event_reviews) == 10, f"Expected 10 event review items, got {len(event_reviews)}"
    assert len(risk_reviews) == 4, f"Expected 4 risk review items, got {len(risk_reviews)}"

    for q_item in identity_reviews:
        assert q_item.segment_id is not None
        assert "no_gallery_players" in q_item.reasons
    print(
        "[OK] 待复核队列统一聚合 25 个待复核项 (11 个待定身份段 + 10 个未关联事件 + 4 条质检风险)，结构完全符合规范"
    )

    # =========================================================================
    # TC-1.4: 幂等重试机制核查
    # =========================================================================
    print("\n" + "=" * 80)
    print("【TC-1.4】幂等重试机制多维度核查...")
    print("=" * 80)

    # 维度 A: 完全一致请求重放 (相同的 request_id, expected_revision=0, 相同 payload)
    print("[TEST] 维度 A: 相同 request_id 重放（期望从 workflow_operations 命中缓存重放）...")
    replay_resp = initialize_library(conn, init_request, feature_values=values, settings=settings)
    assert replay_resp.revision == 1
    assert replay_resp.data.model_dump() == report.model_dump(), (
        "Replay response must be identical to original"
    )
    print("       [PASS] 成功命中幂等缓存，响应与首次初始化完全一致，revision 保持为 1")

    # 维度 B: 提交相同证据包，但更新 expected_revision=1 且使用新 request_id
    print(
        "[TEST] 维度 B: 递进 revision 重复提交相同证据包（期望 changed=False, duplicate_count=1）..."
    )
    new_req = InitializeRequest(
        batch=batch,
        expected_revision=1,
        request_id="session-1-second-attempt",
        actor=actor,
        reason="dialogue 1 duplicate batch resubmission",
        seeds=[],
    )
    dup_resp = initialize_library(conn, new_req, feature_values=values, settings=settings)
    assert dup_resp.revision == 1, (
        f"Revision should not advance on duplicate data, got {dup_resp.revision}"
    )
    assert dup_resp.data.changed is False, "changed flag must be False for identical data"
    assert dup_resp.data.duplicate_count == 1, "duplicate_count must be 1"
    assert dup_resp.data.inserted_count == 0, "inserted_count must be 0"
    print(
        "       [PASS] 重复提交被底层来源指纹精确拦截，changed=False, duplicate_count=1，未递增 revision"
    )

    # 维度 C: 过期版本冲突校验（使用新 request_id，但残留过期的 expected_revision=0）
    print("[TEST] 维度 C: 过期 revision 冲突检测（期望触发 RevisionConflict）...")
    stale_req = InitializeRequest(
        batch=batch,
        expected_revision=0,
        request_id="session-1-stale-attempt",
        actor=actor,
        reason="stale revision test",
        seeds=[],
    )
    try:
        initialize_library(conn, stale_req, feature_values=values, settings=settings)
        raise AssertionError("Should have raised RevisionConflict")
    except RevisionConflict as e:
        print(f"       [PASS] 成功捕获 RevisionConflict: {e}")

    # 维度 D: 相同 request_id 但篡改 payload 内容（期望触发 StorageConflict）
    print("[TEST] 维度 D: 相同 request_id 篡改载荷（期望触发 StorageConflict）...")
    tampered_batch = batch.model_copy(deep=True)
    tampered_batch.batch_id = "tampered-batch-id"
    tampered_req = InitializeRequest(
        batch=tampered_batch,
        expected_revision=0,
        request_id=req_id,  # 沿用原 request_id
        actor=actor,
        reason="tampered payload test",
        seeds=[],
    )
    try:
        initialize_library(conn, tampered_req, feature_values=values, settings=settings)
        raise AssertionError("Should have raised StorageConflict")
    except StorageConflict as e:
        print(f"       [PASS] 成功捕获 StorageConflict: {e}")

    # =========================================================================
    # 数据库状态全量审计摘要与潜在 Bug 暴露
    # =========================================================================
    print("\n" + "=" * 80)
    print("【数据库全量状态审计摘要与潜在 Bug / 隐患分析】")
    print("=" * 80)

    tables = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    ]

    audit_summary = {}
    print("\n[数据库表记录物理统计]:")
    print(f"{'表名 (Table Name)':<30} | {'记录数 (Row Count)':<10}")
    print("-" * 45)
    for t in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        audit_summary[t] = count
        print(f"{t:<30} | {count:<10}")

    # 关闭数据库
    conn.close()

    print("\n[测试结果与结论]:")
    print("1. 全部断言通过！数据库文件已成功生成至: var/test_session_1.sqlite3")
    print("2. 真实数据 10 个 Clips、11 个 Segments、118 个多部位特征、10 条进球事件均无损持久化")
    print("3. 幂等与并发版本安全机制（4 种分支）验证完毕")
    return audit_summary


if __name__ == "__main__":
    run_session_1_evaluation()
