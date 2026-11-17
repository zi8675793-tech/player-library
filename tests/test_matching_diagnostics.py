"""Gate diagnostics must explain failures without turning them into discovery bans."""

from types import SimpleNamespace

import pytest
from test_identity_discovery import angle_vector, synthetic_facts
from test_incremental_workflow import make_incremental_settings

from app.contracts.api import UpdateRequest
from app.contracts.discovery import DiscoveryPolicy
from app.identity.consistency import segment_tracks
from app.identity.discovery import evaluate_existing_player_match
from app.storage.database import connect, transaction
from app.storage.matching_diagnostics import put_matching_diagnostics, read_matching_diagnostics
from app.storage.repositories import Repository, StorageConflict
from app.workflows.incremental import update_incremental


def match(distances, *, forbidden=(), can_match=True, require_all=False, representative_count=2):
    queries = [SimpleNamespace(sample_id=f"q{i}") for i in range(len(distances["P1"]))]
    templates = {pid: [(None, SimpleNamespace(sample_id=pid, clip_id=pid))] for pid in distances}
    components = {s.sample_id: s.sample_id for s in queries}
    components.update({pid: pid for pid in distances})
    diagnostic = {}
    result = evaluate_existing_player_match(
        SimpleNamespace(segment_id="s"),
        queries,
        queries[:representative_count],
        {pid: {} for pid in distances},
        templates,
        SimpleNamespace(
            pair=lambda q, t: SimpleNamespace(distance=distances[t.sample_id][int(q.sample_id[1:])])
        ),
        components,
        DiscoveryPolicy(counter_evidence_margin=0.06),
        forbidden_players=set(forbidden),
        diagnostics=diagnostic,
        can_match=can_match,
        require_all_samples=require_all,
    )
    return result, diagnostic


def test_forbidden_rival_remains_auditable_but_cannot_veto():
    result, trace = match({"P1": [0.2, 0.2], "P2": [0.13, 0.13]}, forbidden={"P2"})
    assert result[0] == "P1"
    assert trace["eligible_player_ids"] == ["P1"]
    assert not trace["candidates"]["P2"]["eligible"]
    assert trace["candidates"]["P2"]["sample_scores"]["q0"] == 0.13
    assert trace["gates"]["competing_sample"]["status"] == "pass"


def test_insufficiently_comparable_identity_cannot_compete():
    result, trace = match({"P1": [0.2, 0.2], "P2": [0.1, None]})
    assert result[0] == "P1"
    assert trace["candidates"]["P2"]["exclusion_reasons"] == [
        "insufficient_comparable_representatives"
    ]


@pytest.mark.parametrize(
    "distances,gate,reason",
    [
        ({"P1": [0.29, 0.29]}, "distance", "best_match_exceeds_threshold"),
        ({"P1": [0.2, 0.2], "P2": [0.22, 0.22]}, "margin", "unresolved_competing_player_margin"),
        ({"P1": [0.2, 0.2, 0.5]}, "far", "minority_sample_far_from_existing_player"),
        (
            {"P1": [0.2, 0.2], "P2": [0.1, 0.6]},
            "competing_sample",
            "minority_sample_supports_competing_player",
        ),
    ],
)
def test_gate_rejections_retain_values_and_later_gates_are_unvisited(distances, gate, reason):
    result, trace = match(distances)
    assert result[0] is None
    assert result[2] == [reason]
    assert trace["gates"][gate]["status"] == "fail"
    assert trace["gates"]["coverage"]["status"] == "not_evaluated"
    assert trace["best_distance"] is not None


def test_disabled_and_external_anchor_traces():
    _, trace = match({"P1": [0.2, 0.2]}, can_match=False)
    assert trace["gates"]["eligibility"]["status"] == "fail"
    result, trace = match({"P1": [0.2, 0.2, 0.35]}, require_all=True)
    assert result[2] == ["external_anchor_requires_all_sample_support"]
    assert trace["gates"]["external_anchor"]["status"] == "fail"


def test_coverage_failure_is_distinct_from_distance_failure():
    result, trace = match({"P1": [0.2, 0.2, 0.35]}, representative_count=3)
    assert result[2] == ["insufficient_existing_player_coverage"]
    assert trace["gates"]["distance"]["status"] == "pass"
    assert trace["gates"]["coverage"]["value"] == pytest.approx(2 / 3)


def test_manual_ban_is_still_a_hard_constraint():
    result, trace = match({"P1": [0.1, 0.1]}, forbidden={"P1"})
    assert result[0] is None
    assert trace["eligible_player_ids"] == []
    assert trace["gates"]["eligibility"]["status"] == "fail"


def test_diagnostic_storage_is_immutable_atomic_and_match_scoped():
    db = connect(":memory:")
    with transaction(db):
        Repository(db).ensure_match("m")
        put_matching_diagnostics(db, "m", "key", 0, "test", {"s": {"distance": 0.29}})
    with transaction(db):
        put_matching_diagnostics(db, "m", "key", 0, "test", {"s": {"distance": 0.29}})
    assert len(read_matching_diagnostics(db, "m")) == 1
    assert read_matching_diagnostics(db, "other") == []
    with pytest.raises(StorageConflict), transaction(db):
        put_matching_diagnostics(db, "m", "key", 0, "test", {})
    with pytest.raises(RuntimeError), transaction(db):
        put_matching_diagnostics(db, "m", "rolled-back", 0, "test", {})
        raise RuntimeError("abort")
    assert len(read_matching_diagnostics(db, "m")) == 1
    db.close()


def test_workflow_persists_failed_match_without_blocking_new_identity(tmp_path):
    # Two real independent synthetic clips can still create a new identity when
    # there is no existing match; diagnostic reasons must not enter blocking reasons.
    raw = synthetic_facts([("a", "1", angle_vector(0), "a"), ("b", "1", angle_vector(0), "b")])
    values = {}
    segmented = segment_tracks(raw, values)
    db = connect(tmp_path / "diagnostics.sqlite3")
    settings = make_incremental_settings(tmp_path)
    first = update_incremental(
        db,
        UpdateRequest(batch=segmented.batch, expected_revision=0),
        segmentation_report=segmented.report,
        feature_values=values,
        settings=settings,
    )
    audits = read_matching_diagnostics(db, raw.match_id)
    assert audits
    assert any(a["attempts"] for a in audits)
    assert first.data["primary_summary"]["new_players_count"] > 0
    before = audits
    replay = update_incremental(
        db,
        UpdateRequest(batch=segmented.batch, expected_revision=first.revision),
        segmentation_report=segmented.report,
        feature_values=values,
        settings=settings,
    )
    assert not replay.data["changed"]
    assert read_matching_diagnostics(db, raw.match_id) == before
    # A separate new group is too far for old templates, but is allowed to create
    # its own identity. Its failed old-match score must survive close/reopen.
    newcomers = synthetic_facts(
        [
            ("c", "1", angle_vector(0.7), "c"),
            ("d", "1", angle_vector(0.7), "d"),
        ]
    )
    segmented_new = segment_tracks(newcomers, {})
    updated = update_incremental(
        db,
        UpdateRequest(batch=segmented_new.batch, expected_revision=first.revision),
        segmentation_report=segmented_new.report,
        feature_values={},
        settings=settings,
    )
    assert updated.data["primary_summary"]["new_players_count"] == 1
    db.close()
    db = connect(tmp_path / "diagnostics.sqlite3")
    failures = [
        attempt["existing_match"]
        for audit in read_matching_diagnostics(db, raw.match_id)
        for attempt in audit["attempts"].values()
        if attempt.get("existing_match", {}).get("reasons") == ["best_match_exceeds_threshold"]
    ]
    assert failures
    assert all(
        d["best_distance"] > 0.28 and d["gates"]["distance"]["status"] == "fail" for d in failures
    )
    db.close()
