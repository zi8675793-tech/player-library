"""Adversarial local recovery: raw evidence and physical exclusions stay authoritative."""

from types import SimpleNamespace

import pytest
from test_track_consistency_v3 import make_v3_batch, source

from app.contracts.discovery import DiscoveryPolicy
from app.contracts.identity import IdentityRisk
from app.contracts.segmentation import SegmentationPolicy
from app.identity.consistency import segment_tracks
from app.identity.discovery import evaluate_existing_player_match
from app.identity.prepare import active_risks


def proposals(result):
    return [p for t in result.report.tracks for p in t.proposals]


def local_policy(**kwargs):
    return SegmentationPolicy(
        version="track-consistency-v3", localize_unresolved_evidence=True, **kwargs
    )


@pytest.mark.parametrize("gap,eligible", [(1.6, True), (4.5, True), (4.6, False)])
def test_archive_candidate_obeys_continuous_v3_gap(gap, eligible):
    # Ambiguous appearance cannot build a new identity, even with continuous tracking.
    batch = make_v3_batch(angles=(0.0, 0.5), sample_frames=[0, round(gap * 10)])
    result = segment_tracks(batch, policy=SegmentationPolicy(version="track-consistency-v3"))
    assert any(p.requires_external_anchor for p in proposals(result)) == eligible
    assert not any(p.can_create_new_identity for p in proposals(result))


def test_observation_interruption_cannot_be_filled_by_archive_continuity():
    batch = make_v3_batch(angles=(0.0, 0.5), sample_frames=[0, 30], missing_frames=range(8, 20))
    result = segment_tracks(batch, policy=local_policy())
    for p in proposals(result):
        if p.segment.start_s <= 1.0 < p.segment.end_s:
            assert not p.can_match_existing
    assert any(p.evidence_semantics == "observation_missing" for p in proposals(result))


def test_minority_foreign_image_does_not_contaminate_anchors_or_clean_flanks():
    batch = make_v3_batch(angles=(0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0))
    before = batch.model_dump_json()
    result = segment_tracks(batch, policy=local_policy())
    supported = [p for p in proposals(result) if p.status == "supported"]
    assert len(supported) == 2
    assert all("s-0-15" not in p.support_sample_ids for p in supported)
    anomaly = next(p for p in proposals(result) if "s-0-15" in p.support_sample_ids)
    assert anomaly.independent_anchor_required and not anomaly.can_create_new_identity
    assert anomaly.segment.end_s - anomaly.segment.start_s == pytest.approx(0.1)
    assert batch.model_dump_json() == before


@pytest.mark.parametrize("occluded,unusable", [((1,), ()), ((), (1,))])
def test_low_quality_has_no_archive_candidate(occluded, unusable):
    batch = make_v3_batch(
        angles=(0.0, 0.5, 0.0), occluded_indices=occluded, unusable_indices=unusable
    )
    result = segment_tracks(batch, policy=local_policy())
    affected = next(p for p in proposals(result) if p.segment.start_s <= 0.5 < p.segment.end_s)
    assert not affected.can_match_existing
    assert affected.evidence_semantics == "low_quality_incomparable"


def archive_match(ds=(0.2, 0.21), clips=("a", "b"), *, components=None, forbidden=False):
    q = SimpleNamespace(sample_id="q", clip_id="query")
    refs = [SimpleNamespace(sample_id=f"t{i}", clip_id=c) for i, c in enumerate(clips)]
    comp = {s.sample_id: s.sample_id for s in [q, *refs]}
    if components:
        comp.update(components)
    trace = {}
    result = evaluate_existing_player_match(
        SimpleNamespace(segment_id="s"),
        [q],
        [q],
        {"P": {}},
        {"P": [(None, t) for t in refs]},
        SimpleNamespace(pair=lambda q, t: SimpleNamespace(distance=ds[int(t.sample_id[1:])])),
        comp,
        DiscoveryPolicy(),
        require_all_samples=True,
        require_independent_anchor=True,
        forbidden_players={"P"} if forbidden else set(),
        diagnostics=trace,
    )
    return result, trace


def test_atomic_image_recovers_only_with_two_independent_external_clip_witnesses():
    result, trace = archive_match()
    assert result[0] == "P"
    assert len(trace["gates"]["external_anchor"]["witnesses"]["q"]) == 2
    assert trace["require_independent_anchor"]


@pytest.mark.parametrize(
    "ds,clips,components",
    [
        ((0.2, 0.21), ("a", "a"), None),
        ((0.2, 0.21), ("query", "a"), None),
        ((0.2, 0.35), ("a", "b"), None),
        ((0.2, 0.21), ("a", "b"), {"t1": "t0"}),
    ],
)
def test_atomic_recovery_rejects_dependent_or_missing_witness(ds, clips, components):
    assert archive_match(ds, clips, components=components)[0][0] is None


def test_foreign_atomic_sample_and_manual_or_spatial_forbid_are_never_overruled():
    assert archive_match((0.5, 0.51))[0][0] is None
    assert archive_match(forbidden=True)[0][0] is None


def test_duplicate_advisory_is_localized_but_human_risk_is_retained():
    batch = make_v3_batch(angles=(0.0,) * 6, duplicate_box=True)
    parent = batch.identity_evidence.segments[0]
    for basis in ("machine_duplicate_candidate", "unspecified"):
        raw = batch.model_copy(deep=True)
        raw.identity_evidence.risks.append(
            IdentityRisk(
                risk_id="advisory",
                kind="duplicate_box",
                basis=basis,
                match_id="m",
                segment_id=parent.segment_id,
                clip_id="c1",
                track_id="0",
                start_s=0.0,
                end_s=parent.end_s,
                source=source("advisory"),
            )
        )
        before = raw.model_dump_json()
        result = segment_tracks(raw, policy=local_policy(localize_duplicate_risks=True))
        clean = next(p for p in proposals(result) if p.status == "supported")
        duplicates = [
            r
            for r in active_risks(result.batch.identity_evidence, clean.segment)
            if r.kind == "duplicate_box"
        ]
        assert bool(duplicates) == (basis == "unspecified")
        affected = next(p for p in proposals(result) if p.segment.start_s <= 1.0 < p.segment.end_s)
        assert not affected.can_match_existing
        assert affected.evidence_semantics == "duplicate_detection"
        assert raw.model_dump_json() == before


def test_localized_duplicate_risks_have_distinct_durable_sources():
    from app.storage.database import connect, transaction
    from app.storage.repositories import Repository

    batch = make_v3_batch(duplicate_box=True)
    other = next(o for o in batch.identity_evidence.observations if o.track_id == "1")
    batch.identity_evidence.observations.append(
        other.model_copy(
            update={
                "observation_id": "duplicate-later",
                "time_s": 1.5,
                "frame_idx": 15,
                "source": source("duplicate-later"),
            }
        )
    )
    batch.identity_evidence.tracks[1].observed_frame_indices.append(15)
    parent = batch.identity_evidence.segments[0]
    batch.identity_evidence.risks.append(
        IdentityRisk(
            risk_id="global-duplicate",
            kind="duplicate_box",
            basis="machine_duplicate_candidate",
            match_id="m",
            segment_id=parent.segment_id,
            clip_id="c1",
            track_id="0",
            start_s=0.0,
            end_s=parent.end_s,
            source=source("global-duplicate"),
        )
    )
    result = segment_tracks(batch, policy=local_policy(localize_duplicate_risks=True))
    db = connect(":memory:")
    with transaction(db):
        Repository(db).ensure_match("m")
        Repository(db).store_batch(result.batch)
    with transaction(db):
        assert not Repository(db).store_batch(result.batch)
    db.close()


def test_atomic_archive_contract_rejects_multiframe_propagation():
    from app.contracts.segmentation import SegmentProposal

    result = segment_tracks(make_v3_batch(angles=(0.0, 0.5)), policy=local_policy())
    proposal = next(p for p in proposals(result) if p.independent_anchor_required)
    data = proposal.model_dump()
    data["segment"]["frame_end_exclusive"] += 2
    with pytest.raises(ValueError, match="beyond one frame"):
        SegmentProposal.model_validate(data)


def test_atomic_candidates_never_cover_inter_sample_gaps_or_create_identity():
    result = segment_tracks(make_v3_batch(angles=(0.0, 0.5, 0.0, 0.5)), policy=local_policy())
    anchors = [p for p in proposals(result) if p.independent_anchor_required]
    assert len(anchors) == 4
    assert all(not p.can_create_new_identity and p.status == "unknown" for p in anchors)
    assert sum(p.segment.end_s - p.segment.start_s for p in anchors) == pytest.approx(0.4)
    pieces = proposals(result)
    assert pieces[0].segment.start_s == 0.0
    assert all(a.segment.end_s == b.segment.start_s for a, b in zip(pieces, pieces[1:]))


def test_cfr_exposure_boundaries_use_integer_frames_without_float_overlap():
    # 545/60 + 1/60 is 9.100000000000001: it must not absorb frame 546.
    batch = make_v3_batch(angles=(0.0, 0.5, 0.0, 0.5), sample_frames=[544, 545, 546, 580], fps=60.0)
    result = segment_tracks(batch, policy=local_policy())
    pieces = proposals(result)
    assert all(a.segment.end_s == b.segment.start_s for a, b in zip(pieces, pieces[1:]))
    assert all(
        p.segment.frame_start <= s.frame_idx < p.segment.frame_end_exclusive
        for p in pieces
        for s in result.batch.identity_evidence.samples
        if s.segment_id == p.segment.segment_id
    )


def test_persistent_contradiction_is_distinct_from_unknown_view_and_low_quality():
    result = segment_tracks(
        make_v3_batch(angles=(0.0, 0.0, 0.0, 2.0, 2.0, 2.0)), policy=local_policy()
    )
    findings = result.report.tracks[0].findings
    sustained = [f for f in findings if f.kind == "sustained_appearance_contradiction"]
    assert len(sustained) == 1
    assert not sustained[0].details["identity_switch_proven"]
    assert all(
        p.support_sample_ids != ["s-0-0", "s-0-5", "s-0-10", "s-0-15"] for p in proposals(result)
    )


def test_workflow_partial_archive_recovery_keeps_foreign_sample_and_gaps_unknown(tmp_path):
    import math

    from test_identity_discovery import angle_vector, synthetic_facts
    from test_incremental_workflow import make_incremental_settings

    from app.contracts.api import UpdateRequest
    from app.storage.database import connect
    from app.storage.snapshot import read_identity_snapshot
    from app.workflows.incremental import update_incremental

    settings = make_incremental_settings(tmp_path)
    archive = synthetic_facts([("a", "1", angle_vector(0), "a"), ("b", "1", angle_vector(0), "b")])
    db = connect(tmp_path / "local.sqlite3")
    first_seg = segment_tracks(archive, policy=local_policy())
    first = update_incremental(
        db,
        UpdateRequest(batch=first_seg.batch, expected_revision=0),
        segmentation_report=first_seg.report,
        settings=settings,
    )
    initial = read_identity_snapshot(db, archive.match_id)
    assert len(initial.gallery.players) == 1
    raw = synthetic_facts([("query", "1", angle_vector(0), "query")])
    for sample, angle in zip(
        raw.identity_evidence.samples, (0.5, 0.0, 2.0, 0.5, 0.0, 0.5), strict=True
    ):
        sample.features[0].values = [math.cos(angle), math.sin(angle)] * 3
    segmented = segment_tracks(raw, policy=local_policy())
    second = update_incremental(
        db,
        UpdateRequest(batch=segmented.batch, expected_revision=first.revision),
        segmentation_report=segmented.report,
        settings=settings,
    )
    final = read_identity_snapshot(db, archive.match_id)
    assignments = {a.segment_id: a.player_id for a in final.gallery.assignments}
    query = [s for s in final.gallery.evidence.samples if s.clip_id == "query"]
    assert sum(bool(assignments[s.segment_id]) for s in query) == 5
    foreign = next(s for s in query if s.frame_idx == 10)
    assert assignments[foreign.segment_id] is None
    assert len(final.gallery.templates) == len(initial.gallery.templates)
    assert len(final.gallery.players) == 1
    for p in proposals(segmented):
        if not p.support_sample_ids:
            assert assignments[p.segment.segment_id] is None
    replay = update_incremental(
        db,
        UpdateRequest(batch=segmented.batch, expected_revision=second.revision),
        segmentation_report=segmented.report,
        settings=settings,
    )
    assert not replay.data["changed"]
    assert replay.revision == second.revision
    db.close()


def test_separated_concurrent_track_cannot_use_two_external_archive_witnesses(tmp_path):
    import math

    from test_identity_discovery import angle_vector, synthetic_facts
    from test_incremental_workflow import make_incremental_settings

    from app.contracts.api import UpdateRequest
    from app.storage.database import connect
    from app.storage.snapshot import read_identity_snapshot
    from app.workflows.incremental import update_incremental

    settings = make_incremental_settings(tmp_path)
    raw = synthetic_facts([(c, "1", angle_vector(0), c) for c in ("a", "b", "c")])
    db = connect(":memory:")
    seg = segment_tracks(raw, policy=local_policy())
    first = update_incremental(
        db,
        UpdateRequest(batch=seg.batch, expected_revision=0),
        segmentation_report=seg.report,
        settings=settings,
    )
    query = synthetic_facts([("a", "2", angle_vector(0), "other-body")])
    for sample, angle in zip(query.identity_evidence.samples, (0.5, 0.0) * 3, strict=True):
        sample.features[0].values = [math.cos(angle), math.sin(angle)] * 3
    seg2 = segment_tracks(query, policy=local_policy())
    update_incremental(
        db,
        UpdateRequest(batch=seg2.batch, expected_revision=first.revision),
        segmentation_report=seg2.report,
        settings=settings,
    )
    state = read_identity_snapshot(db, raw.match_id)
    assignments = {a.segment_id: a.player_id for a in state.gallery.assignments}
    assert all(not assignments[p.segment.segment_id] for p in proposals(seg2))
    from app.storage.matching_diagnostics import read_matching_diagnostics

    assert any(
        "forbidden_by_spatiotemporal_or_manual_constraint" in candidate["exclusion_reasons"]
        for audit in read_matching_diagnostics(db, raw.match_id)
        for attempt in audit["attempts"].values()
        for candidate in attempt.get("existing_match", {}).get("candidates", {}).values()
    )
    db.close()


@pytest.mark.parametrize("angles", [(0.5, 2.0, 2.0, 2.0, 2.0, 2.0), (0.5, 2.0, 2.0, 2.0, 2.0, 0.5)])
def test_single_hit_or_far_intruder_between_archive_hits_cannot_certify_recovery(tmp_path, angles):
    import math

    from test_identity_discovery import angle_vector, synthetic_facts
    from test_incremental_workflow import make_incremental_settings

    from app.contracts.api import UpdateRequest
    from app.storage.database import connect
    from app.storage.snapshot import read_identity_snapshot
    from app.workflows.incremental import update_incremental

    settings = make_incremental_settings(tmp_path)
    db = connect(":memory:")
    archive = synthetic_facts([("a", "1", angle_vector(0), "a"), ("b", "1", angle_vector(0), "b")])
    initial = segment_tracks(archive, policy=local_policy())
    first = update_incremental(
        db,
        UpdateRequest(batch=initial.batch, expected_revision=0),
        segmentation_report=initial.report,
        settings=settings,
    )
    query = synthetic_facts([("query", "1", angle_vector(0), "query")])
    for sample, angle in zip(query.identity_evidence.samples, angles, strict=True):
        sample.features[0].values = [math.cos(angle), math.sin(angle)] * 3
    segmented = segment_tracks(query, policy=local_policy())
    update_incremental(
        db,
        UpdateRequest(batch=segmented.batch, expected_revision=first.revision),
        segmentation_report=segmented.report,
        settings=settings,
    )
    final = read_identity_snapshot(db, archive.match_id)
    query_sids = {p.segment.segment_id for p in proposals(segmented)}
    assert not any(a.player_id for a in final.gallery.assignments if a.segment_id in query_sids)
    assert len(final.gallery.players) == 1
    db.close()
