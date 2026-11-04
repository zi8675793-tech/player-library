"""Conservative decisions over actual synthetic prepare/associate results."""

import pytest

from app.contracts.common import Provenance
from app.contracts.evidence import (
    EvidenceBatch,
    FeatureSpace,
    GallerySnapshot,
    IdentityEvidence,
    InlineFeature,
    Preprocessing,
    Sample,
    TrackingQuality,
    VisibilityContract,
)
from app.contracts.identity import (
    Assignment,
    Calibration,
    ManualConstraint,
    MatchingPolicy,
    NewIdentityRequest,
    Player,
    ReviewRecord,
    Segment,
    Template,
)
from app.identity.associate import associate_segments
from app.identity.prepare import prepare_evidence
from app.identity.resolve import resolve_associations


def policy_for(*, calibrated=False, enabled=False):
    policy = MatchingPolicy(feature_space_id="space")
    if not calibrated:
        return policy
    # This fixture is a synthetic external calibration assertion, not an accuracy claim.
    values = policy.model_dump()
    values.update(tau=0.2, delta=0.2)
    policy = MatchingPolicy(**values)
    values.update(
        auto_assign_enabled=enabled,
        calibration=Calibration(
            status="validated",
            dataset_version="synthetic-resolve-fixture",
            policy_fingerprint=policy.fingerprint,
        ),
    )
    return MatchingPolicy(**values)


def sample_for(segment, index, vector=(1.0, 0.0), *, clean=True):
    sample_id = f"{segment.segment_id}-sample-{index}"
    return Sample(
        match_id="match",
        sample_id=sample_id,
        clip_id=segment.clip_id,
        track_id=segment.track_id,
        segment_id=segment.segment_id,
        time_s=[0.1, 0.6, 1.1][index],
        source=Provenance(
            producer="synthetic",
            record_type="sample",
            source_record_id=sample_id,
            evidence_version="1",
        ),
        features=[
            InlineFeature(feature_space_id="space", values=list(vector) * 2, visibility=[1.0, 1.0])
        ],
        tracking_quality=TrackingQuality(clean_review="approved" if clean else "unreviewed"),
    )


def package(policy, *, query_ids=("query",), vector=(1.0, 0.0), clean=True):
    space = FeatureSpace(
        feature_space_id="space",
        backend="kpr",
        model_hash="a" * 64,
        preprocessing=Preprocessing(version="fixture", color_space="RGB"),
        layout="parts",
        shape=[2, 2],
        normalization="l2",
        distance="euclidean_sqrt_visibility_mean_div2",
        visibility=VisibilityContract(shape=[2], semantics="per-part visibility score in [0,1]"),
    )
    seed_segments, seed_samples, assignments, templates = [], [], [], []
    players = [Player(match_id="match", player_id=f"pl_{name}") for name in ("a", "b")]
    for name, seed_vector in [("a", (1.0, 0.0)), ("b", (0.0, 1.0))]:
        segment = Segment(
            segment_id=f"seed-{name}",
            match_id="match",
            clip_id=f"seed-{name}",
            track_id="track",
            start_s=0.0,
            end_s=1.5,
        )
        seed_segments.append(segment)
        assignments.append(
            Assignment(
                match_id="match",
                segment_id=segment.segment_id,
                player_id=f"pl_{name}",
                status="confirmed",
                method="seed",
                revision=4,
            )
        )
        for index in range(2):
            sample = sample_for(segment, index, seed_vector)
            seed_samples.append(sample)
            templates.append(
                Template(
                    template_id=f"template-{name}-{index}",
                    match_id="match",
                    player_id=f"pl_{name}",
                    sample_id=sample.sample_id,
                    feature_space_id="space",
                    review_status="approved",
                    active=True,
                    revision=4,
                    approval=ReviewRecord(actor="reviewer", reason="clean seed", revision=4),
                )
            )
    snapshot = GallerySnapshot(
        match_id="match",
        revision=4,
        players=players,
        assignments=assignments,
        templates=templates,
        evidence=IdentityEvidence(
            segments=seed_segments, samples=seed_samples, feature_spaces=[space]
        ),
    )
    query_segments = [
        Segment(
            segment_id=name,
            match_id="match",
            clip_id=name,
            track_id="track",
            start_s=0.0,
            end_s=1.5,
        )
        for name in query_ids
    ]
    batch = EvidenceBatch(
        match_id="match",
        batch_id="query",
        producer="synthetic",
        identity_evidence=IdentityEvidence(
            segments=query_segments,
            feature_spaces=[space],
            samples=[
                sample_for(segment, index, vector, clean=clean)
                for segment in query_segments
                for index in range(3)
            ],
        ),
    )
    prepared = prepare_evidence(batch, policy)
    associations = associate_segments(prepared, snapshot, policy)
    return batch, prepared, snapshot, associations


def human_assignment(*, player_id="pl_b", status="confirmed", reason=None):
    return Assignment(
        match_id="match",
        segment_id="query",
        player_id=player_id,
        status=status,
        method="manual",
        unassigned_reason=reason,
        revision=4,
    )


def constraint(kind, *, segment_id="query", related=None, player_id=None, identifier="constraint"):
    return ManualConstraint(
        constraint_id=identifier,
        match_id="match",
        segment_id=segment_id,
        kind=kind,
        related_segment_id=related,
        player_id=player_id,
        actor="reviewer",
        reason="explicit independent constraint",
        revision=4,
    )


def test_uncalibrated_rankings_stay_pending_and_all_inputs_are_unchanged():
    policy = policy_for()
    batch, prepared, snapshot, associations = package(policy)
    before = [item.model_dump(mode="json") for item in (batch, prepared, snapshot, *associations)]
    result = resolve_associations(prepared, snapshot, associations, policy)[0]
    assert [candidate.player_id for candidate in associations[0].candidates] == ["pl_a", "pl_b"]
    assert result.decision == "pending"
    assert result.reasons == ["policy_uncalibrated"]
    assert result.assignment is None
    assert not result.automatic_confirmation_eligible
    assert before == [
        item.model_dump(mode="json") for item in (batch, prepared, snapshot, *associations)
    ]


@pytest.mark.parametrize("enabled,status", [(False, "proposed"), (True, "confirmed")])
def test_calibrated_gates_distinguish_proposal_from_automatic_confirmation(enabled, status):
    policy = policy_for(calibrated=True, enabled=enabled)
    _, prepared, snapshot, associations = package(policy)
    result = resolve_associations(prepared, snapshot, associations, policy)[0]
    assert result.decision == "match_existing"
    assert result.player_id == "pl_a"
    assert result.assignment.status == status
    assert result.assignment.method == "automatic"
    assert result.assignment.revision == snapshot.revision
    assert result.requires_review is not enabled
    assert result.automatic_confirmation_eligible is enabled


def test_human_confirmation_is_preserved_and_conflict_is_reported():
    policy = policy_for(calibrated=True, enabled=True)
    _, prepared, snapshot, associations = package(policy)
    manual = human_assignment()
    result = resolve_associations(
        prepared, snapshot, associations, policy, manual_assignments=[manual]
    )[0]
    assert result.player_id == "pl_b" and result.assignment == manual
    assert result.assignment is not manual
    assert result.requires_review
    assert "manual_candidate_conflict" in result.reasons
    assert not result.automatic_confirmation_eligible


@pytest.mark.parametrize(
    "reason,decision",
    [("rejected", "rejected"), ("other_person", "rejected"), ("unknown", "pending")],
)
def test_manual_unassigned_decisions_are_not_overwritten(reason, decision):
    policy = policy_for(calibrated=True, enabled=True)
    _, prepared, snapshot, associations = package(policy)
    manual = human_assignment(player_id=None, status="unassigned", reason=reason)
    result = resolve_associations(
        prepared, snapshot, associations, policy, manual_assignments=[manual]
    )[0]
    assert result.decision == decision and result.player_id is None
    assert result.assignment == manual


def test_forbidding_one_candidate_does_not_establish_another_identity():
    policy = policy_for(calibrated=True, enabled=True)
    _, prepared, snapshot, associations = package(policy)
    forbidden = constraint("forbid_player", player_id="pl_a")
    result = resolve_associations(
        prepared, snapshot, associations, policy, constraints=[forbidden]
    )[0]
    assert result.decision == "pending" and result.player_id is None
    assert "forbidden_player" in result.reasons


@pytest.mark.parametrize("case", ["empty_gallery", "single_candidate", "distant"])
def test_unmatched_or_insufficient_evidence_never_creates_a_player(case):
    policy = policy_for(calibrated=True, enabled=True)
    vector = (-1.0, 0.0) if case == "distant" else (1.0, 0.0)
    _, prepared, snapshot, _ = package(policy, vector=vector)
    if case != "distant":
        players = snapshot.players[: 0 if case == "empty_gallery" else 1]
        ids = {player.player_id for player in players}
        snapshot = snapshot.model_copy(
            update={
                "players": players,
                "templates": [item for item in snapshot.templates if item.player_id in ids],
                "assignments": [item for item in snapshot.assignments if item.player_id in ids],
            }
        )
    associations = associate_segments(prepared, snapshot, policy)
    result = resolve_associations(prepared, snapshot, associations, policy)[0]
    assert result.decision == "pending"
    assert result.player_id is None and result.assignment is None


@pytest.mark.parametrize("clean,expected", [(True, "create_new_proposal"), (False, "pending")])
def test_new_identity_requires_separate_manual_intent_and_clean_support(clean, expected):
    policy = policy_for()
    _, prepared, snapshot, associations = package(policy, clean=clean)
    request = NewIdentityRequest(
        match_id="match",
        segment_id="query",
        support_sample_ids=["query-sample-0"],
        actor="reviewer",
        reason="independently reviewed new identity",
        revision=4,
    )
    result = resolve_associations(
        prepared, snapshot, associations, policy, new_identity_requests=[request]
    )[0]
    assert result.decision == expected
    assert result.player_id is None and result.assignment is None
    if clean:
        assert result.evidence[0].reference_id == "query-sample-0"


def test_new_identity_support_must_belong_to_the_segment():
    policy = policy_for()
    _, prepared, snapshot, associations = package(policy)
    request = NewIdentityRequest(
        match_id="match",
        segment_id="query",
        support_sample_ids=["seed-a-sample-0"],
        actor="reviewer",
        reason="new identity",
        revision=4,
    )
    with pytest.raises(ValueError, match="support.*usable evidence"):
        resolve_associations(
            prepared, snapshot, associations, policy, new_identity_requests=[request]
        )


def test_block_automatic_does_not_override_separate_manual_new_identity_intent():
    policy = policy_for()
    _, prepared, snapshot, associations = package(policy)
    request = NewIdentityRequest(
        match_id="match",
        segment_id="query",
        support_sample_ids=["query-sample-0"],
        actor="reviewer",
        reason="separately reviewed new identity",
        revision=4,
    )
    result = resolve_associations(
        prepared,
        snapshot,
        associations,
        policy,
        constraints=[constraint("block_automatic")],
        new_identity_requests=[request],
    )[0]
    assert result.decision == "create_new_proposal"
    assert result.assignment is None and result.player_id is None


def test_cannot_link_blocks_both_automatic_sides_and_is_order_independent():
    policy = policy_for(calibrated=True, enabled=True)
    _, prepared, snapshot, associations = package(policy, query_ids=("left", "right"))
    blocked = constraint("cannot_link", segment_id="left", related="right")
    results = resolve_associations(prepared, snapshot, associations, policy, constraints=[blocked])
    reversed_prepared = prepared.model_copy(update={"segments": list(reversed(prepared.segments))})
    again = resolve_associations(
        reversed_prepared, snapshot, list(reversed(associations)), policy, constraints=[blocked]
    )
    assert results == again
    assert [item.segment_id for item in results] == ["left", "right"]
    assert all(item.decision == "pending" for item in results)
    assert all("cannot_link_conflict" in item.reasons for item in results)


def test_cannot_link_chain_checks_all_preliminary_targets_before_changing_results():
    policy = policy_for(calibrated=True, enabled=True)
    _, prepared, snapshot, associations = package(policy, query_ids=("a", "b", "c"))
    constraints = [
        constraint("cannot_link", segment_id="a", related="b", identifier="ab"),
        constraint("cannot_link", segment_id="b", related="c", identifier="bc"),
    ]
    results = resolve_associations(
        prepared, snapshot, associations, policy, constraints=constraints
    )
    assert all(item.decision == "pending" for item in results)
    assert all("cannot_link_conflict" in item.reasons for item in results)
