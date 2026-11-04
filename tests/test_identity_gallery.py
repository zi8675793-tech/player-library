"""Final ownership and human template admission remain independent plans."""

import pytest
from pydantic import ValidationError
from test_identity_resolve import package, policy_for

from app.contracts.evidence import MediaAsset, SourceFile
from app.contracts.identity import Assignment, IdentityRisk, ManualTemplateReview, MatchingPolicy
from app.identity.gallery import plan_gallery
from app.identity.prepare import prepare_evidence


def assignment(
    *, segment_id="query", player_id="pl_a", status="confirmed", method="automatic", revision=4
):
    return Assignment(
        match_id="match",
        segment_id=segment_id,
        player_id=player_id,
        status=status,
        method=method,
        revision=revision,
    )


def review(
    *, sample_id="query-sample-0", player_id="pl_a", status="approved", revision=4, template_id=None
):
    return ManualTemplateReview(
        match_id="match",
        sample_id=sample_id,
        player_id=player_id,
        feature_space_id="space",
        review_status=status,
        actor="reviewer",
        reason="independent template review",
        revision=revision,
        template_id=template_id,
    )


@pytest.mark.parametrize("method", ["automatic", "manual"])
def test_confirmed_identity_alone_never_approves_a_template(method):
    policy = policy_for()
    _, prepared, snapshot, _ = package(policy)
    final = assignment(method=method)
    plan = plan_gallery(prepared, snapshot, policy, final_assignments=[final])
    assert len(plan.changes) == 3
    assert all(change.action == "await_review" for change in plan.changes)
    assert all("manual_template_approval_required" in change.reasons for change in plan.changes)
    assert not any(change.action == "deactivate" for change in plan.changes)


def test_manual_approval_activates_only_the_reviewed_qualified_sample():
    policy = policy_for()
    _, prepared, snapshot, _ = package(policy)
    accepted = review()
    plan = plan_gallery(
        prepared, snapshot, policy, final_assignments=[assignment()], reviews=[accepted]
    )
    activations = [change for change in plan.changes if change.action == "activate"]
    assert [change.sample_id for change in activations] == ["query-sample-0"]
    assert activations[0].review == accepted
    assert activations[0].player_id == "pl_a"
    assert activations[0].template_id is None
    assert plan.gallery_revision == snapshot.revision
    assert [change.sample_id for change in plan.changes if change.action == "await_review"] == [
        "query-sample-1",
        "query-sample-2",
    ]


def test_template_approval_cannot_confirm_a_proposed_identity():
    policy = policy_for()
    _, prepared, snapshot, _ = package(policy)
    plan = plan_gallery(
        prepared,
        snapshot,
        policy,
        final_assignments=[assignment(status="proposed")],
        reviews=[review()],
    )
    assert not any(change.action == "activate" for change in plan.changes)


def test_template_review_requires_an_explicit_player_binding():
    values = review().model_dump()
    del values["player_id"]
    with pytest.raises(ValidationError, match="player_id"):
        ManualTemplateReview.model_validate(values)


def test_approval_for_another_owner_is_rejected_without_mutating_inputs():
    policy = policy_for()
    _, prepared, snapshot, _ = package(policy)
    before = snapshot.model_dump(mode="json")
    with pytest.raises(ValueError, match="approval.*final confirmed owner"):
        plan_gallery(
            prepared,
            snapshot,
            policy,
            final_assignments=[assignment()],
            reviews=[review(player_id="pl_b")],
        )
    assert snapshot.model_dump(mode="json") == before


def test_approval_predating_final_assignment_stays_awaiting_review():
    policy = policy_for()
    _, prepared, snapshot, _ = package(policy)
    plan = plan_gallery(
        prepared,
        snapshot,
        policy,
        final_assignments=[assignment(revision=4)],
        reviews=[review(revision=3)],
    )
    change = next(change for change in plan.changes if change.sample_id == "query-sample-0")
    assert change.action == "await_review"
    assert "approval_predates_final_assignment" in change.reasons


def test_capacity_proposes_replacement_choices_without_evicting_manual_anchors():
    policy = MatchingPolicy(feature_space_id="space", max_templates_per_player=2)
    _, prepared, snapshot, _ = package(policy)
    plan = plan_gallery(
        prepared, snapshot, policy, final_assignments=[assignment()], reviews=[review()]
    )
    replacements = [change for change in plan.changes if change.action == "replacement_required"]
    assert len(replacements) == 1
    assert replacements[0].sample_id == "query-sample-0"
    assert replacements[0].replacement_template_ids == ["template-a-0", "template-a-1"]
    assert not any(change.action in {"activate", "deactivate"} for change in plan.changes)


def test_reduced_capacity_requires_explicit_selection_of_existing_anchors():
    policy = MatchingPolicy(feature_space_id="space", max_templates_per_player=1)
    _, prepared, snapshot, _ = package(policy)
    plan = plan_gallery(prepared, snapshot, policy)
    assert len(plan.changes) == 4
    assert all(change.action == "replacement_required" for change in plan.changes)
    assert all(len(change.replacement_template_ids) == 2 for change in plan.changes)
    assert not any(change.action == "deactivate" for change in plan.changes)


def test_duplicate_evidence_cannot_be_approved_for_two_different_owners():
    policy = policy_for()
    batch, _, snapshot, _ = package(policy, query_ids=("left", "right"))
    samples = {sample.sample_id: sample for sample in batch.identity_evidence.samples}
    samples["right-sample-0"].source = samples["left-sample-0"].source.model_copy(deep=True)
    prepared = prepare_evidence(batch, policy)
    finals = [assignment(segment_id="left"), assignment(segment_id="right", player_id="pl_b")]
    reviews = [
        review(sample_id="left-sample-0"),
        review(sample_id="right-sample-0", player_id="pl_b"),
    ]
    plan = plan_gallery(prepared, snapshot, policy, final_assignments=finals, reviews=reviews)
    reviewed_changes = [
        change for change in plan.changes if change.sample_id in {"left-sample-0", "right-sample-0"}
    ]
    assert len(reviewed_changes) == 2
    assert all(change.action == "await_review" for change in reviewed_changes)
    assert all(
        "conflicting_gallery_evidence_owner" in change.reasons for change in reviewed_changes
    )


def test_duplicate_of_existing_other_owner_stays_awaiting_review():
    policy = policy_for()
    batch, _, snapshot, _ = package(policy)
    batch.identity_evidence.samples[0].source = snapshot.evidence.samples[2].source.model_copy(
        deep=True
    )
    prepared = prepare_evidence(batch, policy)
    plan = plan_gallery(
        prepared, snapshot, policy, final_assignments=[assignment()], reviews=[review()]
    )
    change = next(change for change in plan.changes if change.sample_id == "query-sample-0")
    assert change.action == "await_review"
    assert "conflicting_gallery_evidence_owner" in change.reasons


def test_ownership_change_revokes_old_templates_and_requires_fresh_admission():
    policy = policy_for()
    _, prepared, snapshot, _ = package(policy)
    revised = assignment(segment_id="seed-a", player_id="pl_b", method="manual")
    plan = plan_gallery(prepared, snapshot, policy, final_assignments=[revised])
    revoked = [change for change in plan.changes if change.action == "deactivate"]
    assert [change.template_id for change in revoked] == ["template-a-0", "template-a-1"]
    assert all("template_ownership_invalidated" in change.reasons for change in revoked)
    assert not any(change.action == "activate" for change in plan.changes)


def test_rejected_template_review_revokes_gallery_use_without_rewriting_identity():
    policy = policy_for()
    _, prepared, snapshot, _ = package(policy)
    before_assignments = [item.model_dump() for item in snapshot.assignments]
    rejected = review(sample_id="seed-a-sample-0", status="rejected", template_id="template-a-0")
    plan = plan_gallery(prepared, snapshot, policy, reviews=[rejected])
    revoked = [change for change in plan.changes if change.action == "deactivate"]
    assert [change.template_id for change in revoked] == ["template-a-0"]
    assert "template_review_not_approved" in revoked[0].reasons
    assert [item.model_dump() for item in snapshot.assignments] == before_assignments


def test_stale_rejection_cannot_override_a_newer_approved_anchor():
    policy = policy_for()
    _, prepared, snapshot, _ = package(policy)
    before = snapshot.model_dump(mode="json")
    rejected = review(
        sample_id="seed-a-sample-0",
        status="rejected",
        revision=3,
        template_id="template-a-0",
    )
    with pytest.raises(ValueError, match="stale|predates|older"):
        plan_gallery(prepared, snapshot, policy, reviews=[rejected])
    assert snapshot.model_dump(mode="json") == before


def test_risk_blocks_approved_sample_without_overriding_its_identity():
    policy = policy_for()
    _, prepared, snapshot, _ = package(policy)
    prepared.segments[0].risks.append(
        IdentityRisk(risk_id="mixed", kind="mixed_identity", match_id="match", segment_id="query")
    )
    plan = plan_gallery(
        prepared, snapshot, policy, final_assignments=[assignment()], reviews=[review()]
    )
    change = next(change for change in plan.changes if change.sample_id == "query-sample-0")
    assert change.action == "await_review"
    assert "unresolved_identity_risk" in change.reasons


def test_planning_is_order_independent_and_does_not_change_evidence_snapshot_or_reviews():
    policy = policy_for()
    batch, prepared, snapshot, _ = package(policy)
    finals = [assignment()]
    reviews = [review(sample_id=f"query-sample-{index}") for index in range(3)]
    originals = [batch, prepared, snapshot, *finals, *reviews]
    before = [item.model_dump(mode="json") for item in originals]
    plan = plan_gallery(prepared, snapshot, policy, final_assignments=finals, reviews=reviews)
    reordered = snapshot.model_copy(
        update={
            "templates": list(reversed(snapshot.templates)),
            "players": list(reversed(snapshot.players)),
            "assignments": list(reversed(snapshot.assignments)),
        }
    )
    again = plan_gallery(
        prepared,
        reordered,
        policy,
        final_assignments=list(reversed(finals)),
        reviews=list(reversed(reviews)),
    )
    assert plan == again
    assert before == [item.model_dump(mode="json") for item in originals]
    # Mutating a returned plan cannot mutate any supplied human review record.
    plan.changes[0].review.reason = "changed result only"
    assert before == [item.model_dump(mode="json") for item in originals]


def image_asset(identifier, digest="c" * 64, *, size=None, media_type="image"):
    return MediaAsset(
        asset_id=identifier,
        media_type=media_type,
        file=SourceFile(
            file_id=f"file-{identifier}",
            path=f"{identifier}.png",
            sha256=digest,
            size_bytes=size,
        ),
    )


def connect_chain(batch, first_id, second_id, connector_id):
    """First shares a source observation; second shares a photo with connector."""
    samples = {sample.sample_id: sample for sample in batch.identity_evidence.samples}
    samples[connector_id].source = samples[first_id].source.model_copy(deep=True)
    samples[second_id].image_asset_id = "second-image"
    samples[connector_id].image_asset_id = "connector-image"
    batch.assets.extend([image_asset("second-image"), image_asset("connector-image")])


@pytest.mark.parametrize("approve_connector", [True, False])
def test_transitive_duplicate_group_activates_once_even_if_connector_is_sorted_last(
    approve_connector,
):
    policy = policy_for()
    batch, _, snapshot, _ = package(policy)
    connect_chain(batch, "query-sample-0", "query-sample-1", "query-sample-2")
    prepared = prepare_evidence(batch, policy)
    reviewed_ids = ["query-sample-0", "query-sample-1"]
    if approve_connector:
        reviewed_ids.append("query-sample-2")
    reviews = [review(sample_id=sample_id) for sample_id in reviewed_ids]
    before = [item.model_dump(mode="json") for item in (batch, prepared, snapshot, *reviews)]
    plan = plan_gallery(
        prepared, snapshot, policy, final_assignments=[assignment()], reviews=reviews
    )
    assert [change.sample_id for change in plan.changes if change.action == "activate"] == [
        "query-sample-0"
    ]
    duplicate = next(change for change in plan.changes if change.sample_id == "query-sample-1")
    assert "duplicate_proposed_template_evidence" in duplicate.reasons
    reordered = prepared.model_copy(
        update={
            "segments": [
                segment.model_copy(update={"samples": list(reversed(segment.samples))})
                for segment in reversed(prepared.segments)
            ],
            "assets": list(reversed(prepared.assets)),
        }
    )
    assert (
        plan_gallery(
            reordered,
            snapshot,
            policy,
            final_assignments=[assignment()],
            reviews=list(reversed(reviews)),
        )
        == plan
    )
    assert before == [
        item.model_dump(mode="json") for item in (batch, prepared, snapshot, *reviews)
    ]


def test_transitive_cross_owner_conflict_blocks_every_approved_component_member():
    policy = policy_for()
    batch, _, snapshot, _ = package(policy, query_ids=("a", "b", "z"))
    connect_chain(batch, "a-sample-0", "b-sample-0", "z-sample-0")
    prepared = prepare_evidence(batch, policy)
    finals = [
        assignment(segment_id="a"),
        assignment(segment_id="b", player_id="pl_b"),
        assignment(segment_id="z"),
    ]
    reviews = [
        review(sample_id="a-sample-0"),
        review(sample_id="b-sample-0", player_id="pl_b"),
        review(sample_id="z-sample-0"),
    ]
    plan = plan_gallery(prepared, snapshot, policy, final_assignments=finals, reviews=reviews)
    involved = [
        change
        for change in plan.changes
        if change.sample_id in {"a-sample-0", "b-sample-0", "z-sample-0"}
    ]
    assert len(involved) == 3
    assert all(change.action == "await_review" for change in involved)
    assert all("conflicting_gallery_evidence_owner" in change.reasons for change in involved)


def test_unreviewed_connector_still_links_candidate_to_existing_gallery_evidence():
    policy = policy_for()
    batch, _, snapshot, _ = package(policy)
    samples = {sample.sample_id: sample for sample in batch.identity_evidence.samples}
    samples["query-sample-0"].source = snapshot.evidence.samples[0].source.model_copy(deep=True)
    samples["query-sample-0"].image_asset_id = "connector-image"
    samples["query-sample-1"].image_asset_id = "candidate-image"
    batch.assets.extend([image_asset("connector-image"), image_asset("candidate-image")])
    prepared = prepare_evidence(batch, policy)
    plan = plan_gallery(
        prepared,
        snapshot,
        policy,
        final_assignments=[assignment()],
        reviews=[review(sample_id="query-sample-1")],
    )
    change = next(change for change in plan.changes if change.sample_id == "query-sample-1")
    assert change.action == "await_review"
    assert "duplicate_gallery_evidence" in change.reasons
    assert not any(change.action == "activate" for change in plan.changes)


@pytest.mark.parametrize("conflict", ["hash", "size", "type"])
def test_gallery_rejects_conflicting_asset_content_across_snapshot_and_query(conflict):
    policy = policy_for()
    batch, _, snapshot, _ = package(policy)
    batch.assets.append(image_asset("shared-image", size=1))
    snapshot.assets.append(
        image_asset(
            "shared-image",
            "d" * 64 if conflict == "hash" else "c" * 64,
            size=2 if conflict == "size" else 1,
            media_type="video" if conflict == "type" else "image",
        )
    )
    prepared = prepare_evidence(batch, policy)
    before = [item.model_dump(mode="json") for item in (batch, prepared, snapshot)]
    with pytest.raises(ValueError, match="conflicting content"):
        plan_gallery(
            prepared, snapshot, policy, final_assignments=[assignment()], reviews=[review()]
        )
    assert before == [item.model_dump(mode="json") for item in (batch, prepared, snapshot)]


def test_known_asset_hash_in_query_enriches_comparison_view_without_mutating_snapshot():
    policy = policy_for()
    batch, _, snapshot, _ = package(policy)
    snapshot.assets.append(image_asset("seed-image", digest=None))
    snapshot.evidence.samples[0].image_asset_id = "seed-image"
    batch.assets.extend([image_asset("seed-image"), image_asset("query-image")])
    batch.identity_evidence.samples[0].image_asset_id = "query-image"
    prepared = prepare_evidence(batch, policy)
    before = snapshot.model_dump(mode="json")
    plan = plan_gallery(
        prepared, snapshot, policy, final_assignments=[assignment()], reviews=[review()]
    )
    change = next(change for change in plan.changes if change.sample_id == "query-sample-0")
    assert change.action == "await_review"
    assert "duplicate_gallery_evidence" in change.reasons
    assert snapshot.assets[0].file.sha256 is None
    assert snapshot.model_dump(mode="json") == before


def test_other_space_template_outside_narrowed_current_scope_loses_old_qualification():
    policy = policy_for()
    batch, _, snapshot, _ = package(policy)
    snapshot.templates[0].feature_space_id = "other-space"
    snapshot.evidence.samples[0].features[0].feature_space_id = "other-space"
    narrowed = snapshot.evidence.segments[0].model_copy(update={"start_s": 0.5})
    batch.identity_evidence.segments = [narrowed]
    batch.identity_evidence.samples = []
    prepared = prepare_evidence(batch, policy)
    plan = plan_gallery(prepared, snapshot, policy)
    change = next(change for change in plan.changes if change.template_id == "template-a-0")
    assert change.action == "deactivate"
    assert "template_sample_outside_current_scope" in change.reasons
