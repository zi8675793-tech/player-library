"""Independent regressions for identity-wide invalidation and detached plans."""

from test_identity_resolve import package, policy_for

from app.contracts.evidence import EvidenceBatch, IdentityEvidence
from app.contracts.identity import Assignment, ManualTemplateReview, MatchingPolicy
from app.identity.gallery import plan_gallery
from app.identity.prepare import prepare_evidence


def confirmed(segment_id, player_id="pl_a"):
    return Assignment(
        match_id="match",
        segment_id=segment_id,
        player_id=player_id,
        status="confirmed",
        method="manual",
        revision=4,
    )


def test_gallery_ownership_invalidation_revokes_templates_in_all_feature_spaces():
    old_policy = policy_for()
    _, _, snapshot, _ = package(old_policy)
    old_segment = snapshot.evidence.segments[0]
    new_space = snapshot.evidence.feature_spaces[0].model_copy(
        update={"feature_space_id": "new-space"}
    )
    policy = MatchingPolicy(feature_space_id="new-space")
    samples = []
    for sample in snapshot.evidence.samples[:2]:
        copied = sample.model_copy(deep=True)
        copied.features[0].feature_space_id = "new-space"
        samples.append(copied)
    batch = EvidenceBatch(
        match_id="match",
        batch_id="new-model-query",
        producer="synthetic",
        identity_evidence=IdentityEvidence(
            segments=[old_segment], samples=samples, feature_spaces=[new_space]
        ),
    )
    prepared = prepare_evidence(batch, policy)
    plan = plan_gallery(
        prepared, snapshot, policy, final_assignments=[confirmed(old_segment.segment_id, "pl_b")]
    )
    revoked = {item.template_id for item in plan.changes if item.action == "deactivate"}
    assert {"template-a-0", "template-a-1"} <= revoked


def test_split_to_new_segment_requires_reapproval_of_old_template_qualification():
    policy = policy_for()
    _, _, snapshot, _ = package(policy)
    old_segment = snapshot.evidence.segments[0]
    new_segment = old_segment.model_copy(
        update={
            "segment_id": "split-child",
            "end_s": 0.75,
            "supersedes_segment_id": old_segment.segment_id,
        }
    )
    samples = [
        sample.model_copy(deep=True, update={"segment_id": new_segment.segment_id})
        for sample in snapshot.evidence.samples[:2]
    ]
    batch = EvidenceBatch(
        match_id="match",
        batch_id="split-query",
        producer="synthetic",
        identity_evidence=IdentityEvidence(
            segments=[new_segment], samples=samples, feature_spaces=snapshot.evidence.feature_spaces
        ),
    )
    prepared = prepare_evidence(batch, policy)
    plan = plan_gallery(
        prepared, snapshot, policy, final_assignments=[confirmed(new_segment.segment_id)]
    )
    revoked = {item.template_id for item in plan.changes if item.action == "deactivate"}
    assert {"template-a-0", "template-a-1"} <= revoked


def test_gallery_plan_reviews_are_detached_from_input_human_records():
    policy = policy_for()
    _, prepared, snapshot, _ = package(policy)
    review = ManualTemplateReview(
        match_id="match",
        sample_id="query-sample-0",
        player_id="pl_a",
        feature_space_id="space",
        review_status="approved",
        actor="reviewer",
        reason="approved clean source",
        revision=4,
    )
    before = review.model_dump_json()
    plan = plan_gallery(
        prepared, snapshot, policy, final_assignments=[confirmed("query")], reviews=[review]
    )
    change = next(item for item in plan.changes if item.sample_id == review.sample_id)
    assert change.review is not review
    change.review.reason = "mutated returned plan"
    assert review.model_dump_json() == before
