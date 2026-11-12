"""Transaction-local checks across all current facts, including late references."""

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence

from app.contracts.evidence import EvidenceBatch, GallerySnapshot, InlineFeature
from app.contracts.identity import MatchingPolicy, Segment
from app.identity._eligibility import eligible_template, prepare_gallery
from app.identity.fingerprints import sample_binding
from app.identity.prepare import _within, are_duplicate_samples, resolve_feature
from app.storage.repositories import Repository


def validate_evidence_context(repo: Repository, raw: GallerySnapshot) -> None:
    """Revalidate current raw evidence with all currently known clips and assets.

    A batch can validly refer to a target which has not arrived yet. Once an
    identity target arrives its bounds and type must agree with older identity
    evidence. Historical events retain their original references: an unavailable
    or incompatible support remains unmapped under event association and must
    not block valid identity evidence. Use dictionaries so Pydantic revalidates
    nested facts, without reading a formal snapshot during R+1 writes.
    """
    EvidenceBatch.model_validate(
        {
            "match_id": raw.match_id,
            "batch_id": "workflow-context",
            "producer": "workflows",
            "clips": [clip.model_dump(mode="json") for clip in repo.clips(raw.match_id)],
            "assets": [asset.model_dump(mode="json") for asset in raw.assets],
            "identity_evidence": raw.evidence.model_dump(mode="json"),
        }
    )


def validate_segment_frames(segments: Sequence[Segment]) -> None:
    """Check every active framed range, even across intervening unframed segments.

    Time and frame coordinates remain independent supplied observations. No FPS
    conversion is inferred. Their order must agree, and half-open ranges may
    touch but must not overlap on a single track in the same clip and match.
    """
    grouped = defaultdict(list)
    for segment in segments:
        if segment.active and segment.frame_start is not None:
            grouped[(segment.match_id, segment.clip_id, segment.track_id)].append(segment)
    for group in grouped.values():
        ordered = sorted(group, key=lambda segment: (segment.start_s, segment.segment_id))
        for left, right in zip(ordered, ordered[1:]):
            if left.frame_end_exclusive > right.frame_start:
                raise ValueError(
                    "active identity segment frame ranges overlap or reverse time order: "
                    f"{left.segment_id!r}, {right.segment_id!r}"
                )


def validate_templates(
    final: GallerySnapshot,
    feature_values: Mapping[tuple[str, int], InlineFeature],
    policy_for: Callable[[str], MatchingPolicy],
    *,
    discovery_reports: Sequence[dict] = (),
) -> None:
    """Reject an active template that the shared identity rules cannot use."""
    prepared = {}
    policies = {}
    counts = defaultdict(int)
    for template in final.templates:
        if not template.active:
            continue
        if template.admission == "automatic_experimental":
            _validate_automatic_template(template, final, feature_values, discovery_reports)
            continue
        space = template.feature_space_id
        if space not in prepared:
            policies[space] = policy_for(space)
            prepared[space] = prepare_gallery(final, policies[space], feature_values)
        _, reasons = eligible_template(
            template,
            final,
            policies[space],
            feature_values,
            prepared_gallery=prepared[space],
        )
        if reasons:
            raise ValueError(
                f"active template {template.template_id!r} is ineligible: {', '.join(reasons)}"
            )
        key = (template.player_id, space)
        counts[key] += 1
        if counts[key] > policies[space].max_templates_per_player:
            raise ValueError("active template count exceeds the selected policy capacity")


def _validate_automatic_template(template, final, feature_values, discovery_reports):
    """Experimental admission must retain its exact independent raw support."""
    record = next(
        (
            item
            for item in discovery_reports
            if item["report"]["report_id"] == template.automatic_support_id
        ),
        None,
    )
    if record is None:
        raise ValueError("automatic template is missing its discovery support report")
    report = record["report"]
    group = next(
        (
            item
            for item in report["groups"]
            if record["player_ids_by_group"].get(item["group_id"]) == template.player_id
        ),
        None,
    )
    support = next(
        (
            item
            for item in (group["templates"] if group else [])
            if (item["sample_id"], item["feature_space_id"])
            == (template.sample_id, template.feature_space_id)
        ),
        None,
    )
    if support is None or report["match_id"] != final.match_id:
        raise ValueError("automatic template has no matching group admission")
    samples = {sample.sample_id: sample for sample in final.evidence.samples}
    segments = {segment.segment_id: segment for segment in final.evidence.segments}
    sample = samples.get(template.sample_id)
    segment = segments.get(support["segment_id"])
    assignment = next((a for a in final.assignments if a.segment_id == support["segment_id"]), None)
    if (
        sample is None
        or segment is None
        or not segment.active
        or not _within(sample, segment)
        or sample.segment_id != segment.segment_id
        or assignment is None
        or assignment.player_id != template.player_id
        or assignment.status not in {"proposed", "confirmed"}
        or template.review_status != "unreviewed"
        or template.approval is not None
    ):
        raise ValueError("automatic template scope or current ownership is invalid")
    space = next(
        (
            s
            for s in final.evidence.feature_spaces
            if s.feature_space_id == template.feature_space_id
        ),
        None,
    )
    if space is None or resolve_feature(sample, space, feature_values) is None:
        raise ValueError("automatic template has no current feature")
    admitted_pairs = {
        item["pair_id"]: item for item in report["pairs"] if item["status"] == "supported"
    }
    if not support["support_pair_ids"] or any(
        pair_id not in admitted_pairs or pair_id not in group["support_pair_ids"]
        for pair_id in support["support_pair_ids"]
    ):
        raise ValueError("automatic template has invalid pair support")
    bindings = record.get("sample_sources", {})
    pair_context_ids = {
        sample_id
        for pair_id in support["support_pair_ids"]
        for comparison in admitted_pairs[pair_id]["comparisons"]
        for sample_id in (comparison["left_sample_id"], comparison["right_sample_id"])
    }
    for sample_id in pair_context_ids:
        current = samples.get(sample_id)
        if current is None or bindings.get(sample_id) != sample_binding(current, final.assets):
            raise ValueError("automatic pair comparison context source version has changed")
    witnesses = [samples.get(sid) for sid in support["witness_sample_ids"]]
    if len(witnesses) < 2 or any(witness is None for witness in witnesses):
        raise ValueError("automatic template requires multiple raw witnesses")
    for witness in witnesses:
        witness_segment = segments.get(witness.segment_id)
        witness_assignment = next(
            (a for a in final.assignments if a.segment_id == witness.segment_id), None
        )
        if (
            witness_segment is None
            or not witness_segment.active
            or not _within(witness, witness_segment)
            or witness_assignment is None
            or witness_assignment.player_id != template.player_id
            or witness_assignment.status not in {"proposed", "confirmed"}
        ):
            raise ValueError("automatic template witness ownership is no longer valid")
    for item in [sample, *witnesses]:
        if bindings.get(item.sample_id) != sample_binding(item, final.assets):
            raise ValueError("automatic support sample source version or feature has changed")
    if any(
        witness.clip_id == sample.clip_id
        or witness.segment_id not in group["member_segment_ids"]
        or are_duplicate_samples(sample, witness, final.assets)
        for witness in witnesses
    ) or any(
        are_duplicate_samples(left, right, final.assets)
        for index, left in enumerate(witnesses)
        for right in witnesses[index + 1 :]
    ):
        raise ValueError("automatic template witnesses are not independent raw observations")
