"""Pure contracts enforce local facts while leaving stateful rules to workflows."""

import json

import pytest
from pydantic import ValidationError

from app.contracts import (
    Assignment,
    Clip,
    Event,
    EvidenceBatch,
    FeatureFile,
    FeatureSpace,
    IdentityEvidence,
    ImportRequest,
    IntervalEventTime,
    Player,
    PointEventTime,
    Provenance,
    ResolvedEvent,
    ReviewRequest,
    Sample,
    Segment,
    SourceMapping,
    Subject,
    SuccessResponse,
    Template,
    Track,
)
from app.contracts.api import ErrorDetail, EventQuery, SetAssignmentCommand, SplitSegmentCommand
from app.contracts.common import SourceLocation
from app.contracts.events import ResolvedSubject, SourceEventTime, TrackReference
from app.contracts.evidence import (
    EvidenceWarning,
    FileFeatureReference,
    InlineFeature,
    MediaAsset,
    SourceFile,
)
from app.contracts.identity import AssignmentEvidence, Candidate, Resolution


def source(record_id="record-1", record_type="sample"):
    return Provenance(
        producer="upstream",
        record_type=record_type,
        source_record_id=record_id,
        evidence_version="v1",
    )


def segment(**overrides):
    values = dict(
        match_id="match",
        segment_id="seg-1",
        clip_id="clip",
        track_id="track",
        start_s=0,
        end_s=5,
    )
    values.update(overrides)
    return Segment.model_validate(values)


def sample(**overrides):
    values = dict(
        match_id="match",
        sample_id="sample",
        clip_id="clip",
        track_id="track",
        time_s=1,
        source=source(),
    )
    values.update(overrides)
    return Sample.model_validate(values)


def feature_space(parts=3, dimension=7, **overrides):
    values = dict(
        feature_space_id="space",
        backend="kpr",
        model_hash="a" * 64,
        preprocessing={"version": "resize-v1", "color_space": "RGB"},
        layout="parts",
        shape=[parts, dimension],
        normalization="l2",
        distance="mean_part_euclidean",
        visibility={"shape": [parts], "semantics": "confidence"},
    )
    values.update(overrides)
    return FeatureSpace.model_validate(values)


def event(**overrides):
    values = dict(
        match_id="match",
        event_id="event",
        source_event_key="stable-action-key",
        clip_id="clip",
        event_type="shot",
        time={"kind": "point", "time_s": 3.5},
        source=source("action", "event"),
        subjects=[
            {
                "subject_id": "shooter",
                "role": "shooter",
                "references": [{"kind": "track", "clip_id": "not-arrived", "track_id": "future"}],
            },
            {
                "subject_id": "assist",
                "role": "assister",
                "references": [{"kind": "segment", "segment_id": "future-segment"}],
            },
        ],
    )
    values.update(overrides)
    return Event.model_validate(values)


def round_trip(model):
    encoded = model.model_dump_json()
    json.loads(encoded, parse_constant=lambda value: pytest.fail(f"nonfinite JSON: {value}"))
    decoded = type(model).model_validate_json(encoded)
    assert decoded == model
    return decoded


@pytest.mark.parametrize("mode", ["identity", "events", "mixed", "empty", "omitted"])
def test_independent_evidence_branches_round_trip(mode):
    fields = {}
    if mode in {"identity", "mixed"}:
        fields["identity_evidence"] = IdentityEvidence(segments=[segment()], samples=[sample()])
    if mode in {"events", "mixed"}:
        fields["events"] = [event()]
    if mode == "empty":
        fields.update(identity_evidence=IdentityEvidence(), events=[], clips=[])
    batch = EvidenceBatch(match_id="match", batch_id="handoff-1", producer="collector", **fields)
    assert round_trip(batch).batch_id == "handoff-1"
    if mode == "events":
        assert batch.identity_evidence is None
        assert batch.clips is None  # No reference-existence check, even for the event clip.


def test_batch_is_not_an_identity_namespace_or_a_deduplication_key():
    original = EvidenceBatch(
        match_id="match", batch_id="one", producer="collector", events=[event()]
    )
    later = EvidenceBatch.model_validate({**original.model_dump(), "batch_id": "two"})
    assert original.events == later.events
    assert original.events[0].source == later.events[0].source
    with pytest.raises(ValidationError, match="match_id"):
        EvidenceBatch.model_validate({**original.model_dump(), "match_id": "another-match"})


def test_raw_tracks_and_identity_segments_are_independent_and_may_overlap_locally():
    raw = Track(
        match_id="match",
        clip_id="clip",
        track_id="track",
        start_s=0,
        end_s=10,
        frame_start=0,
        frame_end_exclusive=300,
        source=source("track", "track"),
    )
    evidence = IdentityEvidence(tracks=[raw], segments=[segment(), segment(segment_id="seg-2")])
    assert round_trip(evidence).tracks[0].end_s == 10
    assert len(evidence.segments) == 2  # Active-overlap policy belongs to workflow state.


@pytest.mark.parametrize(
    "time", [{"kind": "point", "time_s": 3.5}, {"kind": "interval", "start_s": 2, "end_s": 4}]
)
def test_multi_role_events_and_reference_localization(time):
    original = event(
        time=time,
        outcome="made",
        points=2,
        game_clock={"period": "OT1", "clock_s": 23.4, "direction": "count_down"},
        raw_scores={"shot": 0.8},
    )
    original.subjects[0].references[0].time = PointEventTime(time_s=3)
    resolved = ResolvedEvent(
        event=original,
        subjects=[
            ResolvedSubject(
                subject_id="shooter", role="shooter", status="mapped", player_id="pl_a"
            ),
            ResolvedSubject(
                subject_id="assist", role="assister", status="mapped", player_id="pl_b"
            ),
        ],
        source_time=SourceEventTime(source_video_id="full-video", time=time),
    )
    decoded = round_trip(resolved)
    assert decoded.subjects[0].player_id != decoded.subjects[1].player_id
    assert "player_id" not in original.model_dump()
    if time["kind"] == "point":
        assert original.time.time_s == 3.5
        assert "end_s" not in original.time.model_dump()


def test_unknown_result_and_open_event_roles_and_types():
    assert event().outcome is None
    other = event(
        event_type="deflection",
        outcome="unknown",
        subjects=[
            Subject(subject_id="defender", role="defender-left"),
            Subject(subject_id="defender-2", role="defender-left"),
        ],
    )
    assert round_trip(other).outcome == "unknown"


@pytest.mark.parametrize(
    "mapping", [None, SourceMapping(source_video_id="full", source_start_s=120, time_scale=0.5)]
)
def test_optional_source_mapping_and_media_round_trip(mapping):
    clip = Clip(
        match_id="match", clip_id="clip", source_mapping=mapping, frame_timing="vfr", duration_s=12
    )
    decoded = round_trip(clip)
    assert decoded.source_mapping == mapping
    assert decoded.fps is None
    assert (
        ResolvedEvent(
            event=event(),
            subjects=[
                ResolvedSubject(subject_id="shooter", role="shooter", status="unmapped"),
                ResolvedSubject(subject_id="assist", role="assister", status="unmapped"),
            ],
        ).source_time
        is None
    )


def test_jersey_number_and_optional_external_references_survive_json():
    player = Player(match_id="match", player_id="pl_zero", jersey_number="00", alias_of="pl_future")
    assert round_trip(player).jersey_number == "00"
    assignment = Assignment(
        match_id="match",
        segment_id="not-arrived",
        player_id="pl_future",
        status="confirmed",
        method="manual",
        revision=9,
    )
    assert round_trip(assignment).player_id == "pl_future"
    template = Template(
        match_id="match",
        template_id="template",
        player_id="pl_future",
        feature_space_id="future-space",
        sample_id="future-sample",
        revision=9,
        review_status="rejected",
        active=False,
    )
    assert round_trip(template).sample_id == "future-sample"
    with pytest.raises(ValidationError):
        Player(match_id="match", player_id="pl_zero", jersey_number=0)


@pytest.mark.parametrize(("parts", "dimension"), [(1, 2), (3, 7), (9, 256)])
def test_feature_dimensions_come_from_contract(parts, dimension):
    space = feature_space(parts, dimension)
    feature = InlineFeature(
        feature_space_id="space", values=[0.0] * (parts * dimension), visibility=[0.5] * parts
    )
    evidence = IdentityEvidence(feature_spaces=[space], samples=[sample(features=[feature])])
    assert round_trip(evidence).feature_spaces[0].shape == [parts, dimension]
    with pytest.raises(ValidationError, match="values length"):
        IdentityEvidence(
            feature_spaces=[space],
            samples=[
                sample(
                    features=[
                        InlineFeature(
                            feature_space_id="space", values=[0.0], visibility=[1.0] * parts
                        )
                    ]
                )
            ],
        )


def test_model_contract_is_independent_of_file_count_and_integrity():
    space = feature_space()

    def file(file_id, count, digest):
        return FeatureFile(
            feature_file_id=file_id,
            feature_space_id="space",
            sample_count=count,
            values_file=SourceFile(
                file_id=f"{file_id}-values", path="features.npy", sha256=digest * 64
            ),
            index_file=SourceFile(file_id=f"{file_id}-index", path="index.json"),
            visibility_file=SourceFile(file_id=f"{file_id}-visibility", path="visibility.npy"),
        )

    files = [file("a", 2, "b"), file("b", 42, "c")]
    evidence = IdentityEvidence(feature_spaces=[space], feature_files=files)
    assert round_trip(evidence).feature_files[1].sample_count == 42
    assert "sample_count" not in space.model_dump()
    assert "values_file" not in space.model_dump()
    with pytest.raises(ValidationError, match="row_index"):
        IdentityEvidence(
            feature_files=files,
            samples=[
                sample(
                    features=[
                        FileFeatureReference(
                            feature_space_id="space", feature_file_id="a", row_index=2
                        )
                    ]
                )
            ],
        )
    # A later/existing feature space or feature file remains a valid reference.
    round_trip(
        IdentityEvidence(
            samples=[
                sample(
                    features=[
                        FileFeatureReference(
                            feature_space_id="future-space", feature_file_id="future", row_index=9
                        )
                    ]
                )
            ]
        )
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True, "1"])
def test_non_finite_or_coerced_numbers_are_rejected(value):
    with pytest.raises(ValidationError):
        PointEventTime(time_s=value)
    with pytest.raises(ValidationError):
        InlineFeature(feature_space_id="space", values=[value])
    with pytest.raises(ValidationError):
        event(raw_scores={"confidence": value})


@pytest.mark.parametrize(
    ("start", "end"), [(1, 1), (2, 1), (-1, 2), (0, float("nan")), (0, float("inf"))]
)
def test_invalid_intervals_have_clear_errors(start, end):
    with pytest.raises(ValidationError, match="end_s|start_s"):
        IntervalEventTime(start_s=start, end_s=end)
    with pytest.raises(ValidationError, match="end_s|start_s"):
        segment(start_s=start, end_s=end)


@pytest.mark.parametrize(
    "frames",
    [
        {"frame_start": 0},
        {"frame_end_exclusive": 5},
        {"frame_start": 2, "frame_end_exclusive": 2},
        {"frame_start": True, "frame_end_exclusive": 2},
    ],
)
def test_invalid_half_open_frame_intervals(frames):
    with pytest.raises(ValidationError):
        segment(**frames)


def test_local_status_and_provenance_rules():
    with pytest.raises(ValidationError, match="requires player_id"):
        Assignment(
            match_id="match", segment_id="seg", status="confirmed", method="manual", revision=0
        )
    with pytest.raises(ValidationError, match="unmapped"):
        ResolvedSubject(subject_id="s", role="shooter", status="unmapped", player_id="pl_one")
    with pytest.raises(ValidationError, match="different source version"):
        Provenance(
            **source().model_dump(exclude_none=True),
            supersedes=source().model_dump(
                include={"producer", "record_type", "source_record_id", "evidence_version"}
            ),
        )
    superseded = source().model_dump(
        include={"producer", "record_type", "source_record_id", "evidence_version"}
    )
    newer = Provenance(
        **{**superseded, "evidence_version": "v2"}, supersedes=superseded, content_hash="d" * 64
    )
    assert round_trip(newer).supersedes.evidence_version == "v1"
    with pytest.raises(ValidationError, match="feature_space_id"):
        AssignmentEvidence(kind="distance", distance=0.2)
    with pytest.raises(ValidationError, match="independent supporting evidence"):
        Resolution(match_id="match", segment_id="s", decision="create_new_proposal")


def test_api_requests_responses_and_review_discriminators():
    batch = EvidenceBatch(match_id="match", batch_id="b", producer="collector", events=[event()])
    assert round_trip(ImportRequest(batch=batch, expected_revision=0)).batch == batch
    response = SuccessResponse[list[Player]](
        data=[Player(player_id="pl_1", match_id="match")], revision=2
    )
    assert round_trip(response).data[0].player_id == "pl_1"
    review = ReviewRequest(
        match_id="match",
        expected_revision=2,
        actor="reviewer",
        commands=[
            {
                "kind": "set_assignment",
                "segment_id": "future",
                "player_id": None,
                "reason": "unknown",
            },
            {
                "kind": "review_template",
                "template_id": "future",
                "review_status": "rejected",
                "reason": "blurred",
            },
        ],
    )
    assert round_trip(review).commands[1].review_status == "rejected"
    for model in [EvidenceBatch, ReviewRequest, SuccessResponse[list[Player]]]:
        assert model.model_json_schema()  # All models are usable by future HTTP bindings.
    with pytest.raises(ValidationError):
        ImportRequest(batch=batch, expected_revision=True)
    with pytest.raises(ValidationError, match="end_s"):
        EventQuery(match_id="match", start_s=5, end_s=4)


def test_candidate_and_json_error_details_are_finite():
    with pytest.raises(ValidationError):
        Candidate(
            match_id="match",
            segment_id="s",
            player_id="pl_1",
            feature_space_id="space",
            distance=float("nan"),
            common_parts=1,
            gallery_revision=0,
        )
    with pytest.raises(ValidationError):
        ErrorDetail(code="invalid", message="invalid input", details={"nested": [float("inf")]})


def test_unknown_fields_and_wrong_time_discriminators_are_rejected():
    with pytest.raises(ValidationError):
        PointEventTime(time_s=1, end_s=2)
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        event(time={"kind": "approximate", "time_s": 1})
    with pytest.raises(ValidationError, match="subject_id"):
        event(
            subjects=[
                Subject(subject_id="same", role="shooter"),
                Subject(subject_id="same", role="assist"),
            ]
        )
    with pytest.raises(ValidationError):
        TrackReference(clip_id="clip", track_id=2)


@pytest.mark.parametrize("reason", ["unknown", "other_person", "ambiguous", "rejected"])
def test_unassigned_reasons_do_not_create_players_or_control_templates(reason):
    assignment = Assignment(
        match_id="match",
        segment_id="future",
        method="manual",
        revision=0,
        unassigned_reason=reason,
    )
    assert round_trip(assignment).player_id is None
    command = SetAssignmentCommand(
        segment_id="future",
        player_id=None,
        unassigned_reason=reason,
        reason="reviewed",
    )
    assert round_trip(command).unassigned_reason == reason
    with pytest.raises(ValidationError, match="unassigned_reason"):
        Assignment(
            match_id="match",
            segment_id="future",
            method="manual",
            revision=0,
            status="confirmed",
            player_id="pl_future",
            unassigned_reason=reason,
        )
    with pytest.raises(ValidationError, match="unassigned_reason"):
        SetAssignmentCommand(
            segment_id="future",
            player_id="pl_future",
            unassigned_reason=reason,
            reason="reviewed",
        )


@pytest.mark.parametrize(
    "change",
    [
        {"shape": [3]},
        {"shape": [0, 7]},
        {"shape": [True, 7]},
        {"visibility": {"shape": [2], "semantics": "confidence"}},
        {"normalization": "none"},
        {"visibility": None},
        {"model_hash": "bad-hash"},
    ],
)
def test_inconsistent_feature_space_contracts_rejected(change):
    with pytest.raises(ValidationError):
        feature_space(**change)


def test_known_feature_visibility_contract_and_unknown_backend_description():
    file = FeatureFile(
        feature_file_id="f",
        feature_space_id="space",
        sample_count=1,
        values_file=SourceFile(file_id="values", path="values.npy"),
        index_file=SourceFile(file_id="index", path="index.json"),
    )
    with pytest.raises(ValidationError, match="visibility_file"):
        IdentityEvidence(feature_spaces=[feature_space()], feature_files=[file])
    with pytest.raises(ValidationError, match="visibility length"):
        IdentityEvidence(
            feature_spaces=[feature_space()],
            samples=[
                sample(
                    features=[
                        InlineFeature(
                            feature_space_id="space", values=[0.0] * 21, visibility=[1.0]
                        ),
                    ]
                )
            ],
        )
    with pytest.raises(ValidationError):
        InlineFeature(feature_space_id="space", values=[0], visibility=[1.01])
    space = feature_space(
        backend="future-backend",
        layout="tensor",
        shape=[2, 3, 4],
        normalization="none",
        visibility=None,
    )
    evidence = IdentityEvidence(
        feature_spaces=[space],
        samples=[
            sample(
                features=[
                    InlineFeature(feature_space_id="space", values=[0] * 24),
                ]
            )
        ],
    )
    assert round_trip(evidence).feature_spaces[0].shape == [2, 3, 4]


def test_asset_integrity_is_a_declaration_and_does_not_open_files(tmp_path):
    missing_file = tmp_path / "not-created.png"
    asset = MediaAsset(
        asset_id="image",
        media_type="image",
        file=SourceFile(
            file_id="image-file", path=str(missing_file), sha256="a" * 64, size_bytes=42
        ),
    )
    batch = EvidenceBatch(match_id="match", batch_id="b", producer="upstream", assets=[asset])
    assert round_trip(batch).assets[0].file.sha256 == "a" * 64
    assert not missing_file.exists()


def test_record_locations_and_warnings_preserve_source_without_reading_files():
    location = SourceLocation(file_id="annotations", line_number=4, record_key="clip-20")
    event_source = source("goal:20:1", "event")
    event_source.locations = [location]
    event_source.content_hash = "a" * 64
    original = event(source=event_source)
    original.subjects[0].event_jersey_number = "00"
    original.subjects[0].event_team_id = "green"
    batch = EvidenceBatch(
        match_id="match",
        batch_id="check",
        producer="adapter",
        events=[original],
        warnings=[
            EvidenceWarning(
                code="identity_unconfirmed",
                message="The selection is event evidence only.",
                clip_id="clip",
                locations=[location],
            )
        ],
    )
    decoded = round_trip(batch)
    assert decoded.events[0].subjects[0].event_jersey_number == "00"
    assert decoded.events[0].source.locations[0].line_number == 4
    assert decoded.warnings[0].locations == [location]
    assert decoded.events[0].source.content_hash == "a" * 64
    with pytest.raises(ValidationError):
        SourceLocation(file_id="annotations", line_number=0)
    with pytest.raises(ValidationError):
        Subject(subject_id="shooter", role="shooter", event_jersey_number=27)


def test_track_observed_frames_retain_gaps_without_fabricating_observations():
    raw = Track(
        match_id="match",
        clip_id="clip",
        track_id="6",
        start_s=0,
        end_s=1,
        frame_start=0,
        frame_end_exclusive=6,
        source=source("track", "track"),
        observed_frame_indices=[0, 2, 5],
    )
    assert round_trip(raw).observed_frame_indices == [0, 2, 5]
    for frames in [[0, 0], [2, 0], [0, 6], [-1], [True], ["2"]]:
        with pytest.raises(ValidationError):
            Track.model_validate({**raw.model_dump(), "observed_frame_indices": frames})


@pytest.mark.parametrize(("start_frame", "end_frame"), [(20, 30), (5, 20), (0, 10)])
def test_split_command_checks_only_its_own_replacement_ranges(start_frame, end_frame):
    first = segment(
        segment_id="a",
        start_s=0,
        end_s=2,
        frame_start=0,
        frame_end_exclusive=10,
        supersedes_segment_id="not-in-database",
    )
    second = segment(
        segment_id="b",
        start_s=2,
        end_s=5,
        frame_start=start_frame,
        frame_end_exclusive=end_frame,
        supersedes_segment_id="not-in-database",
    )
    if start_frame >= 10:
        command = SplitSegmentCommand(
            segment_id="not-in-database", segments=[first, second], reason="identity switched"
        )
        assert round_trip(command).segments[1].segment_id == "b"
    else:
        with pytest.raises(ValidationError, match="frame ranges"):
            SplitSegmentCommand(
                segment_id="not-in-database", segments=[first, second], reason="split"
            )


def test_mapping_result_requires_one_resolution_per_original_role():
    with pytest.raises(ValidationError, match="every original subject_id and role"):
        ResolvedEvent(event=event(), subjects=[])
    with pytest.raises(ValidationError, match="point or interval"):
        ResolvedEvent(
            event=event(subjects=[]),
            subjects=[],
            source_time=SourceEventTime(
                source_video_id="full",
                time=IntervalEventTime(start_s=1, end_s=2),
            ),
        )
