"""Batch-local facts are checked without requiring late evidence or stored state."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.contracts.evidence import EvidenceBatch, IdentityEvidence


def provenance(record_type, record_id):
    return {
        "producer": "upstream",
        "record_type": record_type,
        "source_record_id": record_id,
        "evidence_version": "v1",
    }


def source_file(file_id, path):
    return {"file_id": file_id, "path": path, "sha256": "a" * 64, "size_bytes": 10}


@pytest.fixture
def evidence_data():
    video = source_file("video-file", "video.mp4")
    image = source_file("image-file", "crop.png")
    values = source_file("values-file", "features.npy")
    index = source_file("index-file", "samples.jsonl")
    return {
        "match_id": "match",
        "batch_id": "handoff",
        "producer": "adapter",
        "sources": [video, image, values, index],
        "assets": [
            {"asset_id": "video", "media_type": "video", "file": deepcopy(video)},
            {"asset_id": "image", "media_type": "image", "file": deepcopy(image)},
        ],
        "clips": [
            {
                "match_id": "match",
                "clip_id": "clip-a",
                "asset_id": "video",
                "duration_s": 10.0,
                "frame_count": 100,
                "frame_timing": "cfr",
                "fps": 10.0,
            },
            {
                "match_id": "match",
                "clip_id": "clip-b",
                "duration_s": 20.0,
                "frame_count": 200,
            },
        ],
        "identity_evidence": {
            "tracks": [
                {
                    "match_id": "match",
                    "clip_id": "clip-a",
                    "track_id": "7",
                    "start_s": 1.0,
                    "end_s": 9.0,
                    "frame_start": 10,
                    "frame_end_exclusive": 90,
                    "observed_frame_indices": [10, 20, 30, 89],
                    "source": provenance("track", "track-a"),
                },
                {
                    "match_id": "match",
                    "clip_id": "clip-b",
                    "track_id": "7",
                    "start_s": 10.0,
                    "end_s": 16.0,
                    "frame_start": 100,
                    "frame_end_exclusive": 160,
                    "source": provenance("track", "track-b"),
                },
            ],
            "segments": [
                {
                    "match_id": "match",
                    "segment_id": "segment",
                    "clip_id": "clip-a",
                    "track_id": "7",
                    "start_s": 2.0,
                    "end_s": 8.0,
                    "frame_start": 20,
                    "frame_end_exclusive": 80,
                }
            ],
            "samples": [
                {
                    "match_id": "match",
                    "sample_id": "sample",
                    "clip_id": "clip-a",
                    "track_id": "7",
                    "segment_id": "segment",
                    "time_s": 3.0,
                    "frame_idx": 30,
                    "image_asset_id": "image",
                    "source": provenance("sample", "sample"),
                    "features": [
                        {
                            "kind": "file",
                            "feature_space_id": "space",
                            "feature_file_id": "features",
                            "row_index": 0,
                        }
                    ],
                }
            ],
            "feature_spaces": [
                {
                    "feature_space_id": "space",
                    "backend": "declared-backend",
                    "model_hash": "b" * 64,
                    "preprocessing": {"version": "v1", "color_space": "RGB"},
                    "layout": "vector",
                    "shape": [2],
                    "normalization": "none",
                    "distance": "euclidean",
                }
            ],
            "feature_files": [
                {
                    "feature_file_id": "features",
                    "feature_space_id": "space",
                    "sample_count": 1,
                    "values_file": deepcopy(values),
                    "index_file": deepcopy(index),
                }
            ],
        },
        "events": [
            {
                "match_id": "match",
                "event_id": "event",
                "source_event_key": "goal:clip-a:1",
                "clip_id": "clip-a",
                "event_type": "goal",
                "time": {"kind": "point", "time_s": 7.0},
                "source": provenance("event", "event"),
                "subjects": [
                    {
                        "subject_id": "shooter",
                        "role": "shooter",
                        "references": [
                            {
                                "kind": "segment",
                                "segment_id": "segment",
                                "time": {"kind": "interval", "start_s": 2.0, "end_s": 8.0},
                            },
                            {
                                "kind": "track",
                                "clip_id": "clip-b",
                                "track_id": "7",
                                "time": {"kind": "point", "time_s": 12.0},
                            },
                        ],
                    }
                ],
            }
        ],
    }


def records(data, name):
    if name in {"sources", "assets", "clips", "events"}:
        return data[name]
    return data["identity_evidence"][name]


@pytest.mark.parametrize(
    "name",
    [
        "sources",
        "assets",
        "clips",
        "tracks",
        "segments",
        "samples",
        "feature_spaces",
        "feature_files",
        "events",
    ],
)
@pytest.mark.parametrize("conflicting", [False, True])
def test_batch_entity_ids_are_unique_even_for_identical_records(evidence_data, name, conflicting):
    collection = records(evidence_data, name)
    duplicate = deepcopy(collection[0])
    if conflicting:
        changes = {
            "sources": {"path": "other-source"},
            "assets": {"file": source_file("other-file", "other.mp4")},
            "clips": {"duration_s": 11.0},
            "tracks": {"end_s": 9.5},
            "segments": {"end_s": 7.5},
            "samples": {"time_s": 3.5},
            "feature_spaces": {"shape": [3]},
            "feature_files": {"sample_count": 2},
            "events": {"time": {"kind": "point", "time_s": 6.0}},
        }
        duplicate.update(changes[name])
    collection.append(duplicate)
    with pytest.raises(ValidationError):
        EvidenceBatch.model_validate(evidence_data)


@pytest.mark.parametrize(
    "name", ["tracks", "segments", "samples", "feature_spaces", "feature_files"]
)
def test_identity_envelope_also_rejects_duplicate_entity_keys(evidence_data, name):
    collection = records(evidence_data, name)
    collection.append(deepcopy(collection[0]))
    with pytest.raises(ValidationError):
        IdentityEvidence.model_validate(evidence_data["identity_evidence"])


def test_track_ids_are_local_to_clips_and_entity_namespaces_are_independent(evidence_data):
    evidence_data["events"][0]["event_id"] = "sample"
    batch = EvidenceBatch.model_validate(evidence_data)
    assert batch.events[0].event_id == batch.identity_evidence.samples[0].sample_id
    assert [track.track_id for track in batch.identity_evidence.tracks] == ["7", "7"]
    assert EvidenceBatch.model_validate_json(batch.model_dump_json()) == batch


@pytest.mark.parametrize(
    "field,value", [("path", "changed.png"), ("sha256", "c" * 64), ("size_bytes", 11)]
)
def test_same_file_id_cannot_have_conflicting_nested_declarations(evidence_data, field, value):
    evidence_data["assets"][1]["file"][field] = value
    with pytest.raises(ValidationError):
        EvidenceBatch.model_validate(evidence_data)


def test_nested_feature_and_asset_files_share_the_same_file_id_registry(evidence_data):
    file = evidence_data["identity_evidence"]["feature_files"][0]["values_file"]
    file.update(file_id="image-file", path="unrelated.npy")
    with pytest.raises(ValidationError):
        EvidenceBatch.model_validate(evidence_data)


def test_identical_files_can_be_reused_by_multiple_assets_and_feature_roles(evidence_data):
    evidence_data["assets"].append(
        {
            "asset_id": "second-video",
            "media_type": "video",
            "file": deepcopy(evidence_data["sources"][0]),
        }
    )
    feature_file = evidence_data["identity_evidence"]["feature_files"][0]
    feature_file["index_file"] = deepcopy(feature_file["values_file"])
    batch = EvidenceBatch.model_validate(evidence_data)
    assert batch.assets[0].file == batch.assets[2].file
    assert (
        batch.identity_evidence.feature_files[0].values_file
        == batch.identity_evidence.feature_files[0].index_file
    )


@pytest.mark.parametrize("change", [{"clip_id": "future-clip"}, {"track_id": "other-track"}])
def test_sample_must_match_its_supplied_segment_scope(evidence_data, change):
    sample = evidence_data["identity_evidence"]["samples"][0]
    sample.update(change)
    with pytest.raises(ValidationError):
        IdentityEvidence.model_validate(evidence_data["identity_evidence"])


@pytest.mark.parametrize("time_s,frame_idx", [(1.5, 30), (8.0, 30), (3.0, 19), (3.0, 80)])
def test_sample_must_fall_inside_its_known_segment(evidence_data, time_s, frame_idx):
    sample = evidence_data["identity_evidence"]["samples"][0]
    sample.update(time_s=time_s, frame_idx=frame_idx)
    with pytest.raises(ValidationError):
        IdentityEvidence.model_validate(evidence_data["identity_evidence"])


@pytest.mark.parametrize("time_s,frame_idx", [(0.5, 30), (9.0, 30), (3.0, 9), (3.0, 90)])
def test_sample_must_fall_inside_its_known_track_even_without_segment(
    evidence_data, time_s, frame_idx
):
    sample = evidence_data["identity_evidence"]["samples"][0]
    sample.pop("segment_id")
    sample.update(time_s=time_s, frame_idx=frame_idx)
    with pytest.raises(ValidationError):
        IdentityEvidence.model_validate(evidence_data["identity_evidence"])


def test_selected_segment_can_include_unobserved_track_edges(evidence_data):
    identity = evidence_data["identity_evidence"]
    identity["segments"][0].update(start_s=0.0, end_s=10.0, frame_start=0, frame_end_exclusive=100)
    evidence_data["events"][0]["subjects"][0]["references"][0]["time"] = {
        "kind": "interval",
        "start_s": 0.0,
        "end_s": 10.0,
    }
    batch = EvidenceBatch.model_validate(evidence_data)
    track = batch.identity_evidence.tracks[0]
    segment = batch.identity_evidence.segments[0]
    assert segment.frame_start < track.frame_start
    assert segment.frame_end_exclusive > track.frame_end_exclusive
    assert track.observed_frame_indices == [10, 20, 30, 89]


@pytest.mark.parametrize(
    "name,change",
    [
        ("tracks", {"end_s": 10.5}),
        ("tracks", {"frame_end_exclusive": 101}),
        ("segments", {"end_s": 10.5}),
        ("segments", {"frame_end_exclusive": 101}),
        ("samples", {"time_s": 10.0}),
        ("samples", {"frame_idx": 100}),
    ],
)
def test_identity_records_cannot_exceed_their_supplied_clip(evidence_data, name, change):
    evidence_data["identity_evidence"]["tracks"] = []
    evidence_data["identity_evidence"]["segments"] = []
    if name == "tracks":
        template = {
            "match_id": "match",
            "clip_id": "clip-a",
            "track_id": "unrelated",
            "start_s": 0.0,
            "end_s": 10.0,
            "frame_start": 0,
            "frame_end_exclusive": 100,
            "source": provenance("track", "independent-track"),
        }
        evidence_data["identity_evidence"]["tracks"] = [template]
    elif name == "segments":
        evidence_data["identity_evidence"]["segments"] = [
            {
                "match_id": "match",
                "clip_id": "clip-a",
                "track_id": "unrelated",
                "segment_id": "unrelated",
                "start_s": 0.0,
                "end_s": 10.0,
                "frame_start": 0,
                "frame_end_exclusive": 100,
            }
        ]
    records(evidence_data, name)[0].update(change)
    with pytest.raises(ValidationError):
        EvidenceBatch.model_validate(evidence_data)


@pytest.mark.parametrize(
    "time", [{"kind": "point", "time_s": 10.0}, {"kind": "interval", "start_s": 9.0, "end_s": 10.5}]
)
def test_event_time_cannot_exceed_supplied_event_clip(evidence_data, time):
    evidence_data["events"][0]["time"] = time
    with pytest.raises(ValidationError):
        EvidenceBatch.model_validate(evidence_data)


@pytest.mark.parametrize("target", ["segment", "track"])
@pytest.mark.parametrize("kind", ["point", "interval"])
def test_explicit_subject_time_uses_its_target_range_not_event_clip(evidence_data, target, kind):
    reference = evidence_data["events"][0]["subjects"][0]["references"][
        0 if target == "segment" else 1
    ]
    end = 8.0 if target == "segment" else 16.0
    reference["time"] = (
        {"kind": "point", "time_s": end}
        if kind == "point"
        else {"kind": "interval", "start_s": end - 1, "end_s": end + 0.5}
    )
    with pytest.raises(ValidationError):
        EvidenceBatch.model_validate(evidence_data)


def test_explicit_unknown_track_reference_still_obeys_known_target_clip(evidence_data):
    reference = evidence_data["events"][0]["subjects"][0]["references"][1]
    reference.update(track_id="future-track", time={"kind": "point", "time_s": 20.0})
    with pytest.raises(ValidationError):
        EvidenceBatch.model_validate(evidence_data)


@pytest.mark.parametrize("consumer", ["clip", "sample"])
def test_known_asset_type_must_suit_its_consumer(evidence_data, consumer):
    if consumer == "clip":
        evidence_data["clips"][0]["asset_id"] = "image"
    else:
        evidence_data["identity_evidence"]["samples"][0]["image_asset_id"] = "video"
    with pytest.raises(ValidationError):
        EvidenceBatch.model_validate(evidence_data)


def test_half_open_endpoints_and_cross_clip_support_are_valid(evidence_data):
    identity = evidence_data["identity_evidence"]
    identity["segments"][0].update(start_s=1.0, end_s=9.0, frame_start=10, frame_end_exclusive=90)
    identity["samples"][0].update(time_s=1.0, frame_idx=10)
    evidence_data["events"][0]["time"] = {"kind": "interval", "start_s": 9.0, "end_s": 10.0}
    batch = EvidenceBatch.model_validate(evidence_data)
    assert batch.events[0].subjects[0].references[1].time.time_s == 12.0
    assert batch.identity_evidence.samples[0].frame_idx == 10


def test_missing_timing_metadata_does_not_invent_bounds_or_cfr_mapping(evidence_data):
    evidence_data["clips"][0] = {"match_id": "match", "clip_id": "clip-a", "frame_timing": "vfr"}
    identity = evidence_data["identity_evidence"]
    identity["tracks"][0].pop("frame_start")
    identity["tracks"][0].pop("frame_end_exclusive")
    identity["segments"][0].pop("frame_start")
    identity["segments"][0].pop("frame_end_exclusive")
    identity["samples"][0]["frame_idx"] = 123456
    batch = EvidenceBatch.model_validate(evidence_data)
    assert batch.clips[0].duration_s is None
    assert batch.identity_evidence.samples[0].frame_idx == 123456


def test_omitted_reference_time_allows_unresolved_identity_and_track_gaps(evidence_data):
    event = evidence_data["events"][0]
    event["time"] = {"kind": "point", "time_s": 9.5}
    for reference in event["subjects"][0]["references"]:
        reference.pop("time")
    batch = EvidenceBatch.model_validate(evidence_data)
    assert all(reference.time is None for reference in batch.events[0].subjects[0].references)


def test_unknown_targets_remain_valid_for_late_arriving_evidence(evidence_data):
    evidence_data["sources"] = []
    evidence_data["assets"] = []
    evidence_data["clips"] = []
    identity = evidence_data["identity_evidence"]
    identity["tracks"] = []
    identity["segments"] = []
    identity["feature_spaces"] = []
    identity["feature_files"] = []
    sample = identity["samples"][0]
    sample["time_s"] = 1000.0
    sample["frame_idx"] = 10000
    event = evidence_data["events"][0]
    event["time"] = {"kind": "point", "time_s": 1000.0}
    batch = EvidenceBatch.model_validate(evidence_data)
    assert batch.identity_evidence.samples[0].segment_id == "segment"
    assert batch.events[0].subjects[0].references


@pytest.mark.parametrize("known_first", [False, True])
def test_optional_file_integrity_facts_can_be_missing_without_rewriting_declarations(
    evidence_data, known_first
):
    listed = evidence_data["sources"][1]
    nested = evidence_data["assets"][1]["file"]
    missing = nested if known_first else listed
    missing["sha256"] = None
    missing["size_bytes"] = None
    batch = EvidenceBatch.model_validate(evidence_data)
    listed_result = batch.sources[1]
    nested_result = batch.assets[1].file
    unknown_result = nested_result if known_first else listed_result
    known_result = listed_result if known_first else nested_result
    assert unknown_result.sha256 is None and unknown_result.size_bytes is None
    assert known_result.sha256 == "a" * 64 and known_result.size_bytes == 10


@pytest.mark.parametrize("field,conflict", [("sha256", "b" * 64), ("size_bytes", 11)])
def test_conflicting_later_file_facts_are_detected_after_an_unknown_first_value(
    evidence_data, field, conflict
):
    evidence_data["sources"][1][field] = None
    third = deepcopy(evidence_data["assets"][1])
    third["asset_id"] = "second-image"
    third["file"][field] = conflict
    evidence_data["assets"].append(third)
    with pytest.raises(ValidationError):
        EvidenceBatch.model_validate(evidence_data)


@pytest.mark.parametrize("target", ["segment", "track"])
@pytest.mark.parametrize("kind", ["point", "interval"])
def test_explicit_subject_time_cannot_precede_its_target_start(evidence_data, target, kind):
    reference = evidence_data["events"][0]["subjects"][0]["references"][
        0 if target == "segment" else 1
    ]
    start = 2.0 if target == "segment" else 10.0
    reference["time"] = (
        {"kind": "point", "time_s": start - 0.5}
        if kind == "point"
        else {"kind": "interval", "start_s": start - 0.5, "end_s": start + 0.5}
    )
    with pytest.raises(ValidationError):
        EvidenceBatch.model_validate(evidence_data)


def test_identity_and_event_intervals_can_end_at_the_clip_boundary(evidence_data):
    identity = evidence_data["identity_evidence"]
    identity["tracks"][0].update(start_s=0.0, end_s=10.0, frame_start=0, frame_end_exclusive=100)
    identity["segments"][0].update(start_s=0.0, end_s=10.0, frame_start=0, frame_end_exclusive=100)
    evidence_data["events"][0]["time"] = {"kind": "interval", "start_s": 9.0, "end_s": 10.0}
    batch = EvidenceBatch.model_validate(evidence_data)
    assert batch.identity_evidence.tracks[0].frame_end_exclusive == batch.clips[0].frame_count
    assert batch.events[0].time.end_s == batch.clips[0].duration_s
