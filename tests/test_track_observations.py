"""Structured observation geometry remains machine evidence, not identity truth."""

from copy import deepcopy

import pytest
from pydantic import ValidationError
from test_machine_input import machine_settings, make_machine_input

from app.adapters.machine import load_machine_input
from app.contracts.evidence import EvidenceBatch, IdentityEvidence, Track, TrackObservation


def observation(**changes):
    data = {
        "observation_id": "obs-a",
        "match_id": "match",
        "clip_id": "clip",
        "track_id": "track",
        "time_s": 1.0,
        "frame_idx": 10,
        "bbox_xyxy": [1.0, 2.0, 20.0, 30.0],
        "detection_score": 0.99,
        "source": {
            "producer": "tracker",
            "record_type": "observation",
            "source_record_id": "obs-a",
            "evidence_version": "1",
        },
    }
    data.update(changes)
    return TrackObservation.model_validate(data)


def track(**changes):
    data = {
        "match_id": "match",
        "clip_id": "clip",
        "track_id": "track",
        "start_s": 0.0,
        "end_s": 2.0,
        "frame_start": 0,
        "frame_end_exclusive": 20,
        "observed_frame_indices": [0, 10],
        "source": {
            "producer": "tracker",
            "record_type": "track",
            "source_record_id": "track",
            "evidence_version": "1",
        },
    }
    data.update(changes)
    return Track.model_validate(data)


@pytest.mark.parametrize(
    "changes",
    [
        {"bbox_xyxy": [1.0, 2.0, 1.0, 3.0]},
        {"bbox_xyxy": [1.0, 4.0, 3.0, 3.0]},
        {"bbox_xyxy": [1.0, 2.0, float("nan"), 3.0]},
        {"bbox_xyxy": [1.0, 2.0, 3.0]},
        {"detection_score": 1.01},
        {"detection_score": True},
        {"frame_idx": -1},
    ],
)
def test_observation_rejects_invalid_local_geometry_and_scalars(changes):
    with pytest.raises(ValidationError):
        observation(**changes)


def test_observation_supports_unknown_frame_and_score_without_guessing():
    item = observation(frame_idx=None, detection_score=None)
    evidence = IdentityEvidence(observations=[item])
    assert IdentityEvidence.model_validate_json(evidence.model_dump_json()) == evidence
    assert item.frame_idx is None and item.detection_score is None


@pytest.mark.parametrize(
    "changes",
    [{"time_s": 2.0}, {"frame_idx": 20}, {"frame_idx": 11}, {"match_id": "other"}],
)
def test_observation_checks_known_track_scope_and_actual_frame(changes):
    with pytest.raises(ValidationError):
        IdentityEvidence(tracks=[track()], observations=[observation(**changes)])


def test_observations_allow_duplicate_boxes_but_require_distinct_ids():
    first = observation()
    second = observation(observation_id="obs-b")
    evidence = IdentityEvidence(tracks=[track()], observations=[first, second])
    assert len(evidence.observations) == 2
    with pytest.raises(ValidationError, match="duplicate key"):
        IdentityEvidence(observations=[first, first])


def test_observation_batch_checks_match_and_clip_bounds():
    data = {
        "match_id": "match",
        "batch_id": "batch",
        "producer": "adapter",
        "clips": [{"match_id": "match", "clip_id": "clip", "duration_s": 2.0, "frame_count": 20}],
        "identity_evidence": {"observations": [observation().model_dump()]},
    }
    EvidenceBatch.model_validate(data)
    for changes in [{"time_s": 2.0}, {"frame_idx": 20}, {"match_id": "other"}]:
        changed = deepcopy(data)
        changed["identity_evidence"]["observations"][0].update(changes)
        with pytest.raises(ValidationError):
            EvidenceBatch.model_validate(changed)


def test_machine_adapter_keeps_every_actual_box_with_stable_source(tmp_path):
    root = make_machine_input(tmp_path / "input", clips=("alpha",), track_count=2)
    settings = machine_settings(root, tmp_path)
    batch, _ = load_machine_input(settings, batch_id="first")
    replay, _ = load_machine_input(settings, batch_id="second")
    observations = batch.identity_evidence.observations
    assert observations == replay.identity_evidence.observations
    assert len(observations) == 6
    assert {(item.track_id, item.frame_idx) for item in observations} == {
        (str(track_id), frame) for track_id in range(2) for frame in (0, 2, 4)
    }
    assert all(item.bbox_xyxy == (1.0, 2.0, 12.0, 16.0) for item in observations)
    assert all(item.detection_score == 0.9 for item in observations)
    assert len({item.observation_id for item in observations}) == 6
    assert len({item.source.content_hash for item in observations}) == 6
    files = {file.file_id: file.path for file in batch.sources}
    for item in observations:
        assert item.source.record_type == "track_observation"
        assert len(item.source.locations) == 1
        location = item.source.locations[0]
        assert files[location.file_id] == "track/alpha/tracks.jsonl"
        assert location.line_number > 0
    assert EvidenceBatch.model_validate_json(batch.model_dump_json()) == batch
