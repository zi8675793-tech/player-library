"""Late facts cannot invalidate historical evidence at a business commit."""

import pytest

from app.contracts.common import Provenance
from app.contracts.events import Event, IntervalEventTime, PointEventTime, Subject, TrackReference
from app.contracts.evidence import (
    Clip,
    EvidenceBatch,
    FeatureSpace,
    GallerySnapshot,
    IdentityEvidence,
    InlineFeature,
    MediaAsset,
    Sample,
    SourceFile,
    Track,
)
from app.contracts.identity import Assignment, MatchingPolicy, Player, Segment, Template
from app.storage.database import connect, transaction
from app.storage.repositories import Repository
from app.storage.snapshot import read_identity_snapshot
from app.workflows._consistency import (
    validate_evidence_context,
    validate_segment_frames,
    validate_templates,
)


def segment(segment_id="segment", *, start=0, end=10, frames=None, **updates):
    return Segment(
        match_id="match",
        segment_id=segment_id,
        clip_id="clip",
        track_id="track",
        start_s=start,
        end_s=end,
        frame_start=frames[0] if frames else None,
        frame_end_exclusive=frames[1] if frames else None,
        **updates,
    )


def source(key):
    return Provenance(
        producer="test", record_type="evidence", source_record_id=key, evidence_version="1"
    )


def test_late_clip_bounds_revalidate_historical_segments_and_rollback(tmp_path):
    connection = connect(tmp_path / "library.sqlite3")
    repo = Repository(connection)
    try:
        with transaction(connection):
            repo.store_batch(
                EvidenceBatch(
                    match_id="match",
                    batch_id="early",
                    producer="test",
                    identity_evidence=IdentityEvidence(segments=[segment()]),
                )
            )
            repo.advance_revision("match", 0, actor="test", reason="early", changes={})
        with pytest.raises(ValueError, match="exceeds clip"), transaction(connection):
            repo.store_batch(
                EvidenceBatch(
                    match_id="match",
                    batch_id="late",
                    producer="test",
                    clips=[Clip(match_id="match", clip_id="clip", duration_s=1)],
                )
            )
            validate_evidence_context(repo, read_identity_snapshot(connection, "match").gallery)
        assert repo.revision("match") == 1
        assert repo.clips("match") == []
        assert repo.receipts("match", "late") == []
    finally:
        connection.close()


def test_late_media_type_is_checked_against_older_sample(tmp_path):
    connection = connect(tmp_path / "library.sqlite3")
    repo = Repository(connection)
    sample = Sample(
        match_id="match",
        sample_id="sample",
        clip_id="clip",
        track_id="track",
        time_s=0.5,
        image_asset_id="media",
        source=source("sample"),
    )
    try:
        with transaction(connection):
            repo.store_batch(
                EvidenceBatch(
                    match_id="match",
                    batch_id="early",
                    producer="test",
                    identity_evidence=IdentityEvidence(samples=[sample]),
                )
            )
        with pytest.raises(ValueError, match="image_asset_id"), transaction(connection):
            repo.store_batch(
                EvidenceBatch(
                    match_id="match",
                    batch_id="late",
                    producer="test",
                    assets=[
                        MediaAsset(
                            asset_id="media",
                            media_type="video",
                            file=SourceFile(file_id="file", path="input.mp4"),
                        )
                    ],
                )
            )
            validate_evidence_context(repo, read_identity_snapshot(connection, "match").gallery)
        assert read_identity_snapshot(connection, "match").gallery.assets == []
    finally:
        connection.close()


def test_unknown_event_references_remain_valid_and_match_scoped(tmp_path):
    connection = connect(tmp_path / "library.sqlite3")
    repo = Repository(connection)
    try:
        with transaction(connection):
            repo.store_batch(
                EvidenceBatch(
                    match_id="match",
                    batch_id="event-first",
                    producer="test",
                    events=[
                        Event(
                            match_id="match",
                            event_id="event",
                            source_event_key="event",
                            clip_id="clip",
                            event_type="shot",
                            source=source("event"),
                            time=PointEventTime(time_s=3),
                            subjects=[
                                Subject(
                                    subject_id="shooter",
                                    role="shooter",
                                    references=[TrackReference(clip_id="clip", track_id="track")],
                                )
                            ],
                        )
                    ],
                )
            )
            repo.store_batch(
                EvidenceBatch(
                    match_id="other",
                    batch_id="other",
                    producer="test",
                    clips=[Clip(match_id="other", clip_id="clip", duration_s=1)],
                )
            )
            validate_evidence_context(repo, read_identity_snapshot(connection, "match").gallery)
    finally:
        connection.close()


def test_incompatible_late_event_support_does_not_block_valid_identity_evidence(tmp_path):
    connection = connect(tmp_path / "library.sqlite3")
    repo = Repository(connection)
    event = Event(
        match_id="match",
        event_id="event",
        source_event_key="event",
        clip_id="clip",
        event_type="shot",
        source=source("event"),
        time=PointEventTime(time_s=1.5),
        subjects=[
            Subject(
                subject_id="shooter",
                role="shooter",
                references=[
                    TrackReference(
                        clip_id="clip",
                        track_id="track",
                        time=IntervalEventTime(start_s=0, end_s=5),
                    )
                ],
            )
        ],
    )
    try:
        with transaction(connection):
            repo.store_batch(
                EvidenceBatch(
                    match_id="match", batch_id="event-first", producer="test", events=[event]
                )
            )
        with transaction(connection):
            repo.store_batch(
                EvidenceBatch(
                    match_id="match",
                    batch_id="identity-later",
                    producer="test",
                    identity_evidence=IdentityEvidence(
                        tracks=[
                            Track(
                                match_id="match",
                                clip_id="clip",
                                track_id="track",
                                start_s=1,
                                end_s=2,
                                source=source("track"),
                            )
                        ]
                    ),
                )
            )
            validate_evidence_context(repo, read_identity_snapshot(connection, "match").gallery)
        assert repo.events("match") == [event]
        assert len(read_identity_snapshot(connection, "match").gallery.evidence.tracks) == 1
    finally:
        connection.close()


@pytest.mark.parametrize("last_frames", [(10, 30), (0, 5)])
def test_frame_overlap_or_reversal_cannot_hide_behind_unframed_segment(last_frames):
    segments = [
        segment("early", start=0, end=1, frames=(0, 20)),
        segment("unframed", start=1, end=2),
        segment("late", start=2, end=3, frames=last_frames),
    ]
    with pytest.raises(ValueError, match="frame ranges overlap or reverse"):
        validate_segment_frames(segments)


def test_half_open_frame_boundaries_and_inactive_segments_are_valid():
    segments = [
        segment("early", start=0, end=1, frames=(0, 20)),
        segment("unframed", start=1, end=2),
        segment("late", start=2, end=3, frames=(20, 30)),
        segment("retired", frames=(0, 30), active=False),
    ]
    before = [item.model_dump() for item in segments]
    validate_segment_frames(list(reversed(segments)))
    assert [item.model_dump() for item in segments] == before


def template_snapshot():
    space = FeatureSpace(
        feature_space_id="space",
        backend="kpr",
        model_hash="a" * 64,
        preprocessing={"version": "1", "color_space": "RGB"},
        layout="parts",
        shape=[2, 2],
        normalization="l2",
        distance="euclidean_sqrt_visibility_mean_div2",
        visibility={"shape": [2], "semantics": "per part"},
    )
    sample = Sample(
        match_id="match",
        sample_id="sample",
        clip_id="clip",
        track_id="track",
        segment_id="segment",
        time_s=0.5,
        source=source("sample"),
        features=[
            InlineFeature(feature_space_id="space", values=[1.0, 0.0] * 2, visibility=[1.0] * 2)
        ],
    )
    return GallerySnapshot(
        match_id="match",
        revision=1,
        players=[Player(match_id="match", player_id="pl_player")],
        evidence=IdentityEvidence(segments=[segment()], samples=[sample], feature_spaces=[space]),
        assignments=[
            Assignment(
                match_id="match",
                segment_id="segment",
                player_id="pl_player",
                status="confirmed",
                method="manual",
                revision=1,
            )
        ],
        templates=[
            Template(
                match_id="match",
                template_id="template",
                player_id="pl_player",
                sample_id="sample",
                feature_space_id="space",
                review_status="approved",
                revision=1,
            )
        ],
    )


def test_active_template_requires_current_confirmed_owner():
    snapshot = template_snapshot()

    def policy(space):
        return MatchingPolicy(feature_space_id=space)

    validate_templates(snapshot, {}, policy)
    snapshot.assignments[0].status = "proposed"
    with pytest.raises(ValueError, match="template_assignment_not_confirmed"):
        validate_templates(snapshot, {}, policy)
    snapshot.templates[0].active = False
    validate_templates(snapshot, {}, policy)
