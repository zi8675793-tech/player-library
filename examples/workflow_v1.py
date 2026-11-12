"""Run initialize -> update -> human confirmation -> current personal events.

Uses synthetic evidence, explicit human approvals, and no calibrated/automatic identity claim.
Run: uv run python -m examples.workflow_v1 [--database var/workflow-v1.sqlite3]
"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from app.contracts.api import (
    ConfirmAssignmentCommand,
    EventQuery,
    InitializeRequest,
    PlayerQuery,
    ReviewRequest,
    SeedIdentity,
    UpdateRequest,
)
from app.contracts.common import Provenance
from app.contracts.events import Event, PointEventTime, SegmentReference, Subject
from app.contracts.evidence import (
    Clip,
    EvidenceBatch,
    FeatureSpace,
    IdentityEvidence,
    InlineFeature,
    Sample,
    Track,
)
from app.contracts.identity import Segment
from app.storage.database import connect
from app.storage.repositories import Repository
from app.workflows.initialize import initialize_library
from app.workflows.read import get_events, get_players
from app.workflows.review import apply_review
from app.workflows.update import update_library


def source(key, record_type="sample"):
    return Provenance(
        producer="workflow-example",
        record_type=record_type,
        source_record_id=key,
        evidence_version="1",
    )


def synthetic_batch(name, *, match_id="workflow-demo", vector=(1.0, 0.0), with_event=True):
    """A parameterized fixture; identifiers, number of clips and people are unconstrained."""
    space = FeatureSpace(
        feature_space_id="demo-kpr",
        backend="kpr",
        model_hash="0" * 64,
        preprocessing={"version": "demo-1", "color_space": "RGB"},
        layout="parts",
        shape=[3, 2],
        normalization="l2",
        distance="euclidean_sqrt_visibility_mean_div2",
        visibility={"shape": [3], "semantics": "per-part visibility"},
    )
    segment = Segment(
        match_id=match_id,
        segment_id=name,
        clip_id=name,
        track_id="track",
        start_s=0.0,
        end_s=3.0,
        frame_start=0,
        frame_end_exclusive=30,
    )
    samples = [
        Sample(
            match_id=match_id,
            sample_id=f"{name}-{i}",
            clip_id=name,
            track_id="track",
            segment_id=name,
            time_s=time,
            frame_idx=frame,
            source=source(f"{name}-{i}"),
            features=[
                InlineFeature(
                    feature_space_id=space.feature_space_id,
                    values=list(vector) * 3,
                    visibility=[1.0] * 3,
                )
            ],
        )
        for i, (time, frame) in enumerate([(0.2, 2), (1.2, 12), (2.7, 27)])
    ]
    event = Event(
        match_id=match_id,
        event_id=f"event-{name}",
        source_event_key=f"event-{name}",
        clip_id=name,
        event_type="shot",
        outcome="made",
        time=PointEventTime(time_s=1.2),
        source=source(f"event-{name}", "event"),
        subjects=[
            Subject(
                subject_id="shooter", role="shooter", references=[SegmentReference(segment_id=name)]
            )
        ],
    )
    return EvidenceBatch(
        match_id=match_id,
        batch_id=f"delivery-{name}",
        producer="workflow-example",
        clips=[
            Clip(
                match_id=match_id,
                clip_id=name,
                duration_s=3.0,
                frame_count=30,
                frame_timing="cfr",
                fps=10.0,
            )
        ],
        identity_evidence=IdentityEvidence(
            feature_spaces=[space],
            segments=[segment],
            samples=samples,
            tracks=[
                Track(
                    match_id=match_id,
                    clip_id=name,
                    track_id="track",
                    start_s=0.0,
                    end_s=3.0,
                    frame_start=0,
                    frame_end_exclusive=30,
                    observed_frame_indices=list(range(30)),
                    source=source(f"track-{name}", "track"),
                )
            ],
        ),
        events=[event] if with_event else None,
    )


def run(database):
    connection = connect(database)
    try:
        repo = Repository(connection)
        try:
            revision = repo.revision("workflow-demo")
        except KeyError:
            revision = 0
        initialized = initialize_library(
            connection,
            InitializeRequest(
                batch=synthetic_batch("anchor"),
                expected_revision=revision,
                request_id="initial-anchors",
                actor="example-reviewer",
                reason="manual synthetic initialization",
                seeds=[
                    SeedIdentity(
                        request_id="athlete-a",
                        display_name="Example player",
                        segment_ids=["anchor"],
                        clean_sample_ids=["anchor-0", "anchor-2"],
                        template_sample_ids=["anchor-0", "anchor-2"],
                        actor="example-reviewer",
                        reason="explicit clean anchor and template approval",
                    )
                ],
            ),
        )
        player_id = initialized.data.created_player_ids[0]
        updated = update_library(
            connection,
            UpdateRequest(
                batch=synthetic_batch("later"), expected_revision=repo.revision("workflow-demo")
            ),
        )
        reviewed = apply_review(
            connection,
            ReviewRequest(
                match_id="workflow-demo",
                expected_revision=repo.revision("workflow-demo"),
                request_id="confirm-later",
                actor="example-reviewer",
                commands=[
                    ConfirmAssignmentCommand(
                        segment_id="later",
                        player_id=player_id,
                        reason="human inspected the synthetic later segment",
                    )
                ],
            ),
        )
        # Close and reopen: all reads use the same official persisted state/rules.
        connection.close()
        connection = connect(database)
        return {
            "initialize_revision": initialized.revision,
            "update_revision": updated.revision,
            "review_revision": reviewed.revision,
            "players": get_players(connection, PlayerQuery(match_id="workflow-demo")).model_dump(
                mode="json"
            ),
            "personal_events": get_events(
                connection, EventQuery(match_id="workflow-demo", player_id=player_id)
            ).model_dump(mode="json"),
        }
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path)
    args = parser.parse_args()
    if args.database:
        result = run(args.database)
    else:
        with TemporaryDirectory(prefix="player-library-example-") as directory:
            result = run(Path(directory) / "library.sqlite3")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
