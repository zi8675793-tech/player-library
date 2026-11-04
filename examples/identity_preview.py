"""Run the pure identity chain on synthetic evidence; no files, database or calibration."""

import json

from app.contracts.common import Provenance
from app.contracts.evidence import (
    EvidenceBatch,
    FeatureSpace,
    GallerySnapshot,
    IdentityEvidence,
    InlineFeature,
    Sample,
)
from app.contracts.identity import Assignment, Player, ReviewRecord, Segment, Template
from app.identity import (
    associate_segments,
    plan_gallery,
    prepare_evidence,
    resolve_associations,
)
from app.settings import load_settings


def preview() -> dict:
    space = FeatureSpace(
        feature_space_id="synthetic-kpr",
        backend="kpr",
        model_hash="0" * 64,
        preprocessing={"version": "synthetic-v1", "color_space": "RGB"},
        layout="parts",
        shape=[3, 2],
        normalization="l2",
        distance="euclidean_sqrt_visibility_mean_div2",
        visibility={"shape": [3], "semantics": "synthetic per-part visibility"},
    )

    def evidence(name, vector, times):
        segment = Segment(
            match_id="synthetic-match",
            segment_id=name,
            clip_id=name,
            track_id="track",
            start_s=0.0,
            end_s=3.0,
        )
        samples = [
            Sample(
                match_id=segment.match_id,
                sample_id=f"{name}-{index}",
                clip_id=segment.clip_id,
                track_id=segment.track_id,
                segment_id=name,
                time_s=time,
                source=Provenance(
                    producer="synthetic",
                    record_type="sample",
                    source_record_id=f"{name}-{index}",
                    evidence_version="1",
                ),
                features=[
                    InlineFeature(
                        feature_space_id=space.feature_space_id,
                        values=vector * 3,
                        visibility=[1.0] * 3,
                    )
                ],
            )
            for index, time in enumerate(times)
        ]
        return segment, samples

    seed_segments, seed_samples, assignments, players, templates = [], [], [], [], []
    for name, vector in [("anchor-a", [1.0, 0.0]), ("anchor-b", [0.0, 1.0])]:
        segment, samples = evidence(name, vector, [0.2, 2.7])
        player_id = f"pl_{name}"
        seed_segments.append(segment)
        seed_samples.extend(samples)
        players.append(Player(match_id=segment.match_id, player_id=player_id))
        assignments.append(
            Assignment(
                match_id=segment.match_id,
                segment_id=name,
                player_id=player_id,
                status="confirmed",
                method="seed",
                revision=1,
            )
        )
        templates.extend(
            Template(
                match_id=segment.match_id,
                template_id=f"template-{sample.sample_id}",
                sample_id=sample.sample_id,
                player_id=player_id,
                feature_space_id=space.feature_space_id,
                review_status="approved",
                approval=ReviewRecord(actor="example-human", reason="synthetic seed", revision=1),
                revision=1,
            )
            for sample in samples
        )
    snapshot = GallerySnapshot(
        match_id="synthetic-match",
        revision=1,
        players=players,
        assignments=assignments,
        templates=templates,
        evidence=IdentityEvidence(
            feature_spaces=[space], segments=seed_segments, samples=seed_samples
        ),
    )
    query_segment, query_samples = evidence("query", [1.0, 0.0], [0.2, 1.2, 2.7])
    batch = EvidenceBatch(
        match_id=snapshot.match_id,
        batch_id="synthetic-query",
        producer="synthetic",
        identity_evidence=IdentityEvidence(
            feature_spaces=[space], segments=[query_segment], samples=query_samples
        ),
    )
    # The example keeps all settings defaults, including no tau/delta/calibration.
    from pathlib import Path

    settings = load_settings(Path(__file__).resolve().parents[1] / "config.example.toml")
    policy = settings.matching.policy_for(space.feature_space_id)
    prepared = prepare_evidence(batch, policy)
    associations = associate_segments(prepared, snapshot, policy)
    resolutions = resolve_associations(prepared, snapshot, associations, policy)
    gallery = plan_gallery(
        prepared,
        snapshot,
        policy,
        final_assignments=[item.assignment for item in resolutions if item.assignment is not None],
    )
    return {
        "synthetic_only": True,
        "policy": policy.model_dump(mode="json"),
        "prepared": prepared.model_dump(mode="json"),
        "associations": [item.model_dump(mode="json") for item in associations],
        "resolutions": [item.model_dump(mode="json") for item in resolutions],
        "gallery_plan": gallery.model_dump(mode="json"),
    }


if __name__ == "__main__":
    print(json.dumps(preview(), ensure_ascii=False, indent=2))
