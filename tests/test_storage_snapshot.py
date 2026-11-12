"""Round-trip real feature bytes into the existing identity algorithms."""

import pytest

from app.contracts.common import Provenance
from app.contracts.evidence import (
    EvidenceBatch,
    FeatureFile,
    FeatureSpace,
    FileFeatureReference,
    IdentityEvidence,
    InlineFeature,
    Sample,
    SourceFile,
    Track,
    TrackingQuality,
)
from app.contracts.identity import (
    Assignment,
    IdentityRisk,
    ManualConstraint,
    ManualTemplateReview,
    MatchingPolicy,
    Player,
    RiskReview,
    Segment,
    Template,
)
from app.identity.associate import associate_segments
from app.identity.prepare import prepare_evidence
from app.storage.database import connect, transaction
from app.storage.features import FeatureConflictError, save_feature_value
from app.storage.repositories import Repository
from app.storage.snapshot import read_identity_snapshot


def source(record_id):
    return Provenance(
        producer="test", record_type="sample", source_record_id=record_id, evidence_version="1"
    )


def test_snapshot_feature_round_trip_and_algorithm(tmp_path):
    path = tmp_path / "library.db"
    space = FeatureSpace(
        feature_space_id="fs",
        backend="kpr",
        model_hash="a" * 64,
        preprocessing={"version": "1", "color_space": "RGB"},
        layout="parts",
        shape=[3, 2],
        normalization="l2",
        distance="euclidean_sqrt_visibility_mean_div2",
        visibility={"shape": [3], "semantics": "visible"},
    )
    feature = InlineFeature(feature_space_id="fs", values=[1.0, 0.0] * 3, visibility=[1.0] * 3)
    ff = FeatureFile(
        feature_file_id="file",
        feature_space_id="fs",
        sample_count=2,
        values_file=SourceFile(file_id="f", path="values.npz"),
        index_file=SourceFile(file_id="i", path="index.json"),
        visibility_file=SourceFile(file_id="v", path="visibility.npz"),
    )
    seg = Segment(match_id="m", clip_id="c", track_id="t", segment_id="s", start_s=0.0, end_s=2.0)

    def sample(sid, time, value, segment=seg):
        return Sample(
            match_id="m",
            sample_id=sid,
            clip_id=segment.clip_id,
            track_id=segment.track_id,
            segment_id=segment.segment_id,
            time_s=time,
            source=source(sid),
            tracking_quality=TrackingQuality(usability="usable", clean_review="approved"),
            features=[value],
        )

    samples = [
        sample(
            f"seed{i}",
            0.1 + i * 0.5,
            FileFeatureReference(feature_space_id="fs", feature_file_id="file", row_index=i),
        )
        for i in range(2)
    ]
    samples.append(sample("inline", 1.1, feature))
    batch = EvidenceBatch(
        match_id="m",
        batch_id="b",
        producer="test",
        identity_evidence=IdentityEvidence(
            segments=[seg],
            tracks=[
                Track(
                    match_id="m",
                    clip_id="c",
                    track_id="t",
                    start_s=0.0,
                    end_s=2.0,
                    source=source("track"),
                )
            ],
            samples=samples,
            feature_spaces=[space],
            feature_files=[ff],
            risks=[
                IdentityRisk(risk_id="risk", kind="quality_review", match_id="m", segment_id="s")
            ],
        ),
    )
    conn = connect(path)
    repo = Repository(conn)
    with transaction(conn):
        assert repo.store_batch(batch)
        assert save_feature_value(conn, "m", ff, space, 0, feature)
        assert not save_feature_value(conn, "m", ff, space, 0, feature)
        with pytest.raises(FeatureConflictError):
            save_feature_value(
                conn, "m", ff, space, 0, feature.model_copy(update={"values": [0.0, 1.0] * 3})
            )
        assert save_feature_value(conn, "m", ff, space, 1, feature)
        repo.put_player(Player(match_id="m", player_id="pl_one"))
        repo.put_player(Player(match_id="m", player_id="pl_no_template"))
        repo.put_assignment(
            Assignment(
                match_id="m",
                segment_id="s",
                player_id="pl_one",
                status="confirmed",
                method="manual",
                revision=0,
            ),
            committed_revision=1,
        )
        for i in range(2):
            repo.put_template(
                Template(
                    match_id="m",
                    template_id=f"t{i}",
                    player_id="pl_one",
                    feature_space_id="fs",
                    sample_id=f"seed{i}",
                    review_status="approved",
                    revision=0,
                    approval={"actor": "human", "reason": "clean", "revision": 1},
                ),
                committed_revision=1,
            )
        repo.put_template_review(
            ManualTemplateReview(
                match_id="m",
                sample_id="seed0",
                player_id="pl_one",
                feature_space_id="fs",
                review_status="approved",
                actor="human",
                reason="clean",
                revision=1,
            )
        )
        repo.put_risk_review(
            RiskReview(
                match_id="m",
                risk_id="risk",
                segment_id="s",
                resolution="cleared",
                clean_sample_ids=["seed0"],
                actor="human",
                reason="reviewed",
                revision=1,
            )
        )
        repo.put_template_review(
            ManualTemplateReview(
                match_id="m",
                sample_id="inline",
                player_id="pl_one",
                feature_space_id="fs",
                review_status="rejected",
                actor="human",
                reason="unsuitable",
                revision=1,
            )
        )
        repo.put_constraint(
            ManualConstraint(
                match_id="m",
                segment_id="s",
                constraint_id="forbid",
                kind="forbid_player",
                player_id="pl_no_template",
                actor="human",
                reason="different",
                revision=1,
            )
        )
        repo.advance_revision("m", 0, actor="human", reason="seed", changes={"seed": True})
    conn.close()

    conn = connect(path)
    result = read_identity_snapshot(conn, "m")
    assert result.gallery.revision == 1
    assert {player.player_id for player in result.gallery.players} == {"pl_one", "pl_no_template"}
    assert result.feature_values[("file", 0)] == feature
    assert next(s for s in result.gallery.evidence.samples if s.sample_id == "inline").features == [
        feature
    ]
    assert (
        conn.execute("SELECT typeof(values_blob) FROM sample_feature_values").fetchone()[0]
        == "blob"
    )
    assert result.gallery.assignments[0].revision == 1
    assert result.gallery.templates[0].approval.actor == "human"
    assert {review.review_status for review in result.template_reviews} == {"approved", "rejected"}
    assert result.gallery.risk_reviews[0].resolution == "cleared"
    assert result.gallery.risk_reviews[0].actor == "human"
    assert result.constraints[0].player_id == "pl_no_template"
    query_seg = Segment(
        match_id="m", clip_id="query", track_id="t", segment_id="q", start_s=0.0, end_s=2.0
    )
    query = EvidenceBatch(
        match_id="m",
        batch_id="q",
        producer="test",
        identity_evidence=IdentityEvidence(
            segments=[query_seg],
            samples=[sample(f"q{i}", 0.1 + i * 0.5, feature, query_seg) for i in range(3)],
            feature_spaces=[space],
        ),
    )
    policy = MatchingPolicy(feature_space_id="fs")
    prepared = prepare_evidence(query, policy, feature_values=result.feature_values)
    association = associate_segments(
        prepared, result.gallery, policy, feature_values=result.feature_values
    )[0]
    assert association.local_candidates[0].player_id == "pl_one"
    assert association.local_candidates[0].distance == 0.0
    assert association.gallery_revision == 1
    conn.close()


def test_inline_source_versions_retain_binary_history_and_old_resubmission(tmp_path):
    import struct

    space = FeatureSpace(
        feature_space_id="fs",
        backend="kpr",
        model_hash="a" * 64,
        preprocessing={"version": "1", "color_space": "RGB"},
        layout="parts",
        shape=[2, 2],
        normalization="l2",
        distance="euclidean_sqrt_visibility_mean_div2",
        visibility={"shape": [2], "semantics": "visible"},
    )
    old = InlineFeature(feature_space_id="fs", values=[1.0, 0.0, 1.0, 0.0], visibility=[1.0, 1.0])
    new = InlineFeature(feature_space_id="fs", values=[0.0, 1.0, 0.0, 1.0], visibility=[1.0, 1.0])
    initial = Sample(
        match_id="m",
        sample_id="s",
        clip_id="c",
        track_id="t",
        time_s=0.1,
        source=source("s"),
        features=[old],
    )
    updated = initial.model_copy(
        update={
            "source": Provenance(
                producer="test",
                record_type="sample",
                source_record_id="s",
                evidence_version="2",
                supersedes={
                    "producer": "test",
                    "record_type": "sample",
                    "source_record_id": "s",
                    "evidence_version": "1",
                },
            ),
            "features": [new],
        }
    )
    batch1 = EvidenceBatch(
        match_id="m",
        batch_id="1",
        producer="test",
        identity_evidence=IdentityEvidence(samples=[initial], feature_spaces=[space]),
    )
    batch2 = EvidenceBatch(
        match_id="m",
        batch_id="2",
        producer="test",
        identity_evidence=IdentityEvidence(samples=[updated]),
    )
    path = tmp_path / "versions.db"
    conn = connect(path)
    repo = Repository(conn)
    with transaction(conn):
        assert repo.store_batch(batch1)
        repo.advance_revision("m", 0, actor="operator", reason="ingest", changes={"sample": "1"})
    with transaction(conn):
        assert repo.store_batch(batch2)
        repo.advance_revision(
            "m", 1, actor="operator", reason="new evidence", changes={"sample": "2"}
        )
    with transaction(conn):
        assert not repo.store_batch(batch1)
    conn.close()
    conn = connect(path)
    snap = read_identity_snapshot(conn, "m")
    assert snap.gallery.evidence.samples[0].features == [new]
    row = conn.execute(
        "SELECT evidence_version, values_blob FROM sample_feature_history WHERE match_id=? AND sample_id=?",
        ("m", "s"),
    ).fetchone()
    assert row[0] == "1"
    assert struct.unpack("<4f", row[1]) == tuple(old.values)
    assert snap.gallery.revision == 2
    conn.close()
