"""Formal references are complete and remain within their match scope."""

import pytest

from app.contracts.common import Provenance
from app.contracts.evidence import Sample, Track
from app.contracts.identity import (
    Assignment,
    IdentityRisk,
    ManualConstraint,
    ManualTemplateReview,
    RiskReview,
    Segment,
    Template,
)
from app.storage.database import connect, transaction
from app.storage.relations import (
    MissingReference,
    check_assignment_references,
    check_constraint_references,
    check_risk_review_references,
    check_template_references,
    check_template_review_references,
)


def test_formal_reference_checks_reject_missing_and_cross_match(tmp_path):
    conn = connect(tmp_path / "library.sqlite")
    source = Provenance(
        producer="source", record_type="track", source_record_id="t", evidence_version="1"
    )
    segment = Segment(match_id="a", segment_id="s", clip_id="c", track_id="t", start_s=0, end_s=2)
    track = Track(match_id="a", clip_id="c", track_id="t", start_s=0, end_s=2, source=source)
    sample = Sample(
        match_id="a",
        sample_id="sample",
        clip_id="c",
        track_id="t",
        time_s=1,
        segment_id="s",
        source=source,
    )
    risk = IdentityRisk(match_id="a", risk_id="risk", kind="mixed_identity", segment_id="s")
    with transaction(conn):
        conn.executemany("INSERT INTO matches(match_id) VALUES (?)", [("a",), ("b",)])
        conn.execute(
            "INSERT INTO segments VALUES (?,?,?,?,?,?,?,?)",
            ("a", "s", "c", "t", 0, 2, 1, segment.model_dump_json()),
        )
        conn.execute(
            "INSERT INTO tracks VALUES (?,?,?,?)", ("a", "c", "t", track.model_dump_json())
        )
        conn.execute(
            "INSERT INTO samples VALUES (?,?,?,?,?,?)",
            ("a", "sample", "c", "t", "s", sample.model_dump_json()),
        )
        conn.execute("INSERT INTO risks VALUES (?,?,?)", ("a", "risk", risk.model_dump_json()))
        conn.execute("INSERT INTO players VALUES (?,?,?,?)", ("b", "pl_b", None, "{}"))
        conn.execute("INSERT INTO feature_spaces VALUES (?,?,?)", ("a", "space", "{}"))
        assignment = Assignment(
            match_id="a",
            segment_id="s",
            player_id="pl_b",
            status="confirmed",
            method="manual",
            revision=0,
        )
        with pytest.raises(MissingReference, match="players"):
            check_assignment_references(conn, assignment)
        constraint = ManualConstraint(
            match_id="a",
            constraint_id="x",
            segment_id="s",
            kind="forbid_player",
            player_id="pl_b",
            actor="reviewer",
            reason="test",
            revision=1,
        )
        with pytest.raises(MissingReference, match="players"):
            check_constraint_references(conn, constraint)
        review = RiskReview(
            match_id="a",
            risk_id="risk",
            segment_id="s",
            resolution="cleared",
            clean_sample_ids=["missing"],
            actor="reviewer",
            reason="test",
            revision=1,
        )
        with pytest.raises(MissingReference, match="samples"):
            check_risk_review_references(conn, review)
        conn.execute("INSERT INTO players VALUES (?,?,?,?)", ("a", "pl_b", None, "{}"))
        check_assignment_references(conn, assignment)
        check_constraint_references(conn, constraint)
        template = Template(
            match_id="a",
            template_id="template",
            sample_id="sample",
            player_id="pl_b",
            feature_space_id="space",
            revision=1,
        )
        with pytest.raises(MissingReference, match="does not declare"):
            check_template_references(conn, template)
        template_review = ManualTemplateReview(
            match_id="a",
            sample_id="sample",
            player_id="pl_b",
            feature_space_id="space",
            review_status="approved",
            actor="reviewer",
            reason="test",
            revision=1,
        )
        with pytest.raises(MissingReference, match="does not declare"):
            check_template_review_references(conn, template_review)
        conn.execute("DELETE FROM tracks WHERE match_id=?", ("a",))
        with pytest.raises(MissingReference, match="track"):
            check_assignment_references(conn, assignment)
    conn.close()
