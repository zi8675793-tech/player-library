"""File integrity and locations are provenance, not business record identity."""

from app.contracts.common import Provenance, SourceLocation
from app.contracts.evidence import EvidenceBatch, IdentityEvidence, SourceFile, Track
from app.storage.database import connect, transaction
from app.storage.repositories import Repository


def test_file_hash_and_line_change_does_not_advance_business_revision(tmp_path):
    conn = connect(tmp_path / "library.sqlite")
    repo = Repository(conn)

    def evidence(batch_id, line, digest, path):
        source = Provenance(
            producer="upstream",
            record_type="track",
            source_record_id="track-1",
            evidence_version="v1",
            locations=[SourceLocation(file_id="file", line_number=line)],
        )
        track = Track(match_id="m", clip_id="c", track_id="t", start_s=0, end_s=3, source=source)
        return EvidenceBatch(
            match_id="m",
            batch_id=batch_id,
            producer="upstream",
            sources=[SourceFile(file_id="file", path=path, sha256=digest)],
            identity_evidence=IdentityEvidence(tracks=[track]),
        )

    with transaction(conn):
        assert repo.store_batch(evidence("first", 1, "a" * 64, "one.csv"))
        repo.advance_revision("m", 0, actor="tester", reason="first", changes=["track"])
    with transaction(conn):
        assert not repo.store_batch(evidence("next", 30, "b" * 64, "moved.csv"))
    assert repo.revision("m") == 1
    assert conn.execute("SELECT count(*) FROM tracks").fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM source_records").fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM source_locations").fetchone()[0] == 2
    assert conn.execute("SELECT count(*) FROM source_file_observations").fetchone()[0] == 2
    conn.close()
