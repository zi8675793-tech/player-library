"""Immutable matching attempt diagnostics, separate from identity blocking reasons."""

import json

from app.storage.repositories import StorageConflict


def put_matching_diagnostics(connection, match_id, audit_key, snapshot_revision, phase, attempts):
    if not connection.in_transaction:
        raise RuntimeError("matching diagnostics require a transaction")
    payload = {
        "schema_version": "matching-attempt-audit-v1",
        "snapshot_revision": snapshot_revision,
        "phase": phase,
        "attempts": attempts,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    old = connection.execute(
        "SELECT json FROM matching_diagnostics WHERE match_id=? AND audit_key=?",
        (match_id, audit_key),
    ).fetchone()
    if old:
        if old[0] != encoded:
            raise StorageConflict("matching diagnostics are immutable")
        return
    connection.execute(
        "INSERT INTO matching_diagnostics VALUES (?,?,?,?,?)",
        (match_id, audit_key, snapshot_revision, phase, encoded),
    )


def read_matching_diagnostics(connection, match_id):
    return [
        {"audit_key": row[0], **json.loads(row[1])}
        for row in connection.execute(
            "SELECT audit_key,json FROM matching_diagnostics WHERE match_id=? "
            "ORDER BY snapshot_revision,phase,audit_key",
            (match_id,),
        )
    ]
