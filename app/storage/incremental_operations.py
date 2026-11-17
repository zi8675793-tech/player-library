"""Durable operation states for two-stage incremental batch updates and replay resumption."""

import json
import sqlite3

from app.storage.repositories import StorageConflict


def put_incremental_operation(
    connection: sqlite3.Connection,
    match_id: str,
    operation_key: str,
    phase: str,
    status: str,
    snapshot_revision: int,
    committed_revision: int,
    payload_hash: str,
    result_data: dict,
) -> None:
    if not connection.in_transaction:
        raise RuntimeError("incremental operations require a transaction")
    encoded_result = json.dumps(
        result_data, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    old = connection.execute(
        "SELECT status, snapshot_revision, committed_revision, payload_hash, result_json "
        "FROM incremental_operations WHERE match_id=? AND operation_key=? AND phase=?",
        (match_id, operation_key, phase),
    ).fetchone()
    if old:
        if (old[0], old[1], old[2], old[3], old[4]) != (
            status,
            snapshot_revision,
            committed_revision,
            payload_hash,
            encoded_result,
        ):
            if old[0] == "in_progress" and status == "completed":
                connection.execute(
                    "UPDATE incremental_operations SET status=?, committed_revision=?, result_json=? "
                    "WHERE match_id=? AND operation_key=? AND phase=?",
                    (status, committed_revision, encoded_result, match_id, operation_key, phase),
                )
                return
            raise StorageConflict(
                f"incremental operation {operation_key!r} phase {phase!r} is immutable"
            )
        return
    connection.execute(
        "INSERT INTO incremental_operations VALUES (?,?,?,?,?,?,?,?)",
        (
            match_id,
            operation_key,
            phase,
            status,
            snapshot_revision,
            committed_revision,
            payload_hash,
            encoded_result,
        ),
    )


def get_incremental_operation(
    connection: sqlite3.Connection, match_id: str, operation_key: str, phase: str
) -> dict | None:
    row = connection.execute(
        "SELECT phase, status, snapshot_revision, committed_revision, payload_hash, result_json "
        "FROM incremental_operations WHERE match_id=? AND operation_key=? AND phase=?",
        (match_id, operation_key, phase),
    ).fetchone()
    if not row:
        return None
    return {
        "phase": row[0],
        "status": row[1],
        "snapshot_revision": row[2],
        "committed_revision": row[3],
        "payload_hash": row[4],
        "result": json.loads(row[5]),
    }


def read_incremental_operations(connection: sqlite3.Connection, match_id: str) -> list[dict]:
    return [
        {
            "operation_key": row[0],
            "phase": row[1],
            "status": row[2],
            "snapshot_revision": row[3],
            "committed_revision": row[4],
            "payload_hash": row[5],
            "result": json.loads(row[6]),
        }
        for row in connection.execute(
            "SELECT operation_key, phase, status, snapshot_revision, committed_revision, "
            "payload_hash, result_json FROM incremental_operations WHERE match_id=? "
            "ORDER BY snapshot_revision, phase, operation_key",
            (match_id,),
        )
    ]
