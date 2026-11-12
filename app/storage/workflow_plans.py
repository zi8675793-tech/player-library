"""Durable review diagnostics; callers own transactions and revisions."""

import json
import sqlite3


def read_workflow_plans(connection: sqlite3.Connection, match_id: str) -> list[dict]:
    return [
        {**json.loads(row[0]), "committed_revision": row[1]}
        for row in connection.execute(
            "SELECT json,revision FROM workflow_plans WHERE match_id=? ORDER BY segment_id,feature_space_id",
            (match_id,),
        )
    ]


def put_workflow_plan(connection, match_id, segment_id, feature_space_id, revision, payload):
    if not connection.in_transaction:
        raise RuntimeError("workflow plans require a transaction")
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    old = connection.execute(
        "SELECT json,revision FROM workflow_plans WHERE match_id=? AND segment_id=? AND feature_space_id=?",
        (match_id, segment_id, feature_space_id),
    ).fetchone()
    if old and old[0] == encoded:
        return False
    connection.execute(
        "INSERT INTO workflow_plans VALUES (?,?,?,?,?) ON CONFLICT(match_id,segment_id,feature_space_id) "
        "DO UPDATE SET revision=excluded.revision,json=excluded.json",
        (match_id, segment_id, feature_space_id, revision, encoded),
    )
    return True
