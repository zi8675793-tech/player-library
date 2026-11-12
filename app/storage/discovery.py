"""Immutable machine support records; no human approval is implied by storage."""

import json
import sqlite3
import struct

from app.contracts.common import SourceKey
from app.contracts.discovery import DiscoveryReport
from app.storage.repositories import StorageConflict


def read_discovery_reports(connection: sqlite3.Connection, match_id: str) -> list[dict]:
    return [
        {
            "report": json.loads(row[0]),
            "player_ids_by_group": json.loads(row[1]),
            "committed_revision": row[2],
            "sample_sources": json.loads(row[3]),
        }
        for row in connection.execute(
            "SELECT json,player_map_json,revision,sample_sources_json FROM identity_discovery_reports "
            "WHERE match_id=? ORDER BY revision,report_id",
            (match_id,),
        )
    ]


def sample_binding(sample, assets):
    """Bind support to raw source versions, declared features and image contents."""
    features = [feature.model_dump(mode="json") for feature in sample.features]
    for feature in features:
        if feature["kind"] == "inline":
            for field in ("values", "visibility"):
                if feature[field] is not None:
                    values = feature[field]
                    feature[field] = list(
                        struct.unpack(f"<{len(values)}f", struct.pack(f"<{len(values)}f", *values))
                    )
    return {
        "source_key": sample.source.model_dump(mode="json", include=set(SourceKey.model_fields)),
        "clip_id": sample.clip_id,
        "track_id": sample.track_id,
        "frame_idx": sample.frame_idx,
        "time_s": sample.time_s,
        "image_asset_id": sample.image_asset_id,
        "image_sha256": next(
            (asset.file.sha256 for asset in assets if asset.asset_id == sample.image_asset_id), None
        ),
        "features": features,
    }


def put_discovery_report(
    connection, report: DiscoveryReport, player_ids_by_group, revision, sample_sources
):
    """Persist the exact accepted plan and ID allocation under the caller's transaction."""
    if not connection.in_transaction:
        raise RuntimeError("discovery support requires a transaction")
    if set(player_ids_by_group) != {group.group_id for group in report.groups}:
        raise ValueError("discovery player mapping must cover the exact groups")
    if any(not pid for pid in player_ids_by_group.values()):
        raise ValueError("discovered groups must have valid player IDs")
    new_pids = [
        pid
        for pid in player_ids_by_group.values()
        if not connection.execute(
            "SELECT 1 FROM players WHERE match_id=? AND player_id=?", (report.match_id, pid)
        ).fetchone()
    ]
    if len(set(new_pids)) != len(new_pids):
        raise ValueError("discovered groups creating new players must have distinct player IDs")
    encoded = report.model_dump_json()
    mapping = json.dumps(player_ids_by_group, sort_keys=True, separators=(",", ":"))
    sources = json.dumps(sample_sources, sort_keys=True, separators=(",", ":"))
    previous = connection.execute(
        "SELECT json,player_map_json,sample_sources_json FROM identity_discovery_reports "
        "WHERE match_id=? AND report_id=?",
        (report.match_id, report.report_id),
    ).fetchone()
    if previous:
        if tuple(previous) != (encoded, mapping, sources):
            raise StorageConflict("automatic discovery report is immutable")
        return False
    connection.execute(
        "INSERT INTO identity_discovery_reports VALUES (?,?,?,?,?,?)",
        (report.match_id, report.report_id, revision, encoded, mapping, sources),
    )
    return True
