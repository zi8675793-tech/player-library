CREATE TABLE IF NOT EXISTS incremental_operations (
    match_id TEXT NOT NULL REFERENCES matches(match_id),
    operation_key TEXT NOT NULL,
    phase TEXT NOT NULL,
    status TEXT NOT NULL,
    snapshot_revision INTEGER NOT NULL,
    committed_revision INTEGER NOT NULL,
    payload_hash TEXT NOT NULL,
    result_json TEXT NOT NULL,
    PRIMARY KEY (match_id, operation_key, phase)
);
