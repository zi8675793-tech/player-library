CREATE TABLE matching_diagnostics (
    match_id TEXT NOT NULL REFERENCES matches(match_id),
    audit_key TEXT NOT NULL,
    snapshot_revision INTEGER NOT NULL,
    phase TEXT NOT NULL,
    json TEXT NOT NULL CHECK(json_valid(json)),
    PRIMARY KEY (match_id, audit_key)
);
