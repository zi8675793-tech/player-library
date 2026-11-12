-- Immutable machine support, separate from human reviews and current mappings.
CREATE TABLE identity_discovery_reports (
    match_id TEXT NOT NULL REFERENCES matches(match_id),
    report_id TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (revision > 0),
    json TEXT NOT NULL CHECK (json_valid(json)),
    player_map_json TEXT NOT NULL CHECK (json_valid(player_map_json)),
    sample_sources_json TEXT NOT NULL CHECK (json_valid(sample_sources_json)),
    PRIMARY KEY (match_id, report_id)
);
