-- Multi-prototype template metadata, relational indexing and traceability.
CREATE TABLE IF NOT EXISTS template_prototypes (
    match_id TEXT NOT NULL REFERENCES matches(match_id),
    template_id TEXT NOT NULL,
    player_id TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    prototype_id TEXT NOT NULL,
    time_bucket INTEGER,
    source_component TEXT,
    reliable_parts_json TEXT,
    quality_score REAL,
    json TEXT,
    PRIMARY KEY (match_id, template_id, prototype_id)
);
CREATE INDEX IF NOT EXISTS idx_template_prototypes_player
    ON template_prototypes(match_id, player_id);
CREATE INDEX IF NOT EXISTS idx_template_prototypes_sample
    ON template_prototypes(match_id, sample_id);
