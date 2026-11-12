-- Current reviewable algorithm output. Original evidence remains in its own tables.
CREATE TABLE workflow_plans (
    match_id TEXT NOT NULL REFERENCES matches(match_id),
    segment_id TEXT NOT NULL,
    feature_space_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    json TEXT NOT NULL,
    PRIMARY KEY (match_id, segment_id, feature_space_id)
);
