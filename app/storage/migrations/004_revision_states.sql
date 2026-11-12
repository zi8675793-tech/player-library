-- Review histories remain append-only; current marks the effective selection.
ALTER TABLE risk_reviews ADD COLUMN current INTEGER NOT NULL DEFAULT 1 CHECK (current IN (0,1));
ALTER TABLE template_reviews ADD COLUMN current INTEGER NOT NULL DEFAULT 1 CHECK (current IN (0,1));
ALTER TABLE constraints ADD COLUMN current INTEGER NOT NULL DEFAULT 1 CHECK (current IN (0,1));
ALTER TABLE sample_clean_reviews ADD COLUMN current INTEGER NOT NULL DEFAULT 1 CHECK (current IN (0,1));
UPDATE risk_reviews AS r SET current=0 WHERE revision < (SELECT MAX(revision) FROM risk_reviews x WHERE x.match_id=r.match_id AND x.risk_id=r.risk_id AND x.segment_id=r.segment_id);
UPDATE template_reviews AS r SET current=0 WHERE revision < (SELECT MAX(revision) FROM template_reviews x WHERE x.match_id=r.match_id AND x.sample_id=r.sample_id AND x.player_id=r.player_id AND x.feature_space_id=r.feature_space_id);
UPDATE constraints AS r SET current=0 WHERE revision < (SELECT MAX(revision) FROM constraints x WHERE x.match_id=r.match_id AND x.constraint_id=r.constraint_id);
UPDATE sample_clean_reviews AS r SET current=0 WHERE revision < (SELECT MAX(revision) FROM sample_clean_reviews x WHERE x.match_id=r.match_id AND x.sample_id=r.sample_id);
CREATE TABLE revision_states (
    match_id TEXT NOT NULL REFERENCES matches(match_id),
    revision INTEGER NOT NULL,
    json TEXT NOT NULL CHECK (json_valid(json)),
    PRIMARY KEY (match_id, revision)
);
