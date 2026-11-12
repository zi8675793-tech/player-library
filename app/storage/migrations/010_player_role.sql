-- Index player role for decoupled role filtering without changing the core relational table shape.
-- Role supports: 'player', 'referee', 'staff', 'unknown_role'.
-- Default in Pydantic is 'unknown_role' to ensure historical records are interpreted as anonymous profiles.

CREATE INDEX IF NOT EXISTS idx_players_role ON players(match_id, json_extract(json, '$.role'));
