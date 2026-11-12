-- Frame-level machine evidence is immutable/versioned like tracks and samples.
-- References may arrive before the parent track, so only match scope is a FK.
CREATE TABLE IF NOT EXISTS track_observations (
    match_id TEXT NOT NULL REFERENCES matches(match_id),
    observation_id TEXT NOT NULL,
    clip_id TEXT NOT NULL,
    track_id TEXT NOT NULL,
    time_s REAL NOT NULL CHECK (time_s >= 0),
    frame_idx INTEGER CHECK (frame_idx IS NULL OR frame_idx >= 0),
    json TEXT NOT NULL CHECK (json_valid(json)),
    PRIMARY KEY (match_id, observation_id)
);
CREATE INDEX IF NOT EXISTS track_observations_track
    ON track_observations(match_id, clip_id, track_id, time_s, observation_id);
