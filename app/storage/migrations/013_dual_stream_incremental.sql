-- T3 contracts remain separate from legacy evidence JSON. Existing players and
-- assignments receive valid legacy projections; canonical IDs are unchanged.
CREATE TABLE IF NOT EXISTS dual_stream_segments (
    match_id TEXT NOT NULL REFERENCES matches(match_id),
    segment_id TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    created_revision INTEGER NOT NULL CHECK (created_revision > 0),
    segment_json TEXT NOT NULL CHECK (json_valid(segment_json)),
    feature_json TEXT NOT NULL CHECK (json_valid(feature_json)),
    dtype TEXT NOT NULL CHECK (dtype = 'float32'),
    shape_json TEXT NOT NULL CHECK (shape_json = '[512]'),
    byte_order TEXT NOT NULL CHECK (byte_order = 'little'),
    vector_blob BLOB NOT NULL CHECK (length(vector_blob) = 2048),
    PRIMARY KEY (match_id, segment_id),
    FOREIGN KEY (match_id, segment_id) REFERENCES segments(match_id, segment_id)
);
CREATE TABLE IF NOT EXISTS dual_stream_profiles (
    match_id TEXT NOT NULL,
    player_id TEXT NOT NULL,
    legacy_player_id TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (revision > 0),
    json TEXT NOT NULL CHECK (json_valid(json)),
    PRIMARY KEY (match_id, player_id, revision),
    FOREIGN KEY (match_id, legacy_player_id) REFERENCES players(match_id, player_id)
);
CREATE INDEX IF NOT EXISTS dual_stream_profiles_legacy
    ON dual_stream_profiles(match_id, legacy_player_id);
CREATE TABLE IF NOT EXISTS prototypes (
    match_id TEXT NOT NULL REFERENCES matches(match_id),
    prototype_id TEXT NOT NULL,
    player_id TEXT NOT NULL,
    legacy_player_id TEXT NOT NULL,
    segment_id TEXT NOT NULL,
    created_at_revision INTEGER NOT NULL CHECK (created_at_revision > 0),
    json TEXT NOT NULL CHECK (json_valid(json)),
    PRIMARY KEY (match_id, prototype_id),
    FOREIGN KEY (match_id, legacy_player_id) REFERENCES players(match_id, player_id),
    FOREIGN KEY (match_id, segment_id) REFERENCES dual_stream_segments(match_id, segment_id)
);
CREATE INDEX IF NOT EXISTS prototypes_player ON prototypes(match_id, player_id);
CREATE INDEX IF NOT EXISTS prototypes_segment ON prototypes(match_id, segment_id);
CREATE TABLE IF NOT EXISTS dual_stream_decisions (
    match_id TEXT NOT NULL,
    segment_id TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (revision > 0),
    json TEXT NOT NULL CHECK (json_valid(json)),
    PRIMARY KEY (match_id, segment_id, revision),
    FOREIGN KEY (match_id, segment_id) REFERENCES dual_stream_segments(match_id, segment_id)
);
CREATE INDEX IF NOT EXISTS dual_stream_operation_fingerprint
    ON workflow_operations(match_id, kind, payload_hash);
CREATE TRIGGER IF NOT EXISTS dual_stream_segments_no_update
    BEFORE UPDATE ON dual_stream_segments BEGIN
    SELECT RAISE(ABORT, 'dual-stream evidence is immutable'); END;
CREATE TRIGGER IF NOT EXISTS dual_stream_segments_no_delete
    BEFORE DELETE ON dual_stream_segments BEGIN
    SELECT RAISE(ABORT, 'dual-stream evidence is immutable'); END;
CREATE TRIGGER IF NOT EXISTS dual_stream_profiles_no_update
    BEFORE UPDATE ON dual_stream_profiles BEGIN
    SELECT RAISE(ABORT, 'dual-stream profile history is immutable'); END;
CREATE TRIGGER IF NOT EXISTS dual_stream_profiles_no_delete
    BEFORE DELETE ON dual_stream_profiles BEGIN
    SELECT RAISE(ABORT, 'dual-stream profile history is immutable'); END;
CREATE TRIGGER IF NOT EXISTS prototypes_no_update
    BEFORE UPDATE ON prototypes BEGIN
    SELECT RAISE(ABORT, 'dual-stream prototypes are immutable'); END;
CREATE TRIGGER IF NOT EXISTS prototypes_no_delete
    BEFORE DELETE ON prototypes BEGIN
    SELECT RAISE(ABORT, 'dual-stream prototypes are immutable'); END;
CREATE TRIGGER IF NOT EXISTS dual_stream_decisions_no_update
    BEFORE UPDATE ON dual_stream_decisions BEGIN
    SELECT RAISE(ABORT, 'dual-stream decision history is immutable'); END;
CREATE TRIGGER IF NOT EXISTS dual_stream_decisions_no_delete
    BEFORE DELETE ON dual_stream_decisions BEGIN
    SELECT RAISE(ABORT, 'dual-stream decision history is immutable'); END;
CREATE TRIGGER IF NOT EXISTS dual_stream_operations_no_update
    BEFORE UPDATE ON workflow_operations WHEN OLD.kind = 'dual_stream_incremental' BEGIN
    SELECT RAISE(ABORT, 'dual-stream operation audit is immutable'); END;
CREATE TRIGGER IF NOT EXISTS dual_stream_operations_no_delete
    BEFORE DELETE ON workflow_operations WHEN OLD.kind = 'dual_stream_incremental' BEGIN
    SELECT RAISE(ABORT, 'dual-stream operation audit is immutable'); END;
