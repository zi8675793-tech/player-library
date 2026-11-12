-- Raw evidence references can precede their targets. Formal decisions below
-- instead use composite foreign keys to enforce same-match relationships.
CREATE TABLE matches (
    match_id TEXT PRIMARY KEY,
    revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0)
);
CREATE TABLE batches (
    match_id TEXT NOT NULL REFERENCES matches(match_id), batch_id TEXT NOT NULL, receipt_hash TEXT NOT NULL,
    json TEXT NOT NULL CHECK (json_valid(json)), PRIMARY KEY (match_id, batch_id, receipt_hash)
);
CREATE TABLE source_files (
    match_id TEXT NOT NULL REFERENCES matches(match_id), file_id TEXT NOT NULL,
    json TEXT NOT NULL CHECK (json_valid(json)), PRIMARY KEY (match_id, file_id)
);
CREATE TABLE source_file_observations (
    match_id TEXT NOT NULL, file_id TEXT NOT NULL, observation_hash TEXT NOT NULL,
    json TEXT NOT NULL CHECK (json_valid(json)),
    PRIMARY KEY (match_id, file_id, observation_hash),
    FOREIGN KEY (match_id, file_id) REFERENCES source_files(match_id, file_id)
);
CREATE TABLE source_records (
    match_id TEXT NOT NULL REFERENCES matches(match_id), producer TEXT NOT NULL,
    record_type TEXT NOT NULL, source_record_id TEXT NOT NULL, evidence_version TEXT NOT NULL,
    entity_type TEXT NOT NULL, entity_key TEXT NOT NULL, fingerprint TEXT NOT NULL,
    json TEXT NOT NULL CHECK (json_valid(json)),
    PRIMARY KEY (match_id, producer, record_type, source_record_id, evidence_version)
);
CREATE INDEX source_records_entity ON source_records(match_id, entity_type, entity_key);
CREATE TABLE source_locations (
    match_id TEXT NOT NULL, producer TEXT NOT NULL, record_type TEXT NOT NULL,
    source_record_id TEXT NOT NULL, evidence_version TEXT NOT NULL,
    location_json TEXT NOT NULL CHECK (json_valid(location_json)),
    PRIMARY KEY (match_id, producer, record_type, source_record_id, evidence_version, location_json),
    FOREIGN KEY (match_id, producer, record_type, source_record_id, evidence_version)
      REFERENCES source_records(match_id, producer, record_type, source_record_id, evidence_version)
);
CREATE TABLE assets (
    match_id TEXT NOT NULL REFERENCES matches(match_id), asset_id TEXT NOT NULL,
    json TEXT NOT NULL CHECK (json_valid(json)), PRIMARY KEY (match_id, asset_id)
);
CREATE TABLE clips (
    match_id TEXT NOT NULL REFERENCES matches(match_id), clip_id TEXT NOT NULL, asset_id TEXT,
    json TEXT NOT NULL CHECK (json_valid(json)), PRIMARY KEY (match_id, clip_id)
);
CREATE TABLE tracks (
    match_id TEXT NOT NULL REFERENCES matches(match_id), clip_id TEXT NOT NULL,
    track_id TEXT NOT NULL, json TEXT NOT NULL CHECK (json_valid(json)),
    PRIMARY KEY (match_id, clip_id, track_id)
);
CREATE TABLE segments (
    match_id TEXT NOT NULL REFERENCES matches(match_id), segment_id TEXT NOT NULL,
    clip_id TEXT NOT NULL, track_id TEXT NOT NULL, start_s REAL NOT NULL, end_s REAL NOT NULL,
    active INTEGER NOT NULL CHECK (active IN (0,1)), json TEXT NOT NULL CHECK (json_valid(json)),
    CHECK (end_s > start_s), PRIMARY KEY (match_id, segment_id)
);
CREATE INDEX segments_track ON segments(match_id, clip_id, track_id, active, start_s, end_s);
CREATE TABLE samples (
    match_id TEXT NOT NULL REFERENCES matches(match_id), sample_id TEXT NOT NULL,
    clip_id TEXT NOT NULL, track_id TEXT NOT NULL, segment_id TEXT,
    json TEXT NOT NULL CHECK (json_valid(json)), PRIMARY KEY (match_id, sample_id)
);
CREATE INDEX samples_segment ON samples(match_id, segment_id);
CREATE TABLE feature_spaces (
    match_id TEXT NOT NULL REFERENCES matches(match_id), feature_space_id TEXT NOT NULL,
    json TEXT NOT NULL CHECK (json_valid(json)), PRIMARY KEY (match_id, feature_space_id)
);
CREATE TABLE feature_files (
    match_id TEXT NOT NULL REFERENCES matches(match_id), feature_file_id TEXT NOT NULL,
    feature_space_id TEXT NOT NULL, json TEXT NOT NULL CHECK (json_valid(json)),
    PRIMARY KEY (match_id, feature_file_id), UNIQUE (match_id, feature_file_id, feature_space_id)
);
CREATE TABLE feature_values (
    match_id TEXT NOT NULL, feature_file_id TEXT NOT NULL, row_index INTEGER NOT NULL CHECK (row_index >= 0),
    feature_space_id TEXT NOT NULL, dtype TEXT NOT NULL, shape_json TEXT NOT NULL CHECK (json_valid(shape_json)),
    byte_order TEXT NOT NULL, values_blob BLOB NOT NULL,
    visibility_shape_json TEXT CHECK (visibility_shape_json IS NULL OR json_valid(visibility_shape_json)),
    visibility_blob BLOB,
    CHECK ((visibility_shape_json IS NULL) = (visibility_blob IS NULL)),
    PRIMARY KEY (match_id, feature_file_id, row_index),
    FOREIGN KEY (match_id, feature_file_id, feature_space_id)
      REFERENCES feature_files(match_id, feature_file_id, feature_space_id),
    FOREIGN KEY (match_id, feature_space_id) REFERENCES feature_spaces(match_id, feature_space_id)
);
CREATE TABLE sample_feature_values (
    match_id TEXT NOT NULL, sample_id TEXT NOT NULL, feature_space_id TEXT NOT NULL,
    dtype TEXT NOT NULL, shape_json TEXT NOT NULL CHECK (json_valid(shape_json)),
    byte_order TEXT NOT NULL, values_blob BLOB NOT NULL,
    visibility_shape_json TEXT CHECK (visibility_shape_json IS NULL OR json_valid(visibility_shape_json)),
    visibility_blob BLOB, CHECK ((visibility_shape_json IS NULL) = (visibility_blob IS NULL)),
    PRIMARY KEY (match_id, sample_id, feature_space_id),
    FOREIGN KEY (match_id, sample_id) REFERENCES samples(match_id, sample_id),
    FOREIGN KEY (match_id, feature_space_id) REFERENCES feature_spaces(match_id, feature_space_id)
);
CREATE TABLE players (
    match_id TEXT NOT NULL REFERENCES matches(match_id), player_id TEXT NOT NULL,
    alias_of TEXT, json TEXT NOT NULL CHECK (json_valid(json)),
    PRIMARY KEY (match_id, player_id), CHECK (alias_of IS NULL OR alias_of <> player_id),
    FOREIGN KEY (match_id, alias_of) REFERENCES players(match_id, player_id)
);
CREATE TABLE assignments (
    match_id TEXT NOT NULL, segment_id TEXT NOT NULL, revision INTEGER NOT NULL CHECK (revision >= 0),
    player_id TEXT, current INTEGER NOT NULL CHECK (current IN (0,1)),
    json TEXT NOT NULL CHECK (json_valid(json)), PRIMARY KEY (match_id, segment_id, revision),
    FOREIGN KEY (match_id, segment_id) REFERENCES segments(match_id, segment_id),
    FOREIGN KEY (match_id, player_id) REFERENCES players(match_id, player_id)
);
CREATE UNIQUE INDEX assignments_current ON assignments(match_id, segment_id) WHERE current = 1;
CREATE TABLE templates (
    match_id TEXT NOT NULL, template_id TEXT NOT NULL, player_id TEXT NOT NULL,
    sample_id TEXT NOT NULL, feature_space_id TEXT NOT NULL, revision INTEGER NOT NULL CHECK (revision >= 0),
    current INTEGER NOT NULL CHECK (current IN (0,1)), json TEXT NOT NULL CHECK (json_valid(json)),
    PRIMARY KEY (match_id, template_id, revision),
    FOREIGN KEY (match_id, player_id) REFERENCES players(match_id, player_id),
    FOREIGN KEY (match_id, sample_id) REFERENCES samples(match_id, sample_id),
    FOREIGN KEY (match_id, feature_space_id) REFERENCES feature_spaces(match_id, feature_space_id)
);
CREATE UNIQUE INDEX templates_current ON templates(match_id, template_id) WHERE current = 1;
CREATE INDEX templates_player ON templates(match_id, player_id, feature_space_id, current);
CREATE TABLE risks (
    match_id TEXT NOT NULL REFERENCES matches(match_id), risk_id TEXT NOT NULL,
    json TEXT NOT NULL CHECK (json_valid(json)), PRIMARY KEY (match_id, risk_id)
);
CREATE TABLE risk_reviews (
    match_id TEXT NOT NULL, risk_id TEXT NOT NULL, segment_id TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (revision >= 0), json TEXT NOT NULL CHECK (json_valid(json)),
    PRIMARY KEY (match_id, risk_id, segment_id, revision),
    FOREIGN KEY (match_id, risk_id) REFERENCES risks(match_id, risk_id),
    FOREIGN KEY (match_id, segment_id) REFERENCES segments(match_id, segment_id)
);
CREATE TABLE constraints (
    match_id TEXT NOT NULL, constraint_id TEXT NOT NULL, segment_id TEXT NOT NULL,
    player_id TEXT, related_segment_id TEXT, revision INTEGER NOT NULL CHECK (revision >= 0),
    json TEXT NOT NULL CHECK (json_valid(json)), PRIMARY KEY (match_id, constraint_id, revision),
    FOREIGN KEY (match_id, segment_id) REFERENCES segments(match_id, segment_id),
    FOREIGN KEY (match_id, related_segment_id) REFERENCES segments(match_id, segment_id),
    FOREIGN KEY (match_id, player_id) REFERENCES players(match_id, player_id)
);
CREATE TABLE template_reviews (
    match_id TEXT NOT NULL, sample_id TEXT NOT NULL, player_id TEXT NOT NULL,
    feature_space_id TEXT NOT NULL, revision INTEGER NOT NULL CHECK (revision >= 0),
    json TEXT NOT NULL CHECK (json_valid(json)),
    PRIMARY KEY (match_id, sample_id, player_id, feature_space_id, revision),
    FOREIGN KEY (match_id, sample_id) REFERENCES samples(match_id, sample_id),
    FOREIGN KEY (match_id, player_id) REFERENCES players(match_id, player_id),
    FOREIGN KEY (match_id, feature_space_id) REFERENCES feature_spaces(match_id, feature_space_id)
);
CREATE TABLE events (
    match_id TEXT NOT NULL REFERENCES matches(match_id), event_id TEXT NOT NULL, clip_id TEXT NOT NULL,
    json TEXT NOT NULL CHECK (json_valid(json)), PRIMARY KEY (match_id, event_id)
);
CREATE TABLE subjects (
    match_id TEXT NOT NULL, event_id TEXT NOT NULL, subject_id TEXT NOT NULL, role TEXT NOT NULL,
    json TEXT NOT NULL CHECK (json_valid(json)), PRIMARY KEY (match_id, event_id, subject_id),
    FOREIGN KEY (match_id, event_id) REFERENCES events(match_id, event_id)
);
CREATE TABLE subject_refs (
    match_id TEXT NOT NULL, event_id TEXT NOT NULL, subject_id TEXT NOT NULL,
    ref_index INTEGER NOT NULL CHECK (ref_index >= 0), kind TEXT NOT NULL,
    clip_id TEXT, track_id TEXT, segment_id TEXT, json TEXT NOT NULL CHECK (json_valid(json)),
    PRIMARY KEY (match_id, event_id, subject_id, ref_index),
    FOREIGN KEY (match_id, event_id, subject_id) REFERENCES subjects(match_id, event_id, subject_id)
);
CREATE INDEX subject_refs_track ON subject_refs(match_id, clip_id, track_id);
CREATE INDEX subject_refs_segment ON subject_refs(match_id, segment_id);
CREATE TABLE revisions (
    match_id TEXT NOT NULL REFERENCES matches(match_id), revision INTEGER NOT NULL CHECK (revision > 0),
    previous_revision INTEGER NOT NULL CHECK (previous_revision = revision - 1), actor TEXT NOT NULL,
    reason TEXT NOT NULL, changes_json TEXT NOT NULL CHECK (json_valid(changes_json)),
    PRIMARY KEY (match_id, revision)
);
-- Each replaced sample retains the exact old binary row under its original source version.
CREATE TABLE sample_feature_history (
    match_id TEXT NOT NULL, producer TEXT NOT NULL, record_type TEXT NOT NULL,
    source_record_id TEXT NOT NULL, evidence_version TEXT NOT NULL,
    sample_id TEXT NOT NULL, feature_space_id TEXT NOT NULL,
    dtype TEXT NOT NULL, shape_json TEXT NOT NULL CHECK (json_valid(shape_json)),
    byte_order TEXT NOT NULL, values_blob BLOB NOT NULL,
    visibility_shape_json TEXT CHECK (visibility_shape_json IS NULL OR json_valid(visibility_shape_json)),
    visibility_blob BLOB, CHECK ((visibility_shape_json IS NULL) = (visibility_blob IS NULL)),
    PRIMARY KEY (match_id, producer, record_type, source_record_id, evidence_version, sample_id, feature_space_id),
    FOREIGN KEY (match_id, producer, record_type, source_record_id, evidence_version)
      REFERENCES source_records(match_id, producer, record_type, source_record_id, evidence_version),
    FOREIGN KEY (match_id, sample_id) REFERENCES samples(match_id, sample_id),
    FOREIGN KEY (match_id, feature_space_id) REFERENCES feature_spaces(match_id, feature_space_id)
);
