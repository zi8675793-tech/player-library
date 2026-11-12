CREATE TABLE workflow_operations (
    match_id TEXT NOT NULL REFERENCES matches(match_id),
    kind TEXT NOT NULL,
    request_id TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    PRIMARY KEY (match_id, kind, request_id)
);
CREATE TABLE workflow_reviews (
    match_id TEXT NOT NULL REFERENCES matches(match_id),
    revision INTEGER NOT NULL,
    actor TEXT NOT NULL,
    json TEXT NOT NULL,
    PRIMARY KEY (match_id, revision)
);
CREATE TABLE segment_review_history (
    match_id TEXT NOT NULL,
    segment_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    json TEXT NOT NULL,
    PRIMARY KEY (match_id, segment_id, revision),
    FOREIGN KEY (match_id, segment_id) REFERENCES segments(match_id, segment_id)
);
CREATE TABLE sample_clean_reviews (
    match_id TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    json TEXT NOT NULL,
    PRIMARY KEY (match_id, sample_id, revision),
    FOREIGN KEY (match_id, sample_id) REFERENCES samples(match_id, sample_id)
);
