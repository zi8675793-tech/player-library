-- Machine segmentation proposals and eligibility fingerprints.
-- Retains segment eligibility (supported/unknown) across incremental batches.
CREATE TABLE IF NOT EXISTS segmentation_proposals (
    match_id TEXT NOT NULL REFERENCES matches(match_id),
    segment_id TEXT NOT NULL,
    parent_segment_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('supported', 'unknown')),
    evidence_fingerprint TEXT,
    json TEXT NOT NULL CHECK (json_valid(json)),
    PRIMARY KEY (match_id, segment_id)
);
CREATE INDEX IF NOT EXISTS idx_segmentation_proposals_status
    ON segmentation_proposals(match_id, status);
