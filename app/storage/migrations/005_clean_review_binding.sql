-- Bind only unambiguous legacy approvals. Multiple versions need fresh review.
UPDATE sample_clean_reviews AS c SET json=json_set(c.json, '$.source_key', json((
    SELECT json_object('producer',json_extract(s.json,'$.source.producer'),
                       'record_type',json_extract(s.json,'$.source.record_type'),
                       'source_record_id',json_extract(s.json,'$.source.source_record_id'),
                       'evidence_version',json_extract(s.json,'$.source.evidence_version'))
    FROM samples s WHERE s.match_id=c.match_id AND s.sample_id=c.sample_id
))) WHERE (SELECT COUNT(*) FROM source_records r WHERE r.match_id=c.match_id
    AND r.entity_type='samples' AND r.entity_key=json_array(c.sample_id))=1;
