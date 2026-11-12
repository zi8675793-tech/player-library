"""Order-invariant binding of machine proposals to their actual evidence/features."""

import hashlib
import json
import struct

from app.contracts.common import SourceKey
from app.identity.prepare import resolve_feature


def sample_binding(sample, assets):
    """Bind support to raw source versions, declared features and image contents."""
    features = [feature.model_dump(mode="json") for feature in sample.features]
    for feature in features:
        if feature["kind"] == "inline":
            for field in ("values", "visibility"):
                if feature[field] is not None:
                    values = feature[field]
                    feature[field] = list(
                        struct.unpack(f"<{len(values)}f", struct.pack(f"<{len(values)}f", *values))
                    )
    return {
        "source_key": sample.source.model_dump(mode="json", include=set(SourceKey.model_fields)),
        "clip_id": sample.clip_id,
        "track_id": sample.track_id,
        "frame_idx": sample.frame_idx,
        "time_s": sample.time_s,
        "image_asset_id": sample.image_asset_id,
        "image_sha256": next(
            (asset.file.sha256 for asset in assets if asset.asset_id == sample.image_asset_id), None
        ),
        "features": features,
    }


def evidence_fingerprint(batch, feature_values=None):
    data = batch.model_dump(mode="json", exclude={"batch_id"})
    for key, value in data.items():
        if isinstance(value, list):
            data[key] = sorted(value, key=lambda v: json.dumps(v, sort_keys=True))
    if data.get("identity_evidence"):
        for key, value in data["identity_evidence"].items():
            if isinstance(value, list):
                data["identity_evidence"][key] = sorted(
                    value, key=lambda v: json.dumps(v, sort_keys=True)
                )
    evidence = batch.identity_evidence
    resolved = {}
    if evidence:
        for sample in evidence.samples:
            for space in evidence.feature_spaces:
                feature = resolve_feature(sample, space, feature_values)
                if feature is not None:
                    resolved[sample.sample_id, space.feature_space_id] = feature.model_dump(
                        mode="json"
                    )
    payload = {
        "batch": data,
        "resolved_features": [[list(key), value] for key, value in sorted(resolved.items())],
    }
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()
