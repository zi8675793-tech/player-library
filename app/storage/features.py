"""Durable float32 feature rows; file parsing stays at the input boundary.

Rows contain little-endian IEEE float32 in C order. A missing row means that a
file reference has not yet been hydrated; it is never replaced by zeros.
"""

import json
import math
import sqlite3
import struct
from math import prod

from app.contracts.evidence import FeatureFile, FeatureSpace, InlineFeature, Sample


class FeatureConflictError(ValueError):
    """The same file row has already been saved with different values."""


def _encode(values: list[float], size: int, name: str) -> bytes:
    if len(values) != size:
        raise ValueError(f"{name} length does not match the feature space shape")
    try:
        packed = struct.pack(f"<{size}f", *values)
    except (OverflowError, struct.error) as exc:
        raise ValueError(f"{name} cannot be represented as float32") from exc
    if not all(math.isfinite(value) for value in struct.unpack(f"<{size}f", packed)):
        raise ValueError(f"{name} cannot be represented as finite float32")
    return packed


def _feature_payload(feature_space: FeatureSpace, value: InlineFeature) -> tuple:
    shape = json.dumps(feature_space.shape, separators=(",", ":"))
    values_blob = _encode(value.values, prod(feature_space.shape), "values")
    if feature_space.visibility is None:
        if value.visibility is not None:
            raise ValueError("visibility supplied for a space without visibility")
        visibility_shape, visibility_blob = None, None
    else:
        if value.visibility is None:
            raise ValueError("visibility is required by the feature space")
        visibility_shape = json.dumps(feature_space.visibility.shape, separators=(",", ":"))
        visibility_blob = _encode(
            value.visibility, prod(feature_space.visibility.shape), "visibility"
        )
    return (
        feature_space.feature_space_id,
        feature_space.dtype,
        shape,
        "little",
        values_blob,
        visibility_shape,
        visibility_blob,
    )


def save_feature_value(
    connection: sqlite3.Connection,
    match_id: str,
    feature_file: FeatureFile,
    feature_space: FeatureSpace,
    row_index: int,
    value: InlineFeature,
) -> bool:
    """Save a safely parsed file row without committing; return whether it is new.

    The file and space must already be stored in the same match. Their identities
    are protected by database foreign keys. Reusing a file row with different
    bytes or metadata is a conflict, never an overwrite.
    """
    if not connection.in_transaction:
        raise RuntimeError("feature writes require an active transaction")
    if type(row_index) is not int or not 0 <= row_index < feature_file.sample_count:
        raise ValueError("row_index is outside the feature file")
    if feature_file.feature_space_id != feature_space.feature_space_id or (
        value.feature_space_id != feature_space.feature_space_id
    ):
        raise ValueError("feature file, value and contract must use the same feature space")
    stored_file = connection.execute(
        "SELECT json FROM feature_files WHERE match_id=? AND feature_file_id=?",
        (match_id, feature_file.feature_file_id),
    ).fetchone()
    stored_space = connection.execute(
        "SELECT json FROM feature_spaces WHERE match_id=? AND feature_space_id=?",
        (match_id, feature_space.feature_space_id),
    ).fetchone()
    if stored_file is None or stored_space is None:
        raise ValueError("feature file and feature space must already exist in this match")
    if FeatureFile.model_validate_json(stored_file[0]) != feature_file:
        raise ValueError("feature file does not match its stored contract")
    if FeatureSpace.model_validate_json(stored_space[0]) != feature_space:
        raise ValueError("feature space does not match its stored contract")
    payload = _feature_payload(feature_space, value)
    existing = connection.execute(
        """SELECT feature_space_id, dtype, shape_json, byte_order, values_blob,
                  visibility_shape_json, visibility_blob
           FROM feature_values WHERE match_id=? AND feature_file_id=? AND row_index=?""",
        (match_id, feature_file.feature_file_id, row_index),
    ).fetchone()
    if existing is not None:
        if tuple(existing) != payload:
            raise FeatureConflictError("feature file row already contains different values")
        return False
    connection.execute(
        """INSERT INTO feature_values
           (match_id, feature_file_id, row_index, feature_space_id, dtype, shape_json,
            byte_order, values_blob, visibility_shape_json, visibility_blob)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (match_id, feature_file.feature_file_id, row_index, *payload),
    )
    return True


def load_feature_values(
    connection: sqlite3.Connection, match_id: str
) -> dict[tuple[str, int], InlineFeature]:
    """Read available rows for direct use as identity's ``feature_values`` mapping."""
    values = {}
    rows = connection.execute(
        """SELECT feature_file_id, row_index, feature_space_id, dtype, shape_json,
                  byte_order, values_blob, visibility_shape_json, visibility_blob
           FROM feature_values WHERE match_id=? ORDER BY feature_file_id, row_index""",
        (match_id,),
    )
    for file_id, row, space_id, dtype, shape_json, byte_order, data, vis_shape, vis_data in rows:
        if dtype != "float32" or byte_order != "little":
            raise ValueError("unsupported stored feature representation")
        shape = json.loads(shape_json)
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(x) is not int or x <= 0 for x in shape)
        ):
            raise ValueError("invalid stored feature shape")
        count = prod(shape)
        if len(data) != count * 4:
            raise ValueError("stored feature byte count does not match shape")
        flattened = list(struct.unpack(f"<{count}f", data))
        visibility = None
        if vis_shape is not None or vis_data is not None:
            if vis_shape is None or vis_data is None:
                raise ValueError("incomplete stored visibility")
            vshape = json.loads(vis_shape)
            if (
                not isinstance(vshape, list)
                or not vshape
                or any(type(x) is not int or x <= 0 for x in vshape)
            ):
                raise ValueError("invalid stored visibility shape")
            vcount = prod(vshape)
            if len(vis_data) != vcount * 4:
                raise ValueError("stored visibility byte count does not match shape")
            visibility = list(struct.unpack(f"<{vcount}f", vis_data))
        values[(file_id, row)] = InlineFeature(
            feature_space_id=space_id, values=flattened, visibility=visibility
        )
    return values


def save_inline_feature(
    connection: sqlite3.Connection,
    match_id: str,
    sample_id: str,
    feature_space: FeatureSpace,
    value: InlineFeature,
) -> bool:
    """Persist a sample's inline feature with the same representation as file rows."""
    if not connection.in_transaction:
        raise RuntimeError("feature writes require an active transaction")
    if feature_space.feature_space_id != value.feature_space_id:
        raise ValueError("feature and space IDs differ")
    stored_space = connection.execute(
        "SELECT json FROM feature_spaces WHERE match_id=? AND feature_space_id=?",
        (match_id, feature_space.feature_space_id),
    ).fetchone()
    if stored_space is None or FeatureSpace.model_validate_json(stored_space[0]) != feature_space:
        raise ValueError("feature space does not match its stored contract")
    stored_sample = connection.execute(
        "SELECT json FROM samples WHERE match_id=? AND sample_id=?", (match_id, sample_id)
    ).fetchone()
    if stored_sample is None:
        raise ValueError("sample must already exist in this match")
    sample = Sample.model_validate_json(stored_sample[0])
    if not any(isinstance(item, InlineFeature) and item == value for item in sample.features):
        raise ValueError("inline feature does not match its stored sample")
    payload = _feature_payload(feature_space, value)
    existing = connection.execute(
        """SELECT feature_space_id, dtype, shape_json, byte_order, values_blob,
                  visibility_shape_json, visibility_blob FROM sample_feature_values
           WHERE match_id=? AND sample_id=? AND feature_space_id=?""",
        (match_id, sample_id, feature_space.feature_space_id),
    ).fetchone()
    if existing is not None:
        if tuple(existing) != payload:
            raise FeatureConflictError("sample feature already contains different values")
        return False
    connection.execute(
        """INSERT INTO sample_feature_values
           (match_id, sample_id, feature_space_id, dtype, shape_json, byte_order,
            values_blob, visibility_shape_json, visibility_blob)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (match_id, sample_id, *payload),
    )
    return True


def load_inline_features(
    connection: sqlite3.Connection, match_id: str
) -> dict[tuple[str, str], InlineFeature]:
    """Return sample features reconstructed from stored binary values."""
    rows = connection.execute(
        """SELECT sample_id, feature_space_id, dtype, shape_json, byte_order,
                  values_blob, visibility_shape_json, visibility_blob
           FROM sample_feature_values WHERE match_id=? ORDER BY sample_id, feature_space_id""",
        (match_id,),
    )
    result = {}
    for sample_id, space_id, dtype, shape, byte_order, data, vis_shape, vis_data in rows:
        if dtype != "float32" or byte_order != "little":
            raise ValueError("unsupported stored feature representation")

        def unpack(raw_shape, raw_data):
            sizes = json.loads(raw_shape)
            if (
                not isinstance(sizes, list)
                or not sizes
                or any(type(x) is not int or x <= 0 for x in sizes)
            ):
                raise ValueError("invalid stored feature shape")
            count = prod(sizes)
            if len(raw_data) != 4 * count:
                raise ValueError("stored feature byte count does not match shape")
            return list(struct.unpack(f"<{count}f", raw_data))

        values = unpack(shape, data)
        if (vis_shape is None) != (vis_data is None):
            raise ValueError("incomplete stored visibility")
        visibility = unpack(vis_shape, vis_data) if vis_shape is not None else None
        result[(sample_id, space_id)] = InlineFeature(
            feature_space_id=space_id, values=values, visibility=visibility
        )
    return result
