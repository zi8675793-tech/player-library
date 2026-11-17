"""Frozen OMZ v1 file contract; independent of the business evidence schema.

Frame/step indices are zero based. Boxes are original-image pixel xyxy; native
history boxes can be smoothed/interpolated. time_s = frame_idx / source fps;
there is no cross-source match clock. Keys identify native Track allocations,
not mutable/recycled native IDs. See docs/openvino-running.md for file semantics.
"""

import hashlib
import json
from pathlib import Path
from typing import Literal, TypedDict

TRACKING_SCHEMA = "omz-tracking-v1"


class TrackingFrame(TypedDict):
    source: int
    frame_idx: int
    step: int
    time_s: float
    decoded_sha256: str
    boxes: list[list[int]]
    scores: list[float]


class NativeAppearance(TypedDict):
    mean: list[float] | None
    count: int
    clusters: list[list[float]]
    cluster_counts: list[int]


class NativeTrackInstance(TypedDict):
    instance: str
    source: int
    native_id: int
    native_retained: bool
    timestamps: list[int]
    boxes: list[list[int]]
    origins: list[Literal["detection", "interpolation", "prediction", "unknown"]]
    appearance: NativeAppearance


def verify_tracking_bundle(run_dir: Path) -> dict:
    """Read-only freeze gate, to be called before downstream scoring/import."""
    hashes = json.loads((run_dir / "artifact-hashes.json").read_text())
    for name, digest in hashes.items():
        path = (run_dir / name).resolve()
        if not path.is_relative_to(run_dir.resolve()):
            raise ValueError("Artifact path escapes run directory")
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        if h.hexdigest() != digest:
            raise ValueError(f"Frozen artifact changed: {name}")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest["schema_version"] != TRACKING_SCHEMA:
        raise ValueError("Unsupported tracking contract")
    return manifest
