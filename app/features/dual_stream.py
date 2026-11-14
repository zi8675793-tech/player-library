"""Track-scoped image loading and frozen T1 dual-stream feature packaging.

Paths are relative to the supplied track root (e.g. ``../test/track``) and must
have the form ``<video_id>/samples/<name>.png``. Absolute paths are allowed only
inside that same directory. No directories are scanned and no labels are read.

Example::

    extractor = DualStreamExtractor(
        "../test/track",
        jersey_extractor=JerseyNumberExtractor(allow_download=True),
        appearance_extractor=AppearanceExtractor(allow_download=True),
    )
    features = extractor.extract_observations(observations)

The documented T1 dataclasses are not yet in app/contracts. Their exact frozen
definitions live in these permitted files and are all importable here. The
legacy evidence.TrackObservation is a different contract and is not adapted
implicitly. Move these definitions to app/contracts in a separately scoped task.
"""

import math
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.features.appearance import AppearanceEmbedding, AppearanceExtractor
from app.features.jersey_number import JerseyNumberExtractor, JerseyNumberPrediction

__all__ = [
    "AppearanceEmbedding",
    "AppearanceExtractor",
    "DualStreamExtractor",
    "DualStreamFeature",
    "JerseyNumberExtractor",
    "JerseyNumberPrediction",
    "TrackObservation",
    "TrackletSegment",
]


@dataclass(frozen=True)
class TrackObservation:
    video_id: str
    local_track_id: int
    frame_idx: int
    timestamp_s: float
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    crop_path: str | None = None


@dataclass(frozen=True)
class TrackletSegment:
    segment_id: str
    video_id: str
    local_track_id: int
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    observation_count: int
    observations: tuple[TrackObservation, ...]


@dataclass(frozen=True)
class DualStreamFeature:
    segment_id: str
    jersey: JerseyNumberPrediction | None
    appearance: AppearanceEmbedding


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _validate_observation(observation: TrackObservation) -> None:
    if not isinstance(observation, TrackObservation):
        raise TypeError("Expected the documented T1 TrackObservation")
    if (
        not isinstance(observation.video_id, str)
        or re.fullmatch(r"[A-Za-z0-9_-]+", observation.video_id) is None
    ):
        raise ValueError("video_id must be a single safe path component")
    for name in ("local_track_id", "frame_idx"):
        value = getattr(observation, name)
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if not _finite_number(observation.timestamp_s) or observation.timestamp_s < 0:
        raise ValueError("timestamp_s must be finite and nonnegative")
    if not _finite_number(observation.confidence) or not 0 <= observation.confidence <= 1:
        raise ValueError("confidence must be finite and in [0, 1]")
    box = observation.bbox_xyxy
    if (
        not isinstance(box, tuple)
        or len(box) != 4
        or not all(_finite_number(value) for value in box)
    ):
        raise ValueError("bbox_xyxy must contain four finite coordinates")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 and 0 <= y1 < y2):
        raise ValueError("bbox_xyxy must be an ordered nonnegative box")
    if observation.crop_path is not None and (
        not isinstance(observation.crop_path, str) or not observation.crop_path.strip()
    ):
        raise ValueError("crop_path must be a nonempty string or None")


class DualStreamExtractor:
    """Extract each track independently; fail closed on invalid or missing evidence.

    Batch grouping never merges the same local ID from different videos. Input
    observations contain real frame metadata supplied by the upstream adapter;
    sample filename suffixes are not interpreted as frame numbers. Empty/missing
    crop references can be skipped, but an explicitly referenced bad file fails.
    Metadata and path validation precede model work for the whole batch.
    """

    def __init__(
        self,
        crop_root: str | Path,
        *,
        jersey_extractor: JerseyNumberExtractor | None = None,
        appearance_extractor: AppearanceExtractor | None = None,
    ) -> None:
        self.crop_root = Path(crop_root).resolve()
        self.jersey_extractor = (
            jersey_extractor if jersey_extractor is not None else JerseyNumberExtractor()
        )
        self.appearance_extractor = (
            appearance_extractor if appearance_extractor is not None else AppearanceExtractor()
        )

    def _crop_path(self, observation: TrackObservation) -> Path | None:
        if observation.crop_path is None:
            return None
        supplied = Path(observation.crop_path)
        candidate = supplied if supplied.is_absolute() else self.crop_root / supplied
        expected = self.crop_root / observation.video_id / "samples"
        # Check syntax before resolving: disallow traversal and nested directories.
        if candidate.parent != expected or candidate.suffix.lower() != ".png":
            raise ValueError("crop_path must be <video_id>/samples/<name>.png inside crop_root")
        resolved = candidate.resolve()
        if resolved.parent != expected or resolved != candidate:
            raise ValueError("crop_path must not escape crop_root or use symlinks")
        if not resolved.is_file():
            raise ValueError(f"Referenced crop does not exist: {candidate}")
        return resolved

    def _prepare(
        self, observations: Iterable[TrackObservation]
    ) -> tuple[list[TrackObservation], dict[tuple[str, int, int], Path | None]]:
        ordered = list(observations)
        paths: dict[tuple[str, int, int], Path | None] = {}
        seen_files: set[Path] = set()
        for observation in ordered:
            _validate_observation(observation)
            key = observation.video_id, observation.local_track_id, observation.frame_idx
            if key in paths:
                raise ValueError("Duplicate observation frame within a track")
            path = self._crop_path(observation)
            if path is not None:
                if path in seen_files:
                    raise ValueError(
                        "Duplicate crop file cannot provide independent frame evidence"
                    )
                seen_files.add(path)
            paths[key] = path
        ordered.sort(key=lambda item: (item.video_id, item.local_track_id, item.frame_idx))
        previous: dict[tuple[str, int], float] = {}
        for observation in ordered:
            track = observation.video_id, observation.local_track_id
            if track in previous and observation.timestamp_s <= previous[track]:
                raise ValueError("Track timestamps must strictly increase with frame indices")
            previous[track] = observation.timestamp_s
        return ordered, paths

    @staticmethod
    def _segment(observations: list[TrackObservation]) -> TrackletSegment:
        first, last = observations[0], observations[-1]
        return TrackletSegment(
            segment_id=f"{first.video_id}_{first.local_track_id}_{first.frame_idx}",
            video_id=first.video_id,
            local_track_id=first.local_track_id,
            start_frame=first.frame_idx,
            end_frame=last.frame_idx,
            start_time_s=first.timestamp_s,
            end_time_s=last.timestamp_s,
            observation_count=len(observations),
            observations=tuple(observations),
        )

    def _extract(
        self, segment: TrackletSegment, paths: dict[tuple[str, int, int], Path | None]
    ) -> DualStreamFeature:
        images = []
        for observation in sorted(segment.observations, key=lambda item: item.frame_idx):
            path = paths[observation.video_id, observation.local_track_id, observation.frame_idx]
            if path is None:
                continue
            try:
                with Image.open(path) as image:
                    if image.format != "PNG":
                        raise ValueError(f"Crop is not a PNG image: {path}")
                    if "A" in image.getbands() or "transparency" in image.info:
                        # Masked-out pixels must not retain hidden RGB/background evidence.
                        rgba = image.convert("RGBA")
                        background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
                        image = Image.alpha_composite(background, rgba)
                    images.append(np.asarray(image.convert("RGB")).copy())
            except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
                raise ValueError(f"Cannot read crop PNG: {path}") from exc
        if not images:
            raise ValueError(f"No crop images for segment {segment.segment_id}")
        # Appearance raises if every frame fails quality. Never emit a zero feature.
        appearance = self.appearance_extractor.extract_track(images)
        jersey = self.jersey_extractor.extract_track(images)
        return DualStreamFeature(segment.segment_id, jersey, appearance)

    def extract(self, segment: TrackletSegment) -> DualStreamFeature:
        if not isinstance(segment, TrackletSegment):
            raise TypeError("Expected the documented T1 TrackletSegment")
        observations, paths = self._prepare(segment.observations)
        if not observations:
            raise ValueError("TrackletSegment must contain observations")
        expected = self._segment(observations)
        for name in (
            "segment_id",
            "video_id",
            "local_track_id",
            "start_frame",
            "end_frame",
            "start_time_s",
            "end_time_s",
            "observation_count",
        ):
            if getattr(segment, name) != getattr(expected, name):
                raise ValueError(f"TrackletSegment {name} disagrees with its observations")
        if any(
            (item.video_id, item.local_track_id) != (expected.video_id, expected.local_track_id)
            for item in observations
        ):
            raise ValueError("TrackletSegment must contain exactly one track")
        return self._extract(segment, paths)

    def extract_observations(
        self, observations: Iterable[TrackObservation]
    ) -> list[DualStreamFeature]:
        ordered, paths = self._prepare(observations)
        groups: dict[tuple[str, int], list[TrackObservation]] = defaultdict(list)
        for observation in ordered:
            groups[observation.video_id, observation.local_track_id].append(observation)
        return [self._extract(self._segment(group), paths) for group in groups.values()]
