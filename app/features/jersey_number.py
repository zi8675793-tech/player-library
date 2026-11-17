"""Image-only jersey OCR and conservative confidence-weighted track voting.

EasyOCR is optional and loaded on the first usable image, never on import.
Install real backends with ``uv pip install easyocr torch torchvision``.
Models must already be cached unless ``allow_download=True`` is explicit.
Tests may inject an EasyOCR-compatible reader without installing model packages.

The frozen prediction below reproduces docs/data-contracts.md exactly; that
new contract does not yet have a definition under app/contracts/.
"""

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class JerseyNumberPrediction:
    number: int | None
    confidence: float
    legible: bool
    votes: dict[int, float]
    raw_text: str | None = None


class OCRReader(Protocol):
    def readtext(self, image: np.ndarray, **kwargs: object) -> list: ...


def validate_rgb_image(image: np.ndarray) -> np.ndarray:
    """Accept RGB pixel values only; never silently reinterpret BGR or floats 0..1."""
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) == 0:
        raise ValueError("Expected a nonempty H x W x 3 RGB image")
    if array.dtype.kind not in "uif" or not np.isfinite(array).all():
        raise ValueError("RGB image must contain finite numeric pixels")
    if array.min() < 0 or array.max() > 255:
        raise ValueError("RGB pixels must be in [0, 255]")
    if array.dtype.kind == "f" and array.max() <= 1:
        raise ValueError("Use RGB pixels in [0, 255], not a normalized tensor")
    return np.ascontiguousarray(array, dtype=np.uint8)


def _torso(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    # Exclude the head, shorts and crop edges, which often contain other people.
    return image[int(height * 0.16) : int(height * 0.62), int(width * 0.15) : int(width * 0.85)]


def assess_image_quality(image: np.ndarray) -> float:
    """A torso sharpness/exposure gate, not a learned pose/occlusion classifier.

    A zero score rejects severe blur, under/overexposure and undersized crops.
    Passing this gate does not establish that a jersey number is visible.
    """
    rgb = validate_rgb_image(image)
    if rgb.shape[0] < 24 or rgb.shape[1] < 16:
        return 0.0
    roi = _torso(rgb).astype(np.float32)
    gray = roi @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    if min(gray.shape) < 8:
        return 0.0
    brightness = float(gray.mean())
    clipped = float(np.mean((gray < 15) | (gray > 245)))
    if not 35 <= brightness <= 235 or clipped > 0.65:
        return 0.0
    laplacian = (
        gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1]
    )
    sharpness = float(laplacian.var())
    if sharpness < 30:
        return 0.0
    exposure = min(1.0, brightness / 85, (255 - brightness) / 50) * (1 - clipped)
    return float(np.clip(sharpness / (sharpness + 100) * exposure, 0, 1))


class JerseyNumberExtractor:
    """Recognize only complete 1–2 digit tokens inside a person crop's torso.

    Each crop supplies at most one vote. Track confidence is the average detection
    confidence across supporting keyframes of the winning number (divided by its
    support frame count, avoiding artificial dilution across back-view or unnumbered
    frames). Temporal confirmation enforces multi-frame support (>= min_support_frames),
    high average confidence (>= min_track_confidence), and a safety margin over the
    runner candidate (confidence - runner >= min_vote_margin).
    Callers must supply distinct frames of one track in temporal order; the
    DualStreamExtractor enforces frame and file uniqueness at its input boundary.
    """

    def __init__(
        self,
        reader: OCRReader | None = None,
        *,
        min_ocr_confidence: float = 0.6,
        min_track_confidence: float = 0.85,
        min_support_frames: int = 2,
        min_vote_margin: float = 0.15,
        allow_download: bool = False,
    ) -> None:
        for name, value in (
            ("min_ocr_confidence", min_ocr_confidence),
            ("min_track_confidence", min_track_confidence),
            ("min_vote_margin", min_vote_margin),
        ):
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be finite and in (0, 1]")
        if type(min_support_frames) is not int or min_support_frames < 2:
            raise ValueError("min_support_frames must be an integer >= 2")
        self.reader = reader
        self.min_ocr_confidence = min_ocr_confidence
        self.min_track_confidence = min_track_confidence
        self.min_support_frames = min_support_frames
        self.min_vote_margin = min_vote_margin
        self.allow_download = allow_download

    def _get_reader(self) -> OCRReader:
        if self.reader is None:
            try:
                import easyocr
            except ImportError as exc:
                raise RuntimeError(
                    "EasyOCR is required: uv pip install easyocr torch torchvision; "
                    "or inject an OCR reader"
                ) from exc
            try:
                self.reader = easyocr.Reader(
                    ["en"], gpu=False, download_enabled=self.allow_download, verbose=False
                )
            except (OSError, RuntimeError) as exc:
                raise RuntimeError(
                    "Cannot initialize EasyOCR: cache its weights first, or explicitly "
                    "set allow_download=True on the first run"
                ) from exc
        return self.reader

    def extract(self, image: np.ndarray) -> JerseyNumberPrediction:
        """Return a frame prediction; temporal confirmation happens in extract_track."""
        rgb = validate_rgb_image(image)
        if assess_image_quality(rgb) == 0:
            return JerseyNumberPrediction(None, 0.0, False, {})
        roi = _torso(rgb)
        scale = max(1, min(4, math.ceil(160 / roi.shape[0])))
        if scale > 1:
            roi = np.asarray(
                Image.fromarray(roi).resize(
                    (roi.shape[1] * scale, roi.shape[0] * scale), Image.Resampling.BICUBIC
                )
            )
        results = self._get_reader().readtext(
            roi, detail=1, allowlist="0123456789", paragraph=False
        )
        candidates: dict[int, tuple[float, str]] = {}
        for result in results:
            if not isinstance(result, (tuple, list)) or len(result) != 3:
                continue
            box, text, score = result
            if not isinstance(text, str) or re.fullmatch(r"[0-9]{1,2}", text.strip()) is None:
                continue
            try:
                confidence = float(score)
                polygon = np.asarray(box, dtype=np.float64)
            except (TypeError, ValueError):
                continue
            if (
                not math.isfinite(confidence)
                or not self.min_ocr_confidence <= confidence <= 1
                or polygon.shape != (4, 2)
                or not np.isfinite(polygon).all()
            ):
                continue
            x1, y1 = polygon.min(axis=0)
            x2, y2 = polygon.max(axis=0)
            if not (0 <= x1 < x2 <= roi.shape[1] and 0 <= y1 < y2 <= roi.shape[0]):
                continue
            number = int(text.strip())
            if number not in candidates or confidence > candidates[number][0]:
                candidates[number] = confidence, text
        if not candidates:
            return JerseyNumberPrediction(None, 0.0, False, {})
        ranked = sorted(candidates, key=lambda number: (-candidates[number][0], number))
        best = ranked[0]
        confidence, text = candidates[best]
        if len(ranked) > 1 and confidence - candidates[ranked[1]][0] < self.min_vote_margin:
            return JerseyNumberPrediction(None, 0.0, True, {}, text)
        return JerseyNumberPrediction(best, confidence, True, {best: confidence}, text)

    def extract_track(self, images: Iterable[np.ndarray]) -> JerseyNumberPrediction:
        votes: dict[int, float] = {}
        supports: dict[int, int] = {}
        raw: dict[int, tuple[float, str | None]] = {}
        legible = False
        for image in images:
            rgb = validate_rgb_image(image)
            if assess_image_quality(rgb) == 0:
                continue
            prediction = self.extract(rgb)
            legible |= prediction.legible
            if prediction.number is None:
                continue
            number = prediction.number
            votes[number] = votes.get(number, 0.0) + prediction.confidence
            supports[number] = supports.get(number, 0) + 1
            if number not in raw or prediction.confidence > raw[number][0]:
                raw[number] = prediction.confidence, prediction.raw_text
        if not votes:
            return JerseyNumberPrediction(None, 0.0, legible, {})
        ranked = sorted(votes, key=lambda number: (-votes[number], number))
        best = ranked[0]
        confidence = votes[best] / supports[best]
        competing = [num for num in ranked[1:] if supports[num] >= self.min_support_frames]
        runner = votes[competing[0]] / supports[competing[0]] if competing else 0.0
        confirmed = (
            supports[best] >= self.min_support_frames
            and confidence >= self.min_track_confidence
            and confidence - runner >= self.min_vote_margin
        )
        return JerseyNumberPrediction(
            best if confirmed else None, float(confidence), legible, votes, raw[best][1]
        )
