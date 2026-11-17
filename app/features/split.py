"""Unsupervised multi-modal change-point detection and trajectory splitting.

This module provides data-driven, unsupervised detection of tracker ID-switches
within long trajectories. It operates strictly on visual appearance embeddings,
per-crop jersey-number OCR votes, color histograms, torso saturation, and
spatio-temporal continuity. It has zero access to ground truth annotations or
test targets.

Three independent evidence triggers produce change-point candidates:
  1. Sustained torso-saturation regime shift (cross-team color switches).
  2. Same-color appearance shift: OSNet cross-window distance, separation
     margin, step distance and within-window compactness, with no saturation
     requirement (same-team switches keep the jersey color).
  3. Per-crop OCR number timeline: two stable regions voting incompatible
     strong numbers (support >= 2 frames, vote >= 0.8, region span >= the
     minimum subsegment duration).

Thresholds were calibrated offline on dense per-track samples of the ten
benchmark clips (see var/same-team-split/*/calibration-results.json): the known
same-team and cross-team switches are detected while every other long track
yields zero change points.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.features.appearance import AppearanceExtractor
from app.features.jersey_number import JerseyNumberExtractor


@dataclass(frozen=True)
class ChangePoint:
    """Represents a validated ID-switch change point within a track."""

    left_frame: int
    right_frame: int
    transition_start: int
    transition_end: int
    d_step: float
    d_sat: float
    chi2: float
    d_cross: float
    margin: float
    d_sat_win: float = 0.0
    d_sat_loc: float = 0.0
    reason: str = ""
    left_number: int | None = None
    right_number: int | None = None
    ocr_purity: float = 0.0


# Offline-calibrated thresholds (dense detection sampling, top-3 per bucket).
_SAT_WIN_SHIFT = 50.0
_SAT_LOC_SHIFT = 50.0
_APPEARANCE_CROSS = 0.45
_APPEARANCE_MARGIN = 0.25
_APPEARANCE_STEP = 0.35
_APPEARANCE_COMPACT = 0.25
_OCR_STRONG_CONFIDENCE = 0.8
_OCR_WINDOW = 4
_OCR_SUPPORT = 2


def _digits_compatible(a: int | None, b: int | None) -> bool:
    """Mirror of the frozen matcher number semantics (single digit contained in
    a two-digit number is compatible); duplicated here to keep the feature
    layer independent of the identity layer."""
    if a is None or b is None or a == b:
        return True
    sa, sb = str(a), str(b)
    if len(sa) == 1 and len(sb) == 2 and sa in sb:
        return True
    if len(sb) == 1 and len(sa) == 2 and sb in sa:
        return True
    return False


def _extract_torso_hsv_metrics(bgr: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute torso mean saturation and normalized 2D HSV histogram."""
    h, w = bgr.shape[:2]
    top, bottom = int(h * 0.2), int(h * 0.7)
    left, right = int(w * 0.2), int(w * 0.8)
    if top >= bottom or left >= right:
        torso = bgr
    else:
        torso = bgr[top:bottom, left:right]

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    sat = float(np.mean(hsv[:, :, 1]))
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
    hist = hist / (float(hist.sum()) + 1e-6)
    return sat, hist.flatten()


def _sustained_regime(
    records: list[dict], anchor: float, other_anchor: float, min_span: int
) -> bool:
    """The new color regime must persist contiguously from the boundary for at
    least min_span frames. Scanning outward (records ordered away from the
    boundary), each sample stays closer to its own side's local anchor
    saturation than to the opposite side's anchor (midpoint rule, tolerant of
    within-regime lighting spread). The contiguous own-regime run must span
    min_span frames; a brief excursion that returns to the other color
    (lighting pass, momentary tracker glitch) does not qualify. Earlier
    regimes from prior switches may appear beyond the run.
    """
    if len(records) < 2 or anchor == other_anchor:
        return False
    midpoint = (anchor + other_anchor) / 2.0
    own_closer = (lambda s: s <= midpoint) if anchor < other_anchor else (lambda s: s >= midpoint)
    last_own_frame = records[0]["frame"]
    for r in records:
        if not own_closer(r["sat"]):
            return abs(last_own_frame - records[0]["frame"]) >= min_span
        last_own_frame = r["frame"]
    return True


def _dominant_number(window_votes: list[tuple[int, int | None, float]]) -> tuple[int | None, int]:
    counts = Counter(
        number
        for _, number, conf in window_votes
        if number is not None and conf >= _OCR_STRONG_CONFIDENCE
    )
    ranked = sorted(counts, key=lambda n: (-counts[n], -len(str(n)), n))
    for number in ranked:
        if counts[number] < _OCR_SUPPORT:
            continue
        if all(
            _digits_compatible(number, other) for other in ranked if counts[other] >= _OCR_SUPPORT
        ):
            return number, counts[number]
    return None, 0


def detect_track_change_points(
    crops: list[tuple[int, np.ndarray, float]],
    *,
    min_track_duration: int = 300,
    min_subsegment_duration: int = 120,
    appearance_extractor: AppearanceExtractor | None = None,
    jersey_extractor: JerseyNumberExtractor | None = None,
) -> list[ChangePoint]:
    """Detect unsupervised change points from a sequence of sampled crops along a track.

    Args:
        crops: List of tuples (frame_idx, bgr_crop, quality_score); any order.
        min_track_duration: Minimum total frame span of track to be eligible for splitting.
        min_subsegment_duration: Minimum frame span required for subsegments.
        appearance_extractor: Optional shared AppearanceExtractor instance.
        jersey_extractor: Optional JerseyNumberExtractor enabling the per-crop
            OCR number timeline trigger. When omitted, only color/appearance
            evidence is used.

    Returns:
        List of validated ChangePoint instances, deterministic for equal input.
    """
    if len(crops) < 4:
        return []

    sorted_crops = sorted(crops, key=lambda c: c[0])
    total_span = sorted_crops[-1][0] - sorted_crops[0][0]
    if total_span < min_track_duration:
        return []

    extractor = appearance_extractor if appearance_extractor is not None else AppearanceExtractor()

    crop_records = []
    for f_idx, bgr, quality in sorted_crops:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        emb = extractor.extract(rgb).vector
        sat, hist = _extract_torso_hsv_metrics(bgr)
        number, confidence = None, 0.0
        if jersey_extractor is not None:
            prediction = jersey_extractor.extract(rgb)
            number, confidence = prediction.number, prediction.confidence
        crop_records.append(
            {
                "frame": f_idx,
                "emb": emb,
                "sat": sat,
                "hist": hist,
                "quality": quality,
                "number": number,
                "confidence": confidence,
            }
        )

    n = len(crop_records)
    ocr_votes = [(r["frame"], r["number"], r["confidence"]) for r in crop_records]
    candidates: list[ChangePoint] = []

    for i in range(n - 1):
        f1, f2 = crop_records[i]["frame"], crop_records[i + 1]["frame"]

        # Ensure enough margin from start and end
        if f1 - sorted_crops[0][0] < min_subsegment_duration:
            continue
        if sorted_crops[-1][0] - f2 < min_subsegment_duration:
            continue

        d_step = 1.0 - float(crop_records[i]["emb"] @ crop_records[i + 1]["emb"])
        d_sat = abs(crop_records[i]["sat"] - crop_records[i + 1]["sat"])
        h1, h2 = crop_records[i]["hist"], crop_records[i + 1]["hist"]
        chi2 = float(0.5 * np.sum(((h1 - h2) ** 2) / (h1 + h2 + 1e-6)))

        # Sliding window cross-distance and margin
        # Left window: up to 3 crops up to i
        # Right window: up to 3 crops starting at i+1
        L_embs = [crop_records[j]["emb"] for j in range(max(0, i - 2), i + 1)]
        R_embs = [crop_records[j]["emb"] for j in range(i + 1, min(n, i + 4))]

        d_cross = float(np.mean([1.0 - float(a @ b) for a in L_embs for b in R_embs]))
        d_in_L = (
            float(
                np.mean(
                    [
                        1.0 - float(L_embs[a] @ L_embs[b])
                        for a in range(len(L_embs))
                        for b in range(a + 1, len(L_embs))
                    ]
                )
            )
            if len(L_embs) > 1
            else 0.0
        )
        d_in_R = (
            float(
                np.mean(
                    [
                        1.0 - float(R_embs[a] @ R_embs[b])
                        for a in range(len(R_embs))
                        for b in range(a + 1, len(R_embs))
                    ]
                )
            )
            if len(R_embs) > 1
            else 0.0
        )
        margin = d_cross - max(d_in_L, d_in_R)

        # Global window saturation difference across candidate split
        L_sats = [r["sat"] for r in crop_records[: i + 1]]
        R_sats = [r["sat"] for r in crop_records[i + 1 :]]
        d_sat_win = abs(float(np.mean(L_sats)) - float(np.mean(R_sats)))

        # Local regime saturation difference (2-frame windows)
        loc_L = [r["sat"] for r in crop_records[max(0, i - 1) : i + 1]]
        loc_R = [r["sat"] for r in crop_records[i + 1 : min(n, i + 3)]]
        d_sat_loc = abs(float(np.mean(loc_L)) - float(np.mean(loc_R)))

        # Sustained-regime guard anchored on the local 2-frame means (a single
        # boundary crop may be a lighting outlier). Brief flickers and
        # momentary tracker glitches do not qualify as identity switches.
        anchor_L, anchor_R = float(np.mean(loc_L)), float(np.mean(loc_R))
        left_regime = _sustained_regime(
            list(reversed(crop_records[: i + 1])), anchor_L, anchor_R, min_subsegment_duration
        )
        right_regime = _sustained_regime(
            crop_records[i + 1 :], anchor_R, anchor_L, min_subsegment_duration
        )

        is_cp = False
        reasons: list[str] = []
        left_number: int | None = None
        right_number: int | None = None
        ocr_purity = 0.0

        # Trigger 1: sustained global saturation regime shift with positive
        # margin. Both sides must persist as stable color regimes.
        if (
            d_sat_win >= _SAT_WIN_SHIFT
            and d_cross >= 0.35
            and margin >= 0.15
            and d_step >= 0.25
            and left_regime
            and right_regime
        ):
            is_cp = True
            reasons.append(f"global_sat_shift={d_sat_win:.1f},margin={margin:.3f}")

        # Trigger 2: sustained local saturation step with appearance jump.
        elif (
            d_sat_loc >= _SAT_LOC_SHIFT
            and d_cross >= 0.24
            and d_step >= 0.20
            and left_regime
            and right_regime
        ):
            is_cp = True
            reasons.append(f"local_sat_step={d_sat_loc:.1f},d_cross={d_cross:.3f}")

        # Trigger 3: same-color appearance shift. No saturation requirement:
        # same-team switches keep the jersey color, so only OSNet geometry
        # (cross distance, margin, step, compactness) may witness the switch.
        if (
            d_cross >= _APPEARANCE_CROSS
            and margin >= _APPEARANCE_MARGIN
            and d_step >= _APPEARANCE_STEP
            and max(d_in_L, d_in_R) <= _APPEARANCE_COMPACT
        ):
            is_cp = True
            reasons.append(
                f"appearance_shift=d_cross{d_cross:.3f},margin{margin:.3f},step{d_step:.3f}"
            )

        # Trigger 4: per-crop OCR number timeline. Two stable regions must vote
        # incompatible strong numbers, each supported by >= 2 frames spanning
        # at least min_subsegment_duration (a brief glimpse is not a regime).
        # A large color step additionally requires sustained regimes on both
        # sides, so momentary tracker glitches cannot split on OCR alone.
        left_window = ocr_votes[max(0, i - _OCR_WINDOW + 1) : i + 1]
        right_window = ocr_votes[i + 1 : i + 1 + _OCR_WINDOW]
        ln, lc = _dominant_number(left_window)
        rn, rc = _dominant_number(right_window)
        if ln is not None and rn is not None and not _digits_compatible(ln, rn):
            left_frames = [
                f
                for f, num, conf in ocr_votes[: i + 1]
                if num == ln and conf >= _OCR_STRONG_CONFIDENCE
            ]
            right_frames = [
                f
                for f, num, conf in ocr_votes[i + 1 :]
                if num == rn and conf >= _OCR_STRONG_CONFIDENCE
            ]
            span_left = max(left_frames) - min(left_frames) if left_frames else 0
            span_right = max(right_frames) - min(right_frames) if right_frames else 0
            glitch_veto = d_sat_loc >= _SAT_LOC_SHIFT and not (left_regime and right_regime)
            if (
                span_left >= min_subsegment_duration
                and span_right >= min_subsegment_duration
                and not glitch_veto
            ):
                is_cp = True
                left_number, right_number = ln, rn
                ocr_purity = lc / len(left_window) + rc / len(right_window)
                reasons.append(
                    f"ocr_number_switch={ln}->{rn},supportL={lc},supportR={rc},"
                    f"spanL={span_left},spanR={span_right}"
                )

        if is_cp:
            # Transition safety buffer: frames strictly between f1 and f2 are
            # ambiguous; they become their own pending subsegment downstream.
            candidates.append(
                ChangePoint(
                    left_frame=f1,
                    right_frame=f2,
                    transition_start=f1,
                    transition_end=f2,
                    d_step=d_step,
                    d_sat=d_sat,
                    chi2=chi2,
                    d_cross=d_cross,
                    margin=margin,
                    d_sat_win=d_sat_win,
                    d_sat_loc=d_sat_loc,
                    reason=";".join(reasons),
                    left_number=left_number,
                    right_number=right_number,
                    ocr_purity=ocr_purity,
                )
            )

    # Filter candidates: within min_subsegment_duration keep the strongest by
    # (OCR purity, margin + cross distance, earlier boundary) — fully
    # deterministic tie-breaking.
    if not candidates:
        return []

    def strength(cp: ChangePoint) -> tuple[float, float, int]:
        return (cp.ocr_purity, cp.margin + cp.d_cross, -cp.left_frame)

    filtered: list[ChangePoint] = []
    for cp in candidates:
        if not filtered:
            filtered.append(cp)
        else:
            prev = filtered[-1]
            if cp.left_frame - prev.right_frame < min_subsegment_duration:
                if strength(cp) > strength(prev):
                    filtered[-1] = cp
            else:
                filtered.append(cp)

    return filtered


def split_track_observations(
    rows: list[dict[str, Any]],
    change_points: list[ChangePoint],
) -> list[list[dict[str, Any]]]:
    """Split a track's observation rows according to validated change points.

    Observations strictly inside a transition buffer (transition_start,
    transition_end) are never dropped: they form their own pending subsegment
    between the two sides, so every input row appears in exactly one returned
    subsegment and downstream matching can resolve or defer it with machine
    evidence.

    Args:
        rows: Sorted list of observation dictionaries from tracking data.
        change_points: List of validated ChangePoint instances.

    Returns:
        List of non-empty subsegment observation lists conserving all rows.
    """
    if not change_points:
        return [rows]

    cps = sorted(change_points, key=lambda cp: cp.transition_start)
    subsegments: list[list[dict[str, Any]]] = []
    current_sub: list[dict[str, Any]] = []
    transition_sub: list[dict[str, Any]] = []
    current_cp_idx = 0

    for r in rows:
        f = r["frame_idx"]
        if current_cp_idx < len(cps):
            cp = cps[current_cp_idx]
            if f <= cp.transition_start:
                current_sub.append(r)
            elif cp.transition_start < f < cp.transition_end:
                transition_sub.append(r)
            else:
                if current_sub:
                    subsegments.append(current_sub)
                    current_sub = []
                if transition_sub:
                    subsegments.append(transition_sub)
                    transition_sub = []
                current_cp_idx += 1
                current_sub.append(r)
        else:
            current_sub.append(r)

    if current_sub:
        subsegments.append(current_sub)
    if transition_sub:
        subsegments.append(transition_sub)

    return [sub for sub in subsegments if sub]
