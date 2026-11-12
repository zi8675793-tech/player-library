"""Read the companion target/ evidence without inference or input mutations.

Only this adapter knows the sample CSV, inclusive frame bounds and KPR NPZ keys.
The returned objects describe selected evidence, never confirmed player identities.
"""

import csv
import hashlib
import io
import math
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
from pydantic import ValidationError

from app.adapters._reading import InputReadError, ReadOnlyInput, fingerprint, stable_id
from app.contracts.common import Provenance
from app.contracts.events import Event, IntervalEventTime, PointEventTime, SegmentReference, Subject
from app.contracts.evidence import (
    Clip,
    EvidenceBatch,
    EvidenceWarning,
    FeatureFile,
    FeatureSpace,
    FileFeatureReference,
    IdentityEvidence,
    InlineFeature,
    MediaAsset,
    Preprocessing,
    Sample,
    Track,
    TrackingQuality,
    VisibilityContract,
)
from app.contracts.identity import IdentityRisk, Segment
from app.settings import Settings

PRODUCER = "sample_test"

# External labels belong to this boundary. Generic identity rules consume only
# IdentityRisk, never these strings or TrackingQuality.notes.
_QUALITY_RISK_KINDS = {
    "mixed_identity": "mixed_identity",
    "identity_switch": "identity_switch",
    "duplicate_box_candidate": "duplicate_box",
    "abrupt_motion_review": "quality_review",
}


class SampleTestError(InputReadError):
    """The sample-test handoff could not be safely and completely validated."""


def _need(data: dict, key: str, context: str):
    if key not in data:
        raise InputReadError(f"{context}: missing required field {key}")
    return data[key]


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputReadError(f"{context}: expected a nonempty string")
    return value


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise InputReadError(f"{context}: expected an integer >= {minimum}")
    return value


def _number(value: object, context: str, *, positive: bool = False) -> float:
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise InputReadError(f"{context}: expected a finite number")
    if value < 0 or (positive and value == 0):
        raise InputReadError(
            f"{context}: expected a {'positive' if positive else 'nonnegative'} number"
        )
    return float(value)


def _hash(value: object, context: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise InputReadError(f"{context}: expected a lowercase sha256 hash")
    return value


def _clip_id(value: object, context: str) -> str:
    value = _string(value, context)
    if value in (".", "..") or "/" in value or "\\" in value:
        raise InputReadError(f"{context}: invalid clip ID/path component {value!r}")
    return value


def _track_id(value: object, context: str, *, csv_value: bool = False) -> str:
    if csv_value:
        if not isinstance(value, str) or re.fullmatch(r"T[0-9]+", value) is None:
            raise InputReadError(f"{context}: expected T followed by a nonnegative track ID")
        return str(int(value[1:]))
    return str(_integer(value, context))


def _provenance(kind: str, key: str, content: object, *locations) -> Provenance:
    return Provenance(
        producer="manual_annotation" if kind == "event" else PRODUCER,
        record_type=kind,
        source_record_id=key,
        evidence_version="1",
        content_hash=fingerprint(content),
        locations=list(locations),
    )


def _quality_risks(raw: dict, segment: Segment, location) -> list[IdentityRisk]:
    """Translate selection labels without interpreting detector scores as identity."""
    tags = {tag.strip() for tag in raw.get("quality_review_reasons", "").split(";") if tag.strip()}
    if raw["quality_status"] == "needs_review" and not tags:
        tags.add("needs_review")
    return [
        IdentityRisk(
            risk_id=stable_id("risk", segment.segment_id, tag),
            kind=_QUALITY_RISK_KINDS.get(tag, "quality_review"),
            match_id=segment.match_id,
            segment_id=segment.segment_id,
            clip_id=segment.clip_id,
            track_id=segment.track_id,
            start_s=segment.start_s,
            end_s=segment.end_s,
            source=_provenance("identity_risk", f"{segment.segment_id}:{tag}", raw, location),
        )
        for tag in sorted(tags)
    ]


def _annotations(reader: ReadOnlyInput) -> dict[str, tuple[int, dict]]:
    path = "annotations.csv"
    result = {}
    parser = csv.DictReader(io.StringIO(reader.text(path), newline=""), strict=True)
    required = {"video_id", "filename", "goal_time_s", "jersey_number", "track_id"}
    if parser.fieldnames is None or not required.issubset(parser.fieldnames):
        raise InputReadError(f"{path}: missing required CSV columns {sorted(required)}")
    if len(set(parser.fieldnames)) != len(parser.fieldnames):
        raise InputReadError(f"{path}: duplicate CSV columns")
    try:
        while True:
            line = parser.line_num + 1
            row = next(parser, None)
            if row is None:
                break
            if None in row or any(value is None for value in row.values()):
                raise InputReadError(f"{path}:{line}: wrong number of CSV columns")
            clip_id = _clip_id(row["video_id"], f"{path}:{line}:video_id")
            if clip_id in result:
                raise InputReadError(f"{path}:{line}: duplicate video_id {clip_id}")
            result[clip_id] = (line, row)
    except csv.Error as exc:
        raise InputReadError(f"{path}:{parser.line_num}: invalid CSV: {exc}") from exc
    if not result:
        raise InputReadError(f"{path}: no annotated clips")
    return result


def _feature_space(contract: dict, context: str) -> FeatureSpace:
    if _integer(_need(contract, "schema_version", context), context + ":schema_version") != 1:
        raise InputReadError(f"{context}: unsupported feature contract schema_version")
    p = _integer(_need(contract, "part_count", context), context + ":part_count", minimum=1)
    d = _integer(
        _need(contract, "feature_dimension", context), context + ":feature_dimension", minimum=1
    )
    if _need(contract, "normalize_feature", context) is not True:
        raise InputReadError(f"{context}: KPR requires normalize_feature=true (l2)")
    protocol = _need(contract, "protocol", context)
    if not isinstance(protocol, dict):
        raise InputReadError(f"{context}: protocol must be an object")
    model_hash = _hash(
        _need(protocol, "checkpoint_sha256", context), context + ":checkpoint_sha256"
    )
    config_hash = _hash(_need(protocol, "config_sha256", context), context + ":config_sha256")
    prompt = _string(_need(protocol, "prompt_mode", context), context + ":prompt_mode")
    color = _string(_need(contract, "color_convention", context), context + ":color_convention")
    distance = _string(_need(contract, "distance", context), context + ":distance")
    embeddings = _need(contract, "test_embeddings", context)
    if not isinstance(embeddings, list) or not embeddings:
        raise InputReadError(f"{context}: test_embeddings must be a nonempty ordered list")
    for item in embeddings:
        _string(item, context + ":test_embeddings")
    common = _integer(
        _need(contract, "min_common_parts", context), context + ":min_common_parts", minimum=1
    )
    if common > p:
        raise InputReadError(f"{context}: min_common_parts exceeds part_count")
    descriptor = dict(
        backend="kpr",
        model_hash=model_hash,
        config_hash=config_hash,
        preprocessing=Preprocessing(
            version="sample-test-kpr-v1",
            color_space=color,
            config_hash=fingerprint(
                {"prompt_mode": prompt, "test_embeddings": embeddings, "min_common_parts": common}
            ),
        ),
        layout="parts",
        shape=[p, d],
        normalization="l2",
        distance=distance,
        visibility=VisibilityContract(shape=[p], semantics="per-part visibility score in [0,1]"),
    )
    canonical = {
        key: value.model_dump() if hasattr(value, "model_dump") else value
        for key, value in descriptor.items()
    }
    return FeatureSpace(feature_space_id=stable_id("fs", canonical), **descriptor)


def _arrays(reader, path, contract, samples, space):
    """Validate all rows before constructing any references; never sort array rows."""
    path = str(path)
    n = _integer(_need(contract, "sample_count", path), path + ":sample_count")
    p, d = space.shape
    if n != len(samples):
        raise InputReadError(
            f"{path}: sample_count {n} differs from samples.jsonl rows {len(samples)}"
        )
    required = {"embeddings", "visibility_scores", "sample_ids", "track_ids", "frame_indices"}
    try:
        with np.load(io.BytesIO(reader.data(path)), allow_pickle=False) as archive:
            if not required.issubset(archive.files):
                raise InputReadError(
                    f"{path}: missing required arrays {sorted(required - set(archive.files))}"
                )
            if len(set(archive.files)) != len(archive.files):
                raise InputReadError(f"{path}: duplicate NPZ array keys")
            arrays = {key: archive[key] for key in archive.files}
    except (OSError, ValueError, TypeError, EOFError, BadZipFile) as exc:
        raise InputReadError(f"{path}: cannot read NPZ with allow_pickle=False: {exc}") from exc
    for key, shape in {"embeddings": (n, p, d), "visibility_scores": (n, p)}.items():
        array = arrays[key]
        if array.dtype != np.dtype("float32") or array.shape != shape:
            raise InputReadError(
                f"{path}:{key}: expected float32 shape {shape}, got {array.dtype} {array.shape}"
            )
        if not np.isfinite(array).all():
            raise InputReadError(f"{path}:{key}: contains non-finite values")
    visibility = arrays["visibility_scores"]
    if np.any((visibility < 0) | (visibility > 1)):
        raise InputReadError(f"{path}:visibility_scores: values must be in [0,1]")
    norms = np.linalg.norm(arrays["embeddings"].astype(np.float64), axis=-1)
    bad = np.argwhere((visibility > 0) & ~np.isclose(norms, 1.0, rtol=1e-4, atol=1e-5))
    if bad.size:
        row, part = bad[0]
        raise InputReadError(
            f"{path}: sample row {row}, part {part}: visible embedding is not l2 normalized"
        )
    for key in ("sample_ids", "track_ids", "frame_indices"):
        array = arrays[key]
        valid_dtype = array.dtype.kind == "U" if key == "sample_ids" else array.dtype.kind in "iu"
        if not valid_dtype or array.shape != (n,):
            raise InputReadError(
                f"{path}:{key}: invalid dtype or shape; expected {(n,)} index array"
            )
    for row_index, (line, sample) in enumerate(samples):
        context = f"samples.jsonl:{line}"
        for key, expected in (
            ("sample_ids", _string(_need(sample, "sample_id", context), context + ":sample_id")),
            ("track_ids", _integer(_need(sample, "track_id", context), context + ":track_id")),
            (
                "frame_indices",
                _integer(_need(sample, "frame_idx", context), context + ":frame_idx"),
            ),
        ):
            if arrays[key][row_index] != expected:
                raise InputReadError(
                    f"{path}:{key}: row {row_index}, samples.jsonl:{line} sample {sample.get('sample_id')}: index mismatch"
                )
    return arrays


def _observation(row, context, clip_id, fps, frame_count, width, height):
    if _need(row, "video_id", context) != clip_id:
        raise InputReadError(f"{context}: video_id does not match clip {clip_id}")
    track = _track_id(_need(row, "track_id", context), context + ":track_id")
    frame = _integer(_need(row, "frame_idx", context), context + ":frame_idx")
    time = _number(_need(row, "timestamp_s", context), context + ":timestamp_s")
    if frame >= frame_count or time >= frame_count / fps:
        raise InputReadError(f"{context}: frame/time outside clip range")
    if not math.isclose(time, frame / fps, rel_tol=0, abs_tol=1e-6):
        raise InputReadError(f"{context}: timestamp_s disagrees with frame_idx/fps")
    bbox = _need(row, "bbox_xyxy", context)
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise InputReadError(f"{context}: bbox_xyxy must contain four coordinates")
    x1, y1, x2, y2 = [_number(x, context + ":bbox_xyxy") for x in bbox]
    if not (x1 < x2 <= width and y1 < y2 <= height):
        raise InputReadError(f"{context}: bbox_xyxy outside image or empty")
    score = _number(_need(row, "det_score", context), context + ":det_score")
    if score > 1:
        raise InputReadError(f"{context}: det_score outside [0,1]")
    return track, frame, time


def _load_clip(
    reader, match_id, clip_id, annotation, evidence, assets, warnings, spaces, feature_values=None
):
    csv_line, row = annotation
    base = Path("target") / clip_id
    selection_path, meta_path = base / "selection.json", base / "video_meta.json"
    samples_path, tracks_path = base / "samples.jsonl", base / "tracks.jsonl"
    contract_path, npz_path = base / "feature_contract.json", base / "features.npz"
    selection, meta, contract = (
        reader.json(path) for path in (selection_path, meta_path, contract_path)
    )
    for path, value in ((selection_path, selection), (meta_path, meta)):
        if _need(value, "video_id", str(path)) != clip_id:
            raise InputReadError(f"{path}: video_id does not match clip {clip_id}")
    fps = _number(_need(meta, "fps", str(meta_path)), f"{meta_path}:fps", positive=True)
    count = _integer(
        _need(meta, "frame_count", str(meta_path)), f"{meta_path}:frame_count", minimum=1
    )
    try:
        duration = count / fps
    except OverflowError as exc:
        raise InputReadError(f"{meta_path}: frame_count/fps exceeds finite clip duration") from exc
    _number(duration, f"{meta_path}:frame_count/fps duration", positive=True)
    width = _integer(_need(meta, "width", str(meta_path)), f"{meta_path}:width", minimum=1)
    height = _integer(_need(meta, "height", str(meta_path)), f"{meta_path}:height", minimum=1)
    if "processed_frames" in meta:
        if _integer(meta["processed_frames"], f"{meta_path}:processed_frames") > count:
            raise InputReadError(f"{meta_path}: processed_frames exceeds frame_count")
    video = _string(
        _need(selection, "source_video", str(selection_path)), f"{selection_path}:source_video"
    )
    expected_video = str(
        Path("video") / _string(row["filename"], f"annotations.csv:{csv_line}:filename")
    )
    if video != _need(meta, "input_path", str(meta_path)) or video != expected_video:
        raise InputReadError(
            f"{selection_path}: source_video disagrees with video_meta.json or CSV filename"
        )
    video_file = reader.file(video)
    asset_id = stable_id("asset", match_id, video_file.file_id)
    assets[asset_id] = MediaAsset(asset_id=asset_id, media_type="video", file=video_file)
    if "annotated_video" in selection:
        reader.file(_string(selection["annotated_video"], f"{selection_path}:annotated_video"))
    try:
        goal_time = float(row["goal_time_s"])
    except ValueError as exc:
        raise InputReadError(f"annotations.csv:{csv_line}: invalid goal_time_s") from exc
    goal_time = _number(goal_time, f"annotations.csv:{csv_line}:goal_time_s")
    if goal_time >= duration:
        raise InputReadError(f"annotations.csv:{csv_line}: goal_time_s outside clip range")
    if goal_time != _number(
        _need(selection, "goal_time_s", str(selection_path)), f"{selection_path}:goal_time_s"
    ):
        raise InputReadError(
            f"{selection_path}: goal_time_s disagrees with annotations.csv:{csv_line}"
        )
    jersey = _string(row["jersey_number"], f"annotations.csv:{csv_line}:jersey_number")
    if _need(selection, "event_jersey_number", str(selection_path)) != jersey:
        raise InputReadError(f"{selection_path}: event_jersey_number disagrees with CSV")
    raw_segments = _need(selection, "track_segments", str(selection_path))
    if not isinstance(raw_segments, list) or not raw_segments:
        raise InputReadError(f"{selection_path}: track_segments must be a nonempty list")
    segments_by_track = defaultdict(list)
    quality_risks_by_segment = {}
    selected = []
    for index, raw in enumerate(raw_segments):
        context = f"{selection_path}:track_segments[{index}]"
        if not isinstance(raw, dict):
            raise InputReadError(f"{context}: expected an object")
        track = _track_id(_need(raw, "track_id", context), context + ":track_id")
        start = _integer(_need(raw, "start_frame", context), context + ":start_frame")
        end = _integer(_need(raw, "end_frame", context), context + ":end_frame") + 1
        if not start < end <= count:
            raise InputReadError(f"{context}: invalid frame range or outside clip")
        for previous, _ in segments_by_track[track]:
            if start < previous.frame_end_exclusive and previous.frame_start < end:
                raise InputReadError(f"{context}: overlapping selected ranges for track {track}")
        status = _string(_need(raw, "quality_status", context), context + ":quality_status")
        if status not in {"accepted", "needs_review"}:
            raise InputReadError(f"{context}: unretained or unknown quality_status {status}")
        reasons = raw.get("quality_review_reasons", "")
        if not isinstance(reasons, str):
            raise InputReadError(f"{context}: quality_review_reasons must be a string")
        segment = Segment(
            segment_id=stable_id("seg", match_id, clip_id, track, start, end),
            match_id=match_id,
            clip_id=clip_id,
            track_id=track,
            start_s=start / fps,
            end_s=end / fps,
            frame_start=start,
            frame_end_exclusive=end,
        )
        segments_by_track[track].append((segment, raw))
        selected.append((segment, raw))
        risks = _quality_risks(
            raw, segment, reader.location(selection_path, record=f"track_segments[{index}]")
        )
        quality_risks_by_segment[segment.segment_id] = risks
        evidence.risks.extend(risks)
        if status == "needs_review" or reasons:
            warnings.append(
                EvidenceWarning(
                    code="tracking_quality",
                    clip_id=clip_id,
                    message=f"T{track} [{start},{end}): {status}; {reasons}",
                    locations=[reader.location(selection_path, record=f"track_segments[{index}]")],
                )
            )
    csv_tracks = [
        _track_id(value, f"annotations.csv:{csv_line}:track_id", csv_value=True)
        for value in row["track_id"].split("+")
    ]
    if len(set(csv_tracks)) != len(csv_tracks) or set(csv_tracks) != set(segments_by_track):
        raise InputReadError(
            f"{selection_path}: selected track IDs disagree with annotations.csv:{csv_line}"
        )
    ranges = row.get("target_frame_ranges", "")
    if ranges:
        csv_ranges = []
        for item in ranges.split(";"):
            match = re.fullmatch(r"(T[0-9]+):([0-9]+)-([0-9]+)", item)
            if match is None:
                raise InputReadError(f"annotations.csv:{csv_line}: invalid target_frame_ranges")
            track = _track_id(match[1], f"annotations.csv:{csv_line}", csv_value=True)
            csv_ranges.append((track, int(match[2]), int(match[3]) + 1))
        actual = [(s.track_id, s.frame_start, s.frame_end_exclusive) for s, _ in selected]
        if sorted(csv_ranges) != sorted(actual):
            raise InputReadError(
                f"{selection_path}: frame ranges disagree with annotations.csv:{csv_line}"
            )
    observations = reader.jsonl(tracks_path)
    grouped = defaultdict(list)
    observed_keys = {}
    for line, obs in observations:
        context = f"{tracks_path}:{line}"
        track, frame, time = _observation(obs, context, clip_id, fps, count, width, height)
        candidates = [
            s for s, _ in segments_by_track[track] if s.frame_start <= frame < s.frame_end_exclusive
        ]
        if len(candidates) != 1:
            raise InputReadError(f"{context}: observation outside selected track/frame ranges")
        if (track, frame) in observed_keys:
            raise InputReadError(f"{context}: duplicate track/frame observation")
        observed_keys[track, frame] = (line, obs)
        grouped[track].append((line, obs))
    for track in sorted(segments_by_track):
        rows = grouped[track]
        if not rows:
            raise InputReadError(f"{tracks_path}: no observations for selected track {track}")
        rows.sort(key=lambda item: item[1]["frame_idx"])
        frames = [obs["frame_idx"] for _, obs in rows]
        start, end = frames[0], frames[-1] + 1
        evidence.tracks.append(
            Track(
                match_id=match_id,
                clip_id=clip_id,
                track_id=track,
                start_s=start / fps,
                end_s=end / fps,
                frame_start=start,
                frame_end_exclusive=end,
                observed_frame_indices=frames,
                source=_provenance(
                    "track",
                    f"{clip_id}:track:{track}",
                    [obs for _, obs in rows],
                    reader.location(tracks_path, record=f"track_id={track}"),
                    reader.location(selection_path, record=f"track_id={track}"),
                ),
            )
        )
    for segment, raw in selected:
        actual = [
            frame
            for track, frame in observed_keys
            if track == segment.track_id
            and segment.frame_start <= frame < segment.frame_end_exclusive
        ]
        if not actual:
            raise InputReadError(
                f"{tracks_path}: selected segment {segment.segment_id} has no observations"
            )
        missing = segment.frame_end_exclusive - segment.frame_start - len(actual)
        if missing:
            warnings.append(
                EvidenceWarning(
                    code="missing_observations",
                    clip_id=clip_id,
                    message=f"T{segment.track_id} [{segment.frame_start},{segment.frame_end_exclusive}): {missing} frames unobserved; no observations added",
                    locations=[reader.location(tracks_path, record=f"track_id={segment.track_id}")],
                )
            )
        evidence.segments.append(segment)
    samples = reader.jsonl(samples_path)
    space = _feature_space(contract, str(contract_path))
    spaces[space.feature_space_id] = space
    arrays = _arrays(reader, npz_path, contract, samples, space)
    reader.file(
        npz_path,
        _hash(
            _need(contract, "feature_sha256", str(contract_path)), f"{contract_path}:feature_sha256"
        ),
    )
    protocol = contract["protocol"]
    reader.file(
        samples_path,
        _hash(
            _need(protocol, "samples_sha256", str(contract_path)), f"{contract_path}:samples_sha256"
        ),
    )
    feature_file_id = stable_id("features", match_id, clip_id)
    evidence.feature_files.append(
        FeatureFile(
            feature_file_id=feature_file_id,
            feature_space_id=space.feature_space_id,
            sample_count=len(samples),
            values_file=reader.file(npz_path),
            visibility_file=reader.file(npz_path),
            index_file=reader.file(samples_path),
        )
    )
    seen_sample_ids = set()
    for row_index, (line, sample) in enumerate(samples):
        context = f"{samples_path}:{line} sample {sample.get('sample_id')}"
        external_id = _string(_need(sample, "sample_id", context), context + ":sample_id")
        track, frame, time = _observation(sample, context, clip_id, fps, count, width, height)
        observed = observed_keys.get((track, frame))
        if observed is None:
            raise InputReadError(f"{context}: no actual track observation at sample frame")
        if sample["bbox_xyxy"] != observed[1]["bbox_xyxy"]:
            raise InputReadError(f"{context}: bbox_xyxy differs from track observation")
        segment, raw = next(
            (s, raw)
            for s, raw in segments_by_track[track]
            if s.frame_start <= frame < s.frame_end_exclusive
        )
        crop = _string(_need(sample, "crop_path", context), context + ":crop_path")
        image = reader.file(
            crop, _hash(_need(sample, "crop_sha256", context), context + ":crop_sha256")
        )
        image_asset_id = stable_id("asset", match_id, image.file_id)
        assets[image_asset_id] = MediaAsset(asset_id=image_asset_id, media_type="image", file=image)
        sample_id = stable_id("sample", match_id, clip_id, external_id)
        if sample_id in seen_sample_ids:
            raise InputReadError(f"{context}: duplicate sample_id {external_id}")
        seen_sample_ids.add(sample_id)
        feature_hash = hashlib.sha256(
            arrays["embeddings"][row_index].tobytes()
            + arrays["visibility_scores"][row_index].tobytes()
        ).hexdigest()
        evidence.samples.append(
            Sample(
                match_id=match_id,
                sample_id=sample_id,
                clip_id=clip_id,
                track_id=track,
                time_s=time,
                frame_idx=frame,
                segment_id=segment.segment_id,
                image_asset_id=image_asset_id,
                tracking_quality=TrackingQuality(
                    score=float(sample["det_score"]),
                    score_semantics="detection",
                    risks=quality_risks_by_segment[segment.segment_id],
                    mixed_identity=(
                        True
                        if any(
                            risk.kind == "mixed_identity"
                            for risk in quality_risks_by_segment[segment.segment_id]
                        )
                        else None
                    ),
                    notes=f"{raw['quality_status']}; {raw.get('quality_review_reasons', '')}; identity unconfirmed",
                ),
                source=_provenance(
                    "sample",
                    f"{clip_id}:sample:{external_id}",
                    {
                        "sample": sample,
                        "feature_space_id": space.feature_space_id,
                        "feature_hash": feature_hash,
                        "selection": raw,
                    },
                    reader.location(samples_path, line=line, record=external_id),
                    reader.location(npz_path, record=f"row_index={row_index}"),
                    reader.location(
                        selection_path,
                        record=f"track_id={track},frames=[{segment.frame_start},{segment.frame_end_exclusive})",
                    ),
                    reader.location(
                        tracks_path, line=observed[0], record=f"track_id={track},frame_idx={frame}"
                    ),
                ),
                features=[
                    FileFeatureReference(
                        feature_space_id=space.feature_space_id,
                        feature_file_id=feature_file_id,
                        row_index=row_index,
                    )
                ],
            )
        )
        if feature_values is not None:
            feature_values[(feature_file_id, row_index)] = InlineFeature(
                feature_space_id=space.feature_space_id,
                values=arrays["embeddings"][row_index].reshape(-1).tolist(),
                visibility=arrays["visibility_scores"][row_index].tolist(),
            )
    warnings.append(
        EvidenceWarning(
            code="identity_unconfirmed",
            clip_id=clip_id,
            message="Human target selection and event jersey annotation do not confirm whole-track or cross-clip identity",
            locations=[
                reader.location(selection_path),
                reader.location("annotations.csv", line=csv_line),
            ],
        )
    )
    status = contract.get("status")
    if status is not None:
        warnings.append(
            EvidenceWarning(
                code="feature_status",
                clip_id=clip_id,
                message=_string(status, f"{contract_path}:status"),
                locations=[reader.location(contract_path)],
            )
        )
    source_key = f"goal:{clip_id}:1"
    event_id = stable_id("event", match_id, source_key)
    event = Event(
        match_id=match_id,
        event_id=event_id,
        source_event_key=source_key,
        clip_id=clip_id,
        event_type="goal",
        time=PointEventTime(time_s=goal_time),
        outcome="made",
        source=_provenance(
            "event",
            source_key,
            {
                "annotation": row,
                "selected_ranges": sorted(
                    (s.track_id, s.frame_start, s.frame_end_exclusive) for s, _ in selected
                ),
            },
            reader.location("annotations.csv", line=csv_line, record=source_key),
            reader.location(selection_path),
        ),
        subjects=[
            Subject(
                subject_id=stable_id("subject", match_id, source_key, "shooter"),
                role="shooter",
                event_jersey_number=jersey,
                event_team_id=row.get("team_id") or None,
                references=[
                    SegmentReference(
                        segment_id=s.segment_id,
                        time=(
                            None
                            if s.start_s <= goal_time < s.end_s
                            else IntervalEventTime(start_s=s.start_s, end_s=s.end_s)
                        ),
                    )
                    for s, _ in sorted(
                        selected, key=lambda item: (item[0].track_id, item[0].frame_start)
                    )
                ],
            )
        ],
    )
    return Clip(
        match_id=match_id,
        clip_id=clip_id,
        asset_id=asset_id,
        duration_s=duration,
        frame_count=count,
        fps=fps,
        frame_timing="cfr",
    ), event


def _load_sample_test(
    settings: Settings,
    *,
    batch_id: str,
    clip_ids: Iterable[str] | None = None,
    feature_values: dict[tuple[str, int], InlineFeature] | None = None,
) -> EvidenceBatch:
    """Validate the full handoff or selected clips and return an existing EvidenceBatch.

    Input and runtime directories are never written. A fatal error raises
    SampleTestError with its file/record context; no partially valid batch escapes.
    """
    try:
        if settings.input.adapter != "sample_test":
            raise InputReadError("settings.input.adapter must be sample_test")
        _string(batch_id, "batch_id")
        reader = ReadOnlyInput(settings.paths.input_root)
        annotations = _annotations(reader)
        if isinstance(clip_ids, (str, bytes)):
            raise InputReadError("clip_ids must be an iterable of clip ID strings, not a string")
        selected = (
            set(annotations)
            if clip_ids is None
            else {_clip_id(value, "clip_ids") for value in clip_ids}
        )
        if not selected:
            raise InputReadError("clip_ids: select at least one clip")
        unknown = selected - annotations.keys()
        if unknown:
            raise InputReadError(f"annotations.csv: unknown clip_ids {sorted(unknown)}")
        target = reader.path("target", directory=True)
        if clip_ids is None:
            discovered = {p.name for p in target.iterdir() if p.is_dir()}
            if discovered != set(annotations):
                raise InputReadError(
                    "target/: clip directories disagree with annotations.csv: "
                    f"missing={sorted(set(annotations) - discovered)}, extra={sorted(discovered - set(annotations))}"
                )
        evidence = IdentityEvidence()
        clips, events, warnings, assets, spaces = [], [], [], {}, {}
        for clip_id in sorted(selected):
            reader.path(Path("target") / clip_id, directory=True)
            clip, event = _load_clip(
                reader,
                settings.match.match_id,
                clip_id,
                annotations[clip_id],
                evidence,
                assets,
                warnings,
                spaces,
                feature_values,
            )
            clips.append(clip)
            events.append(event)
        evidence.feature_spaces = [spaces[key] for key in sorted(spaces)]
        return EvidenceBatch(
            schema_version="2",
            match_id=settings.match.match_id,
            batch_id=batch_id,
            producer=PRODUCER,
            sources=sorted(reader.sources.values(), key=lambda source: source.path),
            assets=[assets[key] for key in sorted(assets)],
            clips=clips,
            identity_evidence=evidence,
            events=events,
            warnings=warnings,
        )
    except (InputReadError, ValidationError) as exc:
        raise SampleTestError(str(exc)) from exc


def load_sample_test(
    settings: Settings, *, batch_id: str, clip_ids: Iterable[str] | None = None
) -> EvidenceBatch:
    """Validate evidence without retaining parsed feature vectors in memory."""
    return _load_sample_test(settings, batch_id=batch_id, clip_ids=clip_ids)


def load_sample_test_input(
    settings: Settings, *, batch_id: str, clip_ids: Iterable[str] | None = None
) -> tuple[EvidenceBatch, dict[tuple[str, int], InlineFeature]]:
    """Assemble workflow input from the same validated bytes and NPZ row order.

    File parsing stays at this boundary. The batch retains its original file
    references, and values are separately ready for identity and storage calls.
    No clean-sample, identity or template approval is inferred from input labels.
    """
    values: dict[tuple[str, int], InlineFeature] = {}
    batch = _load_sample_test(settings, batch_id=batch_id, clip_ids=clip_ids, feature_values=values)
    return batch, values
