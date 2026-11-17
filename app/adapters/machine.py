"""Validate unlabelled tracking/ReID files and emit the existing evidence contract.

This module has no dependency on sample_test, scoring or manual databases.
A provisional segment spans one raw track; it never confirms that track's identity.
"""

import csv
import hashlib
import io
import math
import re
from collections import defaultdict
from collections.abc import Iterable
from zipfile import BadZipFile

import numpy as np
from pydantic import ValidationError

from app.adapters._machine_reading import MachineInputReader, clip_component
from app.adapters._reading import InputReadError, fingerprint, stable_id
from app.contracts.common import Provenance
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
    TrackObservation,
    VisibilityContract,
)
from app.contracts.identity import IdentityRisk, Segment
from app.settings import Settings

PRODUCER = "machine_tracks_reid_v1"


class MachineInputError(InputReadError):
    """The machine handoff could not be safely and completely validated."""


def _need(record, key, context):
    if key not in record:
        raise InputReadError(f"{context}: missing required field {key}")
    return record[key]


def _string(value, context):
    if not isinstance(value, str) or not value.strip():
        raise InputReadError(f"{context}: expected a nonempty string")
    return value


def _integer(value, context, minimum=0):
    if type(value) is not int or value < minimum:
        raise InputReadError(f"{context}: expected integer >= {minimum}")
    return value


def _number(value, context, *, positive=False):
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < 0 or (positive and value == 0):
        raise InputReadError(
            f"{context}: expected finite {'positive' if positive else 'nonnegative'} number"
        )
    return float(value)


def _hash(value, context):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise InputReadError(f"{context}: expected lowercase sha256")
    return value


def _provenance(kind, key, content, *locations):
    return Provenance(
        producer=PRODUCER,
        record_type=kind,
        source_record_id=key,
        evidence_version="1",
        content_hash=fingerprint(content),
        locations=list(locations),
    )


def _feature_space(contract, context):
    if _integer(_need(contract, "schema_version", context), context + ":schema_version") != 1:
        raise InputReadError(f"{context}: unsupported feature contract schema_version")
    p = _integer(_need(contract, "part_count", context), context, 1)
    d = _integer(_need(contract, "feature_dimension", context), context, 1)
    if _need(contract, "normalize_feature", context) is not True:
        raise InputReadError(f"{context}: KPR requires normalize_feature=true")
    protocol = _need(contract, "protocol", context)
    if not isinstance(protocol, dict):
        raise InputReadError(f"{context}: protocol must be an object")
    embeddings = _need(contract, "test_embeddings", context)
    if not isinstance(embeddings, list) or not embeddings:
        raise InputReadError(f"{context}: test_embeddings must be a nonempty ordered list")
    for embedding in embeddings:
        _string(embedding, context + ":test_embeddings")
    common = _integer(_need(contract, "min_common_parts", context), context, 1)
    if common > p:
        raise InputReadError(f"{context}: min_common_parts exceeds part_count")
    descriptor = dict(
        backend="kpr",
        model_hash=_hash(_need(protocol, "checkpoint_sha256", context), context),
        config_hash=_hash(_need(protocol, "config_sha256", context), context),
        preprocessing=Preprocessing(
            version="machine-kpr-v1",
            color_space=_string(_need(contract, "color_convention", context), context),
            config_hash=fingerprint(
                {
                    "prompt_mode": _string(_need(protocol, "prompt_mode", context), context),
                    "test_embeddings": embeddings,
                }
            ),
        ),
        layout="parts",
        shape=[p, d],
        normalization="l2",
        distance=_string(_need(contract, "distance", context), context),
        visibility=VisibilityContract(shape=[p], semantics="per-part visibility score in [0,1]"),
    )
    canonical = {
        key: val.model_dump() if hasattr(val, "model_dump") else val
        for key, val in descriptor.items()
    }
    return FeatureSpace(feature_space_id=stable_id("fs", canonical), **descriptor)


def _arrays(reader, path, contract, samples, space):
    n = _integer(_need(contract, "sample_count", path), path + ":sample_count")
    if n != len(samples):
        raise InputReadError(f"{path}: sample_count differs from samples.jsonl row count")
    required = {"embeddings", "visibility_scores", "sample_ids", "track_ids", "frame_indices"}
    try:
        with np.load(io.BytesIO(reader.data(path)), allow_pickle=False) as archive:
            if not required.issubset(archive.files) or len(set(archive.files)) != len(
                archive.files
            ):
                raise InputReadError(f"{path}: missing or duplicate NPZ array keys")
            arrays = {key: archive[key] for key in required}
    except (OSError, ValueError, TypeError, EOFError, BadZipFile) as exc:
        raise InputReadError(f"{path}: cannot read NPZ with allow_pickle=False: {exc}") from exc
    p, d = space.shape
    for key, shape in {"embeddings": (n, p, d), "visibility_scores": (n, p)}.items():
        array = arrays[key]
        if (
            array.dtype != np.dtype("float32")
            or array.shape != shape
            or not np.isfinite(array).all()
        ):
            raise InputReadError(f"{path}:{key}: expected finite float32 shape {shape}")
    visibility = arrays["visibility_scores"]
    if np.any((visibility < 0) | (visibility > 1)):
        raise InputReadError(f"{path}:visibility_scores: values outside [0,1]")
    norms = np.linalg.norm(arrays["embeddings"].astype(np.float64), axis=-1)
    if np.any((visibility > 0) & ~np.isclose(norms, 1.0, rtol=1e-4, atol=1e-5)):
        raise InputReadError(f"{path}: visible embeddings are not l2 normalized")
    for key in ("sample_ids", "track_ids", "frame_indices"):
        arr = arrays[key]
        valid_dtype = arr.dtype.kind == "U" if key == "sample_ids" else arr.dtype.kind in "iu"
        if not valid_dtype or arr.shape != (n,):
            raise InputReadError(f"{path}:{key}: invalid index dtype or shape")
    for index, (line, sample) in enumerate(samples):
        for key, field in (
            ("sample_ids", "sample_id"),
            ("track_ids", "track_id"),
            ("frame_indices", "frame_idx"),
        ):
            expected = _need(sample, field, f"samples.jsonl:{line}")
            if key == "sample_ids":
                _string(expected, f"samples.jsonl:{line}:{field}")
            else:
                _integer(expected, f"samples.jsonl:{line}:{field}")
            if arrays[key][index] != expected:
                raise InputReadError(
                    f"{path}:{key}: row {index}, samples.jsonl:{line}: index mismatch"
                )
    return arrays


def _observation(row, context, clip_id, fps, count, width, height):
    if _need(row, "video_id", context) != clip_id:
        raise InputReadError(f"{context}: video_id disagrees with clip {clip_id}")
    track = str(_integer(_need(row, "track_id", context), context + ":track_id"))
    frame = _integer(_need(row, "frame_idx", context), context + ":frame_idx")
    time = _number(_need(row, "timestamp_s", context), context + ":timestamp_s")
    if frame >= count or time >= count / fps:
        raise InputReadError(f"{context}: frame/time outside clip")
    if not math.isclose(time, frame / fps, rel_tol=0, abs_tol=1e-6):
        raise InputReadError(f"{context}: timestamp_s disagrees with frame_idx/fps (CFR handoff)")
    box = _need(row, "bbox_xyxy", context)
    if not isinstance(box, list) or len(box) != 4:
        raise InputReadError(f"{context}: bbox_xyxy must have four coordinates")
    x1, y1, x2, y2 = [_number(value, context + ":bbox_xyxy") for value in box]
    if not (x1 < x2 <= width and y1 < y2 <= height):
        raise InputReadError(f"{context}: bbox_xyxy empty or outside image")
    if _number(_need(row, "det_score", context), context + ":det_score") > 1:
        raise InputReadError(f"{context}: det_score outside [0,1]")
    return track, frame, time


def _track_summary(reader, path):
    parser = csv.DictReader(io.StringIO(reader.text(path), newline=""), strict=True)
    required = {
        "track_id",
        "observations",
        "observed_seconds",
        "start_frame",
        "end_frame",
        "quality_status",
        "review_reasons",
    }
    if (
        parser.fieldnames is None
        or not required.issubset(parser.fieldnames)
        or len(set(parser.fieldnames)) != len(parser.fieldnames)
    ):
        raise InputReadError(f"{path}: missing/duplicate CSV columns")
    result = {}
    try:
        for row in parser:
            context = f"{path}:{parser.line_num}"
            if None in row or any(val is None for val in row.values()):
                raise InputReadError(f"{context}: wrong number of CSV columns")
            for field in ("track_id", "observations", "start_frame", "end_frame"):
                if re.fullmatch(r"[0-9]+", row[field]) is None:
                    raise InputReadError(f"{context}:{field}: expected nonnegative integer")
            track = str(int(row["track_id"]))
            if track in result:
                raise InputReadError(f"{context}: duplicate track_id")
            try:
                seconds = float(row["observed_seconds"])
            except ValueError as exc:
                raise InputReadError(f"{context}: invalid observed_seconds") from exc
            _number(seconds, context + ":observed_seconds")
            result[track] = (parser.line_num, row)
    except csv.Error as exc:
        raise InputReadError(f"{path}: invalid CSV: {exc}") from exc
    return result


def _load_clip(reader, match, clip_id, evidence, assets, warnings, spaces, values):
    track_base, feature_base = f"track/{clip_id}", f"identity/{clip_id}"
    meta_path = f"{track_base}/video_meta.json"
    tracks_path, csv_path = f"{track_base}/tracks.jsonl", f"{track_base}/tracks.csv"
    samples_path, contract_path = (
        f"{feature_base}/samples.jsonl",
        f"{feature_base}/feature_contract.json",
    )
    npz_path = f"{feature_base}/features.npz"
    meta = reader.json(meta_path)
    if _need(meta, "video_id", meta_path) != clip_id:
        raise InputReadError(f"{meta_path}: video_id disagrees with directory")
    fps = _number(_need(meta, "fps", meta_path), meta_path + ":fps", positive=True)
    count = _integer(_need(meta, "frame_count", meta_path), meta_path + ":frame_count", 1)
    width = _integer(_need(meta, "width", meta_path), meta_path + ":width", 1)
    height = _integer(_need(meta, "height", meta_path), meta_path + ":height", 1)
    duration = _number(count / fps, meta_path + ":duration", positive=True)
    if "processed_frames" in meta and _integer(meta["processed_frames"], meta_path) > count:
        raise InputReadError(f"{meta_path}: processed_frames exceeds frame_count")
    expected_video = f"video/{clip_id}.mp4"
    video = _string(_need(meta, "input_path", meta_path), meta_path + ":input_path")
    # Test the allowlist even for a reference which will later fail semantic checks.
    reader.path(video)
    if video != expected_video:
        raise InputReadError(f"{meta_path}: input_path must equal {expected_video}")
    video_file = reader.file(video)
    asset_id = stable_id("asset", match, video_file.file_id)
    assets[asset_id] = MediaAsset(asset_id=asset_id, media_type="video", file=video_file)
    summaries = _track_summary(reader, csv_path)
    grouped, observed = defaultdict(list), {}
    for line, row in reader.jsonl(tracks_path):
        track, frame, time = _observation(
            row, f"{tracks_path}:{line}", clip_id, fps, count, width, height
        )
        if (track, frame) in observed:
            raise InputReadError(f"{tracks_path}:{line}: duplicate track/frame observation")
        observed[track, frame] = (line, row)
        grouped[track].append((line, row))
        evidence.observations.append(
            TrackObservation(
                observation_id=stable_id("obs", match, PRODUCER, clip_id, track, frame),
                match_id=match,
                clip_id=clip_id,
                track_id=track,
                time_s=time,
                frame_idx=frame,
                bbox_xyxy=tuple(row["bbox_xyxy"]),
                detection_score=row["det_score"],
                source=_provenance(
                    "track_observation",
                    f"{clip_id}:track:{track}:frame:{frame}",
                    row,
                    reader.location(
                        tracks_path, line=line, record=f"track_id={track},frame_idx={frame}"
                    ),
                ),
            )
        )
    if set(grouped) != set(summaries):
        raise InputReadError(f"{csv_path}: track IDs disagree with tracks.jsonl")
    segments, risks_by_track = {}, {}
    for track in sorted(grouped):
        rows = sorted(grouped[track], key=lambda value: value[1]["frame_idx"])
        frames = [row["frame_idx"] for _, row in rows]
        start, end = frames[0], frames[-1] + 1
        csv_line, summary = summaries[track]
        if (
            int(summary["observations"]),
            int(summary["start_frame"]),
            int(summary["end_frame"]),
        ) != (len(frames), start, end - 1) or not math.isclose(
            float(summary["observed_seconds"]), len(frames) / fps, rel_tol=0, abs_tol=1e-6
        ):
            raise InputReadError(
                f"{csv_path}:{csv_line}: summary disagrees with actual observations"
            )
        segment = Segment(
            segment_id=stable_id("seg", match, PRODUCER, clip_id, track, start, end),
            match_id=match,
            clip_id=clip_id,
            track_id=track,
            start_s=rows[0][1]["timestamp_s"],
            end_s=end / fps,
            frame_start=start,
            frame_end_exclusive=end,
        )
        segments[track] = segment
        evidence.segments.append(segment)
        evidence.tracks.append(
            Track(
                match_id=match,
                clip_id=clip_id,
                track_id=track,
                start_s=segment.start_s,
                end_s=segment.end_s,
                frame_start=start,
                frame_end_exclusive=end,
                observed_frame_indices=frames,
                source=_provenance(
                    "track",
                    f"{clip_id}:track:{track}",
                    {"observations": [row for _, row in rows], "summary": summary},
                    reader.location(tracks_path, record=f"track_id={track}"),
                    reader.location(csv_path, line=csv_line, record=f"track_id={track}"),
                ),
            )
        )
        provisional = IdentityRisk(
            risk_id=stable_id("risk", segment.segment_id, "provisional_track"),
            basis="raw_track_continuity_unverified",
            kind="quality_review",
            match_id=match,
            segment_id=segment.segment_id,
            clip_id=clip_id,
            track_id=track,
            start_s=segment.start_s,
            end_s=segment.end_s,
            source=_provenance(
                "identity_risk",
                f"{clip_id}:track:{track}:provisional",
                {"segment_basis": "raw_track_extent", "identity_continuity": "unverified"},
                reader.location(tracks_path, record=f"track_id={track}"),
            ),
        )
        risks = [provisional]
        for tag in sorted(
            {tag.strip() for tag in summary["review_reasons"].split(";") if tag.strip()}
        ):
            kind = {
                "mixed_identity": "mixed_identity",
                "identity_switch": "identity_switch",
                "duplicate_box_candidate": "duplicate_box",
            }.get(tag, "quality_review")
            risks.append(
                IdentityRisk(
                    risk_id=stable_id("risk", segment.segment_id, tag),
                    basis="machine_duplicate_candidate"
                    if kind == "duplicate_box"
                    else "unspecified",
                    kind=kind,
                    match_id=match,
                    segment_id=segment.segment_id,
                    clip_id=clip_id,
                    track_id=track,
                    start_s=segment.start_s,
                    end_s=segment.end_s,
                    source=_provenance(
                        "identity_risk",
                        f"{clip_id}:track:{track}:{tag}",
                        summary,
                        reader.location(csv_path, line=csv_line),
                    ),
                )
            )
        risks_by_track[track] = risks
        evidence.risks.extend(risks)
        warnings.append(
            EvidenceWarning(
                code="provisional_track_segment",
                message=f"Track {track}: provisional segment from raw track extent; identity continuity unverified; no identity or template approval",
                clip_id=clip_id,
                locations=[
                    reader.location(csv_path, line=csv_line),
                    reader.location(tracks_path, record=f"track_id={track}"),
                ],
            )
        )
        if len(frames) != end - start:
            warnings.append(
                EvidenceWarning(
                    code="missing_observations",
                    message=f"Track {track}: {end - start - len(frames)} frames unobserved; no observations added",
                    clip_id=clip_id,
                    locations=[reader.location(tracks_path, record=f"track_id={track}")],
                )
            )
    contract = reader.json(contract_path)
    reader.file(npz_path, _hash(_need(contract, "feature_sha256", contract_path), contract_path))
    protocol = _need(contract, "protocol", contract_path)
    if not isinstance(protocol, dict):
        raise InputReadError(f"{contract_path}: protocol must be an object")
    reader.file(
        samples_path, _hash(_need(protocol, "samples_sha256", contract_path), contract_path)
    )
    samples = reader.jsonl(samples_path)
    space = _feature_space(contract, contract_path)
    spaces[space.feature_space_id] = space
    arrays = _arrays(reader, npz_path, contract, samples, space)
    feature_file_id = stable_id("features", match, PRODUCER, clip_id)
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
    for row_index, (line, raw) in enumerate(samples):
        context = f"{samples_path}:{line}"
        external_id = _string(_need(raw, "sample_id", context), context)
        track, frame, time = _observation(raw, context, clip_id, fps, count, width, height)
        obs = observed.get((track, frame))
        if obs is None:
            raise InputReadError(f"{context}: sample has no actual track observation")
        for field in ("timestamp_s", "bbox_xyxy", "det_score"):
            if raw[field] != obs[1][field]:
                raise InputReadError(f"{context}: {field} differs from track observation")
        crop = _string(_need(raw, "crop_path", context), context)
        reader.path(crop)
        if not crop.startswith(f"track/{clip_id}/samples/"):
            raise InputReadError(f"{context}: crop_path must belong to the same clip")
        image = reader.file(crop, _hash(_need(raw, "crop_sha256", context), context))
        image_asset_id = stable_id("asset", match, image.file_id)
        assets[image_asset_id] = MediaAsset(asset_id=image_asset_id, media_type="image", file=image)
        feature_hash = hashlib.sha256(
            arrays["embeddings"][row_index].tobytes()
            + arrays["visibility_scores"][row_index].tobytes()
        ).hexdigest()
        evidence.samples.append(
            Sample(
                match_id=match,
                sample_id=stable_id("sample", match, PRODUCER, clip_id, external_id),
                clip_id=clip_id,
                track_id=track,
                time_s=time,
                frame_idx=frame,
                segment_id=segments[track].segment_id,
                image_asset_id=image_asset_id,
                tracking_quality=TrackingQuality(
                    score=float(raw["det_score"]),
                    score_semantics="detection",
                    risks=risks_by_track[track],
                    notes=f"machine quality={summaries[track][1]['quality_status']}; raw track identity continuity unverified",
                ),
                source=_provenance(
                    "sample",
                    f"{clip_id}:sample:{external_id}",
                    {
                        "sample": raw,
                        "feature_space_id": space.feature_space_id,
                        "feature_hash": feature_hash,
                    },
                    reader.location(samples_path, line=line, record=external_id),
                    reader.location(npz_path, record=f"row_index={row_index}"),
                    reader.location(
                        tracks_path, line=obs[0], record=f"track_id={track},frame_idx={frame}"
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
        if values is not None:
            values[feature_file_id, row_index] = InlineFeature(
                feature_space_id=space.feature_space_id,
                values=arrays["embeddings"][row_index].reshape(-1).tolist(),
                visibility=arrays["visibility_scores"][row_index].tolist(),
            )
    return Clip(
        match_id=match,
        clip_id=clip_id,
        asset_id=asset_id,
        duration_s=duration,
        frame_count=count,
        fps=fps,
        frame_timing="cfr",
    )


def _load_machine(settings, *, batch_id, clip_ids=None, values=None):
    try:
        if settings.input.adapter != "machine":
            raise InputReadError("settings.input.adapter must be machine")
        _string(batch_id, "batch_id")
        reader = MachineInputReader(settings.paths.input_root)
        discovered = reader.discover_clips()
        if isinstance(clip_ids, (str, bytes)):
            raise InputReadError("clip_ids must be an iterable of strings")
        selected = (
            set(discovered) if clip_ids is None else {clip_component(value) for value in clip_ids}
        )
        if not selected or selected - set(discovered):
            raise InputReadError(f"unknown or empty machine clip selection: {sorted(selected)}")
        evidence, assets, warnings, spaces = IdentityEvidence(), {}, [], {}
        clips = [
            _load_clip(
                reader, settings.match.match_id, clip, evidence, assets, warnings, spaces, values
            )
            for clip in sorted(selected)
        ]
        evidence.feature_spaces = [spaces[key] for key in sorted(spaces)]
        return EvidenceBatch(
            match_id=settings.match.match_id,
            batch_id=batch_id,
            producer=PRODUCER,
            sources=sorted(reader.sources.values(), key=lambda source: source.path),
            assets=[assets[key] for key in sorted(assets)],
            clips=clips,
            identity_evidence=evidence,
            warnings=warnings,
        )
    except (InputReadError, ValidationError, OverflowError, OSError, RuntimeError) as exc:
        raise MachineInputError(str(exc)) from exc


def load_machine(
    settings: Settings, *, batch_id: str, clip_ids: Iterable[str] | None = None
) -> EvidenceBatch:
    """Return standard evidence without retaining array values."""
    return _load_machine(settings, batch_id=batch_id, clip_ids=clip_ids)


def load_machine_input(
    settings: Settings, *, batch_id: str, clip_ids: Iterable[str] | None = None
) -> tuple[EvidenceBatch, dict[tuple[str, int], InlineFeature]]:
    """Return standard evidence plus verified in-memory feature rows for workflows."""
    values = {}
    batch = _load_machine(settings, batch_id=batch_id, clip_ids=clip_ids, values=values)
    return batch, values
