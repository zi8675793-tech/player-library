"""Current, role-specific event association without storage or ReID access."""

import math
from bisect import bisect_left
from collections.abc import Iterable
from dataclasses import dataclass

from app.contracts.events import (
    Event,
    EventTime,
    IntervalEventTime,
    PointEventTime,
    ResolvedEvent,
    ResolvedSubject,
    SourceEventTime,
    SubjectReference,
)
from app.contracts.evidence import Clip, Track
from app.contracts.identity import Assignment, Segment
from app.events.validate import time_within


def _affine_time(time: EventTime, offset: float, scale: float) -> EventTime | None:
    if time.kind == "point":
        value = offset + time.time_s * scale
        return PointEventTime(time_s=value) if value >= 0 else None
    start, end = offset + time.start_s * scale, offset + time.end_s * scale
    return IntervalEventTime(start_s=start, end_s=end) if start >= 0 else None


@dataclass(frozen=True)
class _Context:
    clips: dict[str, Clip]
    tracks: dict[tuple[str, str], Track]
    segments: dict[str, Segment]
    assignments: dict[str, Assignment]


def _reference_time(event: Event, reference: SubjectReference, clip_id: str, ctx: _Context):
    if reference.time is not None:
        return reference.time, None
    if event.clip_id == clip_id:
        return event.time, None
    origin, target = ctx.clips.get(event.clip_id), ctx.clips.get(clip_id)
    if not origin or not target or not origin.source_mapping or not target.source_mapping:
        return None, "cross_clip_time_mapping_missing"
    a, b = origin.source_mapping, target.source_mapping
    if a.source_video_id != b.source_video_id:
        return None, "cross_clip_source_mismatch"
    mapped = _affine_time(
        event.time,
        (a.source_start_s - b.source_start_s) / b.time_scale,
        a.time_scale / b.time_scale,
    )
    return (mapped, None) if mapped else (None, "cross_clip_time_outside_target")


def _replacement_leaves(root: Segment, ctx: _Context):
    """Follow only explicit replacement edges, checking each ancestor's scope."""

    def visit(segment, path):
        if segment.segment_id in path:
            return [], "replacement_cycle"
        children = [
            s for s in ctx.segments.values() if s.supersedes_segment_id == segment.segment_id
        ]
        if segment.active:
            if children:
                return [], "replacement_parent_still_active"
            return [segment], None
        if not children:
            return [], "segment_inactive_without_replacement"
        result = []
        for child in children:
            if (
                (child.clip_id, child.track_id) != (segment.clip_id, segment.track_id)
                or child.start_s < segment.start_s
                or child.end_s > segment.end_s
            ):
                return [], "replacement_outside_original_range"
            if segment.frame_start is not None and (
                child.frame_start is None
                or child.frame_start < segment.frame_start
                or child.frame_end_exclusive > segment.frame_end_exclusive
            ):
                return [], "replacement_frame_range_invalid"
            leaves, reason = visit(child, path | {segment.segment_id})
            if reason:
                return [], reason
            result.extend(leaves)
        return result, None

    return visit(root, set())


def _covered_segments(time: EventTime, segments: list[Segment]):
    if time.kind == "point":
        found = [s for s in segments if s.start_s <= time.time_s < s.end_s]
        if len(found) != 1:
            return [], "identity_conflict" if found else "identity_gap"
        return found, None
    found = sorted(
        (s for s in segments if s.start_s < time.end_s and time.start_s < s.end_s),
        key=lambda s: (s.start_s, s.end_s, s.segment_id),
    )
    cursor = time.start_s
    for segment in found:
        if segment.start_s > cursor:
            return [], "identity_gap"
        if segment.start_s < cursor and cursor > time.start_s:
            return [], "identity_conflict"
        cursor = min(segment.end_s, time.end_s)
    if cursor < time.end_s:
        return [], "identity_gap"
    # Two segments both containing the interval's left edge also conflict.
    if sum(s.start_s <= time.start_s < s.end_s for s in found) > 1:
        return [], "identity_conflict"
    return found, None


def _frame_position(seconds: float, fps: float) -> float:
    position = seconds * fps
    nearest = round(position)
    return nearest if math.isclose(position, nearest, rel_tol=0, abs_tol=1e-9) else position


def _frame_range(time: EventTime, fps: float) -> tuple[int, int]:
    if time.kind == "point":
        first = math.floor(_frame_position(time.time_s, fps))
        return first, first + 1
    return (
        math.floor(_frame_position(time.start_s, fps)),
        math.ceil(_frame_position(time.end_s, fps)),
    )


def _observations(time: EventTime, track: Track | None, clip: Clip | None) -> str | None:
    if track is None:
        return "track_not_received"
    if not time_within(time, track.start_s, track.end_s):
        return "support_outside_track_observations"
    if track.observed_frame_indices is None:
        return "observation_information_missing"
    if not track.observed_frame_indices:
        return "no_actual_observations"
    if clip is None or clip.frame_timing != "cfr" or clip.fps is None:
        return "observation_time_mapping_missing"
    first, end = _frame_range(time, clip.fps)
    observed = track.observed_frame_indices
    if bisect_left(observed, end) - bisect_left(observed, first) != end - first:
        return "missing_observation_frames"
    return None


def _resolve_reference(event: Event, reference: SubjectReference, ctx: _Context):
    if reference.kind == "segment":
        root = ctx.segments.get(reference.segment_id)
        if root is None:
            return None, "segment_not_received"
        clip_id, track_id = root.clip_id, root.track_id
        segments, reason = _replacement_leaves(root, ctx)
        if reason:
            return None, reason
    else:
        clip_id, track_id = reference.clip_id, reference.track_id
        segments = [
            s
            for s in ctx.segments.values()
            if s.active and (s.clip_id, s.track_id) == (clip_id, track_id)
        ]
    time, reason = _reference_time(event, reference, clip_id, ctx)
    if reason:
        return None, reason
    if reference.kind == "segment" and not time_within(time, root.start_s, root.end_s):
        return None, "support_outside_referenced_segment"
    clip = ctx.clips.get(clip_id)
    if clip and clip.duration_s is not None and not time_within(time, 0, clip.duration_s):
        return None, "support_outside_clip"
    supporting, reason = _covered_segments(time, segments)
    if reason:
        return None, reason
    players = set()
    for segment in supporting:
        assignment = ctx.assignments.get(segment.segment_id)
        if assignment is None or assignment.status != "confirmed":
            return None, "identity_not_confirmed"
        players.add(assignment.player_id)
    if len(players) != 1:
        return None, "identity_changes_within_support"
    if clip and clip.frame_timing == "cfr" and clip.fps is not None:
        for segment in supporting:
            if segment.frame_start is None:
                continue
            portion = (
                time
                if time.kind == "point"
                else IntervalEventTime(
                    start_s=max(time.start_s, segment.start_s), end_s=min(time.end_s, segment.end_s)
                )
            )
            first, end = _frame_range(portion, clip.fps)
            if first < segment.frame_start or end > segment.frame_end_exclusive:
                return None, "support_outside_segment_frame_range"
    reason = _observations(time, ctx.tracks.get((clip_id, track_id)), clip)
    return (None, reason) if reason else (next(iter(players)), None)


def associate_event(
    event: Event,
    *,
    assignments: Iterable[Assignment],
    segments: Iterable[Segment],
    tracks: Iterable[Track] = (),
    clips: Iterable[Clip] = (),
) -> ResolvedEvent:
    """Resolve every subject independently using current confirmed support.

    A reference's explicit point/interval always wins over the event time.
    Cross-clip fallback requires an explicit shared source-video mapping.
    Actual observation coverage is required; no nearest-track interpolation is
    performed when frames or their time mapping are unknown.
    """
    assignments, segments, tracks, clips = (
        list(assignments),
        list(segments),
        list(tracks),
        list(clips),
    )
    if any(r.match_id != event.match_id for r in [*assignments, *segments, *tracks, *clips]):
        raise ValueError("event association context must belong to event.match_id")
    for records, key in (
        (clips, lambda item: item.clip_id),
        (tracks, lambda item: (item.clip_id, item.track_id)),
        (segments, lambda item: item.segment_id),
        (assignments, lambda item: item.segment_id),
    ):
        if len({key(record) for record in records}) != len(records):
            raise ValueError("duplicate records in event association context")
    ctx = _Context(
        {c.clip_id: c for c in clips},
        {(t.clip_id, t.track_id): t for t in tracks},
        {s.segment_id: s for s in segments},
        {a.segment_id: a for a in assignments},
    )
    clip = ctx.clips.get(event.clip_id)
    validation_reasons = []
    if clip and clip.duration_s is not None and not time_within(event.time, 0, clip.duration_s):
        validation_reasons.append("event_outside_clip")
    subjects = []
    for subject in event.subjects:
        players, reasons = set(), list(validation_reasons)
        for reference in subject.references:
            player, reason = _resolve_reference(event, reference, ctx)
            if reason:
                reasons.append(reason)
            else:
                players.add(player)
        if not subject.references:
            reasons.append("subject_has_no_support")
        if len(players) > 1:
            reasons.append("subject_supports_disagree")
        if reasons:
            subjects.append(
                ResolvedSubject(
                    subject_id=subject.subject_id,
                    role=subject.role,
                    status="unmapped",
                    reason=";".join(dict.fromkeys(reasons)),
                )
            )
        else:
            subjects.append(
                ResolvedSubject(
                    subject_id=subject.subject_id,
                    role=subject.role,
                    status="mapped",
                    player_id=next(iter(players)),
                )
            )
    source_time = None
    clip = ctx.clips.get(event.clip_id)
    if clip and clip.source_mapping:
        mapping = clip.source_mapping
        source_time = SourceEventTime(
            source_video_id=mapping.source_video_id,
            time=_affine_time(event.time, mapping.source_start_s, mapping.time_scale),
        )
    return ResolvedEvent(
        event=event,
        subjects=subjects,
        source_time=source_time,
        validation_reasons=validation_reasons,
    )


def associate_events(events: Iterable[Event], **context) -> list[ResolvedEvent]:
    """Resolve a set of events using exactly the supplied fixed context."""
    context = {key: list(values) for key, values in context.items()}
    return [associate_event(event, **context) for event in events]
