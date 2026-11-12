"""Pure validation of event facts against the context currently available.

Missing targets are valid late references. Identity assignment and observation
coverage affect association, never the supplied outcome or original event.
"""

from collections.abc import Iterable

from app.contracts.events import Event, EventTime
from app.contracts.evidence import Clip, Track
from app.contracts.identity import Segment


def time_within(time: EventTime, start_s: float, end_s: float) -> bool:
    """Whether a point or entire half-open interval is within a range."""
    if time.kind == "point":
        return start_s <= time.time_s < end_s
    return start_s <= time.start_s and time.end_s <= end_s


def validate_events(
    events: Iterable[Event],
    *,
    match_id: str,
    clips: Iterable[Clip] = (),
    tracks: Iterable[Track] = (),
    segments: Iterable[Segment] = (),
) -> list[Event]:
    """Reject known scope/range contradictions, preserving unknown references.

    An omitted reference time is resolved later; it must not make an original
    event invalid merely because its current identity support is unavailable.
    No result (including ``made``) is inferred from the event type or scores.
    """
    records = list(events)
    clip_records, track_records, segment_records = list(clips), list(tracks), list(segments)
    if any(
        r.match_id != match_id for r in [*records, *clip_records, *track_records, *segment_records]
    ):
        raise ValueError("event context must belong to one match")
    if len({event.event_id for event in records}) != len(records):
        raise ValueError("duplicate event_id")
    clip_map = {clip.clip_id: clip for clip in clip_records}
    track_map = {(track.clip_id, track.track_id): track for track in track_records}
    segment_map = {segment.segment_id: segment for segment in segment_records}
    for event in records:
        clip = clip_map.get(event.clip_id)
        if clip and clip.duration_s is not None and not time_within(event.time, 0, clip.duration_s):
            raise ValueError(f"event {event.event_id!r} is outside its clip")
        for subject in event.subjects:
            for reference in subject.references:
                if reference.time is None:
                    continue
                if reference.kind == "track":
                    target = track_map.get((reference.clip_id, reference.track_id))
                    target_clip = clip_map.get(reference.clip_id)
                else:
                    target = segment_map.get(reference.segment_id)
                    target_clip = clip_map.get(target.clip_id) if target else None
                if target and not time_within(reference.time, target.start_s, target.end_s):
                    raise ValueError(f"event {event.event_id!r} reference time is outside target")
                if (
                    target_clip
                    and target_clip.duration_s is not None
                    and not time_within(reference.time, 0, target_clip.duration_s)
                ):
                    raise ValueError(
                        f"event {event.event_id!r} reference time is outside target clip"
                    )
    return records
