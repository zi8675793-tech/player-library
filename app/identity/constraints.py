"""Physical-frame hard exclusion for T1 segments and legacy overlap safety gates.

T1 segments use exact source/frame cooccurrence, including overlapping boxes.
The legacy representation API retains its persistent separation evidence policy.
Unresolved metadata blocks automatic merges and matches in either mode; absence
of a hard conflict is not positive evidence of identity.
"""

import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from app.contracts.evidence import TrackObservation
from app.features.dual_stream import TrackletSegment
from app.features.dual_stream import TrackObservation as DualStreamObservation
from app.identity.representation import TrackRepresentation


@dataclass(frozen=True)
class CannotLinkConstraint:
    """Frozen T2 contract; segment identifiers denote an unordered hard edge."""

    segment_a: str
    segment_b: str
    reason: str


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _validate_dual_stream_segment(segment: TrackletSegment) -> None:
    """Validate physical frame metadata without opening image or video files."""
    if not isinstance(segment, TrackletSegment):
        raise TypeError("Expected the documented T1 TrackletSegment")
    if (
        not isinstance(segment.video_id, str)
        or re.fullmatch(r"[A-Za-z0-9_-]+", segment.video_id) is None
    ):
        raise ValueError("video_id must be a nonempty source identifier")
    for name in ("local_track_id", "start_frame", "end_frame", "observation_count"):
        value = getattr(segment, name)
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if segment.segment_id != (f"{segment.video_id}_{segment.local_track_id}_{segment.start_frame}"):
        raise ValueError("segment_id must agree with video, track and start frame")
    if not isinstance(segment.observations, tuple) or not segment.observations:
        raise ValueError("TrackletSegment must contain a nonempty observation tuple")
    if segment.observation_count != len(segment.observations):
        raise ValueError("observation_count disagrees with observations")
    for observation in segment.observations:
        if not isinstance(observation, DualStreamObservation):
            raise TypeError("Expected the documented T1 TrackObservation")
        if (
            observation.video_id != segment.video_id
            or type(observation.local_track_id) is not int
            or observation.local_track_id != segment.local_track_id
        ):
            raise ValueError("Observation source and track must agree with its segment")
        if type(observation.frame_idx) is not int or observation.frame_idx < 0:
            raise ValueError("Observation frame_idx must be a nonnegative integer")
        if not _finite_number(observation.timestamp_s) or observation.timestamp_s < 0:
            raise ValueError("Observation timestamp_s must be finite and nonnegative")
        if not _finite_number(observation.confidence) or not 0 <= observation.confidence <= 1:
            raise ValueError("Observation confidence must be finite and in [0, 1]")
        box = observation.bbox_xyxy
        if (
            not isinstance(box, tuple)
            or len(box) != 4
            or not all(_finite_number(value) for value in box)
            or not (0 <= box[0] < box[2] and 0 <= box[1] < box[3])
        ):
            raise ValueError("Observation bbox_xyxy must be a finite ordered nonnegative box")
    observations = sorted(segment.observations, key=lambda observation: observation.frame_idx)
    if any(
        left.frame_idx == right.frame_idx or left.timestamp_s >= right.timestamp_s
        for left, right in zip(observations, observations[1:])
    ):
        raise ValueError("Observation frames and timestamps must strictly increase")
    first, last = observations[0], observations[-1]
    for name, expected in (
        ("start_frame", first.frame_idx),
        ("end_frame", last.frame_idx),
        ("start_time_s", first.timestamp_s),
        ("end_time_s", last.timestamp_s),
    ):
        value = getattr(segment, name)
        if not _finite_number(value) or value != expected:
            raise ValueError(f"TrackletSegment {name} disagrees with observations")


class ForbiddenPairs(set[tuple[int, int]]):
    """Set of unordered pairs (i, j) of conflicting track indices."""

    def __contains__(self, item: object) -> bool:
        if isinstance(item, tuple) and len(item) == 2:
            a, b = item
            if isinstance(a, int) and isinstance(b, int):
                if a == b:
                    return False
                return super().__contains__((min(a, b), max(a, b)))
        return super().__contains__(item)

    def add(self, item: tuple[int, int]) -> None:
        if isinstance(item, tuple) and len(item) == 2:
            a, b = item
            if isinstance(a, int) and isinstance(b, int):
                if a == b:
                    return
                super().add((min(a, b), max(a, b)))
                return
        super().add(item)


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _reliable_observation(obs: Any) -> bool:
    # TrackObservation currently has no quality field; adapters may supply one.
    for quality in (obs, _field(obs, "tracking_quality"), _field(obs, "quality")):
        if quality is not None and (
            _field(quality, "occluded") is True
            or _field(quality, "mixed_identity") is True
            or _field(quality, "usability") == "unusable"
            or _field(quality, "clean_review") == "rejected"
        ):
            return False
    try:
        time_s = float(_field(obs, "time_s"))
        x1, y1, x2, y2 = map(float, _field(obs, "bbox_xyxy"))
    except (TypeError, ValueError):
        return False
    return (
        all(math.isfinite(v) for v in (time_s, x1, y1, x2, y2))
        and time_s >= 0
        and x1 < x2
        and y1 < y2
    )


class SpatioTemporalConstraintEngine:
    """Separate hard spatial/manual conflicts from unresolved overlap blockers.

    T1 ``segments`` use exact physical frame cooccurrence: different local
    tracks in one video and frame are hard cannot-links, regardless of box
    overlap. Frame-index gaps are never filled by interval interpolation.

    The legacy TrackRepresentation API retains its persistent geometry policy:
    disjoint boxes in distinct time buckets establish conflict; all other time
    overlap remains unresolved. Mixing these two input contracts is rejected.
    """

    def __init__(
        self,
        tracks: Sequence[TrackRepresentation | TrackletSegment] = (),
        observations: Sequence[TrackObservation | Any] | None = None,
        manual_constraints: Sequence[Any] | None = None,
        player_assignments: Any | None = None,
        segments_by_id: Any | None = None,
        *,
        bucket_s: float = 0.5,
        cooccurrence_min_buckets: int = 3,
        cooccurrence_separation_heights: float = 0.5,
        segments: Sequence[TrackletSegment] | None = None,
    ) -> None:
        tracks = list(tracks)
        if segments is not None and tracks:
            raise ValueError("Supply either tracks or T1 segments, not both")
        dual_stream_tracks = [isinstance(track, TrackletSegment) for track in tracks]
        if any(dual_stream_tracks) and not all(dual_stream_tracks):
            raise TypeError("Cannot mix T1 TrackletSegment and legacy track representations")
        self._dual_stream_mode = segments is not None or any(dual_stream_tracks)
        if self._dual_stream_mode:
            if observations:
                raise ValueError("T1 segments carry their own physical observations")
            self._init_dual_stream(
                list(segments) if segments is not None else tracks,
                manual_constraints,
                player_assignments,
                segments_by_id,
            )
            return
        self.constraints: tuple[CannotLinkConstraint, ...] = ()
        for name, value in (
            ("bucket_s", bucket_s),
            ("cooccurrence_separation_heights", cooccurrence_separation_heights),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number")
        if (
            isinstance(cooccurrence_min_buckets, bool)
            or not isinstance(cooccurrence_min_buckets, int)
            or cooccurrence_min_buckets < 3
        ):
            raise ValueError("cooccurrence_min_buckets must be an integer >= 3")
        self.bucket_s = bucket_s
        self.cooccurrence_min_buckets = cooccurrence_min_buckets
        self.cooccurrence_separation_heights = cooccurrence_separation_heights
        self.tracks: list[TrackRepresentation] = list(tracks)
        self.observations: list[Any] = list(observations) if observations is not None else []
        self.manual_constraints: list[Any] = list(manual_constraints or [])
        self.player_assignments: dict[str, str] = dict(player_assignments or {})
        self.segments_by_id: dict[str, Any] = {
            t.segment_id: t for t in self.tracks if t.segment_id is not None
        }
        self.segments_by_id.update(segments_by_id or {})
        self._observations: dict[tuple[str, str], list[Any]] = defaultdict(list)
        for obs in self.observations:
            if _reliable_observation(obs):
                key = (str(_field(obs, "clip_id")), str(_field(obs, "track_id")))
                self._observations[key].append(obs)
        self.forbidden_pairs: ForbiddenPairs = ForbiddenPairs()
        self._unresolved_pairs: ForbiddenPairs = ForbiddenPairs()
        for i, left in enumerate(self.tracks):
            for j in range(i + 1, len(self.tracks)):
                relation = self.pair_relation(left, self.tracks[j])
                if relation == "conflict":
                    self.forbidden_pairs.add((i, j))
                elif relation == "unresolved":
                    self._unresolved_pairs.add((i, j))

    @classmethod
    def from_segments(
        cls, segments: Sequence[TrackletSegment], **kwargs: Any
    ) -> "SpatioTemporalConstraintEngine":
        """Build the T2 physical-frame graph, including for an empty batch."""
        return cls(segments=segments, **kwargs)

    def with_player_assignments(
        self, player_assignments: Mapping[str, str] | None
    ) -> "SpatioTemporalConstraintEngine":
        """Return a fast shallow copy with updated player assignments."""
        cloned = object.__new__(self.__class__)
        cloned.__dict__.update(self.__dict__)
        cloned.player_assignments = dict(player_assignments or {})
        return cloned

    def _init_dual_stream(
        self,
        segments: Sequence[TrackletSegment],
        manual_constraints: Sequence[Any] | None,
        player_assignments: Any,
        segments_by_id: Any,
    ) -> None:
        self.tracks = list(segments)
        self.manual_constraints = list(manual_constraints or ())
        self.player_assignments = dict(player_assignments or {})
        self.segments_by_id = {}
        for segment in segments:
            _validate_dual_stream_segment(segment)
            if segment.segment_id in self.segments_by_id:
                raise ValueError(f"Duplicate segment_id: {segment.segment_id}")
            self.segments_by_id[segment.segment_id] = segment
        for segment_id, segment in dict(segments_by_id or {}).items():
            _validate_dual_stream_segment(segment)
            if segment_id != segment.segment_id:
                raise ValueError("segments_by_id keys must agree with segment identifiers")
            previous = self.segments_by_id.get(segment_id)
            if previous is not None and previous != segment:
                raise ValueError(f"Conflicting segment metadata: {segment_id}")
            self.segments_by_id[segment_id] = segment

        by_frame: dict[tuple[str, int], dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
        self.observations = []
        for segment in self.segments_by_id.values():
            for observation in segment.observations:
                self.observations.append(observation)
                by_frame[segment.video_id, observation.frame_idx][segment.local_track_id].add(
                    segment.segment_id
                )
        edges: dict[tuple[str, str], CannotLinkConstraint] = {}
        for occupants in by_frame.values():
            tracks = list(occupants.values())
            for index, left in enumerate(tracks):
                for right in tracks[index + 1 :]:
                    for segment_a in left:
                        for segment_b in right:
                            a, b = sorted((segment_a, segment_b))
                            edges[a, b] = CannotLinkConstraint(
                                a, b, "same_frame_spatial_separation"
                            )
        for constraint in self.manual_constraints:
            if isinstance(constraint, CannotLinkConstraint):
                a, b = constraint.segment_a, constraint.segment_b
                reason = constraint.reason
            elif _field(constraint, "kind") == "cannot_link":
                a, b = _field(constraint, "segment_id"), _field(constraint, "related_segment_id")
                reason = "manual_cannot_link"
            else:
                continue
            if isinstance(a, str) and isinstance(b, str) and a != b:
                a, b = sorted((a, b))
                edges[a, b] = CannotLinkConstraint(a, b, reason)
        self.constraints = tuple(edges[key] for key in sorted(edges))
        self._segment_edges = frozenset(edges)
        self.forbidden_pairs = ForbiddenPairs()
        self._unresolved_pairs = ForbiddenPairs()
        indices = {segment.segment_id: index for index, segment in enumerate(self.tracks)}
        for a, b in edges:
            if a in indices and b in indices:
                self.forbidden_pairs.add((indices[a], indices[b]))

    def cannot_link(self, segment_a: str | Any, segment_b: str | Any) -> bool:
        """Query an unordered hard edge in constant time for T1 segment IDs."""
        return self.pair_relation(segment_a, segment_b) == "conflict"

    def constrained_distance(
        self, segment_a: str | Any, segment_b: str | Any, distance: float
    ) -> float:
        """Preserve distance only for physically clear pairs; fail closed otherwise."""
        return distance if self.pair_relation(segment_a, segment_b) == "clear" else math.inf

    def _segment(self, segment: Any) -> Any:
        if isinstance(segment, str):
            return self.segments_by_id.get(segment)
        return self.segments_by_id.get(_field(segment, "segment_id"), segment)

    def _interval(self, segment: Any) -> tuple[float, float]:
        start, end = float(_field(segment, "start_s")), float(_field(segment, "end_s"))
        # Unsegmented pooled representations carry sample extrema, not declared
        # Segment bounds. Preserve their conservative raw-observation envelope.
        if isinstance(segment, TrackRepresentation) and segment.segment_id is None:
            observations = self._observations.get((str(segment.clip_id), str(segment.track_id)), [])
            for obs in observations:
                time_s = float(_field(obs, "time_s"))
                start, end = min(start, time_s), max(end, time_s)
        return start, end

    def _segment_observations(self, segment: Any, start: float, end: float) -> list[Any]:
        key = (str(_field(segment, "clip_id")), str(_field(segment, "track_id")))
        result = []
        for obs in self._observations.get(key, []):
            if not start <= float(_field(obs, "time_s")) < end:
                continue
            if any(
                _field(segment, name) is not None
                and _field(obs, name) is not None
                and _field(segment, name) != _field(obs, name)
                for name in ("match_id", "segment_id")
            ):
                continue
            frame, frame_start = _field(obs, "frame_idx"), _field(segment, "frame_start")
            if frame is not None and frame_start is not None:
                if not frame_start <= frame < _field(segment, "frame_end_exclusive"):
                    continue
            result.append(obs)
        return result

    def pair_relation(
        self, segment_a: Any, segment_b: Any
    ) -> Literal["conflict", "unresolved", "clear"]:
        """Classify Segment/representation objects or exact segment IDs.

        Manual cannot-links override geometry, including same-track/different-clip
        exemptions. Missing segment metadata fails closed as unresolved. 'clear'
        means no spatial blocker, not proof that the segments are the same person.
        """
        if self._dual_stream_mode:
            sid_a = segment_a if isinstance(segment_a, str) else _field(segment_a, "segment_id")
            sid_b = segment_b if isinstance(segment_b, str) else _field(segment_b, "segment_id")
            if not isinstance(sid_a, str) or not isinstance(sid_b, str):
                return "unresolved"
            if tuple(sorted((sid_a, sid_b))) in self._segment_edges:
                return "conflict"
            if sid_a not in self.segments_by_id or sid_b not in self.segments_by_id:
                return "unresolved"
            return "clear"
        left, right = self._segment(segment_a), self._segment(segment_b)
        sid_a = segment_a if isinstance(segment_a, str) else _field(segment_a, "segment_id")
        sid_b = segment_b if isinstance(segment_b, str) else _field(segment_b, "segment_id")
        match_a, match_b = _field(left, "match_id"), _field(right, "match_id")
        if match_a is not None and match_b is not None and match_a != match_b:
            return "clear"
        if sid_a is not None and sid_a == sid_b:
            return "clear"
        for constraint in self.manual_constraints:
            match_id = _field(constraint, "match_id")
            if match_id is not None and any(
                match is not None and match != match_id for match in (match_a, match_b)
            ):
                continue
            if (
                _field(constraint, "kind") == "cannot_link"
                and sid_a is not None
                and sid_b is not None
            ):
                if {sid_a, sid_b} == {
                    _field(constraint, "segment_id"),
                    _field(constraint, "related_segment_id"),
                }:
                    return "conflict"
        if left is None or right is None:
            return "unresolved"
        if _field(left, "clip_id") != _field(right, "clip_id"):
            return "clear"
        if str(_field(left, "track_id")) == str(_field(right, "track_id")):
            return "clear"
        start_a, end_a = self._interval(left)
        start_b, end_b = self._interval(right)
        start, end = max(start_a, start_b), min(end_a, end_b)
        if start >= end:
            return "clear"

        by_frame, by_time = defaultdict(list), defaultdict(list)
        for obs in self._segment_observations(right, start, end):
            frame = _field(obs, "frame_idx")
            if frame is not None:
                by_frame[frame].append(obs)
            by_time[float(_field(obs, "time_s"))].append(obs)
        witness_times: dict[tuple[str, int | float], float] = {}
        for x in self._segment_observations(left, start, end):
            frame, time_s = _field(x, "frame_idx"), float(_field(x, "time_s"))
            # Prefer shared frame IDs; use exact times if either frame is absent.
            candidates = list(by_frame.get(frame, [])) if frame is not None else []
            candidates.extend(
                y
                for y in by_time.get(time_s, [])
                if frame is None or _field(y, "frame_idx") is None
            )
            for y in candidates:
                x0, t, r, d = map(float, _field(x, "bbox_xyxy"))
                u, v, w, z = map(float, _field(y, "bbox_xyxy"))
                disjoint = r < u or w < x0 or d < v or z < t
                height = ((d - t) + (z - v)) / 2
                separation = math.hypot((x0 + r - u - w) / 2, (t + d - v - z) / 2) / height
                if disjoint and separation >= self.cooccurrence_separation_heights:
                    # Count one event per frame (or exact time), independent of
                    # observation order, duplicate records and argument order.
                    event_frame = frame if frame is not None else _field(y, "frame_idx")
                    key = ("frame", event_frame) if event_frame is not None else ("time", time_s)
                    time = min(time_s, float(_field(y, "time_s")))
                    witness_times[key] = min(witness_times.get(key, time), time)
        buckets = {math.floor(time / self.bucket_s) for time in witness_times.values()}
        return "conflict" if len(buckets) >= self.cooccurrence_min_buckets else "unresolved"

    def is_forbidden(self, idx_a: int, idx_b: int) -> bool:
        """Return only hard geometry/manual conflicts, never unresolved overlaps."""
        return (idx_a, idx_b) in self.forbidden_pairs

    def is_unresolved(self, idx_a: int, idx_b: int) -> bool:
        """Return whether overlap lacks reliable different-person evidence."""
        return not self.is_forbidden(idx_a, idx_b) and (idx_a, idx_b) in self._unresolved_pairs

    def can_merge_clusters(
        self,
        cluster_a: set[int] | Iterable[int],
        cluster_b: set[int] | Iterable[int],
    ) -> bool:
        """Require every cross-cluster pair to be clear, including for iterators."""
        right = tuple(cluster_b)
        return all(
            not self.is_forbidden(a, b) and not self.is_unresolved(a, b)
            for a in cluster_a
            for b in right
        )

    def is_segment_forbidden_for_player(self, segment_id: str, player_id: str) -> bool:
        """Block matches on hard conflicts, manual bans or unresolved overlaps."""
        for constraint in self.manual_constraints:
            if _field(constraint, "segment_id") == segment_id and (
                _field(constraint, "kind") == "block_automatic"
                or (
                    _field(constraint, "kind") == "forbid_player"
                    and _field(constraint, "player_id") == player_id
                )
            ):
                return True
        for assigned_sid, assigned_pid in self.player_assignments.items():
            if assigned_pid != player_id or assigned_sid == segment_id:
                continue
            if self.pair_relation(segment_id, assigned_sid) != "clear":
                return True
            # Preserve explicit additions to the public hard/manual pair set.
            if any(
                self.is_forbidden(i, j)
                for i, t in enumerate(self.tracks)
                if t.segment_id == segment_id
                for j, other in enumerate(self.tracks)
                if other.segment_id == assigned_sid
            ):
                return True
        return False
