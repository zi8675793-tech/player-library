"""T2: pure Revision-R matching, hard exclusions and conservative discovery.

The T1 dataclasses are imported from their existing public module. The remaining
frozen documentation contracts are defined here, with their exact fields, because
they are not yet present in app/contracts and this task cannot modify that layer.

Call IncrementalMatcher(...).plan(features, profiles, segments=all_segments,
batch_id="batch-1", from_revision=R, pending_features=previously_deferred,
player_assignments=historical_assignments). Supply actual T1 segment metadata for
queries, pending evidence, prototypes and every historically assigned segment.
Missing history blocks that player; source/frame identity is never guessed from
an ID. No files, models, labels or database connections are opened here.

Distances are cosine distances in [0, 2]. Prototype weight and image quality are
admission gates, not multipliers that could make a weak prototype artificially
close. A matching number never bypasses appearance, margin or physical safety.
Threshold defaults are conservative configuration, not benchmark calibration.

Handoff to T3: persist the returned plan atomically against from_revision;
new profiles/prototypes belong to to_revision. CREATED_NEW keeps target_player_id
None as specified; evidence['new_player_id'] links it to plan.new_players.
Pass deferred features back explicitly on subsequent calls. Updated prototypes
are additions for existing players, subject to the configured capacity. Since T1
has no pose/view classifier, all newly proposed prototypes use 'generic'.
Validation: uv run pytest tests/test_constraints_and_matching.py;
uv run ruff check .; uv run ruff format --check .
"""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import numpy as np

from app.features.dual_stream import (
    AppearanceEmbedding,
    DualStreamFeature,
    JerseyNumberPrediction,
    TrackletSegment,
)
from app.identity.constraints import SpatioTemporalConstraintEngine

_EPS = 1e-7  # Float32 embedding precision at inclusive distance/margin boundaries.
_VIEWS = {"frontal", "dorsal_with_number", "lateral", "generic"}
_COMPATIBLE_APPEARANCE_MAX_DISTANCE = (
    0.45  # Upper bound for compatible appearance to activate jersey number bonus.
)
_MATCHING_NUMBER_DISTANCE_FACTOR = (
    0.5  # Positive matching discount when confirmed jersey numbers agree.
)


@dataclass(frozen=True)
class PlayerPrototype:
    prototype_id: str
    player_id: str
    view_type: str
    feature: DualStreamFeature
    weight: float
    created_at_revision: int


@dataclass(frozen=True)
class PlayerProfile:
    player_id: str
    display_name: str
    confirmed_jersey_number: int | None
    prototypes: tuple[PlayerPrototype, ...]
    status: str
    created_at_revision: int


class DecisionOutcome(str, Enum):  # noqa: UP042 - preserve the frozen contract exactly.
    MATCHED_EXISTING = "matched_existing"
    CREATED_NEW = "created_new_identity"
    INSUFFICIENT = "insufficient_evidence"


@dataclass(frozen=True)
class IncrementalDecision:
    segment_id: str
    outcome: DecisionOutcome
    target_player_id: str | None
    best_distance: float
    competitive_margin: float
    evidence: dict[str, Any]


@dataclass(frozen=True)
class IncrementalPlan:
    batch_id: str
    from_revision: int
    to_revision: int
    decisions: tuple[IncrementalDecision, ...]
    new_players: tuple[PlayerProfile, ...]
    updated_prototypes: tuple[PlayerPrototype, ...]
    deferred_segments: tuple[str, ...]


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _revision(value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("revision must be a nonnegative integer")


def _finite(value: float, name: str, low: float, high: float) -> None:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(value)
        or not low <= value <= high
    ):
        raise ValueError(f"{name} must be finite and in [{low}, {high}]")


def _number(value: int | None) -> None:
    if value is not None and (type(value) is not int or not 0 <= value <= 99):
        raise ValueError("jersey number must be None or an integer in [0, 99]")


def _jersey(feature: DualStreamFeature) -> int | None:
    return feature.jersey.number if feature.jersey is not None else None


def _validate_feature(feature: DualStreamFeature) -> None:
    if not isinstance(feature, DualStreamFeature):
        raise TypeError("Expected the documented T1 DualStreamFeature")
    _text(feature.segment_id, "segment_id")
    appearance = feature.appearance
    if not isinstance(appearance, AppearanceEmbedding):
        raise TypeError("Expected the documented T1 AppearanceEmbedding")
    vector = appearance.vector
    if (
        not isinstance(vector, np.ndarray)
        or vector.shape != (512,)
        or vector.dtype != np.float32
        or not np.isfinite(vector).all()
        or not np.isclose(np.linalg.norm(vector.astype(np.float64)), 1, rtol=1e-4, atol=1e-5)
    ):
        raise ValueError("appearance must be a finite L2-normalized float32 (512,) vector")
    _finite(appearance.quality_score, "quality_score", 0, 1)
    _text(appearance.model_name, "model_name")
    prediction = feature.jersey
    if prediction is not None:
        if not isinstance(prediction, JerseyNumberPrediction):
            raise TypeError("Expected the documented T1 JerseyNumberPrediction")
        _number(prediction.number)
        _finite(prediction.confidence, "jersey confidence", 0, 1)
        if type(prediction.legible) is not bool:
            raise ValueError("jersey legible must be bool")
        if prediction.number is not None and not prediction.legible:
            raise ValueError("A predicted number must be legible")
        if not isinstance(prediction.votes, dict):
            raise ValueError("jersey votes must be a dictionary")
        for number, score in prediction.votes.items():
            _number(number)
            if number is None:
                raise ValueError("jersey vote keys must be integers")
            _finite(score, "jersey vote", 0, math.inf)
        if prediction.raw_text is not None and not isinstance(prediction.raw_text, str):
            raise ValueError("raw_text must be a string or None")


def _copy_feature(feature: DualStreamFeature) -> DualStreamFeature:
    """Own proposed prototype data without changing the caller's numpy flags."""
    vector = feature.appearance.vector.copy()
    vector.setflags(write=False)
    jersey = feature.jersey
    if jersey is not None:
        jersey = replace(jersey, votes=dict(jersey.votes))
    return replace(feature, jersey=jersey, appearance=replace(feature.appearance, vector=vector))


def _cosine(left: DualStreamFeature, right: DualStreamFeature) -> float:
    a = left.appearance.vector.astype(np.float64)
    b = right.appearance.vector.astype(np.float64)
    similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.clip(1 - similarity, 0, 2))


def _stable_id(kind: str, *parts: Any) -> str:
    return str(uuid5(NAMESPACE_URL, json.dumps(["basketball-t2", kind, *parts])))


class IncrementalMatcher:
    """Plan once against a fixed gallery; never admit a new ID mid-batch.

    Pending proximity components must also satisfy all-pairs compatibility. An
    ambiguous bridge is deferred as a whole rather than arbitrarily attaching it
    to either side. Independent witnesses mean distinct video_id values.
    """

    def __init__(
        self,
        *,
        distance_threshold: float = 0.25,
        competitive_margin: float = 0.04,
        min_quality: float = 0.5,
        min_prototype_weight: float = 0.1,
        max_prototypes: int = 6,
    ) -> None:
        _finite(distance_threshold, "distance_threshold", 0, 2)
        _finite(competitive_margin, "competitive_margin", 0.04, 2)
        _finite(min_quality, "min_quality", 0, 1)
        _finite(min_prototype_weight, "min_prototype_weight", 0, 1)
        if type(max_prototypes) is not int or max_prototypes < 2:
            raise ValueError("max_prototypes must be an integer >= 2")
        self.distance_threshold = float(distance_threshold)
        self.competitive_margin = float(competitive_margin)
        self.min_quality = float(min_quality)
        self.min_prototype_weight = float(min_prototype_weight)
        self.max_prototypes = max_prototypes

    def _profiles(self, players: Sequence[PlayerProfile], revision: int) -> list[PlayerProfile]:
        result = []
        player_ids: set[str] = set()
        prototype_ids: set[str] = set()
        for player in players:
            if not isinstance(player, PlayerProfile):
                raise TypeError("Expected PlayerProfile")
            _text(player.player_id, "player_id")
            if player.player_id in player_ids:
                raise ValueError("Duplicate player_id")
            player_ids.add(player.player_id)
            if player.status not in {"active", "merged", "archived"}:
                raise ValueError("Unknown player status")
            _revision(player.created_at_revision)
            if player.created_at_revision > revision:
                raise ValueError("Player is newer than the Revision R snapshot")
            _text(player.display_name, "display_name")
            _number(player.confirmed_jersey_number)
            if not isinstance(player.prototypes, tuple):
                raise ValueError("prototypes must be a tuple")
            numbers = {player.confirmed_jersey_number} - {None}
            for prototype in player.prototypes:
                if not isinstance(prototype, PlayerPrototype):
                    raise TypeError("Expected PlayerPrototype")
                _text(prototype.prototype_id, "prototype_id")
                if prototype.prototype_id in prototype_ids:
                    raise ValueError("Duplicate prototype_id")
                prototype_ids.add(prototype.prototype_id)
                if prototype.player_id != player.player_id or prototype.view_type not in _VIEWS:
                    raise ValueError("Invalid prototype owner or view_type")
                _revision(prototype.created_at_revision)
                if not player.created_at_revision <= prototype.created_at_revision <= revision:
                    raise ValueError("Prototype is outside the Revision R snapshot")
                _finite(prototype.weight, "prototype weight", 0, 1)
                _validate_feature(prototype.feature)
                if _jersey(prototype.feature) is not None:
                    numbers.add(_jersey(prototype.feature))
            if len(numbers) > 1:
                raise ValueError("Conflicting jersey numbers within one PlayerProfile")
            if player.status == "active":
                result.append(player)
        return sorted(result, key=lambda player: player.player_id)

    def _pair_distance(
        self,
        left: DualStreamFeature,
        right: DualStreamFeature,
        engine: SpatioTemporalConstraintEngine,
    ) -> float:
        a, b = _jersey(left), _jersey(right)
        if a is not None and b is not None and a != b:
            return math.inf
        if (
            left.appearance.model_name != right.appearance.model_name
            or min(left.appearance.quality_score, right.appearance.quality_score) < self.min_quality
            or engine.pair_relation(left.segment_id, right.segment_id) != "clear"
        ):
            return math.inf
        raw_dist = _cosine(left, right)
        if (
            a is not None
            and b is not None
            and a == b
            and raw_dist <= _COMPATIBLE_APPEARANCE_MAX_DISTANCE
        ):
            return raw_dist * _MATCHING_NUMBER_DISTANCE_FACTOR
        return raw_dist

    def _player_score(
        self,
        query: DualStreamFeature,
        player: PlayerProfile,
        engine: SpatioTemporalConstraintEngine,
    ) -> tuple[float, str | None]:
        known = {player.confirmed_jersey_number} - {None}
        known.update(_jersey(proto.feature) for proto in player.prototypes)
        known.discard(None)
        if _jersey(query) is not None and known and _jersey(query) not in known:
            return math.inf, "jersey_conflict"
        # Check all historical assignments AND all prototypes, not only the best.
        history = {sid for sid, pid in engine.player_assignments.items() if pid == player.player_id}
        history.update(proto.feature.segment_id for proto in player.prototypes)
        relations = [
            engine.pair_relation(query.segment_id, sid)
            for sid in sorted(history)
            if sid != query.segment_id
        ]
        if "conflict" in relations:
            return math.inf, "cannot_link"
        if "unresolved" in relations:
            return math.inf, "missing_history"
        query_number = _jersey(query)
        match_number = query_number is not None and (
            player.confirmed_jersey_number == query_number
            or (player.confirmed_jersey_number is None and known == {query_number})
        )
        distances = []
        for proto in player.prototypes:
            if (
                proto.feature.segment_id != query.segment_id
                and proto.weight > 0
                and proto.weight >= self.min_prototype_weight
            ):
                dist = self._pair_distance(query, proto.feature, engine)
                if (
                    match_number
                    and _jersey(proto.feature) is None
                    and math.isfinite(dist)
                    and dist <= _COMPATIBLE_APPEARANCE_MAX_DISTANCE
                ):
                    dist = dist * _MATCHING_NUMBER_DISTANCE_FACTOR
                distances.append(dist)
        best = min(distances, default=math.inf)
        return best, None if math.isfinite(best) else "no_comparable_prototype"

    def plan(
        self,
        features: Sequence[DualStreamFeature],
        players: Sequence[PlayerProfile],
        *,
        segments: Sequence[TrackletSegment],
        batch_id: str,
        from_revision: int,
        pending_features: Sequence[DualStreamFeature] = (),
        player_assignments: Mapping[str, str] | None = None,
    ) -> IncrementalPlan:
        """Return a deterministic, in-memory plan; duplicate evidence is rejected."""
        _text(batch_id, "batch_id")
        _revision(from_revision)
        profiles = self._profiles(players, from_revision)
        queries: dict[str, DualStreamFeature] = {}
        for feature in (*features, *pending_features):
            _validate_feature(feature)
            if feature.segment_id in queries:
                raise ValueError("Duplicate segment_id in candidates/pending evidence")
            queries[feature.segment_id] = _copy_feature(feature)
        assignments = dict(player_assignments or {})
        for sid, pid in assignments.items():
            _text(sid, "assignment segment_id")
            _text(pid, "assignment player_id")
        engine = SpatioTemporalConstraintEngine(segments=segments, player_assignments=assignments)
        if not queries.keys() <= engine.segments_by_id.keys():
            raise ValueError("Missing segment metadata for a candidate/pending feature")
        decisions: dict[str, IncrementalDecision] = {}
        eligible: set[str] = set()
        for sid, query in sorted(queries.items()):
            scores = {
                player.player_id: self._player_score(query, player, engine) for player in profiles
            }
            ranked = sorted(
                (distance, pid) for pid, (distance, _) in scores.items() if math.isfinite(distance)
            )
            best = ranked[0][0] if ranked else math.inf
            margin = (ranked[1][0] - best if len(ranked) > 1 else math.inf) if ranked else 0.0
            quality_ok = query.appearance.quality_score >= self.min_quality
            matched = (
                quality_ok
                and best <= self.distance_threshold + _EPS
                and margin + _EPS >= self.competitive_margin
            )
            evidence: dict[str, Any] = {
                "snapshot_revision": from_revision,
                "distance_metric": "cosine",
                "distance_threshold": self.distance_threshold,
                "required_margin": self.competitive_margin,
                "player_scores": {
                    pid: {"distance": distance, "blocked_by": reason}
                    for pid, (distance, reason) in scores.items()
                },
                "reason": "matched_existing" if matched else "insufficient_evidence",
            }
            decisions[sid] = IncrementalDecision(
                sid,
                DecisionOutcome.MATCHED_EXISTING if matched else DecisionOutcome.INSUFFICIENT,
                ranked[0][1] if matched else None,
                best,
                margin,
                evidence,
            )
            # An incomparable player is not evidence of a new identity. Require
            # explicit physical exclusion or distance beyond the ambiguity band.
            separated = all(
                reason in {"jersey_conflict", "cannot_link"}
                or (
                    math.isfinite(distance)
                    and distance > self.distance_threshold + self.competitive_margin + _EPS
                )
                for distance, reason in scores.values()
            )
            can_seed = query.appearance.quality_score > 0 and query.appearance.quality_score >= max(
                self.min_quality, self.min_prototype_weight
            )
            if not matched and can_seed and separated:
                eligible.add(sid)

        occupied_numbers = {
            player.confirmed_jersey_number
            for player in players
            if player.confirmed_jersey_number is not None
        }
        for player in players:
            for proto in player.prototypes:
                num = _jersey(proto.feature)
                if num is not None:
                    occupied_numbers.add(num)

        self._resolve_batch_conflicts(decisions, queries, engine)
        new_players = self._discover(
            eligible,
            queries,
            decisions,
            engine,
            batch_id,
            from_revision,
            {player.player_id for player in players},
            occupied_numbers,
        )
        updates = self._prototype_updates(decisions, queries, profiles, from_revision + 1)
        ordered = tuple(decisions[sid] for sid in sorted(decisions))
        return IncrementalPlan(
            batch_id,
            from_revision,
            from_revision + 1,
            ordered,
            tuple(new_players),
            tuple(updates),
            tuple(
                item.segment_id for item in ordered if item.outcome == DecisionOutcome.INSUFFICIENT
            ),
        )

    def _resolve_batch_conflicts(
        self,
        decisions: dict[str, IncrementalDecision],
        queries: dict[str, DualStreamFeature],
        engine: SpatioTemporalConstraintEngine,
    ) -> None:
        matched = [d for d in decisions.values() if d.outcome == DecisionOutcome.MATCHED_EXISTING]
        conflicts: dict[str, list[str]] = {}
        for i, left in enumerate(matched):
            for right in matched[i + 1 :]:
                if left.target_player_id != right.target_player_id:
                    continue
                a, b = _jersey(queries[left.segment_id]), _jersey(queries[right.segment_id])
                if engine.pair_relation(left.segment_id, right.segment_id) != "clear" or (
                    a is not None and b is not None and a != b
                ):
                    conflicts.setdefault(left.segment_id, []).append(right.segment_id)
                    conflicts.setdefault(right.segment_id, []).append(left.segment_id)
        for sid, others in conflicts.items():
            old = decisions[sid]
            decisions[sid] = replace(
                old,
                outcome=DecisionOutcome.INSUFFICIENT,
                target_player_id=None,
                evidence={
                    **old.evidence,
                    "reason": "batch_conflict",
                    "conflicting_segments": others,
                },
            )

    def _discover(
        self,
        eligible: set[str],
        queries: dict[str, DualStreamFeature],
        decisions: dict[str, IncrementalDecision],
        engine: SpatioTemporalConstraintEngine,
        batch_id: str,
        revision: int,
        existing_ids: set[str],
        occupied_numbers: set[int] | None = None,
    ) -> list[PlayerProfile]:
        occupied = set(occupied_numbers or ())
        ids = sorted(eligible)
        neighbors: dict[str, set[str]] = {sid: set() for sid in ids}
        distances: dict[tuple[str, str], float] = {}
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                distance = self._pair_distance(queries[a], queries[b], engine)
                distances[a, b] = distance
                if distance <= self.distance_threshold + _EPS:
                    neighbors[a].add(b)
                    neighbors[b].add(a)
        unassigned = set(ids)
        result: list[PlayerProfile] = []
        while unassigned:
            # Seed with highest-quality candidate to anchor a consistent identity.
            seed = min(unassigned, key=lambda sid: (-queries[sid].appearance.quality_score, sid))
            clique = [seed]
            seed_video = engine.segments_by_id[seed].video_id
            candidates = [c for c in unassigned if c != seed and c in neighbors[seed]]
            candidates.sort(
                key=lambda c: (
                    0 if engine.segments_by_id[c].video_id != seed_video else 1,
                    distances[tuple(sorted((seed, c)))],
                    -queries[c].appearance.quality_score,
                    c,
                )
            )
            for c in candidates:
                if all(c in neighbors[m] for m in clique):
                    clique.append(c)

            members = sorted(clique)
            videos = {engine.segments_by_id[sid].video_id for sid in members}

            # Spatio-temporal Cannot-Link physical safety across all pairs in members
            if any(
                engine.pair_relation(a, b) != "clear"
                for i, a in enumerate(members)
                for b in members[i + 1 :]
            ):
                unassigned.remove(seed)
                continue

            numbers = sorted({_jersey(queries[sid]) for sid in members} - {None})
            if len(numbers) > 1:
                unassigned.remove(seed)
                continue
            number = numbers[0] if numbers else None

            # Dual-track admission check
            if number is not None:
                # Track A: Strong physical jersey number pass-through
                quality_ok = queries[seed].appearance.quality_score >= self.min_quality
                number_available = number not in occupied
                if not (len(videos) >= 1 and quality_ok and number_available):
                    unassigned.remove(seed)
                    continue
                track_a_triggered = True
            else:
                # Track B: No-number pure appearance clustering
                if len(videos) < 2:
                    unassigned.remove(seed)
                    continue
                track_a_triggered = False

            unassigned -= set(members)
            player_id = _stable_id("player", batch_id, revision, members)
            if player_id in existing_ids:
                raise ValueError("New player_id already exists in the snapshot")
            existing_ids.add(player_id)
            if number is not None:
                occupied.add(number)
            # First retain independent videos; fill remaining slots by quality.
            ranked = sorted(members, key=lambda sid: (-queries[sid].appearance.quality_score, sid))
            chosen: list[str] = []
            used_videos: set[str] = set()
            for sid in ranked:
                video = engine.segments_by_id[sid].video_id
                if video not in used_videos and len(chosen) < self.max_prototypes:
                    chosen.append(sid)
                    used_videos.add(video)
            chosen.extend(sid for sid in ranked if sid not in chosen)
            prototypes = tuple(
                self._prototype(player_id, queries[sid], revision + 1)
                for sid in chosen[: self.max_prototypes]
            )
            result.append(
                PlayerProfile(
                    player_id,
                    f"{number}号球员" if number is not None else f"未命名球员_{player_id[:8]}",
                    number,
                    prototypes,
                    "active",
                    revision + 1,
                )
            )
            for sid in members:
                support_distance = max(
                    (distances[tuple(sorted((sid, other)))] for other in members if other != sid),
                    default=0.0,
                )
                old = decisions[sid]
                decisions[sid] = replace(
                    old,
                    outcome=DecisionOutcome.CREATED_NEW,
                    target_player_id=None,
                    evidence={
                        **old.evidence,
                        "reason": "confirmed_jersey_number_seed"
                        if track_a_triggered
                        else "independent_cross_video_support",
                        "new_player_id": player_id,
                        "support_segments": members,
                        "support_video_ids": sorted(videos),
                        "max_support_distance": support_distance,
                    },
                )
        return sorted(result, key=lambda player: player.player_id)

    def _prototype(
        self, player_id: str, feature: DualStreamFeature, revision: int
    ) -> PlayerPrototype:
        return PlayerPrototype(
            _stable_id("prototype", player_id, feature.segment_id, revision),
            player_id,
            "generic",
            _copy_feature(feature),
            float(feature.appearance.quality_score),
            revision,
        )

    def _prototype_updates(
        self,
        decisions: dict[str, IncrementalDecision],
        queries: dict[str, DualStreamFeature],
        profiles: list[PlayerProfile],
        revision: int,
    ) -> list[PlayerPrototype]:
        result = []
        for player in profiles:
            available = max(0, self.max_prototypes - len(player.prototypes))
            existing_segments = {p.feature.segment_id for p in player.prototypes}
            candidates = sorted(
                (
                    queries[sid]
                    for sid, decision in decisions.items()
                    if decision.outcome == DecisionOutcome.MATCHED_EXISTING
                    and decision.target_player_id == player.player_id
                    and sid not in existing_segments
                    and queries[sid].appearance.quality_score > 0
                    and queries[sid].appearance.quality_score
                    >= max(self.min_quality, self.min_prototype_weight)
                ),
                key=lambda f: (-f.appearance.quality_score, f.segment_id),
            )
            result.extend(
                self._prototype(player.player_id, f, revision) for f in candidates[:available]
            )
        return sorted(result, key=lambda proto: (proto.player_id, proto.prototype_id))
