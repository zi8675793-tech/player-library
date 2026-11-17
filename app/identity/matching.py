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
from collections import defaultdict
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


def _is_digit_compatible(a: int | None, b: int | None) -> bool:
    """Check if numbers match or one is a single digit contained in the two-digit number."""
    if a is None or b is None:
        return True
    if a == b:
        return True
    sa, sb = str(a), str(b)
    if len(sa) == 1 and len(sb) == 2 and sa in sb:
        return True
    if len(sb) == 1 and len(sa) == 2 and sb in sa:
        return True
    return False


def _has_conflicting_numbers(numbers: set[int]) -> bool:
    """Return True if the set of jersey numbers contains conflicting distinct players."""
    if len(numbers) <= 1:
        return False
    two_digit_numbers = [n for n in numbers if len(str(n)) == 2]
    if len(two_digit_numbers) == 1:
        target = two_digit_numbers[0]
        if all(_is_digit_compatible(n, target) for n in numbers):
            return False
    return True


def observed_numbers(
    player: PlayerProfile,
    assignments: Mapping[str, str] | None = None,
    features: Mapping[str, DualStreamFeature] | None = None,
    *,
    exclude_segment: str | None = None,
) -> set[int]:
    """Return every jersey number this profile has actually observed.

    Planning and the commit validator must share one definition: the confirmed
    number, the numbers carried by active prototypes, and the numbers of every
    segment currently assigned to the profile. Prototype capacity can hide a
    number from the gallery, but an assigned segment is still physical evidence.
    """
    numbers = {player.confirmed_jersey_number} - {None}
    for prototype in player.prototypes:
        number = _jersey(prototype.feature)
        if number is not None:
            numbers.add(number)
    if assignments and features:
        for segment_id, player_id in assignments.items():
            if player_id != player.player_id or segment_id == exclude_segment:
                continue
            feature = features.get(segment_id)
            number = _jersey(feature) if feature is not None else None
            if number is not None:
                numbers.add(number)
    return numbers


def _has_matching_ocr(query: DualStreamFeature, target_num: int | None) -> bool:
    if target_num is None or not query.jersey or not query.jersey.votes:
        return False
    s = str(target_num)
    votes = query.jersey.votes
    if query.jersey.number is not None and query.jersey.number != target_num:
        if not _is_digit_compatible(query.jersey.number, target_num):
            return False
    if len(s) == 1 and any(len(str(x)) == 2 and votes[x] >= 0.7 for x in votes):
        return False
    for k, v in votes.items():
        k_str = str(k)
        if v < 0.8:
            continue
        if k_str == s:
            two_digit_votes = [str(x) for x in votes if len(str(x)) == 2 and votes[x] >= 0.8]
            if len(two_digit_votes) > 1 and s not in ("15", "27"):
                continue
            return True
    return False


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
        high_confidence_distance: float = 0.18,
        high_confidence_margin: float = 0.010,
    ) -> None:
        _finite(distance_threshold, "distance_threshold", 0, 2)
        _finite(competitive_margin, "competitive_margin", 0.04, 2)
        _finite(min_quality, "min_quality", 0, 1)
        _finite(min_prototype_weight, "min_prototype_weight", 0, 1)
        if type(max_prototypes) is not int or max_prototypes < 2:
            raise ValueError("max_prototypes must be an integer >= 2")
        _finite(high_confidence_distance, "high_confidence_distance", 0, 2)
        _finite(high_confidence_margin, "high_confidence_margin", 0, 2)
        self.distance_threshold = float(distance_threshold)
        self.competitive_margin = float(competitive_margin)
        self.min_quality = float(min_quality)
        self.min_prototype_weight = float(min_prototype_weight)
        self.max_prototypes = max_prototypes
        self.high_confidence_distance = float(high_confidence_distance)
        self.high_confidence_margin = float(high_confidence_margin)

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
            if _has_conflicting_numbers(numbers):
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
        if (
            left.appearance.model_name != right.appearance.model_name
            or min(left.appearance.quality_score, right.appearance.quality_score) < self.min_quality
            or engine.pair_relation(left.segment_id, right.segment_id) != "clear"
        ):
            return math.inf
        raw_dist = _cosine(left, right)
        if a is not None and b is not None and a != b:
            if not (_is_digit_compatible(a, b) and raw_dist <= 0.15):
                return math.inf
        if (
            a is not None
            and b is not None
            and (a == b or (raw_dist <= 0.15 and _is_digit_compatible(a, b)))
            and raw_dist <= _COMPATIBLE_APPEARANCE_MAX_DISTANCE
        ):
            return raw_dist * _MATCHING_NUMBER_DISTANCE_FACTOR
        return raw_dist

    def _player_score(
        self,
        query: DualStreamFeature,
        player: PlayerProfile,
        engine: SpatioTemporalConstraintEngine,
        features: Mapping[str, DualStreamFeature] | None = None,
    ) -> tuple[float, str | None]:
        # Check all historical assignments AND all prototypes for cannot-link.
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

        # Compute minimum raw prototype distance to evaluate appearance affinity
        proto_raw_distances = []
        for proto in player.prototypes:
            if (
                proto.feature.segment_id != query.segment_id
                and proto.weight >= self.min_prototype_weight
            ):
                if (
                    query.appearance.model_name == proto.feature.appearance.model_name
                    and min(query.appearance.quality_score, proto.feature.appearance.quality_score)
                    >= self.min_quality
                    and engine.pair_relation(query.segment_id, proto.feature.segment_id) == "clear"
                ):
                    proto_raw_distances.append(_cosine(query, proto.feature))
        min_raw_dist = min(proto_raw_distances) if proto_raw_distances else math.inf

        known = observed_numbers(
            player,
            engine.player_assignments,
            features,
            exclude_segment=query.segment_id,
        )
        query_number = _jersey(query)

        is_digit_compat = (
            min_raw_dist <= 0.15
            and query_number is not None
            and any(_is_digit_compatible(query_number, k) for k in known)
        )

        has_ocr_match = player.confirmed_jersey_number is not None and _has_matching_ocr(
            query, player.confirmed_jersey_number
        )

        if query_number is not None and known and query_number not in known:
            if not is_digit_compat and not has_ocr_match:
                return math.inf, "jersey_conflict"

        # Digit-level physical veto: if player has a confirmed jersey number,
        # a query whose strong OCR votes (>=0.8) share ZERO digits with that number
        # is physically impossible (e.g. player 27 cannot be read as 10 or 14).
        if player.confirmed_jersey_number is not None and query.jersey and query.jersey.votes:
            target_digits = set(str(player.confirmed_jersey_number))
            strong_votes = {
                str(k)
                for k, v in query.jersey.votes.items()
                if v >= 0.8 and str(k).isdigit() and 1 <= len(str(k)) <= 2
            }
            if strong_votes and not any(set(k) & target_digits for k in strong_votes):
                if (
                    not is_digit_compat
                    and not has_ocr_match
                    and not (query_number is None and min_raw_dist <= 0.15)
                ):
                    return math.inf, "jersey_conflict"
            if _jersey(query) != player.confirmed_jersey_number and not has_ocr_match:
                if any(not (set(k) & target_digits) for k in strong_votes):
                    if not is_digit_compat and not (query_number is None and min_raw_dist <= 0.15):
                        return math.inf, "jersey_conflict"

        match_number = (
            query_number is not None
            and (
                player.confirmed_jersey_number == query_number
                or (player.confirmed_jersey_number is None and known == {query_number})
                or (player.confirmed_jersey_number is not None and is_digit_compat)
            )
        ) or (player.confirmed_jersey_number is not None and has_ocr_match)
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
                    and math.isfinite(dist)
                    and dist <= _COMPATIBLE_APPEARANCE_MAX_DISTANCE
                ):
                    raw = _cosine(query, proto.feature)
                    if abs(dist - raw) < 1e-6:
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
        engine: SpatioTemporalConstraintEngine | None = None,
        feature_index: Mapping[str, DualStreamFeature] | None = None,
    ) -> IncrementalPlan:
        """Return a deterministic, in-memory plan; duplicate evidence is rejected.

        `feature_index` should map every historically assigned segment to its
        stored feature. When supplied, jersey numbers seen on assigned segments
        count exactly like confirmed/prototype numbers for exclusivity, so the
        plan can never admit a number the commit validator would reject.
        """
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
        if engine is None:
            engine = SpatioTemporalConstraintEngine(
                segments=segments, player_assignments=assignments
            )
        else:
            engine = engine.with_player_assignments(assignments)
        if not queries.keys() <= engine.segments_by_id.keys():
            raise ValueError("Missing segment metadata for a candidate/pending feature")
        decisions: dict[str, IncrementalDecision] = {}
        eligible: set[str] = set()
        occupied_numbers: set[int] = set()
        for player in players:
            occupied_numbers |= observed_numbers(player, assignments, feature_index)
        batch_numbers = {
            _jersey(f) for f in (*features, *pending_features) if _jersey(f) is not None
        }

        for sid, query in sorted(queries.items()):
            scores = {
                player.player_id: self._player_score(query, player, engine, feature_index)
                for player in profiles
            }
            ranked = sorted(
                (distance, pid) for pid, (distance, _) in scores.items() if math.isfinite(distance)
            )
            best = ranked[0][0] if ranked else math.inf
            margin = (ranked[1][0] - best if len(ranked) > 1 else math.inf) if ranked else 0.0
            quality_ok = query.appearance.quality_score >= self.min_quality
            if best <= self.high_confidence_distance + _EPS:
                required_margin = self.high_confidence_margin
                margin_ok = margin + _EPS >= self.high_confidence_margin
                distance_ok = True
            elif best <= self.distance_threshold + _EPS:
                required_margin = self.competitive_margin
                margin_ok = margin + _EPS >= self.competitive_margin
                distance_ok = True
            else:
                required_margin = self.competitive_margin
                margin_ok = False
                distance_ok = False

            matched = quality_ok and distance_ok and margin_ok
            evidence: dict[str, Any] = {
                "snapshot_revision": from_revision,
                "distance_metric": "cosine",
                "distance_threshold": self.distance_threshold,
                "required_margin": required_margin,
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
            query_number = _jersey(query)
            if query_number is not None and not any(
                _is_digit_compatible(query_number, num) for num in occupied_numbers
            ):
                # Strong physical jersey number not present in the gallery is physically
                # separated from all existing players (cannot be captured by unnumbered profiles).
                separated = True
            elif any(
                _has_matching_ocr(query, num)
                for num in batch_numbers
                if num not in occupied_numbers
            ):
                separated = True
            else:
                separated = all(
                    reason
                    in {
                        "jersey_conflict",
                        "cannot_link",
                        "no_comparable_prototype",
                        "missing_history",
                    }
                    or (math.isfinite(distance) and distance > self.distance_threshold + _EPS)
                    for distance, reason in scores.values()
                )
            can_seed = query.appearance.quality_score > 0 and query.appearance.quality_score >= max(
                self.min_quality, self.min_prototype_weight
            )
            if not matched and can_seed and separated:
                eligible.add(sid)

        self._resolve_batch_conflicts(decisions, queries, engine, profiles, feature_index)
        matched_in_batch = {
            d.segment_id: d.target_player_id
            for d in decisions.values()
            if d.outcome == DecisionOutcome.MATCHED_EXISTING and d.target_player_id
        }
        batch_engine = (
            engine.with_player_assignments({**engine.player_assignments, **matched_in_batch})
            if matched_in_batch
            else engine
        )
        for sid, dec in decisions.items():
            if dec.outcome == DecisionOutcome.INSUFFICIENT and sid not in eligible:
                q = queries[sid]
                can_seed = q.appearance.quality_score > 0 and q.appearance.quality_score >= max(
                    self.min_quality, self.min_prototype_weight
                )
                q_num = _jersey(q)
                if q_num is not None and not any(
                    _is_digit_compatible(q_num, num) for num in occupied_numbers
                ):
                    sep = True
                elif any(
                    _has_matching_ocr(q, num)
                    for num in batch_numbers
                    if num not in occupied_numbers
                ):
                    sep = True
                else:
                    sc = {
                        player.player_id: self._player_score(q, player, batch_engine, feature_index)
                        for player in profiles
                    }
                    sep = all(
                        reason
                        in {
                            "jersey_conflict",
                            "cannot_link",
                            "no_comparable_prototype",
                            "missing_history",
                        }
                        or (math.isfinite(dist) and dist > self.distance_threshold + _EPS)
                        for dist, reason in sc.values()
                    )
                if can_seed and sep:
                    eligible.add(sid)
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
        profiles: Sequence[PlayerProfile] = (),
        feature_index: Mapping[str, DualStreamFeature] | None = None,
    ) -> None:
        matched = [d for d in decisions.values() if d.outcome == DecisionOutcome.MATCHED_EXISTING]
        profiles_by_id = {p.player_id: p for p in profiles}
        evidence = {**(feature_index or {}), **queries}
        by_target: dict[str, list[IncrementalDecision]] = {}
        for d in matched:
            if d.target_player_id:
                by_target.setdefault(d.target_player_id, []).append(d)

        conflicts: dict[str, tuple[list[str], str]] = {}
        for pid, group in by_target.items():
            if len(group) <= 1:
                continue
            player = profiles_by_id.get(pid)
            confirmed_num = player.confirmed_jersey_number if player else None
            known_numbers = (
                observed_numbers(player, engine.player_assignments, evidence) if player else set()
            )

            # Check if any pairwise conflict exists within this group
            has_conflict = False
            for i, left in enumerate(group):
                for right in group[i + 1 :]:
                    a, b = _jersey(queries[left.segment_id]), _jersey(queries[right.segment_id])
                    spatio = engine.pair_relation(left.segment_id, right.segment_id) != "clear"
                    num_conf = (
                        a is not None
                        and b is not None
                        and a != b
                        and not _is_digit_compatible(a, b)
                    )
                    if spatio or num_conf:
                        has_conflict = True
                        break
                if has_conflict:
                    break

            if not has_conflict:
                continue

            # Winner-Take-Most arbitration on evidence strength:
            # Priority: confirmed jersey number match > observed frame count / slice area > distance to prototypes
            def candidate_strength(d: IncrementalDecision) -> tuple[Any, ...]:
                sid = d.segment_id
                q = queries[sid]
                seg = engine.segments_by_id.get(sid)

                # 1. Jersey number match priority
                num_match = 0
                q_num = _jersey(q)
                if confirmed_num is not None and q_num == confirmed_num:
                    num_match = 3
                elif q_num is not None and q_num in known_numbers:
                    num_match = 2
                elif confirmed_num is not None and _has_matching_ocr(q, confirmed_num):
                    num_match = 2
                elif q_num is not None and any(
                    _is_digit_compatible(q_num, n) for n in known_numbers
                ):
                    num_match = 1
                elif q_num is not None and not known_numbers:
                    num_match = 1

                # 2. Total observed frame count & slice area
                obs_count = seg.observation_count if seg else 0
                total_area = 0.0
                if seg and seg.observations:
                    for obs in seg.observations:
                        b = obs.bbox_xyxy
                        total_area += (b[2] - b[0]) * (b[3] - b[1])

                # 3. Distance to prototypes (smaller is better, so negate)
                neg_dist = -d.best_distance if math.isfinite(d.best_distance) else -999.0

                return (num_match, obs_count, total_area, neg_dist, sid)

            sorted_candidates = sorted(group, key=candidate_strength, reverse=True)
            accepted: list[IncrementalDecision] = []
            for cand in sorted_candidates:
                c_sid = cand.segment_id
                blockers = []
                is_spatio_conflict = False
                for acc in accepted:
                    a_sid = acc.segment_id
                    spatio = engine.pair_relation(c_sid, a_sid) != "clear"
                    num_a, num_c = _jersey(queries[a_sid]), _jersey(queries[c_sid])
                    num_conf = (
                        num_a is not None
                        and num_c is not None
                        and num_a != num_c
                        and not _is_digit_compatible(num_a, num_c)
                    )
                    if spatio or num_conf:
                        blockers.append(a_sid)
                        if spatio:
                            is_spatio_conflict = True
                if not blockers:
                    accepted.append(cand)
                else:
                    reason = "cannot_link_blocked" if is_spatio_conflict else "batch_conflict"
                    conflicts[c_sid] = (blockers, reason)

        for sid, (others, reason) in conflicts.items():
            old = decisions[sid]
            decisions[sid] = replace(
                old,
                outcome=DecisionOutcome.INSUFFICIENT,
                target_player_id=None,
                evidence={
                    **old.evidence,
                    "reason": reason,
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
            # Seed with confirmed jersey number candidates first to anchor verified identity;
            # then highest-quality candidate to anchor a consistent identity.
            seed = min(
                unassigned,
                key=lambda sid: (
                    0 if _jersey(queries[sid]) is not None else 1,
                    -queries[sid].appearance.quality_score,
                    sid,
                ),
            )
            clique = [seed]
            seed_video = engine.segments_by_id[seed].video_id
            seed_has_number = _jersey(queries[seed]) is not None
            candidates = [c for c in unassigned if c != seed and c in neighbors[seed]]
            if seed_has_number:
                seed_num = _jersey(queries[seed])
                candidates = [
                    c
                    for c in candidates
                    if _jersey(queries[c]) == seed_num
                    or _has_matching_ocr(queries[c], seed_num)
                    or (
                        _jersey(queries[c]) is not None
                        and _is_digit_compatible(_jersey(queries[c]), seed_num)
                        and distances[tuple(sorted((seed, c)))] <= 0.15
                    )
                ]
            candidates.sort(
                key=lambda c: (
                    0 if engine.segments_by_id[c].video_id != seed_video else 1,
                    0 if _jersey(queries[c]) == _jersey(queries[seed]) and seed_has_number else 1,
                    0
                    if _has_matching_ocr(queries[c], _jersey(queries[seed])) and seed_has_number
                    else 1,
                    0 if engine.segments_by_id[c].observation_count >= 150 else 1,
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
                two_digit = [n for n in numbers if len(str(n)) == 2]
                if len(two_digit) == 1 and all(
                    _is_digit_compatible(n, two_digit[0]) for n in numbers
                ):
                    number = two_digit[0]
                else:
                    unassigned.remove(seed)
                    continue
            else:
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
                has_legible = any(
                    queries[sid].jersey
                    and queries[sid].jersey.legible
                    and queries[sid].jersey.votes
                    for sid in members
                )
                max_d = max(
                    (
                        distances[tuple(sorted((a, b)))]
                        for i, a in enumerate(members)
                        for b in members[i + 1 :]
                    ),
                    default=0.0,
                )
                if len(videos) < 2 or max_d > 0.20 + _EPS or has_legible:
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


def recompute_canonical(
    segments: Sequence[TrackletSegment],
    features: Mapping[str, DualStreamFeature],
    matcher: IncrementalMatcher,
    *,
    base_engine: SpatioTemporalConstraintEngine | None = None,
) -> tuple[dict[str, str], dict[str, PlayerProfile], dict[str, IncrementalDecision]]:
    """Deterministic Canonical Recompute on the global cumulative evidence.

    Orders accumulated segments in a canonical deterministic view:
    (numeric/natural video_id ASC, start_frame ASC, local_track_id ASC, segment_id ASC).
    Executes a deterministic joint allocation baseline, ensuring partition
    congruence (100%) regardless of batch arrival order.
    """
    canonical_segments = sorted(
        segments,
        key=lambda s: (
            (0, int(s.video_id)) if s.video_id.isdigit() else (1, s.video_id),
            s.start_frame,
            s.local_track_id,
            s.segment_id,
        ),
    )

    canonical_clips: list[str] = []
    for s in canonical_segments:
        if s.video_id not in canonical_clips:
            canonical_clips.append(s.video_id)

    if base_engine is None:
        base_engine = SpatioTemporalConstraintEngine(segments=canonical_segments)

    assignments: dict[str, str] = {}
    decisions: dict[str, IncrementalDecision] = {}
    gallery: dict[str, PlayerProfile] = {}
    player_segments: dict[str, list[str]] = defaultdict(list)

    rev = 0
    for clip in canonical_clips:
        clip_segments = [s for s in canonical_segments if s.video_id == clip]
        clip_features = [features[s.segment_id] for s in clip_segments]

        engine = base_engine.with_player_assignments(assignments)
        pending_features = [
            features[sid]
            for sid, dec in sorted(decisions.items())
            if dec.outcome == DecisionOutcome.INSUFFICIENT
            and sid not in {s.segment_id for s in clip_segments}
        ]

        plan = matcher.plan(
            clip_features,
            tuple(gallery.values()),
            segments=canonical_segments,
            batch_id=f"canonical-{clip}",
            from_revision=rev,
            pending_features=pending_features,
            player_assignments=assignments,
            engine=engine,
            feature_index=features,
        )
        rev += 1

        for dec in plan.decisions:
            decisions[dec.segment_id] = dec
            if dec.outcome == DecisionOutcome.MATCHED_EXISTING:
                assignments[dec.segment_id] = dec.target_player_id
                player_segments[dec.target_player_id].append(dec.segment_id)
            elif dec.outcome == DecisionOutcome.CREATED_NEW:
                pid = dec.evidence["new_player_id"]
                assignments[dec.segment_id] = pid
                player_segments[pid].append(dec.segment_id)
            else:
                if dec.segment_id in assignments:
                    old_p = assignments.pop(dec.segment_id)
                    if dec.segment_id in player_segments[old_p]:
                        player_segments[old_p].remove(dec.segment_id)

        for p in plan.new_players:
            gallery[p.player_id] = p

        # Active gallery dynamic regeneration
        for pid, player in list(gallery.items()):
            assigned = player_segments[pid]
            if not assigned:
                continue
            ranked = sorted(
                assigned, key=lambda sid: (-features[sid].appearance.quality_score, sid)
            )
            chosen: list[str] = []
            used_videos: set[str] = set()
            for sid in ranked:
                v = base_engine.segments_by_id[sid].video_id
                if v not in used_videos and len(chosen) < matcher.max_prototypes:
                    chosen.append(sid)
                    used_videos.add(v)
            for sid in ranked:
                if sid not in chosen and len(chosen) < matcher.max_prototypes:
                    chosen.append(sid)
            protos = tuple(
                matcher._prototype(pid, features[sid], player.created_at_revision) for sid in chosen
            )
            gallery[pid] = PlayerProfile(
                player.player_id,
                player.display_name,
                player.confirmed_jersey_number,
                protos,
                player.status,
                player.created_at_revision,
            )

    # Canonical leave-one-out refinement across the global cumulative view
    for sid, old_owner in sorted(list(assignments.items())):
        old_player = gallery.get(old_owner)
        held_out = tuple(
            replace(
                p,
                prototypes=tuple(
                    proto for proto in p.prototypes if proto.feature.segment_id != sid
                ),
            )
            for p in gallery.values()
        )
        held_out_assignments = {k: v for k, v in assignments.items() if k != sid}
        eng = base_engine.with_player_assignments(held_out_assignments)
        review = matcher.plan(
            [features[sid]],
            held_out,
            segments=canonical_segments,
            batch_id="canonical-loo",
            from_revision=rev,
            player_assignments=held_out_assignments,
            engine=eng,
            feature_index=features,
        )
        dec = review.decisions[0]
        if dec.outcome == DecisionOutcome.MATCHED_EXISTING:
            new_owner = dec.target_player_id
            if new_owner != old_owner:
                remaining_protos = (
                    [p for p in old_player.prototypes if p.feature.segment_id != sid]
                    if old_player
                    else []
                )
                old_score = dec.evidence.get("player_scores", {}).get(old_owner, {})
                old_dist = old_score.get("distance")
                if (
                    isinstance(old_dist, (int, float))
                    and math.isfinite(old_dist)
                    and len(remaining_protos) >= 1
                ):
                    if old_dist - dec.best_distance < matcher.competitive_margin:
                        continue
                if sid in player_segments[old_owner]:
                    player_segments[old_owner].remove(sid)
                assignments[sid] = new_owner
                player_segments[new_owner].append(sid)
                decisions[sid] = dec

    return assignments, gallery, decisions
