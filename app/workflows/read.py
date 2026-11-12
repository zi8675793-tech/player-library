"""Shared current-player, event and review views at one database revision."""

import sqlite3
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager

from app.contracts.api import EventQuery, PlayerQuery, SuccessResponse
from app.contracts.events import ResolvedEvent
from app.contracts.evidence import GallerySnapshot
from app.contracts.identity import Candidate
from app.contracts.views import PlayerView, ReviewQueueItem
from app.events.associate import associate_events
from app.identity.prepare import active_risks
from app.storage.discovery import read_discovery_reports
from app.storage.repositories import Repository
from app.storage.snapshot import IdentitySnapshot, read_identity_snapshot
from app.storage.workflow_plans import read_workflow_plans


@contextmanager
def _read_state(connection: sqlite3.Connection, match_id: str) -> Iterator[IdentitySnapshot]:
    own = not connection.in_transaction
    if own:
        connection.execute("BEGIN")
    try:
        yield read_identity_snapshot(connection, match_id)
    finally:
        if own:
            connection.rollback()


def _events(connection: sqlite3.Connection, state: IdentitySnapshot) -> list[ResolvedEvent]:
    gallery = state.gallery
    repo = Repository(connection)
    return associate_events(
        repo.events(gallery.match_id),
        assignments=gallery.assignments,
        segments=gallery.evidence.segments,
        tracks=gallery.evidence.tracks,
        clips=repo.clips(gallery.match_id),
    )


def get_identity_state(
    connection: sqlite3.Connection, match_id: str
) -> SuccessResponse[GallerySnapshot]:
    """Export current identity facts through the shared revision-scoped read path."""
    with _read_state(connection, match_id) as state:
        return SuccessResponse(data=state.gallery, revision=state.gallery.revision)


def get_discovery_reports(connection, match_id: str) -> SuccessResponse[list[dict]]:
    """Read immutable automatic support separately from current human-reviewed facts."""
    with _read_state(connection, match_id) as state:
        return SuccessResponse(
            data=read_discovery_reports(connection, match_id), revision=state.gallery.revision
        )


def _made_counts(events: list[ResolvedEvent]) -> dict[str, int]:
    """Count explicit made outcomes for scorer/shooter, once per canonical fact.

    Conflicting outcomes or confirmed scorers among canonical views contribute
    no official count until reviewed. Other roles and ordinary shots never score.
    """
    groups = defaultdict(list)
    for resolved in events:
        if resolved.validation_reasons:
            continue
        groups[resolved.event.canonical_event_id or resolved.event.event_id].append(resolved)
    counts = defaultdict(int)
    for group in groups.values():
        outcomes = {item.event.outcome for item in group if item.event.outcome is not None}
        if outcomes != {"made"}:
            continue
        players = {
            subject.player_id
            for item in group
            for subject in item.subjects
            if subject.role in {"shooter", "scorer"} and subject.status == "mapped"
        }
        if len(players) == 1:
            counts[next(iter(players))] += 1
    return dict(counts)


def get_players(
    connection: sqlite3.Connection, query: PlayerQuery
) -> SuccessResponse[list[PlayerView]]:
    """Return profiles, active approved templates and current official made counts."""
    with _read_state(connection, query.match_id) as state:
        gallery = state.gallery
        counts = _made_counts(_events(connection, state))
        players = [
            player
            for player in gallery.players
            if player.alias_of is None
            and (query.player_id is None or player.player_id == query.player_id)
            and (query.team_id is None or player.team_id == query.team_id)
        ]
        views = [
            PlayerView(
                player=player,
                confirmed_segment_ids=sorted(
                    a.segment_id
                    for a in gallery.assignments
                    if a.player_id == player.player_id
                    and a.status == "confirmed"
                    and any(
                        s.segment_id == a.segment_id and s.active for s in gallery.evidence.segments
                    )
                ),
                templates=[
                    t
                    for t in gallery.templates
                    if t.player_id == player.player_id
                    and t.active
                    and t.review_status == "approved"
                ],
                made_count=counts.get(player.player_id, 0),
            )
            for player in players[query.offset : query.offset + query.limit]
        ]
        return SuccessResponse(data=views, revision=gallery.revision)


def get_events(
    connection: sqlite3.Connection, query: EventQuery
) -> SuccessResponse[list[ResolvedEvent]]:
    """Filter original events using the shared current, role-specific association."""
    with _read_state(connection, query.match_id) as state:
        found = []
        for resolved in _events(connection, state):
            event = resolved.event
            if query.clip_id is not None and event.clip_id != query.clip_id:
                continue
            if query.event_type is not None and event.event_type != query.event_type:
                continue
            if query.outcome is not None and event.outcome != query.outcome:
                continue
            subjects = [s for s in resolved.subjects if query.role is None or s.role == query.role]
            if query.role is not None and not subjects:
                continue
            if query.player_id is not None and not any(
                s.status == "mapped" and s.player_id == query.player_id for s in subjects
            ):
                continue
            if event.time.kind == "point":
                if query.start_s is not None and event.time.time_s < query.start_s:
                    continue
                if query.end_s is not None and event.time.time_s >= query.end_s:
                    continue
            else:
                if query.start_s is not None and event.time.end_s <= query.start_s:
                    continue
                if query.end_s is not None and event.time.start_s >= query.end_s:
                    continue
            found.append(resolved)
        found.sort(
            key=lambda item: (
                item.event.clip_id,
                item.event.time.time_s
                if item.event.time.kind == "point"
                else item.event.time.start_s,
                item.event.event_id,
            )
        )
        return SuccessResponse(
            data=found[query.offset : query.offset + query.limit], revision=state.gallery.revision
        )


def get_review_queue(
    connection: sqlite3.Connection, match_id: str
) -> SuccessResponse[list[ReviewQueueItem]]:
    """Explain current pending identity, risk, template and event support problems."""
    with _read_state(connection, match_id) as state:
        gallery = state.gallery
        active = {s.segment_id: s for s in gallery.evidence.segments if s.active}
        assignments = {a.segment_id: a for a in gallery.assignments}
        plans = read_workflow_plans(connection, match_id)
        # Events can advance the match revision without invalidating identity
        # diagnostics. Accepted human identity/gallery decisions do invalidate
        # earlier matching context; never present those candidates as current.
        clean_reviews = Repository(connection).clean_reviews(match_id)
        identity_revision = max(
            [
                0,
                *(a.revision for a in gallery.assignments),
                *(t.revision for t in gallery.templates),
                *(r.revision for r in gallery.risk_reviews),
                *(r.revision for r in state.template_reviews),
                *(c.revision for c in state.constraints),
                *(r["revision"] for r in clean_reviews.values()),
            ]
        )
        spaces = {space.feature_space_id for space in gallery.evidence.feature_spaces}
        planned = defaultdict(list)
        for plan in plans:
            if plan["segment_id"] in active and (not spaces or plan["feature_space_id"] in spaces):
                planned[plan["segment_id"]].append(plan)
        queue = []
        for segment_id, segment in active.items():
            assignment = assignments.get(segment_id)
            if assignment is None or assignment.status != "confirmed":
                if planned[segment_id]:
                    for plan in planned[segment_id]:
                        prepared, association, resolution = (
                            plan.get("prepared", {}),
                            plan.get("association", {}),
                            plan.get("resolution", {}),
                        )
                        diagnostic_revision = plan.get(
                            "diagnostic_revision", plan["committed_revision"]
                        )
                        stale = diagnostic_revision < identity_revision
                        reasons = list(
                            dict.fromkeys(
                                [
                                    "identity_not_confirmed",
                                    *(["matching_diagnostics_outdated"] if stale else []),
                                    *(
                                        [f"human_{assignment.unassigned_reason or 'unassigned'}"]
                                        if assignment and assignment.method in {"manual", "seed"}
                                        else []
                                    ),
                                    *prepared.get("reasons", []),
                                    *association.get("blocking_reasons", []),
                                    *resolution.get("reasons", []),
                                    *(
                                        e["reason"]
                                        for e in prepared.get("exclusions", [])
                                        if e.get("reason") != "bucket_not_selected"
                                    ),
                                ]
                            )
                        )
                        queue.append(
                            ReviewQueueItem(
                                kind="identity",
                                segment_id=segment_id,
                                feature_space_id=plan.get("feature_space_id"),
                                reasons=reasons,
                                candidates=[]
                                if stale
                                else [
                                    Candidate.model_validate(c)
                                    for c in (
                                        association.get("candidates")
                                        or association.get("local_candidates", [])
                                    )
                                ],
                                details={
                                    "prepared": prepared,
                                    "association": association,
                                    "resolution": resolution,
                                    "diagnostic_revision": diagnostic_revision,
                                    "current_identity_revision": identity_revision,
                                    "current_assignment": assignment.model_dump(mode="json")
                                    if assignment
                                    else None,
                                },
                            )
                        )
                else:
                    has_samples = any(
                        s.clip_id == segment.clip_id
                        and s.track_id == segment.track_id
                        and segment.start_s <= s.time_s < segment.end_s
                        for s in gallery.evidence.samples
                    )
                    queue.append(
                        ReviewQueueItem(
                            kind="identity",
                            segment_id=segment_id,
                            reasons=[
                                "identity_not_confirmed",
                                "matching_evidence_missing" if has_samples else "samples_missing",
                                *(
                                    [f"human_{assignment.unassigned_reason or 'unassigned'}"]
                                    if assignment and assignment.method in {"manual", "seed"}
                                    else []
                                ),
                            ],
                            details={
                                "current_assignment": assignment.model_dump(mode="json")
                                if assignment
                                else None
                            },
                        )
                    )
            for plan in planned[segment_id]:
                for change in plan.get("gallery_changes", []):
                    action = change.get("action")
                    if action not in {"await_review", "replacement_required"}:
                        continue
                    if (
                        assignment is None
                        or assignment.status != "confirmed"
                        or change.get("player_id") != assignment.player_id
                    ):
                        continue
                    sample_id = change.get("sample_id")
                    if any(
                        t.active
                        and t.review_status == "approved"
                        and t.sample_id == sample_id
                        and t.feature_space_id == plan.get("feature_space_id")
                        and assignment
                        and t.player_id == assignment.player_id
                        for t in gallery.templates
                    ):
                        continue
                    if any(
                        r.sample_id == sample_id
                        and r.feature_space_id == plan.get("feature_space_id")
                        and assignment
                        and r.player_id == assignment.player_id
                        and r.review_status == "rejected"
                        for r in state.template_reviews
                    ):
                        continue
                    queue.append(
                        ReviewQueueItem(
                            kind="template",
                            segment_id=segment_id,
                            sample_id=sample_id,
                            feature_space_id=plan.get("feature_space_id"),
                            reasons=list(dict.fromkeys([action, *change.get("reasons", [])])),
                            details={"change": change},
                        )
                    )
        for segment_id, segment in active.items():
            for risk in active_risks(gallery.evidence, segment, gallery.risk_reviews):
                queue.append(
                    ReviewQueueItem(
                        kind="risk",
                        segment_id=segment_id,
                        reasons=[risk.kind],
                        details={"risk": risk.model_dump(mode="json")},
                    )
                )
        resolved_events = _events(connection, state)
        canonical_groups = defaultdict(list)
        for resolved in resolved_events:
            canonical_groups[resolved.event.canonical_event_id or resolved.event.event_id].append(
                resolved
            )
        for canonical, group in canonical_groups.items():
            outcomes = {item.event.outcome for item in group if item.event.outcome is not None}
            scorers = {
                s.player_id
                for item in group
                for s in item.subjects
                if s.role in {"shooter", "scorer"} and s.status == "mapped"
            }
            reasons = []
            if len(outcomes) > 1:
                reasons.append("canonical_outcomes_conflict")
            if len(scorers) > 1:
                reasons.append("canonical_scorers_conflict")
            if reasons:
                queue.append(
                    ReviewQueueItem(
                        kind="event",
                        reasons=reasons,
                        details={
                            "canonical_event_id": canonical,
                            "event_ids": [r.event.event_id for r in group],
                        },
                    )
                )
        for event in resolved_events:
            unresolved = [s for s in event.subjects if s.status == "unmapped"]
            if unresolved or event.validation_reasons:
                queue.append(
                    ReviewQueueItem(
                        kind="event",
                        event_id=event.event.event_id,
                        reasons=list(
                            dict.fromkeys(
                                [*event.validation_reasons, *(s.reason for s in unresolved)]
                            )
                        ),
                        details={"subjects": [s.model_dump(mode="json") for s in unresolved]},
                    )
                )
        return SuccessResponse(data=queue, revision=gallery.revision)
