"""Shared current-player, event and review views at one database revision."""

import sqlite3
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager

from app.contracts.api import EventQuery, PlayerQuery, SuccessResponse
from app.contracts.events import ResolvedEvent
from app.contracts.evidence import GallerySnapshot
from app.contracts.identity import Candidate, PersonRole
from app.contracts.views import (
    PlayerStatsView,
    PlayerView,
    ReviewActionSuggestion,
    ReviewQueueItem,
)
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
            and (query.role is None or player.role == query.role)
            and (not query.formal_players_only or player.role == PersonRole.PLAYER)
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
                is_formal_player=(player.role == PersonRole.PLAYER),
                profile_type=(
                    "formal_player"
                    if player.role == PersonRole.PLAYER
                    else "referee_profile"
                    if player.role == PersonRole.REFEREE
                    else "staff_profile"
                    if player.role == PersonRole.STAFF
                    else "anonymous_identity_profile"
                ),
            )
            for player in players[query.offset : query.offset + query.limit]
        ]
        return SuccessResponse(data=views, revision=gallery.revision)


def get_formal_player_stats(
    connection: sqlite3.Connection, match_id: str
) -> SuccessResponse[list[PlayerStatsView]]:
    """Return official statistics for confirmed formal players only (excludes unknown/staff/referee)."""
    with _read_state(connection, match_id) as state:
        gallery = state.gallery
        counts = _made_counts(_events(connection, state))
        stats = []
        for player in gallery.players:
            if player.alias_of is not None:
                continue
            if player.role != PersonRole.PLAYER:
                continue
            confirmed_segs = [
                a.segment_id
                for a in gallery.assignments
                if a.player_id == player.player_id
                and a.status == "confirmed"
                and any(
                    s.segment_id == a.segment_id and s.active for s in gallery.evidence.segments
                )
            ]
            approved_templates = [
                t
                for t in gallery.templates
                if t.player_id == player.player_id and t.active and t.review_status == "approved"
            ]
            stats.append(
                PlayerStatsView(
                    match_id=match_id,
                    player_id=player.player_id,
                    role=player.role,
                    display_name=player.display_name,
                    jersey_number=player.jersey_number,
                    confirmed_segments_count=len(confirmed_segs),
                    approved_templates_count=len(approved_templates),
                    official_made_count=counts.get(player.player_id, 0),
                    is_formal_player=True,
                )
            )
        return SuccessResponse(data=stats, revision=gallery.revision)


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

        # Index samples by segment and tracks by clip
        samples_by_seg = defaultdict(list)
        for s in gallery.evidence.samples:
            if s.segment_id and s.segment_id in active:
                samples_by_seg[s.segment_id].append(s)
            else:
                for seg_id, seg in active.items():
                    if (
                        seg.clip_id == s.clip_id
                        and seg.track_id == s.track_id
                        and seg.start_s <= s.time_s < seg.end_s
                    ):
                        samples_by_seg[seg_id].append(s)
                        break

        tracks_by_clip = defaultdict(list)
        for t in gallery.evidence.tracks:
            tracks_by_clip[t.clip_id].append(t)

        samples_by_id = {s.sample_id: s for s in gallery.evidence.samples}

        queue = []
        for segment_id, segment in active.items():
            assignment = assignments.get(segment_id)
            seg_samples = samples_by_seg[segment_id]
            thumbnail_ids = [s.image_asset_id for s in seg_samples if s.image_asset_id]
            seg_risks = active_risks(gallery.evidence, segment, gallery.risk_reviews)
            risk_kinds = [r.kind for r in seg_risks]
            seg_prop = state.segment_proposals.get(segment_id)

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
                        candidates = (
                            []
                            if stale
                            else [
                                Candidate.model_validate(c)
                                for c in (
                                    association.get("candidates")
                                    or association.get("local_candidates", [])
                                )
                            ]
                        )
                        suggested_actions = [
                            *(
                                [
                                    ReviewActionSuggestion(
                                        action="confirm_assignment",
                                        label=f"确认关联至候选球员 {c.player_id}",
                                        target_player_id=c.player_id,
                                        description=f"距离/匹配分: {c.distance:.4f}"
                                        if c.distance is not None
                                        else None,
                                    )
                                    for c in candidates[:3]
                                ]
                            ),
                            ReviewActionSuggestion(
                                action="create_player",
                                label="建立新球员档案",
                                description="为此待定段创建独立身份",
                            ),
                            ReviewActionSuggestion(
                                action="split_segment",
                                label="按断点拆分轨迹段",
                                description="切除遮挡或混人区间",
                            ),
                            ReviewActionSuggestion(
                                action="set_assignment",
                                label="标记为非球员或废弃",
                                description="标记为 unassigned/rejected",
                            ),
                        ]
                        queue.append(
                            ReviewQueueItem(
                                kind="identity",
                                segment_id=segment_id,
                                clip_id=segment.clip_id,
                                track_id=segment.track_id,
                                start_s=segment.start_s,
                                end_s=segment.end_s,
                                feature_space_id=plan.get("feature_space_id"),
                                reasons=reasons,
                                risk_kinds=risk_kinds,
                                review_priority="critical"
                                if any(
                                    k in {"duplicate_box", "mixed_identity", "identity_conflict"}
                                    for k in risk_kinds
                                )
                                else ("high" if risk_kinds else "medium"),
                                candidates=candidates,
                                thumbnail_asset_ids=thumbnail_ids[:6],
                                suggested_actions=suggested_actions,
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
                    has_samples = bool(seg_samples)
                    auto_reasons = []
                    if assignment and assignment.evidence:
                        for ev in assignment.evidence:
                            if ev.description:
                                auto_reasons.append(ev.description)
                            elif ev.kind:
                                auto_reasons.append(ev.kind)
                    if seg_prop and seg_prop.reasons:
                        auto_reasons.extend(seg_prop.reasons)
                    reasons = list(
                        dict.fromkeys(
                            [
                                "identity_not_confirmed",
                                *(
                                    ["matching_evidence_missing"]
                                    if has_samples and not auto_reasons
                                    else []
                                ),
                                *(["samples_missing"] if not has_samples else []),
                                *(
                                    [f"human_{assignment.unassigned_reason or 'unassigned'}"]
                                    if assignment and assignment.method in {"manual", "seed"}
                                    else []
                                ),
                                *auto_reasons,
                            ]
                        )
                    )
                    suggested_actions = [
                        ReviewActionSuggestion(
                            action="create_player",
                            label="建立新球员档案",
                            description="为此待定段创建独立身份",
                        ),
                        ReviewActionSuggestion(
                            action="split_segment",
                            label="按断点拆分轨迹段",
                            description="切除遮挡或混人区间",
                        ),
                        ReviewActionSuggestion(
                            action="set_assignment",
                            label="标记为非球员或废弃",
                            description="标记为 unassigned/rejected",
                        ),
                    ]
                    queue.append(
                        ReviewQueueItem(
                            kind="identity",
                            segment_id=segment_id,
                            clip_id=segment.clip_id,
                            track_id=segment.track_id,
                            start_s=segment.start_s,
                            end_s=segment.end_s,
                            reasons=reasons,
                            risk_kinds=risk_kinds,
                            review_priority="critical"
                            if any(
                                k in {"duplicate_box", "mixed_identity", "identity_conflict"}
                                for k in risk_kinds
                            )
                            else ("high" if risk_kinds else "medium"),
                            thumbnail_asset_ids=thumbnail_ids[:6],
                            suggested_actions=suggested_actions,
                            details={
                                "current_assignment": assignment.model_dump(mode="json")
                                if assignment
                                else None,
                                "proposal": seg_prop.model_dump(mode="json") if seg_prop else None,
                                "samples_count": len(seg_samples),
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
                    t_sample = samples_by_id.get(sample_id)
                    t_thumb = (
                        [t_sample.image_asset_id] if t_sample and t_sample.image_asset_id else []
                    )
                    queue.append(
                        ReviewQueueItem(
                            kind="template",
                            segment_id=segment_id,
                            clip_id=segment.clip_id,
                            track_id=segment.track_id,
                            start_s=segment.start_s,
                            end_s=segment.end_s,
                            sample_id=sample_id,
                            feature_space_id=plan.get("feature_space_id"),
                            reasons=list(dict.fromkeys([action, *change.get("reasons", [])])),
                            review_priority="low",
                            thumbnail_asset_ids=t_thumb,
                            suggested_actions=[
                                ReviewActionSuggestion(
                                    action="review_template",
                                    label="批准模板进入图库",
                                    target_player_id=assignment.player_id if assignment else None,
                                ),
                                ReviewActionSuggestion(
                                    action="review_template",
                                    label="拒绝此模板",
                                    target_player_id=assignment.player_id if assignment else None,
                                ),
                            ],
                            details={"change": change},
                        )
                    )
        for segment_id, segment in active.items():
            seg_samples = samples_by_seg[segment_id]
            thumbnail_ids = [s.image_asset_id for s in seg_samples if s.image_asset_id]
            for risk in active_risks(gallery.evidence, segment, gallery.risk_reviews):
                overlapping = []
                if (
                    risk.kind in {"duplicate_box", "mixed_identity", "identity_conflict"}
                    or "overlap" in risk.kind
                ):
                    overlapping = sorted(
                        {
                            t.track_id
                            for t in tracks_by_clip[segment.clip_id]
                            if t.track_id != segment.track_id
                            and max(segment.start_s, t.start_s) < min(segment.end_s, t.end_s)
                        }
                    )
                risk_priority = (
                    "critical"
                    if risk.kind in {"duplicate_box", "mixed_identity", "identity_conflict"}
                    else "high"
                )
                suggested_actions = [
                    ReviewActionSuggestion(
                        action="review_risk",
                        label=f"解除风险并标记有效: {risk.kind}",
                        description="人工确认该轨迹无混人/重叠错误，准许参与归人",
                    ),
                    ReviewActionSuggestion(
                        action="split_segment",
                        label="切除高风险帧区间",
                        description="按帧拆分并移除异常范围",
                    ),
                    ReviewActionSuggestion(
                        action="set_assignment",
                        label="隔离该轨迹段",
                        description="标记为 rejected/other_person",
                    ),
                ]
                queue.append(
                    ReviewQueueItem(
                        kind="risk",
                        segment_id=segment_id,
                        clip_id=segment.clip_id,
                        track_id=segment.track_id,
                        start_s=segment.start_s,
                        end_s=segment.end_s,
                        reasons=[risk.kind],
                        risk_kinds=[risk.kind],
                        review_priority=risk_priority,
                        thumbnail_asset_ids=thumbnail_ids[:6],
                        suggested_actions=suggested_actions,
                        details={
                            "risk": risk.model_dump(mode="json"),
                            "overlapping_tracks": overlapping,
                        },
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
