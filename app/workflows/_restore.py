"""Restore a historical business state by appending a validated new revision."""

from app.contracts.evidence import GallerySnapshot
from app.contracts.identity import (
    Assignment,
    ManualConstraint,
    MatchingPolicy,
    Player,
    RiskReview,
    Segment,
    Template,
)
from app.storage.discovery import read_discovery_reports
from app.storage.repositories import _json
from app.storage.revision_states import REVIEW_KEYS, historical_state
from app.storage.snapshot import read_identity_snapshot
from app.storage.workflow_plans import put_workflow_plan
from app.workflows._consistency import (
    validate_evidence_context,
    validate_segment_frames,
    validate_templates,
)
from app.workflows._review_ops import current_evidence, validate_constraints


def restore_state(repo, match_id, target, revision, policies, settings):
    db = repo.connection
    state = read_identity_snapshot(db, match_id)
    target_state = historical_state(db, match_id, target)
    # Preserve evidence, players and audit. Later replacement segments are retired.
    target_segments = {s["segment_id"]: s for s in target_state["segments"]}
    for row in db.execute("SELECT json FROM segments WHERE match_id=?", (match_id,)).fetchall():
        segment = Segment.model_validate_json(row[0])
        value = target_segments.get(segment.segment_id)
        if value is not None:
            segment = Segment.model_validate(value)
        elif segment.supersedes_segment_id:
            segment.active = False
            segment.supersedes_segment_id = None
        repo.replace_segment(segment, committed_revision=revision)
    players = {p["player_id"]: p for p in target_state["players"]}
    for row in db.execute("SELECT json FROM players WHERE match_id=?", (match_id,)).fetchall():
        player = Player.model_validate_json(row[0])
        player.alias_of = players.get(player.player_id, {}).get("alias_of")
        repo.put_player(player, update=True)
    # New assignment/template rows make a restore itself a historical fact.
    for table, model, writer in (
        ("assignments", Assignment, repo.put_assignment),
        ("templates", Template, repo.put_template),
    ):
        db.execute(f"UPDATE {table} SET current=0 WHERE match_id=?", (match_id,))
        for value in target_state[table]:
            writer(model.model_validate(value), committed_revision=revision, replace=True)
    for table, keys in REVIEW_KEYS.items():
        db.execute(f"UPDATE {table} SET current=0 WHERE match_id=?", (match_id,))
        for value in target_state[table]:
            value = {**value, "revision": revision}
            columns = ["match_id", *keys, "revision", "json"]
            extra = []
            if table == "constraints":
                extra = ["segment_id", "player_id", "related_segment_id"]
                columns += extra
            db.execute(
                f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                (
                    match_id,
                    *(value[k] for k in keys),
                    revision,
                    _json(value),
                    *(value[k] for k in extra),
                ),
            )
    # Plans are a derived cache; their complete prior values live in revision_states.
    db.execute("DELETE FROM workflow_plans WHERE match_id=?", (match_id,))
    for plan in target_state["workflow_plans"]:
        put_workflow_plan(
            db, match_id, plan["segment_id"], plan["feature_space_id"], revision, plan
        )

    def records(table, model, selected=False):
        return [
            model.model_validate_json(row[0])
            for row in db.execute(
                f"SELECT json FROM {table} WHERE match_id=?"
                + (" AND current=1" if selected else ""),
                (match_id,),
            )
        ]

    final = state.gallery.model_copy(deep=True)
    final.revision = revision
    final.evidence.segments = records("segments", Segment)
    final.players = [p for p in records("players", Player) if p.alias_of is None]
    final.assignments = records("assignments", Assignment, True)
    final.templates = records("templates", Template, True)
    final.risk_reviews = records("risk_reviews", RiskReview, True)
    constraints = records("constraints", ManualConstraint, True)
    final = GallerySnapshot.model_validate(final.model_dump())
    validate_evidence_context(repo, final)
    validate_segment_frames(final.evidence.segments)
    if repo.check_segment_overlaps(match_id) or repo.check_alias_cycles(match_id):
        raise ValueError("restored identity segments overlap or player aliases cycle")
    final.evidence = current_evidence(final.evidence, clean_reviews=repo.clean_reviews(match_id))
    validate_constraints(final, constraints)
    active = {s.segment_id for s in final.evidence.segments if s.active}
    players = {p.player_id for p in final.players}
    for assignment in final.assignments:
        # Inactive parents keep their historical assignment after a split.
        if (
            assignment.status == "confirmed"
            and assignment.segment_id in active
            and assignment.player_id not in players
        ):
            raise ValueError("restored assignment refers to a merged or missing player")
    policy_map = {p.feature_space_id: p for p in policies}
    validate_templates(
        final,
        state.feature_values,
        lambda space: (
            policy_map.get(space)
            or (
                settings.matching.policy_for(space)
                if settings
                else MatchingPolicy(feature_space_id=space)
            )
        ),
        discovery_reports=read_discovery_reports(db, match_id),
    )
