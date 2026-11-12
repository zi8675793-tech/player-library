"""Immutable manifests of effective identity state, separate from raw evidence."""

import json

TABLES = (
    "segments",
    "players",
    "assignments",
    "templates",
    "risk_reviews",
    "template_reviews",
    "constraints",
    "sample_clean_reviews",
    "workflow_plans",
)
SELECTED = {
    "assignments",
    "templates",
    "risk_reviews",
    "template_reviews",
    "constraints",
    "sample_clean_reviews",
}
REVIEW_KEYS = {
    "risk_reviews": ("risk_id", "segment_id"),
    "template_reviews": ("sample_id", "player_id", "feature_space_id"),
    "constraints": ("constraint_id",),
    "sample_clean_reviews": ("sample_id",),
}


def capture_state(db, match_id, revision, *, legacy_checkpoint=False):
    state = {
        table: [
            json.loads(row[0])
            for row in db.execute(
                f"SELECT json FROM {table} WHERE match_id=?"
                + (" AND current=1" if table in SELECTED else ""),
                (match_id,),
            )
        ]
        for table in TABLES
    }
    state["legacy_checkpoint"] = legacy_checkpoint
    state["sample_clean_reviews"] = [
        {**json.loads(row[1]), "sample_id": row[0]}
        for row in db.execute(
            "SELECT sample_id,json FROM sample_clean_reviews WHERE match_id=? AND current=1",
            (match_id,),
        )
    ]
    db.execute(
        "INSERT INTO revision_states VALUES (?,?,?)",
        (match_id, revision, json.dumps(state, sort_keys=True)),
    )


def historical_state(db, match_id, target):
    row = db.execute(
        "SELECT json FROM revision_states WHERE match_id=? AND revision=?", (match_id, target)
    ).fetchone()
    if row:
        return json.loads(row[0])
    # Pre-upgrade restore deleted review history. Never invent missing facts.
    records = [
        (r[0], json.loads(r[1]))
        for r in db.execute(
            "SELECT revision,json FROM workflow_reviews WHERE match_id=? ORDER BY revision",
            (match_id,),
        )
    ]

    def verified_restore(revision):
        row = db.execute(
            "SELECT json FROM revision_states WHERE match_id=? AND revision=?", (match_id, revision)
        ).fetchone()
        return row is not None and not json.loads(row[0]).get("legacy_checkpoint", False)

    if any(
        c["kind"] == "restore_revision" and not verified_restore(revision)
        for revision, commands in records
        for c in commands
    ):
        raise ValueError(
            "legacy revision has no reliable state manifest after destructive restore; explicit review required"
        )
    state = {}
    for table in SELECTED:
        keys = REVIEW_KEYS.get(
            table, ("segment_id",) if table == "assignments" else ("template_id",)
        )
        latest = {}
        for row in db.execute(
            "SELECT json"
            + (",sample_id" if table == "sample_clean_reviews" else "")
            + f" FROM {table} WHERE match_id=? AND revision<=? ORDER BY revision",
            (match_id, target),
        ):
            value = json.loads(row[0])
            if table == "sample_clean_reviews":
                value["sample_id"] = row[1]
            latest[tuple(value[k] for k in keys)] = value
        state[table] = list(latest.values())
    state["players"] = [
        json.loads(r[0])
        for r in db.execute("SELECT json FROM players WHERE match_id=?", (match_id,))
    ]
    for player in state["players"]:
        player["alias_of"] = None
    players = {p["player_id"]: p for p in state["players"]}
    later_children = set()
    for revision, commands in records:
        for command in commands:
            if command["kind"] == "merge_players" and revision <= target:
                for alias in command["alias_player_ids"]:
                    players[alias]["alias_of"] = command["player_id"]
            if command["kind"] == "split_segment" and revision > target:
                later_children.update(s["segment_id"] for s in command["segments"])
    state["segments"] = []
    for row in db.execute("SELECT segment_id,json FROM segments WHERE match_id=?", (match_id,)):
        older = db.execute(
            "SELECT json FROM segment_review_history WHERE match_id=? AND segment_id=? AND revision>? ORDER BY revision LIMIT 1",
            (match_id, row[0], target),
        ).fetchone()
        value = json.loads(older[0] if older else row[1])
        if value["segment_id"] in later_children:
            value["active"] = False
            value["supersedes_segment_id"] = None
        state["segments"].append(value)
    # Legacy plans were a mutable cache, not historical truth. Rebuild by review.
    state["workflow_plans"] = []
    return state
