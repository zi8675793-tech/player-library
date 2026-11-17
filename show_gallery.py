"""Pretty-print the incremental player gallery from a frozen run directory.

Reads ONLY the machine-inference SQLite database produced by run_real_batches.py.
Never touches annotations.csv or test/target/.

Usage:
    uv run python show_gallery.py <run_dir>
"""

import json
import sys
from pathlib import Path

from app.storage.database import connect
from app.storage.dual_stream import read_dual_stream_snapshot


def main() -> None:
    run_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
    if run_dir is None or not (run_dir / "player-library.sqlite3").is_file():
        print("Usage: uv run python show_gallery.py <run_dir containing player-library.sqlite3>")
        raise SystemExit(2)

    conn = connect(run_dir / "player-library.sqlite3")
    try:
        snap = read_dual_stream_snapshot(conn, "t3-real-basketball")
        revision_row = conn.execute(
            "SELECT revision FROM matches WHERE match_id='t3-real-basketball'"
        ).fetchone()
        tables = {
            row[0]: conn.execute(f'SELECT COUNT(*) FROM "{row[0]}"').fetchone()[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
    finally:
        conn.close()

    players = sorted(snap.players, key=lambda p: p.player_id)
    assignments = snap.assignments
    decisions = snap.decisions

    # Segment lookup for detail
    seg_by_id = {s.segment_id: s for s in snap.segments}

    # Count decisions by outcome
    outcome_counts: dict[str, int] = {}
    for d in decisions.values():
        outcome_counts[d] = (
            outcome_counts.get(d, 0) + 1
            if isinstance(d, str)
            else outcome_counts.get(str(d), 0) + 1
        )

    print("=" * 88)
    print(f"球员档案库 (增量建库最终快照)  |  run: {run_dir.name}")
    print(
        f"revision: {revision_row[0] if revision_row else snap.revision}  |  "
        f"球员档案: {len(players)}  |  轨迹段: {len(snap.segments)}  |  "
        f"活跃原型: {sum(len(p.prototypes) for p in players)}"
    )
    print("=" * 88)

    for player in players:
        owned = sorted(sid for sid, pid in assignments.items() if pid == player.player_id)
        print(f"\n■ 球员档案 {player.player_id}")
        meta = getattr(player, "confirmed_jersey_number", None)
        print(f"  确认球衣号码 : {meta if meta is not None else '（未识别到号码，纯外观档案）'}")
        print(f"  活跃原型数   : {len(player.prototypes)}")
        for i, proto in enumerate(player.prototypes):
            dim = getattr(proto, "shape", None)
            desc = f"dim={dim[0] if dim else '?'}" if dim is not None else ""
            print(f"    - 原型 {i + 1:<2} {desc}")
        print(f"  归属轨迹段   : {len(owned)} 条")
        for sid in owned:
            seg = seg_by_id.get(sid)
            if seg is not None:
                print(
                    f"    - {sid}  ({seg.video_id} 帧 {seg.start_frame}~{seg.end_frame}, "
                    f"{seg.observation_count} 次观测, {seg.end_time_s - seg.start_time_s:.2f}s)"
                )

    print()
    print("=" * 88)
    print("决策统计（三态增量决策）:")
    for key in sorted(outcome_counts):
        print(f"  {key:<28} {outcome_counts[key]:>4} 段")
    print("-" * 88)
    print("数据库表行数:", json.dumps(tables, ensure_ascii=False))
    print("=" * 88)


if __name__ == "__main__":
    main()
