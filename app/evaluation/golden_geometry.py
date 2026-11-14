"""Identity-blind, conservative same-frame correspondence for frozen predictions.

No inference, labels, track IDs, time search or appearance features are consulted.
"""

from dataclasses import asdict, dataclass
from math import isfinite


@dataclass(frozen=True)
class GeometryRule:
    min_iou: float = 0.5
    min_margin: float = 0.1
    duplicate_iou: float = 0.85
    physical_separation_iou: float = 0.1

    def as_dict(self):
        return asdict(self)


def iou(a, b):
    for box in (a, b):
        if len(box) != 4 or not all(isfinite(x) for x in box):
            raise ValueError("Invalid box")
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Empty box")
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area = lambda x: (x[2] - x[0]) * (x[3] - x[1])  # noqa: E731
    return intersection / (area(a) + area(b) - intersection)


def match_boxes(left, right, rule=GeometryRule()):
    """Return one result per left box; require mutual best + both-side margin.

    Near-duplicate nodes remain in competition and are never silently dropped.
    This deliberately rejects ambiguous components instead of resolving them by ID.
    """
    matrix = [[iou(a, b) for b in right] for a in left]
    left_dup = {
        i
        for i, a in enumerate(left)
        if any(i != j and iou(a, b) >= rule.duplicate_iou for j, b in enumerate(left))
    }
    right_dup = {
        i
        for i, a in enumerate(right)
        if any(i != j and iou(a, b) >= rule.duplicate_iou for j, b in enumerate(right))
    }
    results = []
    for i, row in enumerate(matrix):
        candidates = sorted(range(len(right)), key=lambda j: (-row[j], j))
        if not candidates:
            results.append({"status": "no_candidate", "index": None, "best_iou": 0.0})
            continue
        j = candidates[0]
        column = sorted((matrix[k][j] for k in range(len(left))), reverse=True)
        row_margin = row[j] - (row[candidates[1]] if len(candidates) > 1 else 0.0)
        col_margin = column[0] - (column[1] if len(column) > 1 else 0.0)
        status = "matched"
        if row[j] < rule.min_iou:
            status = "below_iou"
        elif i in left_dup or j in right_dup:
            status = "duplicate_ambiguity"
        elif row[j] != column[0] or min(row_margin, col_margin) < rule.min_margin:
            status = "competition_ambiguity"
        results.append(
            {
                "status": status,
                "index": j if status == "matched" else None,
                "best_index": j,
                "best_iou": row[j],
                "row_margin": row_margin,
                "column_margin": col_margin,
            }
        )
    return results


def synthetic_checks(rule=GeometryRule()):
    cases = [
        ("exact", [[0, 0, 100, 200]], [[0, 0, 100, 200]], "matched"),
        ("smoothing_20_percent", [[0, 0, 100, 200]], [[20, 0, 120, 200]], "matched"),
        ("large_shift", [[0, 0, 100, 200]], [[40, 0, 140, 200]], "below_iou"),
        ("missing", [[0, 0, 100, 200]], [], "no_candidate"),
        ("duplicate", [[0, 0, 100, 200]], [[0, 0, 100, 200]] * 2, "duplicate_ambiguity"),
        (
            "overlap_competition",
            [[10, 0, 110, 200]],
            [[0, 0, 100, 200], [20, 0, 120, 200]],
            "competition_ambiguity",
        ),
        ("old_duplicates", [[0, 0, 100, 200]] * 2, [[0, 0, 100, 200]], "duplicate_ambiguity"),
    ]
    output = []
    for name, left, right, expected in cases:
        actual = match_boxes(left, right, rule)
        if any(r["status"] != expected for r in actual):
            raise AssertionError(name)
        output.append({"case": name, "expected": expected, "actual": actual, "passed": True})
    return output
