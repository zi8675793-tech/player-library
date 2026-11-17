"""Boundary tests use synthetic geometry/truth only; no evaluation answers."""

import copy
import itertools
import json

import pytest

from app.evaluation.omz_geometry import GeometryRule, match_boxes, synthetic_checks
from app.evaluation.omz_io import array_items, sha, write_json
from app.evaluation.omz_prepare import correspond_frame, physical_pairs
from app.evaluation.omz_score import (
    align_identities,
    identity_diagnostics,
    score_frozen_predictions,
    score_rows,
    verify_geometry,
)


def observation(box=(0, 0, 100, 200)):
    return {"video_id": "v", "track_id": "old", "frame_idx": 10, "bbox_xyxy": list(box)}


def point(identity="new", origin="detection", **extra):
    return {
        "box": [0, 0, 100, 200],
        "identity": identity,
        "origin": origin,
        "instance": "instance",
        "source": 0,
        "frame_idx": 10,
        **extra,
    }


def metric_rows(counts):
    return [
        {"truth": truth, "identity": pid, "status": "matched", "video_id": str(i % 2)}
        for (truth, pid), count in counts.items()
        for i in range(count)
    ]


def test_synthetic_geometry_pre_label_rules():
    assert len(synthetic_checks()) == 7


def test_global_matching_cannot_reuse_one_detection():
    result = match_boxes([[0, 0, 100, 200], [20, 0, 120, 200]], [[0, 0, 100, 200]])
    assert sum(r["status"] == "matched" for r in result) == 1


def test_geometry_and_competition_independent_of_labels_and_ids():
    old, native = [observation()], [point()]
    a, _ = correspond_frame(old, [[0, 0, 100, 200]], native, True, GeometryRule())
    old[0]["truth"] = "arbitrary permuted answer"
    old[0]["track_id"] = "new"
    native[0]["identity"] = "old"
    b, _ = correspond_frame(old, [[0, 0, 100, 200]], native, True, GeometryRule())
    assert a[0]["old_to_raw"] == b[0]["old_to_raw"]
    assert a[0]["raw_to_native"] == b[0]["raw_to_native"]
    assert b[0]["identity"] == "old"


@pytest.mark.parametrize("origin", ["interpolation", "prediction", "unknown"])
def test_non_detection_never_counts_as_real(origin):
    rows, members = correspond_frame(
        [observation()], [[0, 0, 100, 200]], [point(origin=origin)], True, GeometryRule()
    )
    assert rows[0]["identity"] is None
    assert rows[0]["status"] == "no_retained_real_detection"
    assert not members


def test_missing_or_unprocessed_frame_stays_in_denominator():
    for processed in (True, False):
        rows, _ = correspond_frame([observation()], [], [], processed, GeometryRule())
        scored = score_rows([{**rows[0], "truth": "27"}], {"27": "x"})
        assert len(scored) == 1 and not scored[0]["correct"]
        assert scored[0]["status"] == (
            "detection_missing_or_geometry" if processed else "unprocessed"
        )


def test_duplicate_native_points_are_ambiguous_even_same_identity():
    rows, _ = correspond_frame(
        [observation()],
        [[0, 0, 100, 200]],
        [point(), point(instance="other")],
        True,
        GeometryRule(),
    )
    assert rows[0]["status"] == "native_duplicate_ambiguity"
    assert rows[0]["identity"] is None


def test_all_people_one_identity_exposes_known_errors():
    rows = metric_rows({("27", "all"): 70, ("15", "all"): 48})
    aligned = align_identities(rows)
    scored = score_rows(rows, aligned)
    assert sum(r["correct"] for r in scored) == 70
    assert sum(r["outcome"] == "known_wrong_identity" for r in scored) == 48
    assert identity_diagnostics(rows)["mixed_identities"] == {"all": {"27": 70, "15": 48}}


def test_fragments_cannot_be_stitched_for_score_or_duration():
    rows = metric_rows({("27", "a"): 35, ("27", "b"): 35})
    aligned = align_identities(rows)
    assert sum(r["correct"] for r in score_rows(rows, aligned)) == 35
    assert not score_rows(metric_rows({("27", "b"): 1}), {"27": "a"})[0]["correct"]
    assert identity_diagnostics(rows)["split_truth_count"] == 1


def test_alignment_is_global_injective_optimum():
    counts = {("a", "1"): 10, ("a", "2"): 9, ("b", "1"): 9, ("b", "2"): 0}
    rows = metric_rows(counts)
    aligned = align_identities(rows)
    assert aligned == {"a": "2", "b": "1"}
    for n in range(1, 4):
        counts = {(str(i), str(j)): ((i + 3) * (j + 7)) % 5 for i in range(n) for j in range(n + 1)}
        rows = metric_rows(counts)
        result = sum(r["correct"] for r in score_rows(rows, align_identities(rows)))
        exhaustive = max(
            sum(counts[str(i), p[i]] for i in range(n))
            for p in itertools.permutations(map(str, range(n + 1)), n)
        )
        assert result == exhaustive


def test_physical_conflict_requires_same_source_real_frame():
    a = point(raw_detection_confirmed=True, raw_box=[0, 0, 100, 200])
    b = point(instance="b", raw_detection_confirmed=True, raw_box=[300, 0, 400, 200])
    assert physical_pairs([a, b], GeometryRule())[0]["status"] == "physical_conflict"
    for changed in ({"source": 1}, {"frame_idx": 11}, {"raw_detection_confirmed": False}):
        assert not physical_pairs([a, {**b, **changed}], GeometryRule())


def test_matching_reordering_does_not_change_geometric_answer():
    left = [[0, 0, 100, 200], [300, 0, 400, 200]]
    right = copy.deepcopy(left)
    for order in itertools.permutations(right):
        result = match_boxes(left, order)
        assert [order[r["index"]] for r in result] == left


def test_streaming_array_ignores_key_like_strings(tmp_path):
    path = tmp_path / "large.json"
    data = {
        "note": 'string containing "tracks": [',
        "tracks": [{"s": "x" * 80000}, {"a": 1}],
        "history": [[1], [2]],
        "events": ["unused"],
    }
    path.write_text(json.dumps(data))
    assert list(array_items(path, "tracks")) == data["tracks"]
    assert list(array_items(path, "history")) == data["history"]


def test_geometry_tampering_rejected_before_labels(tmp_path, monkeypatch):
    geometry = tmp_path / "geometry"
    geometry.mkdir()
    path = geometry / "rules.json"
    write_json(path, {"threshold": 0.5})
    write_json(geometry / "geometry-freeze.json", {"files": {"rules.json": sha(path)}})
    write_json(path, {"threshold": 0.1})
    called = []
    monkeypatch.setattr("app.evaluation.omz_score.load_truth", lambda *args: called.append(True))
    with pytest.raises(ValueError, match="Changed frozen geometry"):
        score_frozen_predictions(geometry, tmp_path, tmp_path / "out")
    assert not called
    assert not (tmp_path / "out").exists()


def test_geometry_path_escape_rejected(tmp_path):
    external = tmp_path / "external"
    external.write_text("a")
    geometry = tmp_path / "g"
    geometry.mkdir()
    write_json(geometry / "geometry-freeze.json", {"files": {"../external": sha(external)}})
    with pytest.raises(ValueError):
        verify_geometry(geometry)
