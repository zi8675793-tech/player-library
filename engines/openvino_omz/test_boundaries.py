# ruff: noqa: E402, I001
"""Run in the isolated engine environment; generated videos live only under var."""

import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from run import BASE, DEMO, FullCapture, Observer

from mc_tracker.mct import MultiCameraTracker
from mc_tracker.sct import Track
from utils.misc import read_py_config


@pytest.mark.parametrize("lengths", [(1,), (0,), (1, 3), (3, 1), (2, 2), (0, 2)])
def test_exact_eof_and_last_frame(lengths):
    folder = BASE / "boundary-videos"
    folder.mkdir(exist_ok=True)
    paths = []
    for i, length in enumerate(lengths):
        path = folder / (f"{lengths}-{i}.avi")
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (32, 64))
        for n in range(length):
            w.write(np.full((64, 32, 3), n * 20, np.uint8))
        w.release()
        paths.append(path)
    cap = FullCapture(paths)
    seen = [0] * len(lengths)
    config = read_py_config(str(DEMO / "configs/person.py"))
    tracker = MultiCameraTracker(len(paths), None, config.sct_config, **vars(config.mct_config))
    while (item := cap.read()) is not None:
        frames, real = item
        seen = [n + int(r) for n, r in zip(seen, real)]
        tracker.process(frames, [[] for _ in paths], [[] for _ in paths])
    assert seen == list(lengths)
    assert tracker.time == max(lengths)
    assert tracker.get_all_tracks_history() == [[] for _ in paths]
    assert cap.read() is None
    cap.close()


def test_readonly_observer_interpolation_merge_reuse(tmp_path):
    observer = Observer("boundary")
    random.seed(100)
    sys.setprofile(observer.profile)
    try:
        t = Track(1, 0, (0, 0, 10, 30), 0)
        t.add_detection((0, 0, 10, 30), None, 2, 2)
        other = Track(1, 0, (0, 0, 10, 30), 4)
        t.merge_continuation(other, 10)
        recycled = Track(1, 0, (0, 0, 10, 30), 6)
    finally:
        sys.setprofile(None)
    assert len({observer.key(t), observer.key(other), observer.key(recycled)}) == 3
    assert observer.origins[observer.key(t)] == {
        0: "detection",
        1: "interpolation",
        2: "detection",
        3: "interpolation",
        4: "detection",
    }
    assert observer.events[-2]["kind"] == "continuation"
    assert t.timestamps == list(range(5))
    assert t.id == recycled.id


def test_observer_preserves_native_history_and_tracks_deleted_births():
    config = read_py_config(str(DEMO / "configs/person.py"))

    def execute(observe):
        random.seed(100)
        tracker = MultiCameraTracker(2, None, config.sct_config, **vars(config.mct_config))
        observer = Observer("native-test")
        for step in range(45):
            observer.step = step
            frames = [np.zeros((64, 32, 3), np.uint8)] * 2
            detections = [
                [(0, 0, 10, 30)] if step < 3 or step >= 10 else [],
                [(10, 0, 20, 30)] if step < 25 else [],
            ]
            if observe:
                sys.setprofile(observer.profile)
            try:
                tracker.process(frames, detections, [[], []])
            finally:
                sys.setprofile(None)
        return tracker, observer

    plain, _ = execute(False)
    tracked, observer = execute(True)
    assert plain.get_all_tracks_history() == tracked.get_all_tracks_history()
    snapshot = observer.snapshot(tracked)
    assert any(not t["native_retained"] for t in snapshot["tracks"])
    assert len({t["instance"] for t in snapshot["tracks"]}) == len(snapshot["tracks"])
    assert all(len(t["origins"]) == len(t["timestamps"]) for t in snapshot["tracks"])
