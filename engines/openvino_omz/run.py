# ruff: noqa: E402
"""A is an unchanged official subprocess; B observes unchanged stateful SCT/MCT."""

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "var/openvino-migration"
OMZ = BASE / "omz"
DEMO = OMZ / "demos/multi_camera_multi_target_tracking_demo/python"
sys.path.insert(0, str(DEMO))
import cv2
import numpy as np
from mc_tracker.mct import MultiCameraTracker
from openvino.runtime import Core, get_version
from utils.misc import read_py_config
from utils.network_wrappers import Detector, VectorCNN

SCHEMA = "omz-tracking-v1"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def convert(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x))


def dump(path, value):
    path.write_text(json.dumps(value, default=convert, allow_nan=False))


class FullCapture:
    """No prefetch or looping: one read and one submission per real frame."""

    def __init__(self, paths):
        self.caps = [cv2.VideoCapture(str(p)) for p in paths]
        if not all(c.isOpened() for c in self.caps):
            raise RuntimeError("Cannot open input")
        self.counts = [0] * len(paths)
        self.ended = [False] * len(paths)
        self.shapes = [
            (int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), 3)
            for c in self.caps
        ]
        self.fps = [c.get(cv2.CAP_PROP_FPS) for c in self.caps]

    def read(self):
        frames, real = [], []
        for i, cap in enumerate(self.caps):
            ok, frame = (False, None) if self.ended[i] else cap.read()
            if ok:
                self.counts[i] += 1
            else:
                self.ended[i] = True
                frame = np.zeros(self.shapes[i], np.uint8)
            frames.append(frame)
            real.append(ok)
        return (frames, real) if any(real) else None

    def close(self):
        for cap in self.caps:
            cap.release()


class Observer:
    """Python return-event observer. Never writes into any native tracker object."""

    def __init__(self, run_id):
        self.run_id = run_id
        self.objects = {}  # strong refs prevent Python object-address reuse
        self.origins = {}
        self.events = []
        self.step = -1

    def key(self, obj):
        return self.objects[id(obj)][0]

    def profile(self, frame, event, arg):
        if event != "return" or frame.f_code.co_filename != str(DEMO / "mc_tracker/sct.py"):
            return
        d, name = frame.f_locals, frame.f_code.co_name
        obj = d.get("self")
        if name == "__init__" and type(obj).__name__ == "Track":
            key = f"{self.run_id}:s{obj.cam_id}:t{len(self.objects)}"
            self.objects[id(obj)] = (key, obj)
            self.origins[key] = {obj.timestamps[0]: "detection"}
            self.events.append(dict(step=self.step, kind="birth", instance=key, native_id=obj.id))
        elif name == "add_detection":
            self.origins[self.key(obj)][d["timestamp"]] = "detection"
        elif name == "_interpolate":
            origins = self.origins[self.key(obj)]
            for t in obj.timestamps:
                if t not in origins:
                    origins[t] = "interpolation"
        elif name == "merge_continuation":
            key, other = self.key(obj), self.key(d["other"])
            self.origins[key].update(self.origins[other])
            self.events.append(
                dict(step=self.step, kind="continuation", survivor=key, absorbed=other)
            )

    def snapshot(self, tracker):
        retained = {id(t) for s in tracker.scts for t in s.tracks + s.history_tracks}
        tracks = []
        for key, t in self.objects.values():
            tracks.append(
                dict(
                    instance=key,
                    source=t.cam_id,
                    native_id=t.id,
                    native_retained=id(t) in retained,
                    timestamps=t.timestamps,
                    boxes=t.boxes,
                    origins=[self.origins[key].get(i, "unknown") for i in t.timestamps],
                    appearance=dict(
                        mean=t.f_avg.get() if t.f_avg.is_valid() else None,
                        count=t.f_avg.count,
                        clusters=t.f_clust.clusters,
                        cluster_counts=t.f_clust.clusters_sizes,
                    ),
                )
            )
        return dict(
            schema_version=SCHEMA,
            step_count=tracker.time,
            tracks=tracks,
            history=tracker.get_all_tracks_history(),
            events=self.events,
        )


def models():
    return [
        BASE / "models" / n / "FP32" / (n + ".xml")
        for n in ("person-detection-retail-0013", "person-reidentification-retail-0277")
    ]


def run_b(paths, out, prefix):
    out.mkdir()
    config = read_py_config(str(DEMO / "configs/person.py"))
    random.seed(config.random_seed)
    core = Core()
    det_model, reid_model = models()
    detector = Detector(core, str(det_model), config.obj_det.trg_classes, 0.6, "CPU", len(paths))
    reid = VectorCNN(core, str(reid_model), "CPU")
    tracker = MultiCameraTracker(
        len(paths),
        reid,
        config.sct_config,
        **vars(config.mct_config),
        visual_analyze=config.analyzer,
    )
    capture = FullCapture(paths)
    observer = Observer(out.parent.name)
    step = 0
    with (out / "frames.jsonl").open("w") as ledger, (out / "steps.jsonl").open("w") as steps:
        while True:
            item = capture.read()
            if item is None:
                break
            frames, real = item
            detector.run_async(frames, step)
            detections = detector.wait_and_grab()
            for i, valid in enumerate(real):
                if not valid:
                    detections[i] = []
                else:
                    ledger.write(
                        json.dumps(
                            dict(
                                source=i,
                                frame_idx=capture.counts[i] - 1,
                                step=step,
                                time_s=(capture.counts[i] - 1) / capture.fps[i],
                                decoded_sha256=hashlib.sha256(frames[i].tobytes()).hexdigest(),
                                boxes=[d[0] for d in detections[i]],
                                scores=[float(d[1]) for d in detections[i]],
                            )
                        )
                        + "\n"
                    )
            observer.step = step
            sys.setprofile(observer.profile)
            try:
                tracker.process(
                    frames, [[d[0] for d in ds] for ds in detections], [[] for _ in paths]
                )
            finally:
                sys.setprofile(None)
            states = []
            for i, sct in enumerate(tracker.scts):
                for t in sct.tracks:
                    states.append(
                        dict(
                            instance=observer.key(t),
                            source=i,
                            native_id=t.id,
                            start=t.timestamps[0],
                            end=t.timestamps[-1],
                            box=t.boxes[-1],
                            current=t.timestamps[-1] == step,
                            origin=observer.origins[observer.key(t)].get(step),
                            real_source=real[i],
                            label=("ID " + str(t.id if len(t) > sct.time_window else -1))
                            if t.timestamps[-1] == step
                            else None,
                        )
                    )
            steps.write(
                json.dumps(dict(step=step, real=real, tracks=states), default=convert) + "\n"
            )
            step += 1
            if step == prefix:
                dump(out / "prefix.json", observer.snapshot(tracker))
            if step % 50 == 0:
                print("B step", step, capture.counts, flush=True)
    detector.infer_queue.wait_all()
    reid.infer_queue.wait_all()
    assert not detector.outputs and not reid.outputs
    dump(out / "final.json", observer.snapshot(tracker))
    dump(
        out / "summary.json",
        dict(
            step_count=step,
            real_frames=capture.counts,
            ended=capture.ended,
            end_reason="all_sources_eof",
            queues_drained=True,
        ),
    )
    capture.close()


def audit(paths, out):
    cap = FullCapture(paths)
    counts = [0] * len(paths)
    h = [hashlib.sha256() for _ in paths]
    with (out / "decode-audit.jsonl").open("w") as f:
        while True:
            item = cap.read()
            if item is None:
                break
            frames, real = item
            for i, valid in enumerate(real):
                if valid:
                    digest = hashlib.sha256(frames[i].tobytes()).hexdigest()
                    f.write(
                        json.dumps(dict(source=i, frame_idx=counts[i], decoded_sha256=digest))
                        + "\n"
                    )
                    h[i].update(digest.encode())
                    counts[i] += 1
    metadata = [
        dict(
            source=i,
            path=str(p),
            sha256=sha(p),
            decoded_frames=counts[i],
            fps=cap.fps[i],
            shape=cap.shapes[i],
            decoded_digest=h[i].hexdigest(),
        )
        for i, p in enumerate(paths)
    ]
    cap.close()
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, nargs="+")
    args = parser.parse_args()
    out = args.run_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "A").exists() and not (out / "B").exists(), "Never overwrite a run"
    paths = [
        p.resolve()
        for p in (
            args.input
            or sorted((ROOT.parent / "test/video").glob("*.mp4"), key=lambda p: int(p.stem))
        )
    ]
    start = time.time()
    sources = audit(paths, out)
    pinned = [p for p in OMZ.rglob("*") if p.is_file() and ".git" not in p.parts]
    pinned += list((BASE / "models").rglob("*.xml")) + list((BASE / "models").rglob("*.bin"))
    manifest = dict(
        schema_version=SCHEMA,
        run_id=out.name,
        sources=sources,
        implementation={
            str(p.relative_to(ROOT)): sha(p)
            for p in [
                *Path(__file__).parent.glob("*.py"),
                *Path(__file__).parent.glob("*.txt"),
                ROOT / "app/contracts/tracking.py",
            ]
        },
        seed=100,
        detector_threshold=0.6,
        device="CPU",
        commit=subprocess.check_output(
            ["git", "-C", str(OMZ), "rev-parse", "HEAD"], text=True
        ).strip(),
        native_diff=subprocess.check_output(["git", "-C", str(OMZ), "diff", "HEAD"], text=True),
        engine_files={str(p.relative_to(BASE)): sha(p) for p in pinned},
        runtime=get_version(),
        platform=platform.platform(),
        machine=platform.machine(),
        dependencies=subprocess.check_output(
            ["uv", "pip", "freeze", "--python", sys.executable], text=True
        ),
        preprocessing="Official IEModel: OpenCV BGR uint8 resize INTER_LINEAR to NCHW model shape; no external normalization; ReID native SCT crops; FP32 IR",
        condition="A native entry; B EOF scheduling adaptation; no ground truth access",
    )
    assert manifest["native_diff"] == ""
    dump(out / "manifest.json", manifest)
    a = out / "A"
    a.mkdir()
    cmd = [
        sys.executable,
        str(DEMO / "multi_camera_multi_target_tracking_demo.py"),
        "-i",
        *map(str, paths),
        "--m_detector",
        str(models()[0]),
        "--m_reid",
        str(models()[1]),
        "--config",
        str(DEMO / "configs/person.py"),
        "--no_show",
        "--history_file",
        str(a / "history.json"),
        "--save_detections",
        str(a / "detections.json"),
    ]
    dump(a / "command.json", cmd)
    with (a / "stdout.log").open("w") as log:
        subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
    ds = json.loads((a / "detections.json").read_text())
    n = len(ds[0])
    assert all(len(s) == n and [d["frame_id"] for d in s] == list(range(n)) for s in ds)
    dump(
        a / "summary.json",
        dict(
            step_count=n,
            processed_ranges=[[0, n] for _ in paths],
            end_reason="native capture EOF/process flag; pending lookahead not submitted",
            real_processed=n * len(paths),
            first_eof_sources=[
                s["source"]
                for s in sources
                if s["decoded_frames"] == min(x["decoded_frames"] for x in sources)
            ],
        ),
    )
    print("A completed", n, flush=True)
    run_b(paths, out / "B", n)
    prefix = json.loads((out / "B/prefix.json").read_text())
    ah = json.loads((a / "history.json").read_text())
    # Native history order is part of the comparison, including IDs and smoothed boxes.
    same = prefix["history"] == ah
    bframes = [json.loads(s) for s in (out / "B/frames.jsonl").read_text().splitlines()]
    aframes = [json.loads(s) for s in (out / "decode-audit.jsonl").read_text().splitlines()]
    assert [(r["source"], r["frame_idx"], r["decoded_sha256"]) for r in bframes] == [
        (r["source"], r["frame_idx"], r["decoded_sha256"]) for r in aframes
    ]
    detsame = all(
        {"frame_id": r["step"], "boxes": r["boxes"], "scores": r["scores"]}
        == ds[r["source"]][r["step"]]
        for r in bframes
        if r["step"] < n
    )
    dump(
        out / "verification.json",
        dict(
            prefix_step=n,
            history_exact_equal=same,
            detections_exact_equal=detsame,
            frame_ledger_exact_equal=True,
            real_frames=len(bframes),
            elapsed_s=time.time() - start,
        ),
    )
    assert same and detsame, "Investigate prefix discrepancy before handoff"
    dump(
        out / "artifact-hashes.json",
        {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()},
    )
    print("VERIFIED", out, flush=True)


if __name__ == "__main__":
    main()
