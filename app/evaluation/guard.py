"""Strict isolation barrier between prediction inference and ground truth scoring.

Mandates:
1. No ground truth files, annotations or target directories may ever enter the inference pipeline;
2. Inference functions cannot be invoked from inside scoring contexts;
3. Scoring functions cannot read ground truth answers until predictions and artifact hashes are frozen.
"""

import hashlib
import inspect
import json
from pathlib import Path


class GroundTruthLeakageError(RuntimeError):
    """Raised when ground truth or scoring context leaks into inference."""


class PrematureScoringError(RuntimeError):
    """Raised when scoring attempts to read answers before predictions are frozen."""


FORBIDDEN_INFERENCE_KEYWORDS = (
    "annotations.csv",
    "test/target",
    "/target/",
    "ground_truth",
    "person_gt_id",
)

SCORING_CONTEXT_MARKERS = (
    "score_frozen_answers",
    "score_frozen_predictions",
    "evaluate_supervised_metrics",
)


def assert_inference_isolation(caller_context: str = "") -> None:
    """Assert that the current call stack does not originate from a post-freeze scoring context."""
    frame = inspect.currentframe()
    try:
        while frame:
            filename = frame.f_code.co_filename
            for marker in SCORING_CONTEXT_MARKERS:
                if marker in filename:
                    raise GroundTruthLeakageError(
                        f"Architectural isolation violation: inference function invoked within "
                        f"scoring context ({marker} in {filename}): {caller_context}"
                    )
            frame = frame.f_back
    finally:
        del frame


def assert_no_ground_truth_in_evidence(sources: list) -> None:
    """Verify that evidence source list contains no ground truth annotations or targets."""
    for source in sources:
        path_str = getattr(source, "path", str(source))
        for forbidden in FORBIDDEN_INFERENCE_KEYWORDS:
            if forbidden in path_str:
                raise GroundTruthLeakageError(
                    f"Forbidden ground truth source detected in evidence: {path_str}"
                )


def assert_frozen_before_scoring(run_dir: str | Path) -> dict[str, str]:
    """Verify all artifacts exist and match artifact-hashes.json before scoring may open answers."""
    run_path = Path(run_dir).resolve()
    manifest_path = run_path / "artifact-hashes.json"
    if not manifest_path.is_file():
        raise PrematureScoringError(
            f"Run directory {run_path} has not been frozen (artifact-hashes.json missing)"
        )
    with manifest_path.open("r", encoding="utf-8") as f:
        hashes = json.load(f)

    required_artifacts = {
        "predictions.json",
        "summary.json",
        "config.json",
        "input-manifest.json",
        "player-library.sqlite3",
    }
    missing = required_artifacts - set(hashes)
    if missing:
        raise PrematureScoringError(
            f"Frozen manifest in {run_path} is missing required artifacts: {sorted(missing)}"
        )

    for name, expected_hash in hashes.items():
        file_path = run_path / name
        if not file_path.is_file():
            raise PrematureScoringError(f"Frozen artifact file missing: {file_path}")
        digest = hashlib.sha256()
        with file_path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        actual_hash = digest.hexdigest()
        if actual_hash != expected_hash:
            raise PrematureScoringError(
                f"Frozen artifact tampered or modified: {file_path.name} "
                f"(expected {expected_hash}, got {actual_hash})"
            )
    return hashes
