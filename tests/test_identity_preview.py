"""The documented public example is executable and remains a noncommitting preview."""

import json
import subprocess
import sys
from pathlib import Path

from app.contracts.evidence import PreparedEvidence
from app.contracts.identity import AssociationResult, GalleryPlan, MatchingPolicy, Resolution


def test_documented_preview_runs_the_public_chain_without_calibrated_confirmation():
    output = subprocess.run(
        [sys.executable, "-m", "examples.identity_preview"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.stdout)
    policy = MatchingPolicy.model_validate(result["policy"])
    prepared = PreparedEvidence.model_validate(result["prepared"])
    association = AssociationResult.model_validate(result["associations"][0])
    resolution = Resolution.model_validate(result["resolutions"][0])
    gallery = GalleryPlan.model_validate(result["gallery_plan"])
    assert not policy.calibrated and not policy.auto_assign_enabled
    assert len(association.candidates) >= 2
    assert association.common_bucket_count == 3
    assert association.common_query_coverage == 1.0
    assert association.policy_fingerprint == prepared.policy_fingerprint == policy.fingerprint
    assert resolution.decision == "pending" and resolution.reasons == ["policy_uncalibrated"]
    assert resolution.assignment is None
    assert gallery.changes == []
