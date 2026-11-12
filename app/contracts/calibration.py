"""Contracts for distance calibration profiles, candidate evidence, and space-bound diagnostics."""

import hashlib
import json
from datetime import UTC, datetime
from typing import Literal

from pydantic import ConfigDict, Field, NonNegativeFloat, NonNegativeInt, model_validator

from app.contracts.common import (
    ContractModel,
    Identifier,
    PositiveInt,
    Probability,
    Sha256,
)


class DistanceSemantics(ContractModel):
    """Explicit mathematical and backend semantics of the distance metric."""

    backend: Identifier = "kpr"
    layout: Identifier = "parts"
    normalization: Identifier = "l2"
    distance_metric: Identifier = "euclidean_sqrt_visibility_mean_div2"
    part_weighting: Identifier = "sqrt_visibility"


class DistanceQuantiles(ContractModel):
    """Summary statistics and quantiles of distance distributions."""

    count: NonNegativeInt
    min: NonNegativeFloat
    max: NonNegativeFloat
    mean: NonNegativeFloat
    std: NonNegativeFloat
    p01: NonNegativeFloat
    p05: NonNegativeFloat
    p10: NonNegativeFloat
    p25: NonNegativeFloat
    p50: NonNegativeFloat
    p75: NonNegativeFloat
    p90: NonNegativeFloat
    p95: NonNegativeFloat
    p99: NonNegativeFloat


class CandidatePairExclusion(ContractModel):
    """Detailed record of a sample pair excluded from calibration evidence with reason."""

    pair_type: Literal["candidate_positive", "physical_negative"]
    left_sample_id: Identifier
    right_sample_id: Identifier
    clip_id: Identifier
    reason: Identifier
    details: dict = Field(default_factory=dict)


class DistanceCalibrationProfile(ContractModel):
    """A frozen, serializable distance calibration profile strictly bound to a feature space.

    A calibrated profile defines the operational near and far thresholds for a specific
    feature space. It explicitly records its evidence source, extraction config, and
    statistical quantiles. Silent fallback between different feature spaces is forbidden.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)

    schema_version: Literal["distance-calibration-profile-v1"] = "distance-calibration-profile-v1"
    profile_id: Identifier
    feature_space_id: Identifier
    calibration_version: Identifier = "stage1-diagnostic-v1"
    distance_metric: Identifier = "euclidean_sqrt_visibility_mean_div2"
    distance_semantics: DistanceSemantics = Field(default_factory=DistanceSemantics)
    status: Literal["uncalibrated", "unvalidated", "provisional", "calibrated"] = "uncalibrated"
    near_distance: Probability = 0.18
    far_distance: Probability = 0.45
    min_common_parts: PositiveInt = 2
    min_part_visibility: Probability = 0.2
    evidence_source: Identifier = "test-machine-evidence"
    evidence_fingerprint: Sha256 = "0" * 64
    config_fingerprint: Sha256 = "0" * 64
    created_at_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    candidate_positives_summary: DistanceQuantiles | None = None
    physical_negatives_summary: DistanceQuantiles | None = None
    exclusion_counts: dict[str, int] = Field(default_factory=dict)
    notes: str = ""

    @property
    def fingerprint(self) -> str:
        values = self.model_dump(mode="json", exclude={"notes", "created_at_utc"})
        encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @model_validator(mode="after")
    def validate_threshold_bounds(self) -> "DistanceCalibrationProfile":
        if self.near_distance >= self.far_distance:
            raise ValueError(
                f"near_distance ({self.near_distance}) must be strictly less than "
                f"far_distance ({self.far_distance})"
            )
        if self.status == "calibrated" and (
            self.evidence_fingerprint == "0" * 64 or not self.notes
        ):
            raise ValueError(
                "calibrated status requires non-empty evidence fingerprint and provenance notes"
            )
        return self


class CalibrationDiagnosticReport(ContractModel):
    """Full diagnostic report of feature space distance distribution and candidate evidence."""

    schema_version: Literal["calibration-diagnostic-report-v1"] = "calibration-diagnostic-report-v1"
    profile: DistanceCalibrationProfile
    candidate_positive_count: NonNegativeInt
    physical_negative_count: NonNegativeInt
    total_exclusions_count: NonNegativeInt
    exclusion_breakdown: dict[str, int]
    overlap_interval: list[NonNegativeFloat] = Field(default_factory=list)
    separation_diagnosis: str
    positive_pairs: list[dict] = Field(default_factory=list)
    negative_pairs: list[dict] = Field(default_factory=list)
    exclusions: list[CandidatePairExclusion] = Field(default_factory=list)
