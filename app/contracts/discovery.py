"""Uncalibrated, evidence-constrained discovery; independent of human approval."""

from typing import Any, Literal

from pydantic import Field, model_validator

from app.contracts.calibration import DistanceCalibrationProfile
from app.contracts.common import (
    ContractModel,
    Identifier,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    Probability,
    Sha256,
)


class DiscoveryPolicy(ContractModel):
    version: Literal["complete_support_v1", "complete_support_v2", "complete_support_v3"] = (
        "complete_support_v1"
    )
    feature_space_id: Identifier | None = None
    calibration: Literal["unvalidated"] = "unvalidated"
    support_mode: Literal[
        "absolute",
        "cooccurrence_contrastive",
        "compensated_contrastive",
        "constrained_manifold",
    ] = "absolute"
    clustering_threshold: NonNegativeFloat = 0.28
    existing_player_match_threshold: NonNegativeFloat | None = None
    short_segment_match_threshold: NonNegativeFloat | None = None
    linkage: Literal["complete", "average", "single"] = "average"
    enable_geometric_negative_compensation: bool = False
    compensation_min_common_parts: PositiveInt = 4
    compensation_min_part_visibility: Probability = 0.35
    compensation_max_distance: Probability = 0.22
    compensation_max_raw_distance: Probability = 0.28
    compensation_rival_margin: NonNegativeFloat = 0.06
    # Decoupled near thresholds
    intra_segment_near_distance: Probability = 0.18
    inter_segment_tau: Probability = 0.18
    tau: Probability = 0.18
    # Decoupled far and upper bound thresholds
    candidate_far_distance: Probability = 0.45
    negative_evidence_distance: Probability = 0.45
    far_distance: Probability = 0.45
    max_pair_distance: Probability = 0.30
    # Decoupled margin thresholds
    contrastive_margin: NonNegativeFloat = 0.04
    counter_evidence_margin: NonNegativeFloat = 0.04
    rival_margin: NonNegativeFloat = 0.04
    margin: NonNegativeFloat = 0.04
    min_common_parts: PositiveInt = 2
    min_part_visibility: Probability = 0.2
    bucket_s: PositiveFloat = 0.5
    min_support_buckets: PositiveInt = Field(default=3, ge=3)
    min_match_buckets: PositiveInt = Field(default=2, ge=2)
    min_coverage: Probability = 0.8
    max_templates_per_group: PositiveInt = Field(default=6, ge=2)
    cooccurrence_min_buckets: PositiveInt = Field(default=3, ge=3)
    cooccurrence_separation_heights: PositiveFloat = 0.5
    representation_mode: Literal["single_pooled", "multi_prototype"] = "multi_prototype"
    max_prototypes_per_segment: PositiveInt = Field(default=4, ge=1)
    prototype_near_distance: Probability = 0.18
    prototype_far_distance: Probability = 0.45
    enable_group_acceptance: bool = True
    bridge_pruning: bool = True
    rival_margin_check: bool = True
    calibration_profile: DistanceCalibrationProfile | None = None
    calibration_fingerprint: Sha256 | None = None

    @model_validator(mode="before")
    @classmethod
    def sync_decoupled_parameters(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        prof = data.get("calibration_profile")
        if prof is not None:
            if hasattr(prof, "feature_space_id"):
                data.setdefault("feature_space_id", prof.feature_space_id)
                data.setdefault("intra_segment_near_distance", prof.near_distance)
                data.setdefault("inter_segment_tau", prof.near_distance)
                data.setdefault("tau", prof.near_distance)
                data.setdefault("prototype_near_distance", prof.near_distance)
                data.setdefault("candidate_far_distance", prof.far_distance)
                data.setdefault("negative_evidence_distance", prof.far_distance)
                data.setdefault("far_distance", prof.far_distance)
                data.setdefault("prototype_far_distance", prof.far_distance)
                data.setdefault("calibration_fingerprint", prof.fingerprint)
            elif isinstance(prof, dict):
                data.setdefault("feature_space_id", prof.get("feature_space_id"))
                if "near_distance" in prof:
                    data.setdefault("intra_segment_near_distance", prof["near_distance"])
                    data.setdefault("inter_segment_tau", prof["near_distance"])
                    data.setdefault("tau", prof["near_distance"])
                    data.setdefault("prototype_near_distance", prof["near_distance"])
                if "far_distance" in prof:
                    data.setdefault("candidate_far_distance", prof["far_distance"])
                    data.setdefault("negative_evidence_distance", prof["far_distance"])
                    data.setdefault("far_distance", prof["far_distance"])
                    data.setdefault("prototype_far_distance", prof["far_distance"])
                if "fingerprint" in prof:
                    data.setdefault("calibration_fingerprint", prof["fingerprint"])
        if data.get("version") == "complete_support_v3":
            data.setdefault("support_mode", "cooccurrence_contrastive")
            data.setdefault("enable_geometric_negative_compensation", True)
            data.setdefault("contrastive_margin", 0.06)
            data.setdefault("counter_evidence_margin", 0.06)
            data.setdefault("rival_margin", 0.06)
            data.setdefault("margin", 0.06)
        if data.get("support_mode") == "compensated_contrastive":
            data.setdefault("enable_geometric_negative_compensation", True)
        # Tau / near distance synchronization
        if "tau" in data:
            data.setdefault("intra_segment_near_distance", data["tau"])
            data.setdefault("inter_segment_tau", data["tau"])
        elif "inter_segment_tau" in data:
            data.setdefault("tau", data["inter_segment_tau"])
        elif "intra_segment_near_distance" in data:
            data.setdefault("tau", data["intra_segment_near_distance"])

        # Far distance / upper bound synchronization
        if "far_distance" in data:
            data.setdefault("candidate_far_distance", data["far_distance"])
            data.setdefault("negative_evidence_distance", data["far_distance"])
        elif "candidate_far_distance" in data:
            data.setdefault("far_distance", data["candidate_far_distance"])
        elif "negative_evidence_distance" in data:
            data.setdefault("far_distance", data["negative_evidence_distance"])

        # Margin synchronization
        if "margin" in data:
            data.setdefault("contrastive_margin", data["margin"])
            data.setdefault("counter_evidence_margin", data["margin"])
            data.setdefault("rival_margin", data["margin"])
        elif "contrastive_margin" in data:
            data.setdefault("margin", data["contrastive_margin"])
        elif "counter_evidence_margin" in data:
            data.setdefault("margin", data["counter_evidence_margin"])
        elif "rival_margin" in data:
            data.setdefault("margin", data["rival_margin"])

        return data

    @model_validator(mode="after")
    def thresholds(self):
        if self.calibration_profile is not None:
            prof = self.calibration_profile
            if self.feature_space_id is not None and self.feature_space_id != prof.feature_space_id:
                raise ValueError(
                    f"policy feature_space_id {self.feature_space_id} does not match "
                    f"calibration_profile {prof.feature_space_id}"
                )
            if (
                self.calibration_fingerprint is not None
                and self.calibration_fingerprint != prof.fingerprint
            ):
                raise ValueError(
                    "calibration_fingerprint mismatch with attached calibration_profile"
                )
            for attr_name in ("intra_segment_near_distance", "inter_segment_tau", "tau"):
                val = getattr(self, attr_name)
                if abs(val - prof.near_distance) > 1e-6:
                    raise ValueError(
                        f"Policy parameter {attr_name}={val} conflicts with "
                        f"attached calibration_profile.near_distance={prof.near_distance}"
                    )
            for attr_name in (
                "candidate_far_distance",
                "negative_evidence_distance",
                "far_distance",
            ):
                val = getattr(self, attr_name)
                if abs(val - prof.far_distance) > 1e-6:
                    raise ValueError(
                        f"Policy parameter {attr_name}={val} conflicts with "
                        f"attached calibration_profile.far_distance={prof.far_distance}"
                    )
        if self.support_mode != "constrained_manifold":
            if not self.inter_segment_tau <= self.max_pair_distance < self.candidate_far_distance:
                raise ValueError("require tau <= max_pair_distance < far_distance")
        return self


class DiscoveryComparison(ContractModel):
    left_sample_id: Identifier
    right_sample_id: Identifier
    distance: NonNegativeFloat | None = None
    common_parts: int = 0
    reason: Identifier | None = None


class DiscoveryDirection(ContractModel):
    query_segment_id: Identifier
    query_sample_ids: list[Identifier]
    supported_sample_ids: list[Identifier]
    coverage: Probability
    score: NonNegativeFloat | None = None
    competitor_segment_ids: list[Identifier] = Field(default_factory=list)
    sample_contrasts: list[dict] = Field(default_factory=list)


class DiscoveryPair(ContractModel):
    pair_id: Identifier
    left_segment_id: Identifier
    right_segment_id: Identifier
    status: Literal["supported", "conflict", "insufficient", "ambiguous"]
    score: NonNegativeFloat | None = None
    reasons: list[Identifier]
    directions: list[DiscoveryDirection] = Field(default_factory=list)
    comparisons: list[DiscoveryComparison] = Field(default_factory=list)
    conflict_observation_ids: list[Identifier] = Field(default_factory=list)
    prototype_witnesses: list[dict[str, Any]] = Field(default_factory=list)
    rival_margins: dict[str, float] = Field(default_factory=dict)


class AutomaticTemplateSupport(ContractModel):
    sample_id: Identifier
    segment_id: Identifier
    feature_space_id: Identifier
    support_pair_ids: list[Identifier]
    witness_sample_ids: list[Identifier]
    prototype_id: Identifier | None = None
    time_bucket: int | None = None
    reliable_parts: list[int] = Field(default_factory=list)
    quality_score: float | None = None
    risks: list[Identifier] = Field(default_factory=list)
    source_component: Identifier | None = None


class DiscoveredGroup(ContractModel):
    group_id: Identifier
    anchor_segment_id: Identifier
    member_segment_ids: list[Identifier]
    support_pair_ids: list[Identifier]
    templates: list[AutomaticTemplateSupport]
    rejected_member_ids: list[Identifier] = Field(default_factory=list)
    member_evidence: dict[str, Any] = Field(default_factory=dict)
    acceptance_status: str = "accepted"
    acceptance_reasons: list[Identifier] = Field(default_factory=list)


class UnknownSegment(ContractModel):
    segment_id: Identifier
    reasons: list[Identifier]


class DiscoveryReport(ContractModel):
    schema_version: Literal["identity-discovery-v1"] = "identity-discovery-v1"
    report_id: Identifier
    match_id: Identifier
    status: Literal["experimental"] = "experimental"
    policy: DiscoveryPolicy
    policy_fingerprint: Sha256
    input_fingerprint: Sha256
    segmentation_fingerprint: Sha256
    feature_space_id: Identifier
    groups: list[DiscoveredGroup]
    pairs: list[DiscoveryPair]
    unknown: list[UnknownSegment]
    summary: dict
