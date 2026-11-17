"""Machine consistency proposals; these never certify a named or distinct identity."""

from typing import Literal

from pydantic import Field, StrictBool, model_validator

from app.contracts.calibration import DistanceCalibrationProfile
from app.contracts.common import (
    ContractModel,
    Identifier,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    Probability,
    Provenance,
    SegmentRange,
    Sha256,
)
from app.contracts.evidence import EvidenceBatch
from app.contracts.identity import IdentityRisk, SampleExclusion, Segment


class SegmentationSpacePolicy(ContractModel):
    feature_space_id: Identifier
    near_distance: Probability = 0.18
    far_distance: Probability = 0.45
    calibration_version: Identifier | None = None
    profile_fingerprint: Sha256 | None = None
    profile: DistanceCalibrationProfile | None = None

    @model_validator(mode="before")
    @classmethod
    def sync_profile(cls, data: object) -> object:
        if isinstance(data, dict) and "profile" in data and data["profile"] is not None:
            prof = data["profile"]
            if hasattr(prof, "feature_space_id"):
                data.setdefault("feature_space_id", prof.feature_space_id)
                data.setdefault("near_distance", prof.near_distance)
                data.setdefault("far_distance", prof.far_distance)
                data.setdefault("calibration_version", prof.calibration_version)
                data.setdefault("profile_fingerprint", prof.fingerprint)
            elif isinstance(prof, dict):
                data.setdefault("feature_space_id", prof.get("feature_space_id"))
                data.setdefault("near_distance", prof.get("near_distance"))
                data.setdefault("far_distance", prof.get("far_distance"))
                data.setdefault("calibration_version", prof.get("calibration_version"))
        return data

    @model_validator(mode="after")
    def separated(self):
        if self.profile is not None:
            if self.profile.feature_space_id != self.feature_space_id:
                raise ValueError(
                    f"profile feature_space_id {self.profile.feature_space_id} does not match {self.feature_space_id}"
                )
            if (
                self.profile_fingerprint is not None
                and self.profile_fingerprint != self.profile.fingerprint
            ):
                raise ValueError("profile_fingerprint mismatch with attached profile")
            if abs(self.near_distance - self.profile.near_distance) > 1e-6:
                raise ValueError(
                    f"SegmentationSpacePolicy near_distance={self.near_distance} conflicts with "
                    f"attached profile.near_distance={self.profile.near_distance}"
                )
            if abs(self.far_distance - self.profile.far_distance) > 1e-6:
                raise ValueError(
                    f"SegmentationSpacePolicy far_distance={self.far_distance} conflicts with "
                    f"attached profile.far_distance={self.profile.far_distance}"
                )
        if self.near_distance >= self.far_distance:
            raise ValueError("near_distance must be below far_distance")
        return self


class SegmentationPolicy(ContractModel):
    version: Literal["track-consistency-v1", "track-consistency-v2", "track-consistency-v3"] = (
        "track-consistency-v1"
    )
    local_support_windows: StrictBool = False
    enable_transition_zones: StrictBool = False
    transition_uncertainty_threshold: Probability = 0.45
    scope_risks_to_breaks: StrictBool = False
    localize_unresolved_evidence: StrictBool = False
    localize_duplicate_risks: StrictBool = False
    ensure_relative_buckets: StrictBool = False
    # Engineering gates fixed before real-input execution; not identity calibration.
    near_distance: Probability = 0.18
    far_distance: Probability = 0.45
    min_common_parts: PositiveInt = 2
    min_part_visibility: Probability = 0.2
    min_support_buckets: PositiveInt = 3
    min_match_buckets: PositiveInt = 2
    bucket_s: PositiveFloat = 0.5
    max_sample_gap_s: PositiveFloat = 1.5
    max_continuous_sample_gap_s: PositiveFloat = 4.5
    max_observation_gap_s: PositiveFloat = 0.25
    jump_heights: PositiveFloat = 1.5
    jump_heights_per_s: PositiveFloat = 8.0
    duplicate_iou: Probability = 0.85
    camera_min_tracks: PositiveInt = 3
    camera_jump_fraction: Probability = 0.6
    spaces: list[SegmentationSpacePolicy] = Field(default_factory=list)
    calibration_profiles: list[DistanceCalibrationProfile] = Field(default_factory=list)
    strict_space_calibration: StrictBool = False

    @model_validator(mode="before")
    @classmethod
    def set_v3_defaults(cls, data: object) -> object:
        if isinstance(data, dict) and data.get("version") == "track-consistency-v3":
            data.setdefault("local_support_windows", True)
            data.setdefault("enable_transition_zones", True)
            data.setdefault("scope_risks_to_breaks", True)
            data.setdefault("ensure_relative_buckets", True)
            data.setdefault("min_match_buckets", 2)
            data.setdefault("max_continuous_sample_gap_s", 4.5)
        return data

    @model_validator(mode="after")
    def thresholds(self):
        if self.near_distance >= self.far_distance:
            raise ValueError("near_distance must be below far_distance")
        ids = [p.feature_space_id for p in self.spaces]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate segmentation feature_space_id")
        prof_ids = [p.feature_space_id for p in self.calibration_profiles]
        if len(prof_ids) != len(set(prof_ids)):
            raise ValueError("duplicate calibration_profiles feature_space_id")
        prof_map = {p.feature_space_id: p for p in self.calibration_profiles}
        for space in self.spaces:
            if space.feature_space_id in prof_map:
                prof = prof_map[space.feature_space_id]
                if abs(space.near_distance - prof.near_distance) > 1e-6:
                    raise ValueError(
                        f"SegmentationPolicy space {space.feature_space_id} near_distance={space.near_distance} "
                        f"conflicts with calibration profile near_distance={prof.near_distance}"
                    )
                if abs(space.far_distance - prof.far_distance) > 1e-6:
                    raise ValueError(
                        f"SegmentationPolicy space {space.feature_space_id} far_distance={space.far_distance} "
                        f"conflicts with calibration profile far_distance={prof.far_distance}"
                    )
        return self


class ConsistencyPair(ContractModel):
    left_sample_id: Identifier
    right_sample_id: Identifier
    feature_space_id: Identifier
    comparable: bool
    common_parts: int
    distance: NonNegativeFloat | None = None
    reason: Identifier | None = None


class ConsistencyFinding(SegmentRange):
    kind: Identifier
    sample_ids: list[Identifier] = Field(default_factory=list)
    observation_ids: list[Identifier] = Field(default_factory=list)
    details: dict = Field(default_factory=dict)


class SegmentProposal(ContractModel):
    segment: Segment
    parent_segment_id: Identifier
    source: Provenance
    status: Literal["supported", "unknown"]
    reasons: list[Identifier]
    support_sample_ids: list[Identifier] = Field(default_factory=list)
    source_observation_ids: list[Identifier] = Field(default_factory=list)
    can_match_existing: StrictBool = False
    can_create_new_identity: StrictBool = False
    requires_external_anchor: StrictBool = False
    independent_anchor_required: StrictBool = False
    evidence_semantics: Literal[
        "unspecified",
        "supported",
        "unknown_viewpoint",
        "low_quality_incomparable",
        "appearance_contradiction",
        "observation_missing",
        "duplicate_detection",
        "insufficient_temporal_support",
        "unobserved_extent",
    ] = "unspecified"
    # Always false: supported means a conservative consistency gate, not identity truth.
    identity_verified: Literal[False] = False

    @model_validator(mode="before")
    @classmethod
    def set_proposal_defaults(cls, data: object) -> object:
        if isinstance(data, dict):
            status = data.get("status")
            if status == "supported":
                data.setdefault("can_match_existing", True)
                data.setdefault("can_create_new_identity", True)
            elif status == "unknown":
                data.setdefault("can_match_existing", False)
                data.setdefault("can_create_new_identity", False)
        return data

    @model_validator(mode="after")
    def anchor_eligibility(self):
        if self.independent_anchor_required and not self.requires_external_anchor:
            raise ValueError("independent archive witnesses require an external anchor")
        if self.independent_anchor_required:
            if len(self.support_sample_ids) != 1:
                raise ValueError("independent archive recovery requires one atomic sample")
            if (
                self.segment.frame_start is not None
                and self.segment.frame_end_exclusive != self.segment.frame_start + 1
            ):
                raise ValueError("independent archive recovery cannot propagate beyond one frame")
        if self.requires_external_anchor and (
            self.status != "unknown" or not self.can_match_existing or self.can_create_new_identity
        ):
            raise ValueError("external anchor requires unknown, match-only eligibility")
        if self.status == "unknown":
            if self.can_match_existing and not self.requires_external_anchor:
                raise ValueError("unknown matching requires an explicit external anchor")
            if self.can_create_new_identity:
                raise ValueError("unknown proposals cannot create new identities")
        return self


class TrackConsistencyReport(ContractModel):
    parent_segment: Segment
    proposals: list[SegmentProposal] = Field(default_factory=list)
    findings: list[ConsistencyFinding] = Field(default_factory=list)
    comparisons: list[ConsistencyPair] = Field(default_factory=list)
    exclusions: list[SampleExclusion] = Field(default_factory=list)
    risks: list[IdentityRisk] = Field(default_factory=list)


class SegmentationReport(ContractModel):
    schema_version: Literal["track-consistency-report-v1"] = "track-consistency-report-v1"
    policy: SegmentationPolicy
    policy_fingerprint: Sha256
    input_fingerprint: Sha256
    derived_evidence_fingerprint: Sha256 | None = None
    tracks: list[TrackConsistencyReport]
    duplicate_candidates: list[dict] = Field(default_factory=list)
    summary: dict


class SegmentationResult(ContractModel):
    batch: EvidenceBatch
    report: SegmentationReport
