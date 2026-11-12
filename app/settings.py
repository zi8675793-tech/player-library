"""Read validated TOML settings without creating or modifying runtime resources."""

import tomllib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from app.contracts.calibration import DistanceCalibrationProfile
from app.contracts.discovery import DiscoveryPolicy
from app.contracts.identity import MatchingPolicy
from app.contracts.segmentation import SegmentationPolicy, SegmentationSpacePolicy

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, pattern=r"\S")]
PositiveInteger = Annotated[int, Field(strict=True, gt=0)]
NonNegativeSeconds = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]


class SettingsError(ValueError):
    """A configuration file could not be read or did not satisfy its schema."""


class _SettingsModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class PathsSettings(_SettingsModel):
    input_root: Path
    runtime_root: Path
    calibration_profile: Path | None = None

    @field_validator("input_root", "runtime_root", "calibration_profile", mode="before")
    @classmethod
    def resolve_path(cls, value: object, info: ValidationInfo) -> Path | None:
        if value is None:
            return None
        if not isinstance(value, (str, Path)):
            raise ValueError("path must be a non-empty string or Path")
        if isinstance(value, str) and not value.strip():
            raise ValueError("path must not be blank")
        path = Path(value)
        if not path.is_absolute():
            config_dir = (info.context or {}).get("config_dir")
            if config_dir is None:
                raise ValueError(
                    "relative paths require a configuration directory; use load_settings"
                )
            path = config_dir / path
        return path.resolve()


class InputSettings(_SettingsModel):
    adapter: Literal["normalized", "sample_test", "machine"]


class MatchSettings(_SettingsModel):
    match_id: NonEmptyString


class MatchingSettings(_SettingsModel):
    min_common_parts: PositiveInteger
    max_templates_per_player: PositiveInteger
    auto_assign_enabled: bool
    policies: list[MatchingPolicy] = Field(default_factory=list)

    @field_validator("policies", mode="before")
    @classmethod
    def inherit_shared_settings(cls, value: object, info: ValidationInfo) -> object:
        """A space policy inherits the three existing shared settings when omitted."""
        if not isinstance(value, list):
            return value
        shared = ("min_common_parts", "max_templates_per_player", "auto_assign_enabled")
        return [
            {**{key: info.data[key] for key in shared if key in info.data}, **policy}
            if isinstance(policy, dict)
            else policy
            for policy in value
        ]

    @model_validator(mode="after")
    def consistent_policies(self) -> "MatchingSettings":
        ids = [policy.feature_space_id for policy in self.policies]
        if len(ids) != len(set(ids)):
            raise ValueError("matching.policies requires unique feature_space_id values")
        for policy in self.policies:
            for field in ("min_common_parts", "max_templates_per_player", "auto_assign_enabled"):
                if getattr(policy, field) != getattr(self, field):
                    raise ValueError(
                        f"matching.policies {policy.feature_space_id}: {field} must "
                        "agree with matching shared settings"
                    )
        if self.auto_assign_enabled and not self.policies:
            raise ValueError("auto_assign_enabled requires explicit calibrated matching.policies")
        return self

    def policy_for(self, feature_space_id: str) -> MatchingPolicy:
        """Select an exact space policy, keeping unconfigured spaces uncalibrated."""
        for policy in self.policies:
            if policy.feature_space_id == feature_space_id:
                return policy
        if self.auto_assign_enabled:
            raise SettingsError(
                f"No calibrated matching policy for feature_space_id {feature_space_id!r}"
            )
        return MatchingPolicy(
            feature_space_id=feature_space_id,
            min_common_parts=self.min_common_parts,
            max_templates_per_player=self.max_templates_per_player,
            auto_assign_enabled=False,
        )


class PlaybackSettings(_SettingsModel):
    before_event_s: NonNegativeSeconds
    after_event_s: NonNegativeSeconds


class ServerSettings(_SettingsModel):
    host: NonEmptyString
    port: Annotated[int, Field(strict=True, ge=1, le=65535)]


class Settings(_SettingsModel):
    paths: PathsSettings
    input: InputSettings
    match: MatchSettings
    matching: MatchingSettings
    playback: PlaybackSettings
    server: ServerSettings
    segmentation: SegmentationPolicy = Field(default_factory=SegmentationPolicy)
    discovery: DiscoveryPolicy = Field(default_factory=DiscoveryPolicy)


def bind_calibration_profile(settings: Settings, profile: DistanceCalibrationProfile) -> Settings:
    """Explicitly bind a frozen distance calibration profile to segmentation and discovery policies."""
    seg_spaces = [
        s for s in settings.segmentation.spaces if s.feature_space_id != profile.feature_space_id
    ]
    seg_spaces.append(SegmentationSpacePolicy(profile=profile))

    seg_profiles = [
        p
        for p in settings.segmentation.calibration_profiles
        if p.feature_space_id != profile.feature_space_id
    ]
    seg_profiles.append(profile)

    seg_dict = settings.segmentation.model_dump(mode="python")
    seg_dict["spaces"] = [s.model_dump(mode="python") for s in seg_spaces]
    seg_dict["calibration_profiles"] = [p.model_dump(mode="python") for p in seg_profiles]
    seg_dict["strict_space_calibration"] = True
    new_seg = SegmentationPolicy.model_validate(seg_dict)

    disc_dict = settings.discovery.model_dump(mode="python")
    disc_dict["feature_space_id"] = profile.feature_space_id
    disc_dict["calibration_profile"] = profile.model_dump(mode="python")
    disc_dict["calibration_fingerprint"] = profile.fingerprint
    disc_dict["intra_segment_near_distance"] = profile.near_distance
    disc_dict["inter_segment_tau"] = profile.near_distance
    disc_dict["tau"] = profile.near_distance
    disc_dict["candidate_far_distance"] = profile.far_distance
    disc_dict["negative_evidence_distance"] = profile.far_distance
    disc_dict["far_distance"] = profile.far_distance
    new_disc = DiscoveryPolicy.model_validate(disc_dict)

    return settings.model_copy(update={"segmentation": new_seg, "discovery": new_disc})


def load_settings(config_path: str | Path) -> Settings:
    """Load settings, resolving every relative path against the TOML file's directory.

    Loading does not check that data roots exist, create directories, open a database,
    expand environment variables, or write any files. The original six sections are
    required; segmentation is optional. Unknown fields and sections are errors.
    """
    path = Path(config_path).absolute()
    try:
        with path.open("rb") as config_file:
            values = tomllib.load(config_file)
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise SettingsError(f"Cannot read configuration {path}: {exc}") from exc
    try:
        settings = Settings.model_validate(values, context={"config_dir": path.parent})
        if settings.paths.calibration_profile is not None:
            from app.identity.calibration import load_calibration_profile

            profile = load_calibration_profile(settings.paths.calibration_profile)
            settings = bind_calibration_profile(settings, profile)
        return settings
    except ValidationError as exc:
        raise SettingsError(f"Invalid configuration {path}:\n{exc}") from exc
    except Exception as exc:
        raise SettingsError(f"Failed to load calibration profile for {path}: {exc}") from exc
