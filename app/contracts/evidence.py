"""Evidence envelopes, media lineage and explicit, variable-size feature spaces.

No files are opened here. References may refer to evidence already stored or yet to
arrive; validators check only facts available in the supplied object.
"""

from collections.abc import Callable, Hashable, Iterable
from enum import StrEnum
from math import prod
from typing import Annotated, Literal, TypeVar

from pydantic import Field, StrictBool, model_validator

from app.contracts.common import (
    ContractModel,
    FiniteFloat,
    Identifier,
    JerseyNumber,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    Probability,
    Provenance,
    SegmentRange,
    Sha256,
    SourceLocation,
)
from app.contracts.events import Event, EventTime, PointEventTime, TrackReference
from app.contracts.identity import (
    Assignment,
    IdentityRisk,
    Player,
    RiskReview,
    SampleExclusion,
    Segment,
    Template,
)

RecordT = TypeVar("RecordT")


def _unique_index(
    records: Iterable[RecordT], key: Callable[[RecordT], Hashable], label: str
) -> dict[Hashable, RecordT]:
    """Reject repeated entity declarations rather than silently overwriting them."""
    indexed = {}
    for record in records:
        record_key = key(record)
        if record_key in indexed:
            conflict = " with conflicting content" if indexed[record_key] != record else ""
            raise ValueError(f"{label}: duplicate key {record_key!r}{conflict}")
        indexed[record_key] = record
    return indexed


def _point_within(time_s: float, start_s: float, end_s: float, context: str) -> None:
    if not start_s <= time_s < end_s:
        raise ValueError(f"{context}: time_s {time_s} outside [{start_s}, {end_s})")


def _event_time_within(time: EventTime, start_s: float, end_s: float, context: str) -> None:
    if isinstance(time, PointEventTime):
        _point_within(time.time_s, start_s, end_s, context)
    elif time.start_s < start_s or time.end_s > end_s:
        raise ValueError(f"{context}: interval time outside [{start_s}, {end_s})")


class SourceFile(ContractModel):
    """An input locator and optional integrity facts, never a model contract."""

    file_id: Identifier
    path: Identifier
    sha256: Sha256 | None = None
    size_bytes: NonNegativeInt | None = None


class MediaType(StrEnum):
    VIDEO = "video"
    IMAGE = "image"


class MediaAsset(ContractModel):
    asset_id: Identifier
    media_type: MediaType
    file: SourceFile


class SourceMapping(ContractModel):
    """source seconds = source_start_s + clip seconds * time_scale."""

    source_video_id: Identifier
    source_start_s: NonNegativeFloat
    time_scale: PositiveFloat = 1.0


class FrameTiming(StrEnum):
    CFR = "cfr"
    VFR = "vfr"
    UNKNOWN = "unknown"


class Clip(ContractModel):
    match_id: Identifier
    clip_id: Identifier
    asset_id: Identifier | None = None
    duration_s: PositiveFloat | None = None
    frame_count: PositiveInt | None = None
    frame_timing: FrameTiming = FrameTiming.UNKNOWN
    fps: PositiveFloat | None = None
    source_mapping: SourceMapping | None = None

    @model_validator(mode="after")
    def cfr_has_rate(self) -> "Clip":
        if self.frame_timing == FrameTiming.CFR and self.fps is None:
            raise ValueError("CFR clips require fps; VFR timing must use actual timestamps")
        return self


class FeatureLayout(StrEnum):
    VECTOR = "vector"
    PARTS = "parts"
    TENSOR = "tensor"


class Normalization(StrEnum):
    NONE = "none"
    L2 = "l2"


class Preprocessing(ContractModel):
    """An explicit preprocessing version and color convention."""

    version: Identifier
    color_space: Identifier
    config_hash: Sha256 | None = None


class VisibilityContract(ContractModel):
    shape: list[PositiveInt] = Field(min_length=1)
    semantics: Identifier


class FeatureSpace(ContractModel):
    """Per-sample shape; N/sample_count belongs to each file instead.

    Describing a backend does not install or execute it. Runtime backend support is
    the responsibility of the future adapter/distance implementation.
    """

    feature_space_id: Identifier
    backend: Identifier
    model_hash: Sha256
    config_hash: Sha256 | None = None
    preprocessing: Preprocessing
    layout: FeatureLayout
    shape: list[PositiveInt] = Field(min_length=1)
    dtype: Literal["float32"] = "float32"
    normalization: Normalization
    distance: Identifier
    visibility: VisibilityContract | None = None

    @model_validator(mode="after")
    def consistent_layout(self) -> "FeatureSpace":
        expected_rank = {FeatureLayout.VECTOR: 1, FeatureLayout.PARTS: 2}.get(self.layout)
        if expected_rank is not None and len(self.shape) != expected_rank:
            raise ValueError(f"{self.layout} layout requires a rank-{expected_rank} sample shape")
        if self.backend == "kpr":
            if self.layout != FeatureLayout.PARTS or self.normalization != Normalization.L2:
                raise ValueError("kpr requires parts layout and l2 normalization")
            if self.visibility is None or self.visibility.shape != self.shape[:1]:
                raise ValueError("kpr visibility shape must match the number of parts")
        return self


class FeatureFile(ContractModel):
    """Array/index integrity and row count may differ across files in one space."""

    feature_file_id: Identifier
    feature_space_id: Identifier
    sample_count: NonNegativeInt
    values_file: SourceFile
    index_file: SourceFile
    visibility_file: SourceFile | None = None


def _feature_source_files(files: Iterable[FeatureFile]) -> Iterable[SourceFile]:
    for file in files:
        yield file.values_file
        yield file.index_file
        if file.visibility_file is not None:
            yield file.visibility_file


def _consistent_file_declarations(files: Iterable[SourceFile]) -> None:
    """Compare known facts across references without changing partial declarations."""
    known: dict[str, dict[str, str | int | None]] = {}
    for file in files:
        facts = {"path": file.path, "sha256": file.sha256, "size_bytes": file.size_bytes}
        previous = known.setdefault(file.file_id, {})
        for field, value in facts.items():
            if value is None:
                continue
            if previous.get(field) is not None and previous[field] != value:
                raise ValueError(f"file_id {file.file_id!r}: conflicting {field} declarations")
            previous[field] = value


class InlineFeature(ContractModel):
    """Values flattened in row-major order according to FeatureSpace.shape."""

    kind: Literal["inline"] = "inline"
    feature_space_id: Identifier
    values: list[FiniteFloat] = Field(min_length=1)
    visibility: list[Probability] | None = None


class FileFeatureReference(ContractModel):
    kind: Literal["file"] = "file"
    feature_space_id: Identifier
    feature_file_id: Identifier
    row_index: NonNegativeInt


SampleFeature = Annotated[InlineFeature | FileFeatureReference, Field(discriminator="kind")]


class Track(SegmentRange):
    """Raw upstream track extent; it never implies a single accepted identity."""

    match_id: Identifier
    clip_id: Identifier
    track_id: Identifier
    source: Provenance
    observed_frame_indices: list[NonNegativeInt] | None = None

    @model_validator(mode="after")
    def actual_observed_frames(self) -> "Track":
        frames = self.observed_frame_indices
        if frames is not None:
            if frames != sorted(set(frames)):
                raise ValueError("observed_frame_indices must be sorted and unique")
            if self.frame_start is not None and any(
                not self.frame_start <= frame < self.frame_end_exclusive for frame in frames
            ):
                raise ValueError("observed_frame_indices must lie within the track frame range")
        return self


class ScoreSemantics(StrEnum):
    UNKNOWN = "unknown"
    DETECTION = "detection"
    TRACKING = "tracking"


class SampleUsability(StrEnum):
    UNKNOWN = "unknown"
    USABLE = "usable"
    UNUSABLE = "unusable"


class CleanReviewStatus(StrEnum):
    UNREVIEWED = "unreviewed"
    APPROVED = "approved"
    REJECTED = "rejected"


class TrackingQuality(ContractModel):
    score: Probability | None = None
    score_semantics: ScoreSemantics = ScoreSemantics.UNKNOWN
    usability: SampleUsability = SampleUsability.UNKNOWN
    clean_review: CleanReviewStatus = CleanReviewStatus.UNREVIEWED
    risks: list[IdentityRisk] = Field(default_factory=list)
    occluded: StrictBool | None = None
    mixed_identity: StrictBool | None = None
    notes: str | None = None


class AppearanceEvidence(ContractModel):
    """Observations, independently of accepted identity and template admission."""

    team_id: Identifier | None = None
    jersey_number: JerseyNumber | None = None
    confidence: Probability | None = None


class Sample(ContractModel):
    match_id: Identifier
    sample_id: Identifier
    clip_id: Identifier
    track_id: Identifier
    time_s: NonNegativeFloat
    source: Provenance
    frame_idx: NonNegativeInt | None = None
    segment_id: Identifier | None = None
    image_asset_id: Identifier | None = None
    tracking_quality: TrackingQuality | None = None
    appearance: AppearanceEvidence | None = None
    features: list[SampleFeature] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_feature_spaces(self) -> "Sample":
        ids = [feature.feature_space_id for feature in self.features]
        if len(ids) != len(set(ids)):
            raise ValueError("a sample may supply only one feature per feature_space_id")
        return self


def _sample_within(sample: Sample, target: Track | Segment, context: str) -> None:
    if (sample.match_id, sample.clip_id, sample.track_id) != (
        target.match_id,
        target.clip_id,
        target.track_id,
    ):
        raise ValueError(f"{context}: sample match_id/clip_id/track_id must match its target")
    _point_within(sample.time_s, target.start_s, target.end_s, context)
    if sample.frame_idx is not None and target.frame_start is not None:
        if not target.frame_start <= sample.frame_idx < target.frame_end_exclusive:
            raise ValueError(f"{context}: sample frame_idx outside target frame range")


class IdentityEvidence(ContractModel):
    risks: list[IdentityRisk] = Field(default_factory=list)
    tracks: list[Track] = Field(default_factory=list)
    segments: list[Segment] = Field(default_factory=list)
    samples: list[Sample] = Field(default_factory=list)
    feature_spaces: list[FeatureSpace] = Field(default_factory=list)
    feature_files: list[FeatureFile] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_supplied_features(self) -> "IdentityEvidence":
        tracks = _unique_index(
            self.tracks, lambda track: (track.clip_id, track.track_id), "tracks (clip_id, track_id)"
        )
        segments = _unique_index(self.segments, lambda segment: segment.segment_id, "segment_id")
        samples = _unique_index(self.samples, lambda sample: sample.sample_id, "sample_id")
        spaces = _unique_index(
            self.feature_spaces, lambda space: space.feature_space_id, "feature_space_id"
        )
        files = _unique_index(
            self.feature_files, lambda file: file.feature_file_id, "feature_file_id"
        )
        _consistent_file_declarations(_feature_source_files(self.feature_files))
        risks = {}
        for risk in [
            *self.risks,
            *(
                risk
                for sample in self.samples
                if sample.tracking_quality is not None
                for risk in sample.tracking_quality.risks
            ),
        ]:
            previous = risks.get(risk.risk_id)
            if previous is not None and previous != risk:
                raise ValueError(f"risk {risk.risk_id!r}: conflicting risk declarations")
            risks[risk.risk_id] = risk
            segment = segments.get(risk.segment_id)
            sample = samples.get(risk.sample_id)
            for target in [item for item in (segment, sample) if item is not None]:
                if (
                    risk.match_id != target.match_id
                    or (risk.clip_id is not None and risk.clip_id != target.clip_id)
                    or (risk.track_id is not None and risk.track_id != target.track_id)
                ):
                    raise ValueError(f"risk {risk.risk_id!r}: scope does not match known target")
            if segment is not None and sample is not None:
                _sample_within(sample, segment, f"risk {risk.risk_id!r} scope")
            if risk.start_s is not None:
                if segment is not None and (
                    risk.start_s < segment.start_s or risk.end_s > segment.end_s
                ):
                    raise ValueError(f"risk {risk.risk_id!r}: time scope outside segment")
                if sample is not None:
                    _point_within(
                        sample.time_s, risk.start_s, risk.end_s, f"risk {risk.risk_id!r} scope"
                    )
        for owner in self.samples:
            for risk in owner.tracking_quality.risks if owner.tracking_quality is not None else []:
                if (
                    risk.match_id != owner.match_id
                    or (risk.sample_id is not None and risk.sample_id != owner.sample_id)
                    or (risk.clip_id is not None and risk.clip_id != owner.clip_id)
                    or (risk.track_id is not None and risk.track_id != owner.track_id)
                ):
                    raise ValueError(f"sample {owner.sample_id!r}: attached risk scope mismatch")
        for segment in self.segments:
            track = tracks.get((segment.clip_id, segment.track_id))
            if track is not None and segment.match_id != track.match_id:
                raise ValueError(f"segment {segment.segment_id!r}: track match_id must match")
            # Selection bounds may include unobserved edges beyond Track's extent.
        for file in self.feature_files:
            space = spaces.get(file.feature_space_id)
            if space is not None:
                if space.visibility is not None and file.visibility_file is None:
                    raise ValueError("feature space visibility requires visibility_file")
                if space.visibility is None and file.visibility_file is not None:
                    raise ValueError("visibility_file supplied for a space without visibility")
        for sample in self.samples:
            segment = segments.get(sample.segment_id)
            if segment is not None:
                _sample_within(sample, segment, f"sample {sample.sample_id!r} segment reference")
            track = tracks.get((sample.clip_id, sample.track_id))
            if track is not None:
                _sample_within(sample, track, f"sample {sample.sample_id!r} track reference")
            for feature in sample.features:
                space = spaces.get(feature.feature_space_id)
                if isinstance(feature, FileFeatureReference):
                    file = files.get(feature.feature_file_id)
                    if file is not None:
                        if file.feature_space_id != feature.feature_space_id:
                            raise ValueError(
                                "feature file and sample must use the same feature space"
                            )
                        if feature.row_index >= file.sample_count:
                            raise ValueError(
                                "row_index must be less than feature file sample_count"
                            )
                elif space is not None:
                    if len(feature.values) != prod(space.shape):
                        raise ValueError(
                            f"sample {sample.sample_id}: values length must match "
                            f"feature space {space.feature_space_id} shape {space.shape}"
                        )
                    if space.visibility is None:
                        if feature.visibility is not None:
                            raise ValueError("visibility supplied for a space without visibility")
                    elif feature.visibility is None or len(feature.visibility) != prod(
                        space.visibility.shape
                    ):
                        raise ValueError(
                            "visibility length must match feature space visibility shape"
                        )
        return self


class EvidenceWarning(ContractModel):
    """A retained input concern; it does not confirm or reject an identity."""

    code: Identifier
    message: Identifier
    clip_id: Identifier | None = None
    locations: list[SourceLocation] = Field(default_factory=list)


class EvidenceBatch(ContractModel):
    schema_version: Literal["2"] = "2"
    match_id: Identifier
    batch_id: Identifier
    producer: Identifier
    sources: list[SourceFile] = Field(default_factory=list)
    assets: list[MediaAsset] = Field(default_factory=list)
    clips: list[Clip] | None = None
    identity_evidence: IdentityEvidence | None = None
    events: list[Event] | None = None
    warnings: list[EvidenceWarning] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent_match_scope(self) -> "EvidenceBatch":
        records = [*(self.clips or []), *(self.events or [])]
        if self.identity_evidence is not None:
            records.extend(self.identity_evidence.tracks)
            records.extend(self.identity_evidence.segments)
            records.extend(self.identity_evidence.samples)
            records.extend(self.identity_evidence.risks)
            records.extend(
                risk
                for sample in self.identity_evidence.samples
                if sample.tracking_quality is not None
                for risk in sample.tracking_quality.risks
            )
        if any(record.match_id != self.match_id for record in records):
            raise ValueError("all supplied records must belong to EvidenceBatch.match_id")
        clips = _unique_index(self.clips or [], lambda clip: clip.clip_id, "clip_id")
        assets = _unique_index(self.assets, lambda asset: asset.asset_id, "asset_id")
        _unique_index(self.sources, lambda file: file.file_id, "sources file_id")
        _unique_index(self.events or [], lambda event: event.event_id, "event_id")
        identity = self.identity_evidence
        _consistent_file_declarations(
            [
                *self.sources,
                *(asset.file for asset in self.assets),
                *_feature_source_files(identity.feature_files if identity is not None else []),
            ]
        )
        for clip in clips.values():
            asset = assets.get(clip.asset_id)
            if asset is not None and asset.media_type != MediaType.VIDEO:
                raise ValueError(f"clip {clip.clip_id!r}: asset_id must reference a video asset")
        tracks = {}
        segments = {}
        if identity is not None:
            tracks = {(track.clip_id, track.track_id): track for track in identity.tracks}
            segments = {segment.segment_id: segment for segment in identity.segments}
            for record in [*identity.tracks, *identity.segments]:
                clip = clips.get(record.clip_id)
                if clip is None:
                    continue
                context = f"{type(record).__name__} in clip {record.clip_id!r}"
                if clip.duration_s is not None and record.end_s > clip.duration_s:
                    raise ValueError(f"{context}: time range exceeds clip duration_s")
                if clip.frame_count is not None:
                    if (
                        record.frame_end_exclusive is not None
                        and record.frame_end_exclusive > clip.frame_count
                    ):
                        raise ValueError(f"{context}: frame range exceeds clip frame_count")
                    if isinstance(record, Track) and any(
                        frame >= clip.frame_count for frame in record.observed_frame_indices or []
                    ):
                        raise ValueError(f"{context}: observed frame exceeds clip frame_count")
            for sample in identity.samples:
                clip = clips.get(sample.clip_id)
                context = f"sample {sample.sample_id!r} in clip {sample.clip_id!r}"
                if clip is not None:
                    if clip.duration_s is not None:
                        _point_within(sample.time_s, 0, clip.duration_s, context)
                    if (
                        clip.frame_count is not None
                        and sample.frame_idx is not None
                        and sample.frame_idx >= clip.frame_count
                    ):
                        raise ValueError(f"{context}: frame_idx exceeds clip frame_count")
                asset = assets.get(sample.image_asset_id)
                if asset is not None and asset.media_type != MediaType.IMAGE:
                    raise ValueError(
                        f"sample {sample.sample_id!r}: image_asset_id must reference an image asset"
                    )
        for event in self.events or []:
            clip = clips.get(event.clip_id)
            if clip is not None and clip.duration_s is not None:
                _event_time_within(event.time, 0, clip.duration_s, f"event {event.event_id!r}")
            for subject in event.subjects:
                for index, reference in enumerate(subject.references):
                    # Implicit localization and identity ambiguity belong to association.
                    if reference.time is None:
                        continue
                    context = f"event {event.event_id!r} subject {subject.subject_id!r} reference[{index}]"
                    if isinstance(reference, TrackReference):
                        target = tracks.get((reference.clip_id, reference.track_id))
                        target_clip = clips.get(reference.clip_id)
                    else:
                        target = segments.get(reference.segment_id)
                        target_clip = clips.get(target.clip_id) if target is not None else None
                    if target is not None:
                        _event_time_within(reference.time, target.start_s, target.end_s, context)
                    if target_clip is not None and target_clip.duration_s is not None:
                        _event_time_within(reference.time, 0, target_clip.duration_s, context)
        return self


class PreparedSample(ContractModel):
    """A copied sample with hydrated, validated features and sampling coordinates."""

    sample: Sample
    feature: InlineFeature
    bucket_index: NonNegativeInt
    region: Annotated[int, Field(strict=True, ge=0, le=2)]
    visible_parts: NonNegativeInt
    visibility_sum: NonNegativeFloat
    clean_approved: StrictBool = False


class PreparedSegment(ContractModel):
    """Usable evidence, representative selection and retained segment-level risks."""

    segment: Segment
    samples: list[PreparedSample] = Field(default_factory=list)
    representative_sample_ids: list[Identifier] = Field(default_factory=list)
    exclusions: list[SampleExclusion] = Field(default_factory=list)
    risks: list[IdentityRisk] = Field(default_factory=list)
    covered_regions: list[Annotated[int, Field(strict=True, ge=0, le=2)]] = Field(
        default_factory=list
    )
    bucket_count: NonNegativeInt = 0
    coverage_start_s: NonNegativeFloat | None = None
    coverage_end_s: NonNegativeFloat | None = None
    reasons: list[Identifier] = Field(default_factory=list)


class PreparedEvidence(ContractModel):
    match_id: Identifier
    feature_space_id: Identifier
    policy_version: Identifier
    policy_fingerprint: Sha256
    feature_space: FeatureSpace | None = None
    assets: list[MediaAsset] = Field(default_factory=list)
    segments: list[PreparedSegment] = Field(default_factory=list)
    exclusions: list[SampleExclusion] = Field(default_factory=list)


class GallerySnapshot(ContractModel):
    """A complete match-scoped view supplied by a caller, never expanded in-place."""

    match_id: Identifier
    revision: NonNegativeInt
    players: list[Player] = Field(default_factory=list)
    assignments: list[Assignment] = Field(default_factory=list)
    templates: list[Template] = Field(default_factory=list)
    evidence: IdentityEvidence = Field(default_factory=IdentityEvidence)
    risk_reviews: list[RiskReview] = Field(default_factory=list)
    assets: list[MediaAsset] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent_snapshot(self) -> "GallerySnapshot":
        _unique_index(self.players, lambda player: player.player_id, "snapshot player_id")
        _unique_index(
            self.assignments, lambda assignment: assignment.segment_id, "snapshot assignment"
        )
        _unique_index(self.templates, lambda template: template.template_id, "snapshot template_id")
        _unique_index(self.assets, lambda asset: asset.asset_id, "snapshot asset_id")
        records = [
            *self.players,
            *self.assignments,
            *self.templates,
            *self.evidence.tracks,
            *self.evidence.segments,
            *self.evidence.samples,
            *self.evidence.risks,
            *self.risk_reviews,
        ]
        if any(record.match_id != self.match_id for record in records):
            raise ValueError("all snapshot records must belong to snapshot.match_id")
        if any(
            record.revision > self.revision
            for record in [*self.assignments, *self.templates, *self.risk_reviews]
        ):
            raise ValueError("snapshot cannot contain records newer than its revision")
        _consistent_file_declarations(
            [
                *(asset.file for asset in self.assets),
                *_feature_source_files(self.evidence.feature_files),
            ]
        )
        return self
