"""Fixed evaluation unit definitions and projection alignment tools.

Establishes immutable machine evaluation units based on raw samples and observations,
guaranteeing a unified denominator for sample and duration coverage across all segmentations.
"""

from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.adapters._reading import stable_id
from app.contracts.discovery import DiscoveryReport
from app.contracts.evidence import EvidenceBatch
from app.contracts.identity import Assignment, Segment


class EvaluationUnit(BaseModel):
    """An immutable, atomic observation or sample from the machine handoff."""

    model_config = ConfigDict(extra="forbid")

    unit_id: str
    unit_type: Literal["sample", "observation"]
    clip_id: str
    track_id: str
    frame_idx: int | None = None
    time_s: float
    sample_id: str | None = None
    observation_id: str | None = None


class EvaluationManifest(BaseModel):
    """Catalog of fixed evaluation units and global denominators."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "fixed-evaluation-units-v1"
    match_id: str
    total_samples: int
    total_observations: int
    total_duration_s: float
    clip_count: int
    track_count: int
    units: list[EvaluationUnit] = Field(default_factory=list)

    @property
    def sample_units(self) -> list[EvaluationUnit]:
        return [u for u in self.units if u.unit_type == "sample"]

    @property
    def observation_units(self) -> list[EvaluationUnit]:
        return [u for u in self.units if u.unit_type == "observation"]


class ProjectedUnit(BaseModel):
    """An evaluation unit projected onto an experiment's segment and assignment state."""

    model_config = ConfigDict(extra="forbid")

    unit_id: str
    unit_type: Literal["sample", "observation"]
    clip_id: str
    track_id: str
    frame_idx: int | None = None
    time_s: float
    sample_id: str | None = None
    observation_id: str | None = None
    segment_id: str | None = None
    status: Literal["assigned", "unassigned"]
    player_id: str | None = None
    group_id: str | None = None
    unassigned_reason: str | None = None


def extract_evaluation_units(batch: EvidenceBatch) -> EvaluationManifest:
    """Extract all raw samples and observations from an evidence batch into fixed units."""
    evidence = batch.identity_evidence
    if evidence is None:
        raise ValueError("EvidenceBatch contains no identity evidence")

    units: list[EvaluationUnit] = []

    # 1. Raw feature samples
    for sample in evidence.samples:
        frame_str = str(sample.frame_idx) if sample.frame_idx is not None else "noframe"
        time_str = f"{round(sample.time_s, 4):.4f}"
        uid = stable_id(
            "unit_sample",
            sample.clip_id,
            sample.track_id,
            frame_str,
            time_str,
            sample.sample_id,
        )
        units.append(
            EvaluationUnit(
                unit_id=uid,
                unit_type="sample",
                clip_id=sample.clip_id,
                track_id=sample.track_id,
                frame_idx=sample.frame_idx,
                time_s=float(sample.time_s),
                sample_id=sample.sample_id,
            )
        )

    # 2. Raw valid track observations
    for obs in evidence.observations:
        frame_str = str(obs.frame_idx) if obs.frame_idx is not None else "noframe"
        time_str = f"{round(obs.time_s, 4):.4f}"
        uid = stable_id(
            "unit_obs",
            obs.clip_id,
            obs.track_id,
            frame_str,
            time_str,
            obs.observation_id,
        )
        units.append(
            EvaluationUnit(
                unit_id=uid,
                unit_type="observation",
                clip_id=obs.clip_id,
                track_id=obs.track_id,
                frame_idx=obs.frame_idx,
                time_s=float(obs.time_s),
                observation_id=obs.observation_id,
            )
        )

    # Deterministic sorting
    units.sort(
        key=lambda u: (
            u.clip_id,
            u.track_id,
            u.time_s,
            u.frame_idx if u.frame_idx is not None else -1,
            u.unit_type,
            u.unit_id,
        )
    )

    total_duration_s = sum(track.end_s - track.start_s for track in evidence.tracks)
    clip_count = len(batch.clips) if batch.clips else len({u.clip_id for u in units})
    track_count = len(evidence.tracks)

    return EvaluationManifest(
        schema_version="fixed-evaluation-units-v1",
        match_id=batch.match_id,
        total_samples=len(evidence.samples),
        total_observations=len(evidence.observations),
        total_duration_s=total_duration_s,
        clip_count=clip_count,
        track_count=track_count,
        units=units,
    )


def _unit_in_segment(unit: EvaluationUnit, segment: Segment) -> bool:
    """Check if an evaluation unit falls within the bounds of a segment."""
    if (unit.clip_id, unit.track_id) != (segment.clip_id, segment.track_id):
        return False
    if (
        unit.frame_idx is not None
        and segment.frame_start is not None
        and segment.frame_end_exclusive is not None
    ):
        return segment.frame_start <= unit.frame_idx < segment.frame_end_exclusive
    # Fallback to temporal check
    # Check strictly half-open [start_s, end_s)
    return segment.start_s <= unit.time_s < segment.end_s


def project_units_to_segments(
    units: Sequence[EvaluationUnit],
    segments: Sequence[Segment],
    assignments: Sequence[Assignment],
    discovery_report: DiscoveryReport | None = None,
) -> list[ProjectedUnit]:
    """Project fixed evaluation units onto experiment segments and assignment decisions.

    Regardless of how segmentation partitions tracks into fewer or more segments,
    every evaluation unit is mapped to its enclosing active segment's assignment.
    """
    active_segments = [s for s in segments if s.active]
    segments_by_track: dict[tuple[str, str], list[Segment]] = defaultdict(list)
    for seg in active_segments:
        segments_by_track[(seg.clip_id, seg.track_id)].append(seg)

    assignments_by_segment: dict[str, Assignment] = {a.segment_id: a for a in assignments}

    group_id_by_segment: dict[str, str] = {}
    unknown_reasons_by_segment: dict[str, str] = {}
    if discovery_report is not None:
        for group in discovery_report.groups:
            for member_sid in group.member_segment_ids:
                group_id_by_segment[member_sid] = group.group_id
        for unk in discovery_report.unknown:
            if unk.reasons:
                unknown_reasons_by_segment[unk.segment_id] = unk.reasons[0]

    projected: list[ProjectedUnit] = []
    for unit in units:
        candidate_segments = segments_by_track.get((unit.clip_id, unit.track_id), [])
        matched_segment: Segment | None = None
        for seg in candidate_segments:
            if _unit_in_segment(unit, seg):
                matched_segment = seg
                break

        if matched_segment is not None:
            assignment = assignments_by_segment.get(matched_segment.segment_id)
            if (
                assignment is not None
                and assignment.status in ("proposed", "confirmed")
                and assignment.player_id
            ):
                status: Literal["assigned", "unassigned"] = "assigned"
                player_id = assignment.player_id
                group_id = group_id_by_segment.get(matched_segment.segment_id)
                unassigned_reason = None
            else:
                status = "unassigned"
                player_id = None
                group_id = None
                specific_reason = unknown_reasons_by_segment.get(matched_segment.segment_id)
                assigned_reason = assignment.unassigned_reason if assignment is not None else None
                if specific_reason:
                    unassigned_reason = specific_reason
                elif assigned_reason:
                    unassigned_reason = assigned_reason
                else:
                    unassigned_reason = "unknown"
            projected.append(
                ProjectedUnit(
                    unit_id=unit.unit_id,
                    unit_type=unit.unit_type,
                    clip_id=unit.clip_id,
                    track_id=unit.track_id,
                    frame_idx=unit.frame_idx,
                    time_s=unit.time_s,
                    sample_id=unit.sample_id,
                    observation_id=unit.observation_id,
                    segment_id=matched_segment.segment_id,
                    status=status,
                    player_id=player_id,
                    group_id=group_id,
                    unassigned_reason=unassigned_reason,
                )
            )
        else:
            # Not covered by any active segment (e.g. pruned or unsegmented gap)
            projected.append(
                ProjectedUnit(
                    unit_id=unit.unit_id,
                    unit_type=unit.unit_type,
                    clip_id=unit.clip_id,
                    track_id=unit.track_id,
                    frame_idx=unit.frame_idx,
                    time_s=unit.time_s,
                    sample_id=unit.sample_id,
                    observation_id=unit.observation_id,
                    segment_id=None,
                    status="unassigned",
                    player_id=None,
                    group_id=None,
                    unassigned_reason="uncovered_by_active_segments",
                )
            )

    return projected
