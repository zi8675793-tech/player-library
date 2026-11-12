"""Unified coverage funnel and detailed multi-dimensional loss breakdown.

Implements:
1. Unified 7-Stage Coverage Funnel:
   - raw_tracks -> usable_samples -> locally_consistent_segments ->
     cross_clip_comparable_segments -> candidate_matches ->
     final_assigned_identities -> templates_admitted
   - Fixed denominators: total raw tracks, samples, observations, track duration_s
   - Distinct qualifications and explicit handling of inapplicable/uncalculated stages
2. Multi-dimensional Loss Breakdown:
   - by_clip: sample, observation, duration, track loss per clip
   - by_track: granular per-track sample, observation, duration, segment loss and reasons
   - by_reason: counts of affected segments, samples, observations, duration, with
     co-occurrence tracking to prevent naive linear addition into false total loss
   - by_batch: incremental batch arrival loss and cumulative assignment progression
"""

from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.contracts.discovery import DiscoveryReport
from app.contracts.evidence import EvidenceBatch
from app.contracts.identity import Assignment, Segment, Template
from app.contracts.segmentation import SegmentationReport
from app.evaluation.units import EvaluationManifest, ProjectedUnit


class FixedDenominators(BaseModel):
    """Immutable machine input denominators."""

    model_config = ConfigDict(extra="forbid")

    total_tracks: int
    total_samples: int
    total_observations: int
    total_duration_s: float


class FunnelStage(BaseModel):
    """A single stage in the identity coverage pipeline."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=True)

    stage_name: str
    stage_label: str
    qualification_definition: str
    applicable: bool = True
    not_applicable_reason: str | None = None

    tracks_count: int | None = None
    segments_count: int | None = None
    samples_count: int | None = None
    observations_count: int | None = None
    duration_s: float | None = None
    templates_count: int | None = None

    track_coverage: float | None = None
    sample_coverage: float | None = None
    observation_coverage: float | None = None
    duration_coverage: float | None = None


class CoverageFunnel(BaseModel):
    """Unified 7-stage identity coverage funnel."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=True)

    schema_version: str = "coverage-funnel-v1"
    match_id: str
    denominators: FixedDenominators
    stages: list[FunnelStage] = Field(default_factory=list)


class ClipLossDetail(BaseModel):
    """Loss metrics aggregated by clip."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=True)

    clip_id: str
    total_tracks: int
    assigned_tracks: int
    lost_tracks: int

    total_samples: int
    assigned_samples: int
    lost_samples: int
    sample_coverage: float

    total_observations: int
    assigned_observations: int
    lost_observations: int
    observation_coverage: float

    total_duration_s: float
    assigned_duration_s: float
    lost_duration_s: float
    duration_coverage: float

    active_segments: int
    assigned_segments: int
    lost_segments: int


class TrackLossDetail(BaseModel):
    """Loss metrics for an individual raw track (clip_id, track_id)."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=True)

    clip_id: str
    track_id: str

    total_samples: int
    assigned_samples: int
    lost_samples: int
    sample_coverage: float

    total_observations: int
    assigned_observations: int
    lost_observations: int
    observation_coverage: float

    duration_s: float
    assigned_duration_s: float
    lost_duration_s: float
    duration_coverage: float

    active_segments: int
    assigned_segments: int
    lost_segments: int

    is_fully_assigned: bool
    has_any_assignment: bool
    unassigned_reasons: list[str] = Field(default_factory=list)


class ReasonLossDetail(BaseModel):
    """Loss metrics aggregated by rejection / unassigned reason."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=True)

    reason: str
    hit_segments: int
    hit_samples: int
    hit_observations: int
    hit_duration_s: float
    co_occurring_reasons: dict[str, int] = Field(default_factory=dict)
    explanation: str


class BatchLossDetail(BaseModel):
    """Loss metrics for an incremental batch."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=True)

    batch_index: int
    batch_id: str
    clips: list[str]

    total_tracks: int
    assigned_tracks: int
    lost_tracks: int

    total_samples: int
    assigned_samples: int
    lost_samples: int

    total_observations: int
    assigned_observations: int
    lost_observations: int

    total_duration_s: float
    assigned_duration_s: float
    lost_duration_s: float

    matched_existing_segments: int
    created_new_identity_segments: int
    insufficient_evidence_segments: int


class LossBreakdown(BaseModel):
    """Detailed multi-dimensional loss breakdown."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=True)

    schema_version: str = "coverage-loss-breakdown-v1"
    match_id: str
    fixed_denominators: FixedDenominators

    total_unassigned_samples: int
    total_unassigned_observations: int
    total_unassigned_duration_s: float
    total_unassigned_segments: int

    by_clip: dict[str, ClipLossDetail] = Field(default_factory=dict)
    by_track: list[TrackLossDetail] = Field(default_factory=list)
    by_reason: dict[str, ReasonLossDetail] = Field(default_factory=dict)
    by_batch: list[BatchLossDetail] | None = None

    overlap_warning: str = (
        "注意：同一未归属段可同时命中多个原因标签（见各原因co_occurring_reasons）。"
        "各原因损失为独立标签命中数，不可简单线性相加，避免虚假累加总损失。"
    )


def build_coverage_funnel(
    manifest: EvaluationManifest,
    projected_units: Sequence[ProjectedUnit],
    raw_batch: EvidenceBatch,
    *,
    segments: Sequence[Segment] | None = None,
    assignments: Sequence[Assignment] | None = None,
    segmentation_report: SegmentationReport | None = None,
    discovery_report: DiscoveryReport | None = None,
    templates: Sequence[Template] | None = None,
) -> CoverageFunnel:
    """Build the unified 7-stage coverage funnel against fixed denominators."""
    total_tracks = manifest.track_count
    total_samples = manifest.total_samples
    total_obs = manifest.total_observations
    total_dur = manifest.total_duration_s

    denominators = FixedDenominators(
        total_tracks=total_tracks,
        total_samples=total_samples,
        total_observations=total_obs,
        total_duration_s=total_dur,
    )

    stages: list[FunnelStage] = []

    # 1. 原始轨迹 (raw_tracks)
    stages.append(
        FunnelStage(
            stage_name="raw_tracks",
            stage_label="原始轨迹",
            qualification_definition="上游机器交付的全部原始轨迹、特征样本与逐帧观测",
            applicable=True,
            tracks_count=total_tracks,
            segments_count=len(segments) if segments is not None else total_tracks,
            samples_count=total_samples,
            observations_count=total_obs,
            duration_s=round(total_dur, 4),
            track_coverage=1.0,
            sample_coverage=1.0,
            observation_coverage=1.0,
            duration_coverage=1.0,
        )
    )

    # 2. 可用样本 (usable_samples)
    evidence = raw_batch.identity_evidence
    usable_samples_list = []
    if evidence is not None and evidence.samples:
        for s in evidence.samples:
            q = s.tracking_quality
            if q is not None and (
                getattr(q, "usability", None) == "unusable"
                or getattr(q, "clean_review", None) == "rejected"
            ):
                continue
            usable_samples_list.append(s)
        usable_sample_count = len(usable_samples_list)
        usable_tracks_count = len({(s.clip_id, s.track_id) for s in usable_samples_list})
    else:
        usable_sample_count = total_samples
        usable_tracks_count = total_tracks

    stages.append(
        FunnelStage(
            stage_name="usable_samples",
            stage_label="可用样本",
            qualification_definition="通过特征合同、数值检查与质量过滤，未被标记为不可用(unusable=false)的特征样本",
            applicable=True,
            tracks_count=usable_tracks_count,
            segments_count=None,
            samples_count=usable_sample_count,
            observations_count=total_obs,
            duration_s=round(total_dur, 4),
            track_coverage=round(usable_tracks_count / total_tracks, 4) if total_tracks else 0.0,
            sample_coverage=round(usable_sample_count / total_samples, 4) if total_samples else 0.0,
            observation_coverage=1.0,
            duration_coverage=1.0,
        )
    )

    # 3. 局部一致段 (locally_consistent_segments)
    if segmentation_report is not None:
        supported_props = [
            p for t in segmentation_report.tracks for p in t.proposals if p.status == "supported"
        ]
        supp_seg_ids = {p.segment.segment_id for p in supported_props}
        supp_tracks = {(p.segment.clip_id, p.segment.track_id) for p in supported_props}
        supp_samples = sum(
            1 for u in projected_units if u.unit_type == "sample" and u.segment_id in supp_seg_ids
        )
        supp_obs = sum(
            1
            for u in projected_units
            if u.unit_type == "observation" and u.segment_id in supp_seg_ids
        )
        supp_dur = sum(p.segment.end_s - p.segment.start_s for p in supported_props)

        stages.append(
            FunnelStage(
                stage_name="locally_consistent_segments",
                stage_label="局部一致段",
                qualification_definition="通过轨迹内逐对一致性检查或局部支持窗口提案，状态为 supported 的派生段",
                applicable=True,
                tracks_count=len(supp_tracks),
                segments_count=len(supported_props),
                samples_count=supp_samples,
                observations_count=supp_obs,
                duration_s=round(supp_dur, 4),
                track_coverage=round(len(supp_tracks) / total_tracks, 4) if total_tracks else 0.0,
                sample_coverage=round(supp_samples / total_samples, 4) if total_samples else 0.0,
                observation_coverage=round(supp_obs / total_obs, 4) if total_obs else 0.0,
                duration_coverage=round(supp_dur / total_dur, 4) if total_dur else 0.0,
            )
        )
    else:
        stages.append(
            FunnelStage(
                stage_name="locally_consistent_segments",
                stage_label="局部一致段",
                qualification_definition="通过轨迹内逐对一致性检查或局部支持窗口提案，状态为 supported 的派生段",
                applicable=False,
                not_applicable_reason="当前运行未执行轨迹内一致性分段 (segment_tracks=false)",
            )
        )

    # 4. 可跨片比较段 (cross_clip_comparable_segments)
    if discovery_report is not None:
        qualified_count = discovery_report.summary.get("qualified_segments", 0)
        # Find which segments were qualified
        qualified_seg_ids = set()
        for g in discovery_report.groups:
            qualified_seg_ids.update(g.member_segment_ids)
        cross_clip_unassigned_reasons = {
            "no_independent_cross_clip_group",
            "isolated_or_insufficient_cross_clip_support",
            "insufficient_cross_clip_support",
            "within_group_conflict",
            "insufficient_independent_template_support",
            "insufficient_profile_evidence_or_templates",
            "ambiguous_competing_existing_players",
            "conflict_with_existing_player_constraints",
        }
        for u in discovery_report.unknown:
            if u.reasons and all(r in cross_clip_unassigned_reasons for r in u.reasons):
                qualified_seg_ids.add(u.segment_id)

        # Fallback if qualified_segments > len(qualified_seg_ids)
        seg_dict = {s.segment_id: s for s in segments} if segments else {}
        q_tracks = {
            (seg_dict[sid].clip_id, seg_dict[sid].track_id)
            for sid in qualified_seg_ids
            if sid in seg_dict
        }
        q_samples = sum(
            1
            for u in projected_units
            if u.unit_type == "sample" and u.segment_id in qualified_seg_ids
        )
        q_obs = sum(
            1
            for u in projected_units
            if u.unit_type == "observation" and u.segment_id in qualified_seg_ids
        )
        q_dur = sum(
            seg_dict[sid].end_s - seg_dict[sid].start_s
            for sid in qualified_seg_ids
            if sid in seg_dict
        )

        stages.append(
            FunnelStage(
                stage_name="cross_clip_comparable_segments",
                stage_label="可跨片比较段",
                qualification_definition="通过特征合同、最小采样数且具备独立时空互斥图节点的跨片段候选合格段",
                applicable=True,
                tracks_count=len(q_tracks),
                segments_count=qualified_count if qualified_count else len(qualified_seg_ids),
                samples_count=q_samples,
                observations_count=q_obs,
                duration_s=round(q_dur, 4),
                track_coverage=round(len(q_tracks) / total_tracks, 4) if total_tracks else 0.0,
                sample_coverage=round(q_samples / total_samples, 4) if total_samples else 0.0,
                observation_coverage=round(q_obs / total_obs, 4) if total_obs else 0.0,
                duration_coverage=round(q_dur / total_dur, 4) if total_dur else 0.0,
            )
        )
    else:
        stages.append(
            FunnelStage(
                stage_name="cross_clip_comparable_segments",
                stage_label="可跨片比较段",
                qualification_definition="通过特征合同、最小采样数且具备独立时空互斥图节点的跨片段候选合格段",
                applicable=False,
                not_applicable_reason="当前运行未执行跨片段身份发现或增量聚类",
            )
        )

    # 5. 候选匹配 (candidate_matches)
    if discovery_report is not None and discovery_report.pairs:
        candidate_seg_ids = {
            p.left_segment_id
            for p in discovery_report.pairs
            if p.status in {"supported", "ambiguous"}
        } | {
            p.right_segment_id
            for p in discovery_report.pairs
            if p.status in {"supported", "ambiguous"}
        }
        cand_tracks = {
            (seg_dict[sid].clip_id, seg_dict[sid].track_id)
            for sid in candidate_seg_ids
            if sid in seg_dict
        }
        cand_samples = sum(
            1
            for u in projected_units
            if u.unit_type == "sample" and u.segment_id in candidate_seg_ids
        )
        cand_obs = sum(
            1
            for u in projected_units
            if u.unit_type == "observation" and u.segment_id in candidate_seg_ids
        )
        cand_dur = sum(
            seg_dict[sid].end_s - seg_dict[sid].start_s
            for sid in candidate_seg_ids
            if sid in seg_dict
        )

        stages.append(
            FunnelStage(
                stage_name="candidate_matches",
                stage_label="候选匹配",
                qualification_definition="在流形聚类或增量图库匹配中，至少与一个跨片候选形成正向距离支持的段",
                applicable=True,
                tracks_count=len(cand_tracks),
                segments_count=len(candidate_seg_ids),
                samples_count=cand_samples,
                observations_count=cand_obs,
                duration_s=round(cand_dur, 4),
                track_coverage=round(len(cand_tracks) / total_tracks, 4) if total_tracks else 0.0,
                sample_coverage=round(cand_samples / total_samples, 4) if total_samples else 0.0,
                observation_coverage=round(cand_obs / total_obs, 4) if total_obs else 0.0,
                duration_coverage=round(cand_dur / total_dur, 4) if total_dur else 0.0,
            )
        )
    else:
        stages.append(
            FunnelStage(
                stage_name="candidate_matches",
                stage_label="候选匹配",
                qualification_definition="在流形聚类或增量图库匹配中，至少与一个跨片候选形成正向距离支持的段",
                applicable=False,
                not_applicable_reason="当前运行无跨片段候选对输出（未开启发现或全部待定）",
            )
        )

    # 6. 最终归人 (final_assigned_identities)
    assigned_units = [u for u in projected_units if u.status == "assigned"]
    assigned_samples = sum(1 for u in assigned_units if u.unit_type == "sample")
    assigned_obs = sum(1 for u in assigned_units if u.unit_type == "observation")
    assigned_seg_ids = {u.segment_id for u in assigned_units if u.segment_id}
    assigned_tracks = {(u.clip_id, u.track_id) for u in assigned_units}
    seg_dict = {s.segment_id: s for s in segments} if segments else {}
    assigned_dur = sum(
        seg_dict[sid].end_s - seg_dict[sid].start_s for sid in assigned_seg_ids if sid in seg_dict
    )

    stages.append(
        FunnelStage(
            stage_name="final_assigned_identities",
            stage_label="最终归人",
            qualification_definition="满足全部物理互斥与跨片段见证约束，最终获得持久 Player ID 关联的段",
            applicable=True,
            tracks_count=len(assigned_tracks),
            segments_count=len(assigned_seg_ids),
            samples_count=assigned_samples,
            observations_count=assigned_obs,
            duration_s=round(assigned_dur, 4),
            track_coverage=round(len(assigned_tracks) / total_tracks, 4) if total_tracks else 0.0,
            sample_coverage=round(assigned_samples / total_samples, 4) if total_samples else 0.0,
            observation_coverage=round(assigned_obs / total_obs, 4) if total_obs else 0.0,
            duration_coverage=round(assigned_dur / total_dur, 4) if total_dur else 0.0,
        )
    )

    # 7. 模板入库 (templates_admitted)
    if templates is not None and len(templates) > 0:
        active_tpls = [t for t in templates if t.active]
        tpl_samples = len({t.sample_id for t in active_tpls})

        stages.append(
            FunnelStage(
                stage_name="templates_admitted",
                stage_label="模板入库",
                qualification_definition="具备跨片段独立见证并被选入图库快照的活跃身份模板样本",
                applicable=True,
                tracks_count=None,
                segments_count=None,
                samples_count=tpl_samples,
                observations_count=None,
                duration_s=None,
                templates_count=len(active_tpls),
                track_coverage=None,
                sample_coverage=round(tpl_samples / total_samples, 4) if total_samples else 0.0,
                observation_coverage=None,
                duration_coverage=None,
            )
        )
    else:
        stages.append(
            FunnelStage(
                stage_name="templates_admitted",
                stage_label="模板入库",
                qualification_definition="具备跨片段独立见证并被选入图库快照的活跃身份模板样本",
                applicable=False,
                not_applicable_reason="当前运行未产生或未入库活跃模板",
            )
        )

    return CoverageFunnel(
        schema_version="coverage-funnel-v1",
        match_id=manifest.match_id,
        denominators=denominators,
        stages=stages,
    )


def build_loss_breakdown(
    manifest: EvaluationManifest,
    projected_units: Sequence[ProjectedUnit],
    raw_batch: EvidenceBatch,
    *,
    segments: Sequence[Segment] | None = None,
    assignments: Sequence[Assignment] | None = None,
    discovery_report: DiscoveryReport | None = None,
    batch_summaries: list[dict[str, Any]] | None = None,
) -> LossBreakdown:
    """Compute granular multi-dimensional coverage loss breakdown."""
    denominators = FixedDenominators(
        total_tracks=manifest.track_count,
        total_samples=manifest.total_samples,
        total_observations=manifest.total_observations,
        total_duration_s=manifest.total_duration_s,
    )

    evidence = raw_batch.identity_evidence
    tracks_by_key = {(t.clip_id, t.track_id): t for t in (evidence.tracks if evidence else [])}

    # Group projected units by track
    sample_units = [u for u in projected_units if u.unit_type == "sample"]
    obs_units = [u for u in projected_units if u.unit_type == "observation"]

    samples_by_track = Counter((u.clip_id, u.track_id) for u in sample_units)
    assigned_samples_by_track = Counter(
        (u.clip_id, u.track_id) for u in sample_units if u.status == "assigned"
    )

    obs_by_track = Counter((u.clip_id, u.track_id) for u in obs_units)
    assigned_obs_by_track = Counter(
        (u.clip_id, u.track_id) for u in obs_units if u.status == "assigned"
    )

    # Active segments and assignments
    active_segments = [s for s in (segments or []) if s.active]
    seg_by_track = defaultdict(list)
    for s in active_segments:
        seg_by_track[(s.clip_id, s.track_id)].append(s)

    assigned_seg_ids = {
        a.segment_id
        for a in (assignments or [])
        if a.status in {"proposed", "confirmed"} and a.player_id is not None
    }

    # Reasons mapping for unassigned segments
    unassigned_reasons_by_seg: dict[str, list[str]] = defaultdict(list)
    if discovery_report is not None:
        for unk in discovery_report.unknown:
            if unk.reasons:
                unassigned_reasons_by_seg[unk.segment_id].extend(unk.reasons)

    for a in assignments or []:
        if a.status == "unassigned":
            found = False
            for ev in a.evidence:
                if ev.kind == "insufficient_evidence" and ev.description:
                    for r in ev.description.split("; "):
                        clean_r = r.strip()
                        if clean_r and clean_r not in unassigned_reasons_by_seg[a.segment_id]:
                            unassigned_reasons_by_seg[a.segment_id].append(clean_r)
                            found = True
            if not found and a.unassigned_reason:
                if a.unassigned_reason not in unassigned_reasons_by_seg[a.segment_id]:
                    unassigned_reasons_by_seg[a.segment_id].append(a.unassigned_reason)

    # 1. By Track Details
    track_details: list[TrackLossDetail] = []
    clip_tracks_map: dict[str, list[TrackLossDetail]] = defaultdict(list)

    for (cid, tid), track in sorted(tracks_by_key.items()):
        tot_samples = samples_by_track[(cid, tid)]
        asg_samples = assigned_samples_by_track[(cid, tid)]
        lost_samples = tot_samples - asg_samples

        tot_obs = obs_by_track[(cid, tid)]
        asg_obs = assigned_obs_by_track[(cid, tid)]
        lost_obs = tot_obs - asg_obs

        dur = float(track.end_s - track.start_s)
        track_segs = seg_by_track[(cid, tid)]
        asg_segs = [s for s in track_segs if s.segment_id in assigned_seg_ids]
        lost_segs = len(track_segs) - len(asg_segs)

        asg_dur = sum(s.end_s - s.start_s for s in asg_segs)
        lost_dur = dur - asg_dur

        reasons = []
        for s in track_segs:
            if s.segment_id not in assigned_seg_ids:
                reasons.extend(unassigned_reasons_by_seg.get(s.segment_id, []))
        clean_track_reasons = sorted(set(reasons)) or (["unknown"] if lost_samples > 0 else [])

        td = TrackLossDetail(
            clip_id=cid,
            track_id=tid,
            total_samples=tot_samples,
            assigned_samples=asg_samples,
            lost_samples=lost_samples,
            sample_coverage=round(asg_samples / tot_samples, 4) if tot_samples else 0.0,
            total_observations=tot_obs,
            assigned_observations=asg_obs,
            lost_observations=lost_obs,
            observation_coverage=round(asg_obs / tot_obs, 4) if tot_obs else 0.0,
            duration_s=round(dur, 4),
            assigned_duration_s=round(asg_dur, 4),
            lost_duration_s=round(max(0.0, lost_dur), 4),
            duration_coverage=round(asg_dur / dur, 4) if dur > 0 else 0.0,
            active_segments=len(track_segs),
            assigned_segments=len(asg_segs),
            lost_segments=lost_segs,
            is_fully_assigned=(tot_obs > 0 and tot_obs == asg_obs),
            has_any_assignment=(asg_samples > 0 or asg_obs > 0 or len(asg_segs) > 0),
            unassigned_reasons=clean_track_reasons,
        )
        track_details.append(td)
        clip_tracks_map[cid].append(td)

    # 2. By Clip Details
    clip_details: dict[str, ClipLossDetail] = {}
    for cid, tds in sorted(clip_tracks_map.items()):
        c_tot_tracks = len(tds)
        c_asg_tracks = sum(1 for t in tds if t.has_any_assignment)
        c_lost_tracks = c_tot_tracks - c_asg_tracks

        c_tot_s = sum(t.total_samples for t in tds)
        c_asg_s = sum(t.assigned_samples for t in tds)
        c_lost_s = c_tot_s - c_asg_s

        c_tot_o = sum(t.total_observations for t in tds)
        c_asg_o = sum(t.assigned_observations for t in tds)
        c_lost_o = c_tot_o - c_asg_o

        c_tot_d = sum(t.duration_s for t in tds)
        c_asg_d = sum(t.assigned_duration_s for t in tds)
        c_lost_d = c_tot_d - c_asg_d

        c_act_segs = sum(t.active_segments for t in tds)
        c_asg_segs = sum(t.assigned_segments for t in tds)
        c_lost_segs = c_act_segs - c_asg_segs

        clip_details[cid] = ClipLossDetail(
            clip_id=cid,
            total_tracks=c_tot_tracks,
            assigned_tracks=c_asg_tracks,
            lost_tracks=c_lost_tracks,
            total_samples=c_tot_s,
            assigned_samples=c_asg_s,
            lost_samples=c_lost_s,
            sample_coverage=round(c_asg_s / c_tot_s, 4) if c_tot_s else 0.0,
            total_observations=c_tot_o,
            assigned_observations=c_asg_o,
            lost_observations=c_lost_o,
            observation_coverage=round(c_asg_o / c_tot_o, 4) if c_tot_o else 0.0,
            total_duration_s=round(c_tot_d, 4),
            assigned_duration_s=round(c_asg_d, 4),
            lost_duration_s=round(max(0.0, c_lost_d), 4),
            duration_coverage=round(c_asg_d / c_tot_d, 4) if c_tot_d > 0 else 0.0,
            active_segments=c_act_segs,
            assigned_segments=c_asg_segs,
            lost_segments=c_lost_segs,
        )

    # 3. By Reason Details with Co-occurrence Tracking
    reason_segments: dict[str, list[Segment]] = defaultdict(list)
    reason_co_occurrence: dict[str, Counter] = defaultdict(Counter)

    for seg in active_segments:
        if seg.segment_id not in assigned_seg_ids:
            reasons = unassigned_reasons_by_seg.get(seg.segment_id, ["unknown"])
            if not reasons:
                reasons = ["unknown"]
            for r in reasons:
                reason_segments[r].append(seg)
                for other_r in reasons:
                    if other_r != r:
                        reason_co_occurrence[r][other_r] += 1

    seg_samples_map = Counter(u.segment_id for u in sample_units)
    seg_obs_map = Counter(u.segment_id for u in obs_units)

    reason_details: dict[str, ReasonLossDetail] = {}
    for r, segs in sorted(reason_segments.items()):
        hit_segs = len(segs)
        hit_s = sum(seg_samples_map[s.segment_id] for s in segs)
        hit_o = sum(seg_obs_map[s.segment_id] for s in segs)
        hit_d = sum(s.end_s - s.start_s for s in segs)

        reason_details[r] = ReasonLossDetail(
            reason=r,
            hit_segments=hit_segs,
            hit_samples=hit_s,
            hit_observations=hit_o,
            hit_duration_s=round(hit_d, 4),
            co_occurring_reasons=dict(reason_co_occurrence[r]),
            explanation=f"未归属原因: {r}",
        )

    # 4. By Batch Details
    batch_details: list[BatchLossDetail] | None = None
    if batch_summaries is not None and len(batch_summaries) > 0:
        batch_details = []
        for idx, bs in enumerate(batch_summaries):
            b_clips = bs.get("clips", [])
            b_tracks = sum(len(clip_tracks_map[c]) for c in b_clips)
            b_asg_tracks = sum(
                sum(1 for t in clip_tracks_map[c] if t.has_any_assignment) for c in b_clips
            )
            b_tot_s = sum(clip_details[c].total_samples for c in b_clips if c in clip_details)
            b_asg_s = sum(clip_details[c].assigned_samples for c in b_clips if c in clip_details)
            b_tot_o = sum(clip_details[c].total_observations for c in b_clips if c in clip_details)
            b_asg_o = sum(
                clip_details[c].assigned_observations for c in b_clips if c in clip_details
            )
            b_tot_d = sum(clip_details[c].total_duration_s for c in b_clips if c in clip_details)
            b_asg_d = sum(clip_details[c].assigned_duration_s for c in b_clips if c in clip_details)

            batch_details.append(
                BatchLossDetail(
                    batch_index=idx,
                    batch_id=bs.get("batch_id", f"batch-{idx}"),
                    clips=b_clips,
                    total_tracks=b_tracks,
                    assigned_tracks=b_asg_tracks,
                    lost_tracks=b_tracks - b_asg_tracks,
                    total_samples=b_tot_s,
                    assigned_samples=b_asg_s,
                    lost_samples=b_tot_s - b_asg_s,
                    total_observations=b_tot_o,
                    assigned_observations=b_asg_o,
                    lost_observations=b_tot_o - b_asg_o,
                    total_duration_s=round(b_tot_d, 4),
                    assigned_duration_s=round(b_asg_d, 4),
                    lost_duration_s=round(max(0.0, b_tot_d - b_asg_d), 4),
                    matched_existing_segments=bs.get("matched_existing", 0),
                    created_new_identity_segments=bs.get("created_new_identity", 0),
                    insufficient_evidence_segments=bs.get("insufficient_evidence", 0),
                )
            )

    unassigned_samples = sum(1 for u in sample_units if u.status == "unassigned")
    unassigned_obs = sum(1 for u in obs_units if u.status == "unassigned")
    unassigned_segs = len(active_segments) - len(assigned_seg_ids)
    unassigned_dur = sum(
        s.end_s - s.start_s for s in active_segments if s.segment_id not in assigned_seg_ids
    )

    return LossBreakdown(
        schema_version="coverage-loss-breakdown-v1",
        match_id=manifest.match_id,
        fixed_denominators=denominators,
        total_unassigned_samples=unassigned_samples,
        total_unassigned_observations=unassigned_obs,
        total_unassigned_duration_s=round(unassigned_dur, 4),
        total_unassigned_segments=unassigned_segs,
        by_clip=clip_details,
        by_track=track_details,
        by_reason=reason_details,
        by_batch=batch_details,
    )
