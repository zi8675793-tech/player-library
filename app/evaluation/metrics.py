"""Dual-track evaluation metric computation engine.

Implements:
1. Ground-truth-free unsupervised metrics:
   - Fixed evaluation unit sample coverage and duration coverage
   - Supported and qualified segment proportions
   - Unknown reason distribution breakdown
   - Identity group size and automatic template source distributions
   - Runtime and peak memory tracking
2. Ground-truth-dependent supervised metrics (default unvalidated):
   - Cross-clip pairwise TP, FP, FN, TN
   - Precision (strictly NaN when predicted positives == 0, never 100%)
   - Recall (0.0 when GT has positives but predictions are all unassigned)
   - Over-merging (impure predicted groups) and over-splitting (fragmented GT individuals)
   - Unlabelled / ambiguous ground truth tolerance
"""

import math
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.contracts.discovery import DiscoveryReport
from app.contracts.identity import Segment
from app.contracts.segmentation import SegmentationReport
from app.evaluation.units import EvaluationManifest, ProjectedUnit


class PairwiseCounts(BaseModel):
    """Cross-clip pairwise evaluation metrics."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=True)

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    precision: float | None = None
    precision_display: str = "NaN (无法计算)"
    recall: float | None = None
    recall_display: str = "0.0%"
    f1: float | None = None
    f1_display: str = "NaN (无法计算)"
    positive_pairs_predicted: int = 0
    positive_pairs_truth: int = 0
    evaluated_pairs_count: int = 0


class ImpurityGroup(BaseModel):
    """Details of a merge error where a predicted group contains multiple ground truths."""

    model_config = ConfigDict(extra="forbid")

    group_id: str
    player_id: str | None = None
    gt_person_ids: list[str]
    unit_count: int


class SplitPerson(BaseModel):
    """Details of a split error where one ground truth person is split across multiple groups."""

    model_config = ConfigDict(extra="forbid")

    gt_person_id: str
    predicted_player_ids: list[str]
    unit_count: int


class UnsupervisedMetrics(BaseModel):
    """Metrics computed purely from machine evidence and model output without ground truth."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=True)

    total_samples: int
    assigned_samples: int
    sample_coverage: float
    total_observations: int
    assigned_observations: int
    observation_coverage: float
    total_duration_s: float
    assigned_duration_s: float
    duration_coverage: float
    supported_segments_ratio: float | None = None
    qualified_segments_ratio: float | None = None
    unknown_reason_distribution: dict[str, int] = Field(default_factory=dict)
    identity_group_size_distribution: dict[str, int] = Field(default_factory=dict)
    template_source_distribution: dict[str, int] = Field(default_factory=dict)
    runtime_s: float | None = None
    peak_memory_mb: float | None = None


class SupervisedMetrics(BaseModel):
    """Metrics computed against ground truth identities (default: unvalidated)."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=True)

    identity_recognition_evaluated: bool = False
    status: Literal["evaluated", "unvalidated"] = "unvalidated"
    message: str | None = "真实身份精度未评估：未提供身份真值（符合数据隔离规范）"
    pairwise: PairwiseCounts | None = None
    merge_errors: list[ImpurityGroup] = Field(default_factory=list)
    merge_error_count: int = 0
    split_errors: list[SplitPerson] = Field(default_factory=list)
    split_error_count: int = 0
    experimental_group_merge_errors: list[ImpurityGroup] = Field(default_factory=list)
    experimental_group_merge_error_count: int = 0
    experimental_group_split_errors: list[SplitPerson] = Field(default_factory=list)
    experimental_group_split_error_count: int = 0
    unlabelled_units_excluded: int = 0


class Stage6BenchmarkMetrics(BaseModel):
    """The 9 explicit evaluation metrics defined for Stage 6 independent audit.

    All metrics explicitly state their denominators, missing ground-truth handling,
    and never report false 100% precision when predictions are zero.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=True)

    # 1. 正确覆盖 (Correct Coverage): 标注范围内正确分配的样本/观测/时长比例
    correct_coverage_samples: float | None = None
    correct_coverage_observations: float | None = None
    correct_coverage_duration: float | None = None
    correct_coverage_denominator_note: str = (
        "有限标注范围（目标范围118样本/10段），全场未标注样本不计入真值分母"
    )

    # 2. 混人率 (Merge Error Rate / Confusion Rate): 预测球员包含多个真值人物的比例
    merge_error_rate: float = 0.0
    merge_error_count: int = 0
    merge_errors_details: list[dict] = Field(default_factory=list)

    # 3. 同人拆档率 (Split Error Rate): 同一真值人物被拆分为多个预测球员的比例
    split_error_rate: float = 0.0
    split_error_count: int = 0
    split_errors_details: list[dict] = Field(default_factory=list)

    # 4. 已归人纯度 (Assigned Purity / Precision): 预测赋予ID的单元中属于主流真值的比例
    assigned_purity: float | None = None
    assigned_purity_note: str = ""

    # 5. 跨片配对召回与精度 (Cross-Clip Pairwise Recall & Precision)
    cross_clip_pairwise_precision: float | None = None
    cross_clip_pairwise_recall: float | None = None
    cross_clip_pairwise_f1: float | None = None
    cross_clip_pairwise_tp: int = 0
    cross_clip_pairwise_fp: int = 0
    cross_clip_pairwise_fn: int = 0

    # 6. 延期率 (Deferred Rate): 证据不足被置于 insufficient_evidence / 延期的段数比例
    deferred_rate: float = 0.0
    deferred_count: int = 0
    total_candidate_segments: int = 0

    # 7. 增量重复建档率 (Incremental Duplicate Identity Rate): 同一球员被重复新建身份而非匹配已有身份的比例
    incremental_duplicate_rate: float | None = None
    incremental_duplicate_count: int = 0
    incremental_duplicate_note: str = ""

    # 8. 批次顺序稳定性 (Batch Order Stability): Forward vs Reverse 联合归属 ARI 与关系一致率
    order_stability_ari: float | None = None
    order_stability_agreement: float | None = None

    # 9. 非球员污染率 (Non-Player Role Intrusion Rate): 裁判/工作人员/非球员被赋予正式球员ID的比例
    non_player_pollution_rate: float | None = None
    non_player_pollution_status: Literal["zero_pollution", "uncomputable_missing_ground_truth"] = (
        "zero_pollution"
    )
    non_player_pollution_note: str = "当前标注集缺少裁判/工作人员真值标注时标记为不可计算，零假阳性"


class DualTrackEvaluationReport(BaseModel):
    """Unified evaluation report containing both unsupervised and supervised tracks."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=True)

    schema_version: str = "dual-track-evaluation-v1"
    match_id: str
    unsupervised: UnsupervisedMetrics
    supervised: SupervisedMetrics
    stage6_metrics: Stage6BenchmarkMetrics | None = None


@contextmanager
def measure_execution():
    """Measure wall-clock time and peak memory allocated via tracemalloc."""
    import tracemalloc

    tracemalloc.start()
    t0 = time.perf_counter()
    metrics: dict[str, float] = {}
    try:
        yield metrics
    finally:
        t1 = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        metrics["runtime_s"] = t1 - t0
        metrics["peak_memory_mb"] = peak / (1024 * 1024)


def compute_unsupervised_metrics(
    manifest: EvaluationManifest,
    projected_units: Sequence[ProjectedUnit],
    *,
    segments: Sequence[Segment] | None = None,
    segmentation_report: SegmentationReport | None = None,
    discovery_report: DiscoveryReport | None = None,
    runtime_s: float | None = None,
    peak_memory_mb: float | None = None,
) -> UnsupervisedMetrics:
    """Compute all ground-truth-free evaluation metrics."""
    # 1. Sample metrics
    sample_units = [u for u in projected_units if u.unit_type == "sample"]
    total_samples = manifest.total_samples
    assigned_samples = sum(1 for u in sample_units if u.status == "assigned")
    sample_coverage = (assigned_samples / total_samples) if total_samples > 0 else 0.0

    # 2. Observation metrics
    obs_units = [u for u in projected_units if u.unit_type == "observation"]
    total_obs = manifest.total_observations
    assigned_obs = sum(1 for u in obs_units if u.status == "assigned")
    obs_coverage = (assigned_obs / total_obs) if total_obs > 0 else 0.0

    # 3. Duration metrics
    # Duration coverage is defined as assigned segment extent over total raw track extent
    assigned_duration_s = 0.0
    if segments is not None:
        active_segments = [s for s in segments if s.active]
        assigned_segment_ids = {
            u.segment_id for u in projected_units if u.status == "assigned" and u.segment_id
        }
        for seg in active_segments:
            if seg.segment_id in assigned_segment_ids:
                assigned_duration_s += seg.end_s - seg.start_s

    total_duration_s = manifest.total_duration_s
    duration_coverage = (assigned_duration_s / total_duration_s) if total_duration_s > 0 else 0.0

    # 4. Segment proportions
    supported_ratio: float | None = None
    qualified_ratio: float | None = None
    if segments is not None:
        active_segments = [s for s in segments if s.active]
        total_active_segments = len(active_segments)
        if total_active_segments > 0:
            if segmentation_report is not None:
                # Count supported proposals
                supported_count = sum(
                    1
                    for track in segmentation_report.tracks
                    for p in track.proposals
                    if p.status == "supported"
                )
                supported_ratio = supported_count / total_active_segments

            if discovery_report is not None:
                qualified_count = discovery_report.summary.get("qualified_segments")
                if qualified_count is not None:
                    qualified_ratio = qualified_count / total_active_segments

    # 5. Unknown reasons distribution
    unknown_reasons = Counter(
        u.unassigned_reason
        for u in projected_units
        if u.status == "unassigned" and u.unassigned_reason
    )

    # 6. Group size distribution
    group_sizes: dict[str, int] = {}
    if discovery_report is not None:
        for group in discovery_report.groups:
            group_sizes[group.group_id] = len(group.member_segment_ids)

    # 7. Template source distribution
    template_counts: dict[str, int] = {}
    if discovery_report is not None:
        for group in discovery_report.groups:
            template_counts[group.group_id] = len(group.templates)

    return UnsupervisedMetrics(
        total_samples=total_samples,
        assigned_samples=assigned_samples,
        sample_coverage=sample_coverage,
        total_observations=total_obs,
        assigned_observations=assigned_obs,
        observation_coverage=obs_coverage,
        total_duration_s=total_duration_s,
        assigned_duration_s=assigned_duration_s,
        duration_coverage=duration_coverage,
        supported_segments_ratio=supported_ratio,
        qualified_segments_ratio=qualified_ratio,
        unknown_reason_distribution=dict(unknown_reasons),
        identity_group_size_distribution=group_sizes,
        template_source_distribution=template_counts,
        runtime_s=runtime_s,
        peak_memory_mb=peak_memory_mb,
    )


def compute_supervised_metrics(
    projected_units: Sequence[ProjectedUnit],
    ground_truth: dict[str, str | None] | None,
    *,
    unit_type_filter: str | None = "sample",
) -> SupervisedMetrics:
    """Compute ground-truth-dependent metrics.

    If ground_truth is None or empty, returns an explicitly unvalidated result.
    Strictly handles edge conditions:
    - If positive predictions == 0, Precision is NaN (never 100%).
    - If positive truth exists and predictions are all unassigned, Recall is 0.0.
    - Unlabelled or ambiguous annotations are excluded from error counting.
    """
    if not ground_truth:
        return SupervisedMetrics(
            identity_recognition_evaluated=False,
            status="unvalidated",
            message="真实身份精度未评估：未提供身份真值（符合数据隔离规范）",
            pairwise=None,
            merge_errors=[],
            merge_error_count=0,
            split_errors=[],
            split_error_count=0,
            unlabelled_units_excluded=0,
        )

    # Filter units by unit_type if specified (default: sample units for ReID evaluation)
    candidate_units = (
        [u for u in projected_units if u.unit_type == unit_type_filter]
        if unit_type_filter is not None
        else list(projected_units)
    )

    unlabelled_tokens = {
        None,
        "",
        "unlabelled",
        "unknown",
        "ambiguous",
        "cannot_determine",
        "unannotated",
    }

    # Separate validly labelled units and unlabelled units
    valid_units: list[ProjectedUnit] = []
    unlabelled_count = 0
    unit_gt: dict[str, str] = {}

    for u in candidate_units:
        # Ground truth can be keyed by unit_id or sample_id
        gt_val = ground_truth.get(u.unit_id)
        if gt_val is None and u.sample_id:
            gt_val = ground_truth.get(u.sample_id)
        if gt_val is None:
            # Fallback to (clip_id, track_id) or (clip_id, track_id, frame_idx) key
            gt_val = ground_truth.get(f"{u.clip_id}:{u.track_id}:{u.frame_idx}")
            if gt_val is None:
                gt_val = ground_truth.get(f"{u.clip_id}:{u.track_id}")

        if gt_val is None or str(gt_val).strip().lower() in unlabelled_tokens:
            unlabelled_count += 1
        else:
            gt_clean = str(gt_val).strip()
            unit_gt[u.unit_id] = gt_clean
            valid_units.append(u)

    # 1. Cross-clip pairwise evaluation
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    evaluated_pairs = 0

    n = len(valid_units)
    for i in range(n):
        u1 = valid_units[i]
        gt1 = unit_gt[u1.unit_id]
        pred1 = u1.player_id if u1.status == "assigned" else None

        for j in range(i + 1, n):
            u2 = valid_units[j]
            # Only evaluate cross-clip pairs!
            if u1.clip_id == u2.clip_id:
                continue

            gt2 = unit_gt[u2.unit_id]
            pred2 = u2.player_id if u2.status == "assigned" else None

            evaluated_pairs += 1
            is_same_gt = gt1 == gt2
            is_same_pred = pred1 is not None and pred2 is not None and pred1 == pred2

            if is_same_gt:
                if is_same_pred:
                    tp += 1
                else:
                    fn += 1
            else:
                if is_same_pred:
                    fp += 1
                else:
                    tn += 1

    pos_pred = tp + fp
    pos_truth = tp + fn

    # Boundary conditions
    if pos_pred == 0:
        precision = float("nan")
        precision_display = "NaN (无法计算)"
    else:
        precision = tp / pos_pred
        precision_display = f"{precision * 100:.2f}%"

    if pos_truth == 0:
        recall = float("nan")
        recall_display = "NaN (无真值正例)"
    else:
        recall = tp / pos_truth
        recall_display = f"{recall * 100:.2f}%"

    f1_denom = 2 * tp + fp + fn
    if f1_denom == 0:
        f1 = float("nan")
        f1_display = "NaN (无法计算)"
    else:
        f1 = (2 * tp) / f1_denom
        f1_display = f"{f1 * 100:.2f}%"

    pairwise = PairwiseCounts(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=None if math.isnan(precision) else precision,
        precision_display=precision_display,
        recall=None if math.isnan(recall) else recall,
        recall_display=recall_display,
        f1=None if math.isnan(f1) else f1,
        f1_display=f1_display,
        positive_pairs_predicted=pos_pred,
        positive_pairs_truth=pos_truth,
        evaluated_pairs_count=evaluated_pairs,
    )

    # 2. Over-merging (误合并) detection
    # Unified evaluation at persistent player_id level to match pairwise identity
    predicted_players: dict[str, set[str]] = defaultdict(set)
    player_units: dict[str, int] = Counter()
    # Experimental grouping statistics at group_id level
    predicted_groups: dict[str, set[str]] = defaultdict(set)
    group_units: dict[str, int] = Counter()

    for u in valid_units:
        if u.status == "assigned" and u.player_id:
            pid = u.player_id
            predicted_players[pid].add(unit_gt[u.unit_id])
            player_units[pid] += 1
            if u.group_id:
                predicted_groups[u.group_id].add(unit_gt[u.unit_id])
                group_units[u.group_id] += 1

    merge_errors: list[ImpurityGroup] = []
    for pid, gt_set in sorted(predicted_players.items()):
        if len(gt_set) > 1:
            merge_errors.append(
                ImpurityGroup(
                    group_id=pid,
                    player_id=pid,
                    gt_person_ids=sorted(list(gt_set)),
                    unit_count=player_units[pid],
                )
            )

    exp_group_merge_errors: list[ImpurityGroup] = []
    for gid, gt_set in sorted(predicted_groups.items()):
        if len(gt_set) > 1:
            exp_group_merge_errors.append(
                ImpurityGroup(
                    group_id=gid,
                    player_id=None,
                    gt_person_ids=sorted(list(gt_set)),
                    unit_count=group_units[gid],
                )
            )

    # 3. Over-splitting (误拆分) detection
    # Unified evaluation at persistent player_id level
    gt_to_pred_players: dict[str, set[str]] = defaultdict(set)
    gt_to_pred_groups: dict[str, set[str]] = defaultdict(set)
    gt_units: dict[str, int] = Counter()

    for u in valid_units:
        gt_id = unit_gt[u.unit_id]
        gt_units[gt_id] += 1
        if u.status == "assigned" and u.player_id:
            gt_to_pred_players[gt_id].add(u.player_id)
            if u.group_id:
                gt_to_pred_groups[gt_id].add(u.group_id)

    split_errors: list[SplitPerson] = []
    for gt_id, pred_set in sorted(gt_to_pred_players.items()):
        if len(pred_set) > 1:
            split_errors.append(
                SplitPerson(
                    gt_person_id=gt_id,
                    predicted_player_ids=sorted(list(pred_set)),
                    unit_count=gt_units[gt_id],
                )
            )

    exp_group_split_errors: list[SplitPerson] = []
    for gt_id, pred_set in sorted(gt_to_pred_groups.items()):
        if len(pred_set) > 1:
            exp_group_split_errors.append(
                SplitPerson(
                    gt_person_id=gt_id,
                    predicted_player_ids=sorted(list(pred_set)),
                    unit_count=gt_units[gt_id],
                )
            )

    return SupervisedMetrics(
        identity_recognition_evaluated=True,
        status="evaluated",
        message="已完成真值关联度量评估",
        pairwise=pairwise,
        merge_errors=merge_errors,
        merge_error_count=len(merge_errors),
        split_errors=split_errors,
        split_error_count=len(split_errors),
        experimental_group_merge_errors=exp_group_merge_errors,
        experimental_group_merge_error_count=len(exp_group_merge_errors),
        experimental_group_split_errors=exp_group_split_errors,
        experimental_group_split_error_count=len(exp_group_split_errors),
        unlabelled_units_excluded=unlabelled_count,
    )


def evaluate_dual_track(
    manifest: EvaluationManifest,
    projected_units: Sequence[ProjectedUnit],
    *,
    ground_truth: dict[str, str | None] | None = None,
    segments: Sequence[Segment] | None = None,
    segmentation_report: SegmentationReport | None = None,
    discovery_report: DiscoveryReport | None = None,
    runtime_s: float | None = None,
    peak_memory_mb: float | None = None,
    unit_type_filter: str | None = "sample",
) -> DualTrackEvaluationReport:
    """Run dual-track evaluation combining unsupervised metrics and optional supervised validation."""
    unsupervised = compute_unsupervised_metrics(
        manifest,
        projected_units,
        segments=segments,
        segmentation_report=segmentation_report,
        discovery_report=discovery_report,
        runtime_s=runtime_s,
        peak_memory_mb=peak_memory_mb,
    )
    supervised = compute_supervised_metrics(
        projected_units,
        ground_truth,
        unit_type_filter=unit_type_filter,
    )
    return DualTrackEvaluationReport(
        schema_version="dual-track-evaluation-v1",
        match_id=manifest.match_id,
        unsupervised=unsupervised,
        supervised=supervised,
    )
