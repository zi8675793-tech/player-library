"""Evaluation framework: fixed evaluation units and dual-track metric calculation."""

from app.evaluation.funnel import (
    BatchLossDetail,
    ClipLossDetail,
    CoverageFunnel,
    FixedDenominators,
    FunnelStage,
    LossBreakdown,
    ReasonLossDetail,
    TrackLossDetail,
    build_coverage_funnel,
    build_loss_breakdown,
)
from app.evaluation.metrics import (
    DualTrackEvaluationReport,
    PairwiseCounts,
    SupervisedMetrics,
    UnsupervisedMetrics,
    compute_supervised_metrics,
    compute_unsupervised_metrics,
    evaluate_dual_track,
)
from app.evaluation.units import (
    EvaluationManifest,
    EvaluationUnit,
    ProjectedUnit,
    extract_evaluation_units,
    project_units_to_segments,
)

__all__ = [
    "BatchLossDetail",
    "ClipLossDetail",
    "CoverageFunnel",
    "DualTrackEvaluationReport",
    "EvaluationManifest",
    "EvaluationUnit",
    "FixedDenominators",
    "FunnelStage",
    "LossBreakdown",
    "PairwiseCounts",
    "ProjectedUnit",
    "ReasonLossDetail",
    "SupervisedMetrics",
    "TrackLossDetail",
    "UnsupervisedMetrics",
    "build_coverage_funnel",
    "build_loss_breakdown",
    "compute_supervised_metrics",
    "compute_unsupervised_metrics",
    "evaluate_dual_track",
    "extract_evaluation_units",
    "project_units_to_segments",
]
