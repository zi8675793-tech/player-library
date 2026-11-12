"""Pure identity rules: prepare, associate, resolve, and plan gallery updates."""

from app.identity.associate import associate_segments
from app.identity.clustering import (
    DiscoveredCluster,
    compute_cluster_distance,
    connected_components,
    constrained_hierarchical_clustering,
    find_graph_bridges,
)
from app.identity.constraints import SpatioTemporalConstraintEngine
from app.identity.discovery import (
    accept_candidate_groups,
    discover_identities,
    evaluate_existing_player_match,
    generate_candidate_clusters,
    group_and_accept_candidates,
    qualify_for_new_identity,
)
from app.identity.distance import FeatureValidationError, kpr_distance
from app.identity.gallery import plan_gallery
from app.identity.prepare import prepare_evidence
from app.identity.representation import (
    AppearancePrototype,
    SegmentRepresentation,
    TrackRepresentation,
    compute_multi_prototype_distance,
    compute_prototype_distance,
    compute_track_distance,
    extract_segment_prototypes,
    pool_segment_features,
    pool_track_features,
)
from app.identity.resolve import resolve_associations

__all__ = [
    "AppearancePrototype",
    "DiscoveredCluster",
    "FeatureValidationError",
    "SegmentRepresentation",
    "SpatioTemporalConstraintEngine",
    "TrackRepresentation",
    "accept_candidate_groups",
    "associate_segments",
    "compute_cluster_distance",
    "compute_multi_prototype_distance",
    "compute_prototype_distance",
    "compute_track_distance",
    "connected_components",
    "constrained_hierarchical_clustering",
    "discover_identities",
    "evaluate_existing_player_match",
    "extract_segment_prototypes",
    "find_graph_bridges",
    "generate_candidate_clusters",
    "group_and_accept_candidates",
    "kpr_distance",
    "plan_gallery",
    "pool_segment_features",
    "pool_track_features",
    "prepare_evidence",
    "qualify_for_new_identity",
    "resolve_associations",
]
