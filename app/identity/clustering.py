"""Spatio-temporally constrained hierarchical manifold clustering.

Implements agglomerative hierarchical clustering subject to hard physical mutual
exclusion constraints: trajectories observed on the same screen (spatio-temporal
overlap within a single clip) are strictly forbidden from being merged. Supports
transitive viewpoint manifold chaining while preventing suicidal bridge execution.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.identity.constraints import SpatioTemporalConstraintEngine
from app.identity.representation import TrackRepresentation, compute_track_distance


@dataclass
class DiscoveredCluster:
    cluster_id: str
    track_indices: list[int]
    tracks: list[TrackRepresentation]
    clip_ids: set[str]
    is_core_player: bool  # len(clip_ids) >= 2 判定为主力核心球员
    average_internal_distance: float
    cluster_type: str = "unassigned"

    @property
    def is_single_clip(self) -> bool:
        """Return True if cluster appears in only one clip but has multiple tracks."""
        return self.cluster_type == "single_clip"

    @property
    def is_unassigned(self) -> bool:
        """Return True if cluster is an isolated/unmerged fragment."""
        return self.cluster_type == "unassigned"

    @property
    def category(self) -> str:
        """Alias for cluster_type ('core', 'single_clip', 'unassigned')."""
        return self.cluster_type

    @property
    def status(self) -> str:
        """Alias for cluster_type ('core', 'single_clip', 'unassigned')."""
        return self.cluster_type


def compute_cluster_distance(
    c_a: Sequence[int],
    c_b: Sequence[int],
    dist_matrix: np.ndarray,
    linkage: str = "complete",
) -> float:
    """Compute the linkage distance between two clusters of track indices.

    Supported linkage methods:
    - 'complete': maximum distance among all pairs. If any pair is incomparable
      (infinite distance), returns infinity.
    - 'average' / 'mean': mean distance among all finite pairwise distances.
    - 'single': minimum distance among all finite pairwise distances.
    """
    pairwise: list[float] = [float(dist_matrix[i, j]) for i in c_a for j in c_b]
    finite_vals: list[float] = [d for d in pairwise if np.isfinite(d)]

    if linkage == "complete":
        if len(finite_vals) < len(pairwise) or not finite_vals:
            return float("inf")
        return float(np.max(finite_vals))

    if linkage in ("average", "mean"):
        if not finite_vals:
            return float("inf")
        return float(np.mean(finite_vals))

    if linkage == "single":
        if not finite_vals:
            return float("inf")
        return float(np.min(finite_vals))

    raise ValueError(
        f"Unsupported linkage criterion: '{linkage}'. Supported: 'complete', 'average', 'single'"
    )


def constrained_hierarchical_clustering(
    tracks: list[TrackRepresentation],
    constraints: SpatioTemporalConstraintEngine,
    threshold: float = 0.28,
    linkage: str = "complete",
    min_common_parts: int = 2,
    min_vis: float = 0.2,
    distance_matrix: np.ndarray | None = None,
    max_pair_threshold: float | None = None,
) -> list[DiscoveredCluster]:
    """Perform agglomerative hierarchical clustering with spatio-temporal hard constraints.

    Args:
        tracks: List of pooled track representations.
        constraints: Spatio-temporal constraint engine defining forbidden track pairs.
        threshold: Distance threshold for merging (default 0.28, matching empirical median ~0.268).
        linkage: Linkage metric ('complete', 'average', or 'single').
        min_common_parts: Minimum overlapping visible parts to compute distance.
        min_vis: Minimum part visibility threshold.
        distance_matrix: Optional precomputed (N, N) pairwise distance matrix.
        max_pair_threshold: Optional upper bound threshold. Merges with any pairwise distance
            exceeding this bound are forbidden.

    Returns:
        List of DiscoveredCluster instances partitioned into core players, single-clip players,
        and unassigned fragments.
    """
    if not tracks:
        return []

    n = len(tracks)
    if n == 1:
        t = tracks[0]
        return [
            DiscoveredCluster(
                cluster_id="cluster_0",
                track_indices=[0],
                tracks=[t],
                clip_ids={t.clip_id},
                is_core_player=False,
                average_internal_distance=0.0,
                cluster_type="unassigned",
            )
        ]

    # 1. Precompute pairwise distance matrix D[i, j]
    if distance_matrix is not None:
        dist_matrix = np.asarray(distance_matrix, dtype=np.float64)
    else:
        dist_matrix = np.full((n, n), np.inf, dtype=np.float64)
        np.fill_diagonal(dist_matrix, 0.0)
        for i in range(n):
            for j in range(i + 1, n):
                d = compute_track_distance(
                    tracks[i], tracks[j], min_common_parts=min_common_parts, min_vis=min_vis
                )
                if d is not None:
                    dist_matrix[i, j] = d
                    dist_matrix[j, i] = d

    # 2. Initialize: Each track in its own cluster
    current_clusters: list[set[int]] = [{i} for i in range(n)]

    # 3. Iterative merge under hard spatio-temporal constraints
    while len(current_clusters) > 1:
        best_dist = float("inf")
        best_pair: tuple[int, int] | None = None

        for i in range(len(current_clusters)):
            c_a = current_clusters[i]
            for j in range(i + 1, len(current_clusters)):
                c_b = current_clusters[j]

                # Hard physical mutual exclusion check
                if not constraints.can_merge_clusters(c_a, c_b):
                    continue

                if max_pair_threshold is not None and any(
                    dist_matrix[u, v] > max_pair_threshold
                    for u in c_a
                    for v in c_b
                    if np.isfinite(dist_matrix[u, v])
                ):
                    continue

                d = compute_cluster_distance(list(c_a), list(c_b), dist_matrix, linkage=linkage)

                if d <= threshold and d < best_dist:
                    best_dist = d
                    best_pair = (i, j)

        if best_pair is None or best_dist > threshold:
            break

        merge_i, merge_j = best_pair
        current_clusters[merge_i] = current_clusters[merge_i] | current_clusters[merge_j]
        current_clusters.pop(merge_j)

    # 4. Partition and format results
    # Sort deterministically by the earliest track index in each cluster
    sorted_clusters = sorted(current_clusters, key=lambda c: min(c))

    results: list[DiscoveredCluster] = []
    for idx, c_set in enumerate(sorted_clusters):
        track_indices = sorted(c_set)
        cluster_tracks = [tracks[i] for i in track_indices]
        clip_ids = {t.clip_id for t in cluster_tracks}
        is_core = len(clip_ids) >= 2

        if len(track_indices) <= 1:
            avg_int_dist = 0.0
        else:
            int_dists = [
                dist_matrix[i, j]
                for i in track_indices
                for j in track_indices
                if i < j and np.isfinite(dist_matrix[i, j])
            ]
            avg_int_dist = float(np.mean(int_dists)) if int_dists else 0.0

        if is_core:
            cluster_type = "core"
        elif len(cluster_tracks) >= 2:
            cluster_type = "single_clip"
        else:
            cluster_type = "unassigned"

        results.append(
            DiscoveredCluster(
                cluster_id=f"cluster_{idx}",
                track_indices=track_indices,
                tracks=cluster_tracks,
                clip_ids=clip_ids,
                is_core_player=is_core,
                average_internal_distance=avg_int_dist,
                cluster_type=cluster_type,
            )
        )

    return results


def find_graph_bridges(
    nodes: Sequence[Any],
    edges: Sequence[tuple[Any, Any]],
) -> list[tuple[Any, Any]]:
    """Find all bridge edges (cut edges) in an undirected graph via Tarjan's algorithm.

    A bridge edge is an edge whose removal increases the number of connected components.
    """
    adj: dict[Any, set[Any]] = {u: set() for u in nodes}
    for u, v in edges:
        if u in adj and v in adj and u != v:
            adj[u].add(v)
            adj[v].add(u)

    tin: dict[Any, int] = {}
    low: dict[Any, int] = {}
    timer = 0
    bridges: list[tuple[Any, Any]] = []

    def dfs(node: Any, parent: Any = None) -> None:
        nonlocal timer
        tin[node] = low[node] = timer
        timer += 1
        for neighbor in sorted(adj[node], key=str):
            if neighbor == parent:
                continue
            if neighbor in tin:
                low[node] = min(low[node], tin[neighbor])
            else:
                dfs(neighbor, node)
                low[node] = min(low[node], low[neighbor])
                if low[neighbor] > tin[node]:
                    bridges.append(tuple(sorted((node, neighbor))))

    for node in sorted(nodes, key=str):
        if node not in tin:
            dfs(node)

    return sorted(bridges)


def connected_components(
    nodes: Sequence[Any],
    edges: Sequence[tuple[Any, Any]],
) -> list[set[Any]]:
    """Partition nodes into connected components based on the given edges."""
    adj: dict[Any, set[Any]] = {u: set() for u in nodes}
    for u, v in edges:
        if u in adj and v in adj and u != v:
            adj[u].add(v)
            adj[v].add(u)

    visited: set[Any] = set()
    components: list[set[Any]] = []
    for node in sorted(nodes, key=str):
        if node in visited:
            continue
        comp: set[Any] = set()
        queue = [node]
        visited.add(node)
        while queue:
            curr = queue.pop(0)
            comp.add(curr)
            for nbr in sorted(adj[curr], key=str):
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
        components.append(comp)
    return components


__all__ = [
    "DiscoveredCluster",
    "compute_cluster_distance",
    "connected_components",
    "constrained_hierarchical_clustering",
    "find_graph_bridges",
]
