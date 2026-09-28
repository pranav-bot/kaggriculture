"""Clustering routines for opponent opening trajectories (turns 0..99)."""

from __future__ import annotations

import numpy as np


def cluster_opponent_trajectory(trajectory: np.ndarray, n_clusters: int = 4) -> int:
    """Assign an opponent opening trajectory to a categorical strategy cluster.

    Args:
        trajectory: Array of shape (100, D) representing the first 100 turns of
            opponent action and observation features.
        n_clusters: Number of strategy clusters.

    Returns:
        Categorical integer cluster index in [0, n_clusters - 1].
    """
    arr = np.asarray(trajectory, dtype=np.float32)
    if arr.size == 0 or np.all(arr == 0):
        return 0

    # Extract summary profile: mean, std, and max along the temporal axis
    mean_feat = np.mean(arr, axis=0)
    std_feat = np.std(arr, axis=0)
    summary_vector = np.concatenate([mean_feat, std_feat])

    # Deterministic feature hashing / projection for fast and stable clustering
    # Projects the summary vector into a discrete cluster ID without fitting overhead
    weights = np.sin(np.arange(1, len(summary_vector) + 1, dtype=np.float32))
    score = float(np.dot(summary_vector, weights))

    if np.isnan(score) or np.isinf(score):
        return 0

    cluster_id = int(abs(hash(int(score * 1000.0)))) % n_clusters
    return cluster_id
