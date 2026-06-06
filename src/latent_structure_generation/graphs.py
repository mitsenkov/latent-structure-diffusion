"""Graph-building utilities for C-alpha protein traces."""

from __future__ import annotations

import numpy as np

from latent_structure_generation.geometry import pairwise_distance_matrix


def knn_edge_index(coords: np.ndarray, k: int = 8, include_reverse: bool = True) -> np.ndarray:
    """Build a k-nearest-neighbour edge index for a C-alpha trace.

    Returns an array with shape ``(2, n_edges)`` where row 0 contains source nodes and
    row 1 contains destination nodes.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n_residues, 3).")
    if k <= 0:
        raise ValueError("k must be positive.")

    n = len(coords)
    if n <= 1:
        return np.empty((2, 0), dtype=int)

    distances = pairwise_distance_matrix(coords)
    np.fill_diagonal(distances, np.inf)
    k_eff = min(k, n - 1)

    edges: set[tuple[int, int]] = set()
    for i in range(n):
        neighbours = np.argsort(distances[i])[:k_eff]
        for j in neighbours:
            edges.add((i, int(j)))
            if include_reverse:
                edges.add((int(j), i))

    edge_array = np.asarray(sorted(edges), dtype=int)
    return edge_array.T if len(edge_array) else np.empty((2, 0), dtype=int)


def sequential_edge_index(n_residues: int, include_reverse: bool = True) -> np.ndarray:
    """Return edges between consecutive residues."""
    if n_residues <= 1:
        return np.empty((2, 0), dtype=int)
    edges: list[tuple[int, int]] = []
    for i in range(n_residues - 1):
        edges.append((i, i + 1))
        if include_reverse:
            edges.append((i + 1, i))
    return np.asarray(edges, dtype=int).T
