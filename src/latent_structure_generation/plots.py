"""Plotting helpers for V0 validation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from latent_structure_generation.geometry import pairwise_distance_matrix


def plot_distance_matrix(coords: np.ndarray, title: str = "C-alpha distance matrix"):
    """Plot a C-alpha pairwise distance matrix."""
    distances = pairwise_distance_matrix(coords)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(distances)
    ax.set_title(title)
    ax.set_xlabel("Residue index")
    ax.set_ylabel("Residue index")
    fig.colorbar(im, ax=ax, label="Distance (Å)")
    fig.tight_layout()
    return fig, ax


def plot_ca_trace(coords: np.ndarray, title: str = "C-alpha trace"):
    """Plot a simple 3D C-alpha trace."""
    coords = np.asarray(coords, dtype=float)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], marker="o", markersize=2)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.tight_layout()
    return fig, ax


def save_metric_table(metrics: pd.DataFrame, path: str | Path) -> None:
    """Save a metric dataframe as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(path, index=False)
