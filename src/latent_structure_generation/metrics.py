"""Validation metrics for C-alpha protein traces."""

from __future__ import annotations

import numpy as np
import pandas as pd

from latent_structure_generation.geometry import (
    adjacent_distances,
    local_bend_angles,
    pairwise_distance_matrix,
    pseudo_dihedrals,
    radius_of_gyration,
    validate_distance_matrix,
)


def ca_validation_summary(coords: np.ndarray, structure_id: str = "structure") -> dict[str, float | int | str | bool]:
    """Return a compact validation summary for one C-alpha coordinate trace."""
    coords = np.asarray(coords, dtype=float)
    distances = pairwise_distance_matrix(coords)
    adjacent = adjacent_distances(coords)
    bends = local_bend_angles(coords)
    pseudo = pseudo_dihedrals(coords)
    dm_checks = validate_distance_matrix(distances)

    if len(coords) > 1:
        non_neighbour_mask = np.ones_like(distances, dtype=bool)
        np.fill_diagonal(non_neighbour_mask, False)
        for offset in (-1, 1):
            diag_indices = np.arange(len(coords) - 1)
            if offset == 1:
                non_neighbour_mask[diag_indices, diag_indices + 1] = False
            else:
                non_neighbour_mask[diag_indices + 1, diag_indices] = False
        non_neighbour_distances = distances[non_neighbour_mask]
        coarse_clash_fraction = float(np.mean(non_neighbour_distances < 3.0)) if len(non_neighbour_distances) else 0.0
    else:
        coarse_clash_fraction = 0.0

    return {
        "structure_id": structure_id,
        "n_residues": int(len(coords)),
        "mean_adjacent_ca": float(np.nanmean(adjacent)) if len(adjacent) else float("nan"),
        "std_adjacent_ca": float(np.nanstd(adjacent)) if len(adjacent) else float("nan"),
        "min_adjacent_ca": float(np.nanmin(adjacent)) if len(adjacent) else float("nan"),
        "max_adjacent_ca": float(np.nanmax(adjacent)) if len(adjacent) else float("nan"),
        "radius_of_gyration": radius_of_gyration(coords),
        "mean_bend_angle": float(np.nanmean(bends)) if len(bends) else float("nan"),
        "std_bend_angle": float(np.nanstd(bends)) if len(bends) else float("nan"),
        "mean_abs_pseudo_dihedral": float(np.nanmean(np.abs(pseudo))) if len(pseudo) else float("nan"),
        "coarse_clash_fraction": coarse_clash_fraction,
        "distance_matrix_is_symmetric": bool(dm_checks["is_symmetric"]),
        "distance_matrix_has_zero_diagonal": bool(dm_checks["has_zero_diagonal"]),
        "distance_matrix_symmetry_error": float(dm_checks["symmetry_error"]),
    }


def summaries_to_dataframe(summaries: list[dict[str, object]]) -> pd.DataFrame:
    """Convert validation summaries to a tidy dataframe."""
    return pd.DataFrame(summaries)
