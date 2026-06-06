"""Geometry utilities for C-alpha and backbone-level validation."""

from __future__ import annotations

import numpy as np


def pairwise_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Return the pairwise Euclidean distance matrix for coordinates.

    Parameters
    ----------
    coords:
        Array with shape ``(n_residues, 3)``.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n_residues, 3).")
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def adjacent_distances(coords: np.ndarray) -> np.ndarray:
    """Return distances between consecutive C-alpha atoms."""
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 2:
        return np.array([], dtype=float)
    return np.linalg.norm(coords[1:] - coords[:-1], axis=-1)


def centre_coordinates(coords: np.ndarray) -> np.ndarray:
    """Centre coordinates at the origin."""
    coords = np.asarray(coords, dtype=float)
    return coords - coords.mean(axis=0, keepdims=True)


def radius_of_gyration(coords: np.ndarray) -> float:
    """Return radius of gyration for a coordinate trace."""
    coords = np.asarray(coords, dtype=float)
    if len(coords) == 0:
        return float("nan")
    centred = centre_coordinates(coords)
    return float(np.sqrt(np.mean(np.sum(centred**2, axis=1))))


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Return the angle between two vectors in degrees."""
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return float("nan")
    cos_theta = np.dot(v1, v2) / denom
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def local_bend_angles(coords: np.ndarray) -> np.ndarray:
    """Return local three-point C-alpha bend angles in degrees."""
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 3:
        return np.array([], dtype=float)
    angles = []
    for i in range(1, len(coords) - 1):
        v1 = coords[i - 1] - coords[i]
        v2 = coords[i + 1] - coords[i]
        angles.append(angle_between(v1, v2))
    return np.asarray(angles, dtype=float)


def dihedral_angle(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Return the dihedral angle for four points in degrees."""
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    norm_b1 = np.linalg.norm(b1)
    if norm_b1 == 0:
        return float("nan")
    b1 = b1 / norm_b1

    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def pseudo_dihedrals(coords: np.ndarray) -> np.ndarray:
    """Return C-alpha pseudo-dihedral angles using four consecutive C-alpha atoms."""
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 4:
        return np.array([], dtype=float)
    values = [dihedral_angle(coords[i], coords[i + 1], coords[i + 2], coords[i + 3]) for i in range(len(coords) - 3)]
    return np.asarray(values, dtype=float)


def validate_distance_matrix(distances: np.ndarray, atol: float = 1e-5) -> dict[str, float | bool]:
    """Return simple distance-matrix validation checks."""
    distances = np.asarray(distances, dtype=float)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distances must be a square matrix.")

    symmetry_error = float(np.max(np.abs(distances - distances.T)))
    diagonal_error = float(np.max(np.abs(np.diag(distances))))
    negative_fraction = float(np.mean(distances < -atol))

    return {
        "is_square": True,
        "is_symmetric": symmetry_error <= atol,
        "has_zero_diagonal": diagonal_error <= atol,
        "symmetry_error": symmetry_error,
        "diagonal_error": diagonal_error,
        "negative_fraction": negative_fraction,
    }


def fixed_length_crop_or_pad(coords: np.ndarray, target_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Crop or zero-pad coordinates to a fixed length.

    Returns
    -------
    padded_coords:
        Array with shape ``(target_length, 3)``.
    mask:
        Boolean array with shape ``(target_length,)`` where true marks real residues.
    """
    coords = np.asarray(coords, dtype=float)
    if target_length <= 0:
        raise ValueError("target_length must be positive.")
    output = np.zeros((target_length, 3), dtype=float)
    mask = np.zeros(target_length, dtype=bool)
    n = min(len(coords), target_length)
    output[:n] = coords[:n]
    mask[:n] = True
    return output, mask
