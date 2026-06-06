import numpy as np

from latent_structure_generation.geometry import (
    adjacent_distances,
    fixed_length_crop_or_pad,
    pairwise_distance_matrix,
    validate_distance_matrix,
)


def test_pairwise_distance_matrix_is_symmetric():
    coords = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [3.0, 4.0, 12.0]])
    distances = pairwise_distance_matrix(coords)
    assert distances.shape == (3, 3)
    assert np.allclose(distances, distances.T)
    assert np.allclose(np.diag(distances), 0.0)
    assert np.isclose(distances[0, 1], 5.0)


def test_adjacent_distances():
    coords = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.0, 0.0]])
    distances = adjacent_distances(coords)
    assert np.allclose(distances, [3.8, 3.8])


def test_validate_distance_matrix():
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    distances = pairwise_distance_matrix(coords)
    checks = validate_distance_matrix(distances)
    assert checks["is_symmetric"]
    assert checks["has_zero_diagonal"]


def test_fixed_length_crop_or_pad():
    coords = np.ones((3, 3))
    padded, mask = fixed_length_crop_or_pad(coords, target_length=5)
    assert padded.shape == (5, 3)
    assert mask.tolist() == [True, True, True, False, False]
