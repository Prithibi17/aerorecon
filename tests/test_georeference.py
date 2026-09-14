import numpy as np

from pipeline.georeference import geodetic_to_ecef, ecef_to_enu_matrix, similarity


def test_similarity_recovers_metric_transform():
    source = np.array([[0, 0, 0], [2, 0, 0], [0, 3, 0], [1, 2, 1]], dtype=float)
    angle = np.radians(23)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    target = (4.2 * (rotation @ source.T)).T + [12, -8, 3]
    scale, recovered_rotation, translation, residuals = similarity(source, target)
    assert np.isclose(scale, 4.2)
    assert np.allclose(recovered_rotation, rotation)
    assert np.allclose(translation, [12, -8, 3])
    assert residuals.max() < 1e-10


def test_local_enu_has_expected_axes():
    anchor = geodetic_to_ecef(28.6, 77.2, 220)
    matrix = ecef_to_enu_matrix(28.6, 77.2)
    assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.norm(anchor), 6.37e6, rtol=.01)
