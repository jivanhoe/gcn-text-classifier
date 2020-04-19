import numpy as np

from data_processing.adjacency_matrix import make_adjacency_matrix_for_doc


def test_make_adjacency_matrix_for_doc() -> None:
    expected_adjacency = np.array(
        [
            [1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1]
        ]
    )
    adjacency = make_adjacency_matrix_for_doc(5, normalize=False)
    assert np.isclose(expected_adjacency, adjacency).all()



