import numpy as np

from data_processing.adjacency_matrix import make_adjacency_matrix_for_doc
from data_processing.model_data import get_model_data

POSITIVE_REVIEWS_PATH = "../../data/movie_reviews/positive_reviews.txt"
NEGATIVE_REVIEWS_PATH = "../../data/movie_reviews/negative_reviews.txt"
EMBEDDINGS_PATH = "../../data/glove/glove_6B_300d.txt"


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


def test_get_data() -> None:
    get_model_data(
        doc_paths=[POSITIVE_REVIEWS_PATH, NEGATIVE_REVIEWS_PATH],
        embeddings_path=None,
        stem_tokens=True
    )

