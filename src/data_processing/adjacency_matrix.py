import numpy as np


def make_adjacency_matrix_for_doc(doc_length: int, normalize: bool = True) -> np.ndarray:
    adjacency = np.eye(doc_length)
    for i in range(doc_length - 1):
        adjacency[i, i+1] = 1
    if normalize:
        sqrt_inv_degree = np.diag(1 / adjacency.sum(1)) ** 0.5
        adjacency = np.matmul(sqrt_inv_degree, np.matmul(adjacency, sqrt_inv_degree))
    return adjacency
