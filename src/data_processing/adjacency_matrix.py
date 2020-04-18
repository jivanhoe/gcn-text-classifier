import numpy as np
from typing import List


def make_adjacency_matrix_for_doc(doc_length: int, normalize: bool = True) -> np.ndarray:
    adjacency = np.eye(doc_length)
    for i in range(doc_length - 1):
        adjacency[i, i+1] = 1
    if normalize:
        sqrt_inv_degree = np.diag(1 / adjacency.sum(1)) ** 0.5
        adjacency = np.matmul(sqrt_inv_degree, np.matmul(adjacency, sqrt_inv_degree))
    return adjacency


def make_custom_adjacency_matrix_for_doc(
        doc_length: int,
        forward_weights: List[float],
        backward_weights: List[float],
        normalize: bool = True
) -> np.ndarray:
    adjacency = np.eye(doc_length)
    for i in range(doc_length - 1):
        for j, weight in enumerate(forward_weights):
            k = i + j + 1
            if k < doc_length:
                adjacency[i, k] = weight
        for j, weight in enumerate(backward_weights):
            k = i - j - 1
            if k >= 0:
                adjacency[i, k] = weight
    if normalize:
        sqrt_inv_degree = np.diag(1 / adjacency.sum(1)) ** 0.5
        adjacency = np.matmul(sqrt_inv_degree, np.matmul(adjacency, sqrt_inv_degree))
    return adjacency
