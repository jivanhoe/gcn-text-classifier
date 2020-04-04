import numpy as np
import torch

from typing import List, Tuple


def generate_random_input_data(
        num_vertices: int,
        num_features: int,
        seed: int = 0
) -> torch.tensor:
    np.random.seed(seed)
    input = np.random.rand(num_vertices, num_features)
    return torch.from_numpy(input).float()


def generate_random_adjacency_matrix(
        num_vertices: int,
        edge_probability: float = 0.25,
        add_self_loops: bool = True,
        normalize: bool = True,
        seed: int = 0
) -> torch.tensor:
    np.random.seed(seed)
    adjacency = np.random.binomial(1, edge_probability / 2, (num_vertices, num_vertices))
    adjacency = adjacency + adjacency.T
    if add_self_loops:
        adjacency = adjacency + np.eye(num_vertices)
    adjacency = np.minimum(adjacency, 1)
    if normalize:
        sqrt_inv_degree = np.diag(1 / adjacency.sum(1)) ** 0.5
        adjacency = np.matmul(sqrt_inv_degree, np.matmul(adjacency, sqrt_inv_degree))
    return torch.from_numpy(adjacency).float()


def generate_random_target_label(
        num_classes: int,
        seed: int = 0
) -> torch.Tensor:
    np.random.seed(seed)
    label = np.random.choice(range(num_classes))
    return torch.tensor(label)


def generate_random_data(
        num_vertices: int,
        num_features: int,
        num_classes: int,
        num_examples: int,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    return [
        (
            generate_random_input_data(num_vertices=num_vertices, num_features=num_features, seed=seed),
            generate_random_adjacency_matrix(num_vertices=num_vertices, seed=seed),
            generate_random_target_label(num_classes=num_classes, seed=seed)
        )
        for seed in range(num_examples)
    ]