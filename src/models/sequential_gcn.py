from typing import List, Callable

import torch
import torch.nn as nn
import torch.nn.functional as f

from models.gcn import GraphConvolutionalNetwork


class SequentialGraphConvolutionalNetwork(nn.Module):

    def __init__(
            self,
            in_features: int,
            gc_hidden_sizes: List[int],
            fc_hidden_sizes: List[int],
            forward_weights_size: int,
            backward_weights_size: int,
            dropout: float,
            gc_activation: Callable = f.selu,
            fc_activation: Callable = f.selu,
            add_residual_connection: bool = False,
            softmax_pooling: bool = False,
            seed: int = 0
    ):
        torch.manual_seed(seed)
        super(SequentialGraphConvolutionalNetwork, self).__init__()
        self.forward_weights_size = forward_weights_size
        self.backward_weights_size = backward_weights_size
        self.forward_weights = nn.Parameter(0.5 * torch.ones(forward_weights_size), requires_grad=True)
        self.backward_weights = nn.Parameter(0.5 * torch.ones(backward_weights_size), requires_grad=True)
        self.gcn = GraphConvolutionalNetwork(
            in_features=in_features,
            gc_hidden_sizes=gc_hidden_sizes,
            fc_hidden_sizes=fc_hidden_sizes,
            dropout=dropout,
            gc_activation=gc_activation,
            fc_activation=fc_activation,
            add_residual_connection=add_residual_connection,
            softmax_pooling=softmax_pooling,
            seed=seed
        )

    def build_adjacency_matrix(self, input: torch.Tensor) -> torch.Tensor:
        num_nodes = input.shape[0]
        adjacency = torch.eye(num_nodes)
        for i in range(num_nodes):
            for j in range(self.forward_weights_size):
                k = i + j + 1
                if k < num_nodes:
                    adjacency[i, k] = self.forward_weights[j]
            for j in range(self.backward_weights_size):
                k = i - j - 1
                if k >= 0:
                    adjacency[i, k] = self.backward_weights[j]
        sqrt_inv_degree = torch.diag(adjacency.sum(dim=1).pow(-0.5))
        adjacency = torch.matmul(sqrt_inv_degree, torch.matmul(adjacency, sqrt_inv_degree))
        return adjacency

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        adjacency = self.build_adjacency_matrix(input)
        return self.gcn(input=input, adjacency=adjacency)

    def save(self, path) -> None:
        torch.save(self.state_dict(), path)

    def load_params(self, path) -> None:
        self.load_state_dict(torch.load(path))
