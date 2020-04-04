
from typing import List, Tuple, Optional

import torch
import torch.nn as nn


class GraphConvolutionalLayer(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super(GraphConvolutionalLayer, self).__init__()
        self.weight_matrix = nn.Linear(in_features, out_features)

    def forward(self, input: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return torch.matmul(adjacency, self.weight_matrix(input))


class SumPoolLayer(nn.Module):

    def __init__(self, in_features: int, normalize: bool = True):
        super(SumPoolLayer, self).__init__()
        self.batch_norm = nn.BatchNorm1d(in_features)
        self.normalize = normalize

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            input = self.batch_norm(input)
        return input.sum(dim=0)


class GraphConvolutionalNetwork(nn.Module):

    def __init__(
            self,
            in_features: int,
            gc_hidden_sizes: List[int],
            fc_hidden_sizes: List[int],
            softmax_outputs: bool = False,
    ):
        super(GraphConvolutionalNetwork, self).__init__()
        self.num_gc_layers = len(gc_hidden_sizes)
        self.num_fc_layers = len(fc_hidden_sizes)
        self.gc_layers = [
            GraphConvolutionalLayer(
                in_features=gc_hidden_sizes[i - 1] if i > 0 else in_features,
                out_features=gc_hidden_sizes[i]
            ) for i in range(self.num_gc_layers)
        ]
        self.fc_layers = [
            nn.Linear(
                in_features=fc_hidden_sizes[i - 1] if i > 0 else gc_hidden_sizes[-1],
                out_features=fc_hidden_sizes[i]
            ) for i in range(len(fc_hidden_sizes))
        ]
        self.sum_pool = SumPoolLayer(in_features=gc_hidden_sizes[-1])
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.softmax_outputs = softmax_outputs

    def forward(self, input: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = input
        for i in range(self.num_gc_layers):
            hidden = self.gc_layers[i](hidden, adjacency)
            hidden = self.relu(hidden)
        if self.num_fc_layers > 0:
            hidden = self.sum_pool(hidden)
        for i in range(self.num_fc_layers):
            hidden = self.fc_layers[i](hidden)
            hidden = self.relu(hidden)
        if self.softmax_outputs:
            return self.softmax(hidden)
        else:
            return hidden


