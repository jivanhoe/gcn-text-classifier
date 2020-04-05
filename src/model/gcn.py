
from typing import List, Tuple, Optional

import torch
import torch.nn as nn


class GraphConvolutionalLayer(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super(GraphConvolutionalLayer, self).__init__()
        self.weight_matrix = nn.Linear(in_features, out_features, bias=False)

    def forward(self, input: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return torch.matmul(adjacency, self.weight_matrix(input))


class SumPoolLayer(nn.Module):

    def __init__(self, in_features: int, normalize: bool = True):
        super(SumPoolLayer, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.softmax(input.mean(dim=0))


class GraphConvolutionalNetwork(nn.Module):

    def __init__(
            self,
            in_features: int,
            gc1_hidden_size: int,
            gc2_hidden_size: int,
            fc1_hidden_size: int,
            fc2_hidden_size: int
    ):
        super(GraphConvolutionalNetwork, self).__init__()
        self.gc1 = GraphConvolutionalLayer(
            in_features=in_features,
            out_features=gc1_hidden_size
        )
        self.gc2 = GraphConvolutionalLayer(
            in_features=gc1_hidden_size,
            out_features=gc2_hidden_size
        )
        self.sumpool = SumPoolLayer(in_features=gc2_hidden_size)
        self.fc1 = nn.Linear(
            in_features=gc2_hidden_size,
            out_features=fc1_hidden_size
        )
        self.fc2 = nn.Linear(
            in_features=fc1_hidden_size,
            out_features=fc2_hidden_size
        )
        self.relu = nn.ReLU()

    def forward(self, input: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = self.gc1(input, adjacency)
        hidden = self.relu(hidden)
        hidden = self.gc2(hidden, adjacency)
        hidden = self.relu(hidden)
        hidden = self.sumpool(hidden)
        hidden = self.fc1(hidden)
        hidden = self.relu(hidden)
        hidden = self.fc2(hidden)
        return hidden



