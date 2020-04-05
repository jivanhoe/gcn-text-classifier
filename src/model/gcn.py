
from typing import List, Tuple, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as f


class GraphConvolutionalLayer(nn.Module):

    def __init__(self, in_features: int, out_features: int, activation: Optional[Callable] = None):
        super(GraphConvolutionalLayer, self).__init__()
        self.weight_matrix = nn.Linear(in_features, out_features, bias=False)
        self.activation = activation

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden, adjacency = inputs
        hidden = torch.matmul(adjacency, self.weight_matrix(hidden))
        if self.activation:
            hidden = self.activation(hidden)
        return hidden, adjacency


class FullyConnectedLayer(nn.Module):

    def __init__(self, in_features: int, out_features: int, activation: Optional[Callable] = None):
        super(FullyConnectedLayer, self).__init__()
        self.weight_matrix = nn.Linear(in_features, out_features, bias=False)
        self.activation = activation

    def forward(self, input: torch.Tensor, activation: Optional[Callable] = None) -> torch.Tensor:
        output = self.weight_matrix(input)
        if self.activation:
            output = self.activation(output)
        return output


class GraphConvolutionalNetwork(nn.Module):

    def __init__(
            self,
            in_features: int,
            gc_hidden_layer_sizes: List[int],
            fc_hidden_layer_sizes: List[int],
            gc_activation: Callable = f.selu,
            fc_activation: Callable = f.relu
    ):
        super(GraphConvolutionalNetwork, self).__init__()

        self.gc_layers = nn.Sequential(
            *[
                GraphConvolutionalLayer(
                    in_features=in_features if i == 0 else gc_hidden_layer_sizes[i - 1],
                    out_features=gc_hidden_layer_sizes[i],
                    activation=gc_activation
                )
                for i in range(len(gc_hidden_layer_sizes))
            ]
        )
        self.fc_layers = nn.Sequential(
            *[
                FullyConnectedLayer(
                    in_features=gc_hidden_layer_sizes[-1] if i == 0 else fc_hidden_layer_sizes[i - 1],
                    out_features=fc_hidden_layer_sizes[i],
                    activation=fc_activation
                )
                for i in range(len(gc_hidden_layer_sizes))
            ]
        )

    def forward(self, input: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.gc_layers((input, adjacency))
        if self.fc_layers:
            hidden = hidden.mean(dim=0)  # f.softmax(hidden.sum(dim=0))
            hidden = self.fc_layers(hidden)
        return hidden



