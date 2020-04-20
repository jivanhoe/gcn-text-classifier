
from typing import List, Tuple, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as f

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GraphConvolutionalLayer(nn.Module):

    def __init__(self, in_features: int, out_features: int, activation: Optional[Callable] = None):
        super(GraphConvolutionalLayer, self).__init__()
        self.weight_matrix = nn.Linear(in_features, out_features, bias=False)
        self.activation = activation

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden, adjacency = inputs
        hidden = torch.matmul(adjacency, self.weight_matrix(hidden.float()).double())
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
            gc_hidden_sizes: List[int],
            gc_activation: Callable = f.relu,
            fc_activation: Callable = f.relu,
            fc_hidden_sizes: Optional[List[int]] = None,
            add_residual_connection: bool = False
    ):
        super(GraphConvolutionalNetwork, self).__init__()
        self.num_gc_layers = len(gc_hidden_sizes)
        self.num_fc_layers = len(fc_hidden_sizes) if fc_hidden_sizes else 0
        self.add_residual_connection = add_residual_connection
        self.gc_layers = nn.Sequential(
            *[
                GraphConvolutionalLayer(
                    in_features=gc_hidden_sizes[i - 1] if i > 0 else in_features,
                    out_features=gc_hidden_sizes[i],
                    activation=gc_activation
                )
                for i in range(self.num_gc_layers)
            ]
        )
        if self.num_fc_layers > 0:
            self.fc_layers = nn.Sequential(
                *[
                    FullyConnectedLayer(
                        in_features=fc_hidden_sizes[i - 1] if i > 0 else (
                            gc_hidden_sizes[-1] + in_features if self.add_residual_connection else
                            gc_hidden_sizes[-1]
                        ),
                        out_features=fc_hidden_sizes[i],
                        activation=fc_activation if (i < self.num_gc_layers - 1) else None
                    )
                    for i in range(self.num_gc_layers)
                ]
            )
        else:
            self.fc_layers = None

    def forward(self, input: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.gc_layers((input, adjacency))
        if self.fc_layers:
            hidden = hidden.mean(dim=0)
            if self.add_residual_connection:
                hidden = torch.cat((hidden, input.mean(dim=0)))
            hidden = self.fc_layers(hidden)
        return hidden

    def save(self, path) -> None:
        torch.save(self.state_dict(), path)

    def load_params(self, path) -> None:
        self.load_state_dict(torch.load(path))
