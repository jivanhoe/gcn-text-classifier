
from typing import List, Tuple, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as f

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GraphConvolutionalLayer(nn.Module):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            dropout: float,
            activation: Optional[Callable] = None
    ):
        super(GraphConvolutionalLayer, self).__init__()
        self.weight_matrix = nn.Linear(in_features, out_features, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden, adjacency = inputs
        hidden = self.weight_matrix(self.dropout(hidden).float()).double()
        hidden = torch.matmul(adjacency, hidden)
        if self.activation:
            hidden = self.activation(hidden)
        return hidden, adjacency


class FullyConnectedLayer(nn.Module):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            dropout: float,
            activation: Optional[Callable] = None
    ):
        super(FullyConnectedLayer, self).__init__()
        self.weight_matrix = nn.Linear(in_features, out_features, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation

    def forward(self, input: torch.Tensor, activation: Optional[Callable] = None) -> torch.Tensor:
        output = self.weight_matrix(self.dropout(input))
        if self.activation:
            output = self.activation(output)
        return output


class GraphConvolutionalNetwork(nn.Module):

    def __init__(
            self,
            in_features: int,
            gc_hidden_sizes: List[int],
            fc_hidden_sizes: Optional[List[int]] = None,
            gc_dropout: float = 0,
            fc_dropout: float = 0,
            gc_activation: Callable = f.selu,
            fc_activation: Callable = f.selu,
            add_residual_connection: bool = False,
            softmax_pooling: bool = False,
            seed: int = 0
    ):
        torch.manual_seed(seed)
        super(GraphConvolutionalNetwork, self).__init__()
        self.num_gc_layers = len(gc_hidden_sizes)
        self.num_fc_layers = len(fc_hidden_sizes) if fc_hidden_sizes else 0
        self.add_residual_connection = add_residual_connection
        self.softmax_pooling = softmax_pooling
        self.gc_layers = nn.Sequential(
            *[
                GraphConvolutionalLayer(
                    in_features=gc_hidden_sizes[i - 1] if i > 0 else in_features,
                    out_features=gc_hidden_sizes[i],
                    activation=gc_activation,
                    dropout=gc_dropout
                )
                for i in range(self.num_gc_layers)
            ]
        )
        if fc_hidden_sizes:
            self.fc_layers = nn.Sequential(
                *[
                    FullyConnectedLayer(
                        in_features=fc_hidden_sizes[i - 1] if i > 0 else (
                            gc_hidden_sizes[-1] + in_features if self.add_residual_connection else gc_hidden_sizes[-1]
                        ),
                        out_features=fc_hidden_sizes[i],
                        activation=fc_activation if (i < self.num_fc_layers - 1) else None,
                        dropout=fc_dropout
                    )
                    for i in range(self.num_fc_layers)
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
            if self.softmax_pooling:
                hidden = f.softmax(hidden, dim=-1)
            hidden = self.fc_layers(hidden)
        return hidden

    def save(self, path) -> None:
        torch.save(self.state_dict(), path)

    def load_params(self, path) -> None:
        self.load_state_dict(torch.load(path))
