from typing import List, Tuple, Callable

import torch
import torch.nn as nn
import logging

from model.gcn import GraphConvolutionalNetwork

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Set up logging
logger = logging.getLogger(__name__)


def train(
        model: GraphConvolutionalNetwork,
        data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        criterion: Callable = nn.CrossEntropyLoss(),
        num_epochs: int = 10,
        learning_rate: float = 1e-3
) -> None:

    # Send model to device and initialize optimize
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for i in range(num_epochs):

        count = 0
        total_loss = 0
        for input, adjacency, target in data:

            # Send data to device
            input = input.to(DEVICE)
            adjacency = adjacency.to(DEVICE)
            target = target.to(DEVICE)

            # Compute prediction and loss
            predicted = model(input=input, adjacency=adjacency).to(DEVICE)
            loss = criterion(predicted.unsqueeze(0), target.unsqueeze(0)).to(DEVICE)

            # Perform gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track progress
            count += 1
            if DEVICE == "cuda":
                loss = loss.cpu()
            total_loss += loss.item()

        # Log progress
        logger.info(f"Epochs completed: \t {i + 1}/{num_epochs}")
        logger.info(f"Mean loss: \t {'{0:.4f}'.format(total_loss / count)}")
        logger.info("-" * 50)
