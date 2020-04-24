from typing import List, Callable, Optional

import torch
import torch.nn as nn
import logging

from model.gcn import GraphConvolutionalNetwork, DEVICE

# Set up logging
logger = logging.getLogger(__name__)


def train(
        model: GraphConvolutionalNetwork,
        inputs: torch.Tensor,
        adjacency: torch.Tensor,
        targets: torch.Tensor,
        train_doc_indices: List[int],
        criterion: Callable = nn.CrossEntropyLoss(),
        num_epochs: int = 10,
        learning_rate: float = 1e-3,
        metrics_to_log: Optional[List[str]] = None,
        model_path: Optional[str] = None
) -> None:

    # Send model to device and initialize optimize
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_targets = targets[train_doc_indices]

    logger.info("training model...")
    for i in range(num_epochs):
        count = 0
        total_loss = 0

        # Send data to device
        inputs = inputs.double().to(DEVICE)
        adjacency = adjacency.double().to(DEVICE)
        targets = targets.double().to(DEVICE)

        # Compute prediction and loss
        predicted = model(input=inputs, adjacency=adjacency).to(DEVICE)

        train_predicted = predicted[train_doc_indices]  # am I allowed to do this in pytorch (grads)?

        # try doing a matrix and compare

        loss = criterion(train_predicted, train_targets.T).to(DEVICE)

        # Perform gradient step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track progress
        count += 1
        total_loss += loss.cpu().item()

        # Log progress
        logger.info(f"epochs completed: \t {i + 1}/{num_epochs}")
        logger.info(f"mean loss: \t {'{0:.3f}'.format(total_loss / count)}")
        logger.info("-" * 50)

    if model_path:
        logger.info("saving model...")
        model.save(path=model_path)
