from typing import List, Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor
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
        val_doc_indices: List[int],
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
    val_targets = targets[val_doc_indices]

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

        train_predicted = predicted[train_doc_indices]
        train_pred_targets: Tensor = train_predicted.max(dim=1)[1]
        val_pred_targets: Tensor = predicted[val_doc_indices].max(dim=1)[1]

        loss = criterion(train_predicted, train_targets.T).to(DEVICE)

        val_acc = sum(val_pred_targets == val_targets).item() / len(val_pred_targets)
        acc = sum(train_pred_targets == train_targets).item() / len(train_pred_targets)

        # Perform gradient step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track progress
        count += 1
        total_loss += loss.cpu().item()

        # Log progress - Note that I think results are delayed
        logger.info(f"epochs completed: \t {i + 1}/{num_epochs}")
        logger.info(f"mean loss: \t {'{0:.3f}'.format(total_loss / count)}")
        logger.info(f"train accuracy: \t {'{0:.3f}'.format(acc)}")
        logger.info(f"validation accuracy: \t {'{0:.3f}'.format(val_acc)}")
        logger.info("-" * 50)

    logger.info("")
    if model_path:
        logger.info("saving model...")
        model.save(path=model_path)
