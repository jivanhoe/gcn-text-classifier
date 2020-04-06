from typing import List, Tuple, Callable, Optional

import torch
import torch.nn as nn
import logging

from model.gcn import GraphConvolutionalNetwork, DEVICE
from utils.metrics import calculate_metrics, log_metrics

# Set up logging
logger = logging.getLogger(__name__)


def train(
        model: GraphConvolutionalNetwork,
        train_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        validation_data: Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
        criterion: Callable = nn.CrossEntropyLoss(),
        num_epochs: int = 10,
        learning_rate: float = 1e-3,
        metrics_to_log: Optional[List[str]] = None,
        model_path: Optional[str] = None
) -> None:

    # Send model to device and initialize optimize
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    logger.info("training model...")
    for i in range(num_epochs):

        count = 0
        total_loss = 0
        for input, adjacency, target in train_data:

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
            total_loss += loss.cpu().item()

        # Log progress
        logger.info(f"epochs completed: \t {i + 1}/{num_epochs}")
        logger.info(f"mean loss: \t {'{0:.3f}'.format(total_loss / count)}")
        if metrics_to_log:
            logger.info("calculating training metrics...")
            log_metrics(
                metrics=calculate_metrics(
                    model=model,
                    data=train_data
                ),
                metrics_to_log=metrics_to_log
            )
            if validation_data:
                logger.info("calculating validation metrics...")
                log_metrics(
                    metrics=calculate_metrics(
                        model=model,
                        data=validation_data
                    ),
                    metrics_to_log=metrics_to_log
                )
        logger.info("-" * 50)

    if model_path:
        logger.info("saving model...")
        model.save(path=model_path)

