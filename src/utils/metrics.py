import logging
from typing import Tuple, Dict, List

import numpy as np
import torch
import torch.nn.functional as f
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

from model.gcn import GraphConvolutionalNetwork, DEVICE

# Set up logging
logger = logging.getLogger(__name__)


def predict_prob(
        model: GraphConvolutionalNetwork,
        input: torch.Tensor,
        adjacency: torch.Tensor,
        positive_class_id: int = 1
) -> float:
    predicted_probs = f.softmax(model(input=input, adjacency=adjacency), dim=-1).detach()
    return predicted_probs.data.cpu().numpy()[positive_class_id]


def calculate_metrics(
        model: GraphConvolutionalNetwork,
        data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        positive_class_id: int = 1
) -> Dict[str, float]:
    predicted_probs = []
    targets = []
    for input, adjacency, target in data:

        # Send data to device
        input = input.to(DEVICE)
        adjacency = adjacency.to(DEVICE)

        # Get predicted probability of positive class from model and target classes
        predicted_probs.append(
            predict_prob(
                model=model,
                input=input,
                adjacency=adjacency,
                positive_class_id=positive_class_id
            )
        )
        targets.append(target.item() == positive_class_id)

    predicted_classes = np.array(predicted_probs) > 0.5
    return {
        "accuracy": accuracy_score(targets, predicted_classes),
        "auc": roc_auc_score(targets, predicted_probs),
        "precision": precision_score(targets, predicted_classes),
        "recall": recall_score(targets, predicted_classes),
        "f1": f1_score(targets, predicted_classes)
    }


def log_metrics(metrics: Dict[str, any], metrics_to_log: List[str]) -> None:
    for metric_name in metrics_to_log:
        logger.info(f"{metric_name}: \t {'{0:.3f}'.format(metrics[metric_name])}")
