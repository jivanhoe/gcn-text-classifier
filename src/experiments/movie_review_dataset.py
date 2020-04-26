import logging
import numpy as np

from data_processing.original_model_data import get_original_model_data
from model.gcn import GraphConvolutionalNetwork
from model.original_model_training import train

logger = logging.basicConfig(level=logging.INFO)

# Paths
POSITIVE_REVIEWS_PATH = "../../data/movie_reviews/positive_reviews.txt"
NEGATIVE_REVIEWS_PATH = "../../data/movie_reviews/negative_reviews.txt"
EMBEDDINGS_PATH = "../../data/glove/glove_6B_300d.txt"
MODEL_PATH = "../../data/movie_reviews_model.pt"

# Model parameters
GC_HIDDEN_SIZES = [256, 128, 2]

# Training parameters
NUM_EPOCHS = 5
LEARNING_RATE = 2e-4
MAX_EXAMPLES_PER_CLASS = None
METRICS_TO_LOG = ["accuracy", "auc"]

if __name__ == "__main__":

    inputs, adjacency, targets, doc_indices, in_features = get_original_model_data(
        doc_paths=[POSITIVE_REVIEWS_PATH, NEGATIVE_REVIEWS_PATH],
        max_examples_per_class=MAX_EXAMPLES_PER_CLASS
    )

    gcn_model = GraphConvolutionalNetwork(
        in_features=in_features,
        gc_hidden_sizes=GC_HIDDEN_SIZES,
        add_residual_connection=False
    )

    # can be optimized
    train_doc_indices = np.random.choice(doc_indices, int(0.7 * len(doc_indices)), replace=False)
    val_doc_indices = [x for x in doc_indices if x not in train_doc_indices]

    train(
        model=gcn_model,
        inputs=inputs,
        adjacency=adjacency,
        targets=targets,
        train_doc_indices=train_doc_indices,
        val_doc_indices=val_doc_indices,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        metrics_to_log=METRICS_TO_LOG,
        model_path=MODEL_PATH
    )
