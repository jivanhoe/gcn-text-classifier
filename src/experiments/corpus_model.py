import logging
import numpy as np

from data_processing.corpus_model_data import get_corpus_model_data
from models.gcn import GraphConvolutionalNetwork
from models.corpus_model_training import train

import torch.nn.functional as f

logger = logging.basicConfig(level=logging.INFO)

# Paths
POSITIVE_REVIEWS_PATH = "../../data/movie_reviews/positive_reviews.txt"
NEGATIVE_REVIEWS_PATH = "../../data/movie_reviews/negative_reviews.txt"

# Model parameters
GC_HIDDEN_SIZES = [200, 2]

# Training parameters
NUM_EPOCHS = 20
LEARNING_RATE = 0.02
MAX_EXAMPLES_PER_CLASS = None

if __name__ == "__main__":

    inputs, adjacency, targets, doc_indices, in_features = get_corpus_model_data(
        doc_paths=[POSITIVE_REVIEWS_PATH, NEGATIVE_REVIEWS_PATH],
        stem_tokens=True,
        clean_tokens=True,
        max_examples_per_class=MAX_EXAMPLES_PER_CLASS
    )

    gcn_model = GraphConvolutionalNetwork(
        in_features=in_features,
        gc_hidden_sizes=GC_HIDDEN_SIZES,
        gc_dropout=0.5,
        gc_activation=f.relu,
        seed=0
    )

    # can be optimized
    train_doc_indices = np.random.choice(doc_indices, int(0.7 * len(doc_indices)), replace=False)
    test_doc_indices = [x for x in doc_indices if x not in train_doc_indices]
    val_doc_indices = np.random.choice(test_doc_indices, int(0.5 * len(test_doc_indices)), replace=False)
    test_doc_indices = [x for x in test_doc_indices if x not in val_doc_indices]

    train(
        model=gcn_model,
        inputs=inputs,
        adjacency=adjacency,
        targets=targets,
        train_doc_indices=train_doc_indices,
        val_doc_indices=val_doc_indices,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE
    )
