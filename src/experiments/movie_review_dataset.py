import logging

from data_processing.model_data import get_model_data
from models.gcn import GraphConvolutionalNetwork
from models.sequential_gcn import SequentialGraphConvolutionalNetwork
from models.training import train

logger = logging.basicConfig(level=logging.INFO)

# Paths
POSITIVE_REVIEWS_PATH = "../../data/movie_reviews/positive_reviews.txt"
NEGATIVE_REVIEWS_PATH = "../../data/movie_reviews/negative_reviews.txt"
EMBEDDINGS_PATH = "../../data/glove/glove_6B_300d.txt"
MODEL_PATH = "../../data/movie_reviews_model.pt"

# Model parameters
GC_HIDDEN_SIZES = [128, 128]
FC_HIDDEN_SIZES = [64, 2]  # Final fully-connected layer size must equal number of classes
FORWARD_WEIGHTS_SIZE = 1
BACKWARD_WEIGHTS_SIZE = 1
USE_SEQUENTIAL_GCN = False

# Training parameters
NUM_EPOCHS = 10
LEARNING_RATE = 3e-4
MAX_EXAMPLES_PER_CLASS = None
METRICS_TO_LOG = ["accuracy", "auc"]

if __name__ == "__main__":

    train_data, test_data, in_features = get_model_data(
        doc_paths=[POSITIVE_REVIEWS_PATH, NEGATIVE_REVIEWS_PATH],
        embeddings_path=EMBEDDINGS_PATH,
        max_examples_per_class=MAX_EXAMPLES_PER_CLASS
    )

    if USE_SEQUENTIAL_GCN:
        gcn_model = SequentialGraphConvolutionalNetwork(
            in_features=in_features,
            gc_hidden_sizes=GC_HIDDEN_SIZES,
            fc_hidden_sizes=FC_HIDDEN_SIZES,
            forward_weights_size=FORWARD_WEIGHTS_SIZE,
            backward_weights_size=BACKWARD_WEIGHTS_SIZE,
            add_residual_connection=False
        )
    else:
        gcn_model = GraphConvolutionalNetwork(
            in_features=in_features,
            gc_hidden_sizes=GC_HIDDEN_SIZES,
            fc_hidden_sizes=FC_HIDDEN_SIZES,
            add_residual_connection=False
        )

    train(
        model=gcn_model,
        train_data=train_data,
        validation_data=test_data,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        metrics_to_log=METRICS_TO_LOG,
        model_path=MODEL_PATH
    )
