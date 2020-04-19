import logging

from data_processing.model_data import get_model_data
from data_processing.original_model_data import get_original_model_data
from model.gcn import GraphConvolutionalNetwork
from model.training import train

logger = logging.basicConfig(level=logging.INFO)

# Paths
POSITIVE_REVIEWS_PATH = "../../data/movie_reviews/positive_reviews.txt"
NEGATIVE_REVIEWS_PATH = "../../data/movie_reviews/negative_reviews.txt"
EMBEDDINGS_PATH = "../../data/glove/glove_6B_300d.txt"
MODEL_PATH = "../../data/movie_reviews_model.pt"

# Model parameters
GC_HIDDEN_SIZES = [256, 128]
FC_HIDDEN_SIZES = [64, 2]  # Final fully-connected layer size must equal number of classes

# Training parameters
NUM_EPOCHS = 5
LEARNING_RATE = 2e-4
MAX_EXAMPLES_PER_CLASS = None
METRICS_TO_LOG = ["accuracy", "auc"]

if __name__ == "__main__":

    adjacency = get_original_model_data(
        doc_paths=[POSITIVE_REVIEWS_PATH, NEGATIVE_REVIEWS_PATH],
        max_examples_per_class=MAX_EXAMPLES_PER_CLASS
    )
    print(adjacency)
    # gcn_model = GraphConvolutionalNetwork(
    #     in_features=in_features,
    #     gc_hidden_sizes=GC_HIDDEN_SIZES,
    #     fc_hidden_sizes=FC_HIDDEN_SIZES,
    #     add_residual_connection=False
    # )
    # train(
    #     model=gcn_model,
    #     train_data=train_data,
    #     validation_data=test_data,
    #     num_epochs=NUM_EPOCHS,
    #     learning_rate=LEARNING_RATE,
    #     metrics_to_log=METRICS_TO_LOG,
    #     model_path=MODEL_PATH
    # )
