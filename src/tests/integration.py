import logging

from data_processing.model_data import get_model_data, partition_data
from experiments.movie_review_dataset_constants import *
from models.gcn import GraphConvolutionalNetwork
from models.sequential_gcn import SequentialGraphConvolutionalNetwork
from models.training import train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    if USE_CUSTOM_ADJACENCY_MATRIX:
        _, data, in_features = get_model_data(
            doc_paths=[POSITIVE_REVIEWS_PATH, NEGATIVE_REVIEWS_PATH],
            embeddings_path=EMBEDDINGS_PATH,
            max_examples_per_class=MAX_EXAMPLES_PER_CLASS,
            forward_weights=FORWARD_WEIGHTS,
            backward_weights=BACKWARD_WEIGHTS,
        )
    else:
        _, data, in_features = get_model_data(
            doc_paths=[POSITIVE_REVIEWS_PATH, NEGATIVE_REVIEWS_PATH],
            embeddings_path=EMBEDDINGS_PATH,
            max_examples_per_class=MAX_EXAMPLES_PER_CLASS
        )

    train_data, val_data, test_data = partition_data(data=data)

    if USE_SEQUENTIAL_GCN:
        gcn_model = SequentialGraphConvolutionalNetwork(
            in_features=in_features,
            gc_hidden_sizes=GC_HIDDEN_SIZES,
            fc_hidden_sizes=FC_HIDDEN_SIZES,
            forward_weights_size=FORWARD_WEIGHTS_SIZE,
            backward_weights_size=BACKWARD_WEIGHTS_SIZE,
            dropout=DROPOUT,
            add_residual_connection=ADD_RESIDUAL_CONNECTION,
            softmax_pooling=SOFTMAX_POOLING,
            seed=SEED
        )
    else:
        gcn_model = GraphConvolutionalNetwork(
            in_features=in_features,
            gc_hidden_sizes=GC_HIDDEN_SIZES,
            fc_hidden_sizes=FC_HIDDEN_SIZES,
            dropout=DROPOUT,
            add_residual_connection=ADD_RESIDUAL_CONNECTION,
            softmax_pooling=SOFTMAX_POOLING,
            seed=SEED
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
    if USE_SEQUENTIAL_GCN:
        logger.info(f"forward weights: \t {gcn_model.forward_weights.data}")
        logger.info(f"backward weights: \t {gcn_model.backward_weights.data}")


