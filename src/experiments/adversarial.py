import logging

from data_processing.model_data import get_model_data
from experiments.movie_review_dataset_constants import *
from models.gcn import GraphConvolutionalNetwork
from models.sequential_gcn import SequentialGraphConvolutionalNetwork
from utils.metrics import log_metrics, calculate_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

POSITIVE_ADVERSARIAL_PHRASE = "cliche"  # Phrase to add to positive examples to trick the classifier
# (e.g. "the movie sucks shit, you fucking stupid ass bitch")
NEGATIVE_ADVERSARIAL_PHRASE = "decent"  # Phrase to add to negative examples to trick the classifier
# (e.g. "one of the greatest movies, possibly ever, truly terrific")


if __name__ == "__main__":

    # Get nominal test data
    _, _, _, test_data, _ = get_model_data(
        doc_paths=[POSITIVE_REVIEWS_PATH, NEGATIVE_REVIEWS_PATH],
        embeddings_path=EMBEDDINGS_PATH,
        max_examples_per_class=MAX_EXAMPLES_PER_CLASS,
    )

    # Make adversarial data test data
    _, _, _, adversarial_test_data, in_features = get_model_data(
        doc_paths=[POSITIVE_REVIEWS_PATH, NEGATIVE_REVIEWS_PATH],
        embeddings_path=EMBEDDINGS_PATH,
        max_examples_per_class=MAX_EXAMPLES_PER_CLASS,
        adversarial_phrases=[POSITIVE_ADVERSARIAL_PHRASE, NEGATIVE_ADVERSARIAL_PHRASE]
    )

    # Load model
    if USE_SEQUENTIAL_GCN:
        model = SequentialGraphConvolutionalNetwork(
            in_features=in_features,
            gc_hidden_sizes=GC_HIDDEN_SIZES,
            fc_hidden_sizes=FC_HIDDEN_SIZES,
            forward_weights_size=FORWARD_WEIGHTS_SIZE,
            backward_weights_size=BACKWARD_WEIGHTS_SIZE,
            add_residual_connection=ADD_RESIDUAL_CONNECTION,
            softmax_pooling=SOFTMAX_POOLING,
            seed=SEED
        )
    else:
        model = GraphConvolutionalNetwork(
            in_features=in_features,
            gc_hidden_sizes=GC_HIDDEN_SIZES,
            fc_hidden_sizes=FC_HIDDEN_SIZES,
            add_residual_connection=ADD_RESIDUAL_CONNECTION,
            softmax_pooling=SOFTMAX_POOLING,
            seed=SEED
        )
    model.load_params(MODEL_PATH)

    logger.info("calculating baseline metrics...")
    log_metrics(
        metrics=calculate_metrics(
            model=model,
            data=test_data
        ),
        metrics_to_log=METRICS_TO_LOG
    )

    logger.info("calculating adversarial metrics...")
    log_metrics(
        metrics=calculate_metrics(
            model=model,
            data=adversarial_test_data
        ),
        metrics_to_log=METRICS_TO_LOG
    )
