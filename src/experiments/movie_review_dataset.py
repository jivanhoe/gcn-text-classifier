from data_processing.movie_review_dataset import get_data
from model.gcn import GraphConvolutionalNetwork
from model.training import train
import logging

logger = logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":

    train_data, test_data, vocab = get_data(max_examples_per_class=1000)

    gcn = GraphConvolutionalNetwork(
        in_features=len(vocab) + 1,
        gc_hidden_layer_sizes=[512, 256],
        fc_hidden_layer_sizes=[128, 2]
    )

    train(
        model=gcn,
        data=train_data,
        num_epochs=300,
        learning_rate=1e-3
    )
