from data_processing.movie_review_dataset import get_data
from model.gcn import GraphConvolutionalNetwork
from model.training import train
import logging

logger = logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":

    train_data, test_data, vocab = get_data(max_examples_per_class=500)

    gcn = GraphConvolutionalNetwork(
        in_features=len(vocab) + 1,
        gc_hidden_sizes=[512, 256],
        fc_hidden_sizes=[128, 2],
        softmax_outputs=True
    )

    train(
        model=gcn,
        data=train_data,
        num_epochs=20,
        learning_rate=1e-2
    )
