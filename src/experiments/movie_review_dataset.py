from data_processing.movie_review_dataset import get_data
from model.gcn import GraphConvolutionalNetwork
from model.training import train
import logging

logger = logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":

    train_data, test_data, vocab = get_data(max_examples_per_class=1000)

    gcn = GraphConvolutionalNetwork(
        in_features=len(vocab) + 1,
        gc1_hidden_size=256,
        gc2_hidden_size=128,
        fc1_hidden_size=64,
        fc2_hidden_size=2
    )

    train(
        model=gcn,
        data=train_data,
        num_epochs=300,
        learning_rate=1e-3
    )
