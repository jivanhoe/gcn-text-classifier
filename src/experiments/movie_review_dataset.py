from data_processing.movie_review_dataset import get_data
from model.gcn import GraphConvolutionalNetwork
from model.training import train
import logging

logger = logging.basicConfig(level=logging.INFO)

EMBEDDINGS_PATH = '../../data/glove/glove_6B_300d.txt'

if __name__ == "__main__":

    train_data, test_data, in_features = get_data(embeddings_path=EMBEDDINGS_PATH, max_examples_per_class=2000)

    gcn = GraphConvolutionalNetwork(
        in_features=in_features,
        gc_hidden_sizes=[256, 128],
        fc_hidden_sizes=[64, 2],
        add_residual_connection=False
    )

    train(
        model=gcn,
        data=train_data,
        num_epochs=300,
        learning_rate=2e-4
    )
