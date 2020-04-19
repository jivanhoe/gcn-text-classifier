from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from data_processing.model_data import get_model_data
from models.gcn import GraphConvolutionalNetwork
from models.sequential_gcn import SequentialGraphConvolutionalNetwork
from experiments.movie_review_dataset_constants import *
from nltk.stem import PorterStemmer


def get_clustering_data(
        doc_paths: List[str],
        embeddings_path: str,
        model: Union[GraphConvolutionalNetwork, SequentialGraphConvolutionalNetwork],
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # Get data for model
    train_docs, test_docs, train_data, test_data, _ = get_model_data(
        doc_paths=doc_paths,
        embeddings_path=embeddings_path,
    )
    docs = train_docs + test_docs
    data = train_data + test_data

    # Extract model input and model embeddings for words
    input_embeddings = []
    model_embeddings = []
    words = []
    targets = []
    stemmer = PorterStemmer()
    for i, (input, adjacency, target) in enumerate(data):
        if type(model) == GraphConvolutionalNetwork:
            output, _ = model.gc_layers((input, adjacency))
        elif type(model) == SequentialGraphConvolutionalNetwork:
            adjacency = model.build_adjacency_matrix(input=input)
            output, _ = model.gcn.gc_layers((input, adjacency))
        else:
            raise NotImplementedError
        input_embeddings.append(input.data.numpy())
        model_embeddings.append(output.data.numpy())
        words += [stemmer.stem(word) for word in docs[i]]
        targets += [target.item() for _  in range(len(docs[i]))]

    # Tidy up
    input_emmbeddings = pd.DataFrame(np.concatenate(input_embeddings))
    input_emmbeddings["word"] = words
    input_emmbeddings = input_emmbeddings.groupby("word").first()

    model_embeddings = pd.DataFrame(np.concatenate(model_embeddings))
    model_embeddings["word"] = words
    model_embeddings["target"] = targets
    model_embeddings = model_embeddings.set_index("word")

    return input_emmbeddings, model_embeddings


if __name__ == "__main__":

    # Load model
    if USE_SEQUENTIAL_GCN:
        pretrained_model = SequentialGraphConvolutionalNetwork(
            in_features=IN_FEATURES,
            gc_hidden_sizes=GC_HIDDEN_SIZES,
            fc_hidden_sizes=FC_HIDDEN_SIZES,
            forward_weights_size=FORWARD_WEIGHTS_SIZE,
            backward_weights_size=BACKWARD_WEIGHTS_SIZE,
            add_residual_connection=ADD_RESIDUAL_CONNECTION,
            softmax_pooling=SOFTMAX_POOLING,
            seed=SEED
        )
    else:
        pretrained_model = GraphConvolutionalNetwork(
            in_features=IN_FEATURES,
            gc_hidden_sizes=GC_HIDDEN_SIZES,
            fc_hidden_sizes=FC_HIDDEN_SIZES,
            add_residual_connection=ADD_RESIDUAL_CONNECTION,
            softmax_pooling=SOFTMAX_POOLING,
            seed=SEED
        )
    pretrained_model.load_params(MODEL_PATH)

    # Load data
    input_emmbeddings, model_embeddings = get_clustering_data(
        doc_paths=[POSITIVE_REVIEWS_PATH, NEGATIVE_REVIEWS_PATH],
        embeddings_path=EMBEDDINGS_PATH,
        model=pretrained_model
    )

    input_emmbeddings.to_csv(f"{CLUSTERING_DATA_DIR}input_embeddings.csv")
    model_embeddings.to_csv(f"{CLUSTERING_DATA_DIR}model_embeddings.csv")
