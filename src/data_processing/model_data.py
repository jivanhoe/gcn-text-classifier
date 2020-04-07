import logging
from typing import List, Tuple, Optional

import torch
from sklearn.model_selection import train_test_split

from data_processing.data_loading import load_docs_by_class
from data_processing.embedding import get_embedding_features
from data_processing.one_hot_encoding import get_one_hot_encoding_features
from data_processing.adjacency_matrix import make_adjacency_matrix_for_doc

# Set up logging
logger = logging.getLogger(__name__)


def get_feature_data(docs: List[List[str]], embeddings_path: Optional[str] = None):
    if embeddings_path:
        return get_embedding_features(docs=docs, path=embeddings_path)
    else:
        return get_one_hot_encoding_features(docs=docs)


def get_model_data(
        doc_paths: List[str],
        embeddings_path: Optional[str] = None,
        stem_tokens: bool = False,
        clean_tokens: bool = False,
        max_examples_per_class: Optional[int] = None,
        test_size: float = 0.3
) -> Tuple[
    List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    int
]:
    # Load docs
    logger.info("loading data...")
    docs, targets = load_docs_by_class(
        paths=doc_paths,
        stem_tokens=stem_tokens,
        clean_tokens=clean_tokens,
        max_examples_per_class=max_examples_per_class
    )
    logger.info(f"number of documents: \t {len(docs)}")

    # Get input feature data
    logger.info("getting input features...")
    inputs, in_features = get_feature_data(
        docs=docs,
        embeddings_path=embeddings_path
    )
    logger.info(f"number of features: \t {in_features}")

    # Get adjacency matrix
    adjacencies = [
        make_adjacency_matrix_for_doc(doc_length=features.shape[0])
        for features in inputs
    ]

    # Partition data
    train_inputs, test_inputs, train_adjacencies, test_adjacencies, train_targets, test_targets = train_test_split(
        inputs,
        adjacencies,
        targets,
        test_size=test_size,
        stratify=targets,
        shuffle=True
    )

    # Convert data to tensors
    numpy_to_tensor = lambda array: torch.from_numpy(array).float()
    train_inputs = map(numpy_to_tensor, train_inputs)
    test_inputs = map(numpy_to_tensor, test_inputs)
    train_adjacencies = map(numpy_to_tensor, train_adjacencies)
    test_adjacencies = map(numpy_to_tensor, test_adjacencies)
    train_targets = map(torch.tensor, train_targets)
    test_targets = map(torch.tensor, test_targets)

    # Zip data and return
    return list(zip(train_inputs, train_adjacencies, train_targets)), \
        list(zip(test_inputs, test_adjacencies, test_targets)), in_features
