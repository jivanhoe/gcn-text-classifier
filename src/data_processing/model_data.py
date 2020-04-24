import logging
from typing import List, Tuple, Optional

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize


from data_processing.adjacency_matrix import make_adjacency_matrix_for_doc, make_custom_adjacency_matrix_for_doc
from data_processing.data_loading import load_docs_by_class
from data_processing.embedding import get_embedding_features
from data_processing.one_hot_encoding import get_one_hot_encoding_features

# Set up logging
logger = logging.getLogger(__name__)


def get_feature_data(
        docs: List[List[str]],
        embeddings_path: Optional[str] = None
) -> Tuple[List[List[str]], List[np.ndarray], int]:
    if embeddings_path:
        return get_embedding_features(docs=docs, path=embeddings_path)
    else:
        return get_one_hot_encoding_features(docs=docs)


def make_docs_adversarial(
        docs: List[List[str]],
        targets: List[int],
        adversarial_phrases: List[str]
) -> List[List[str]]:
    adversarial_phrases = [word_tokenize(phrase) for phrase in adversarial_phrases]
    adversarial_docs = []
    for i, doc in enumerate(docs):
        adversarial_docs.append(doc + adversarial_phrases[targets[i]])
    return adversarial_docs


def get_model_data(
        doc_paths: List[str],
        embeddings_path: Optional[str] = None,
        stem_tokens: bool = False,
        clean_tokens: bool = False,
        max_examples_per_class: Optional[int] = None,
        adversarial_phrases: Optional[List[str]] = None,
        forward_weights: Optional[List[float]] = None,
        backward_weights: Optional[List[float]] = None
) -> Tuple[List[List[str]], List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], int]:
    # Load docs
    logger.info("loading data...")
    docs, targets = load_docs_by_class(
        paths=doc_paths,
        stem_tokens=stem_tokens,
        clean_tokens=clean_tokens,
        max_examples_per_class=max_examples_per_class
    )
    logger.info(f"number of documents: \t {len(docs)}")

    if adversarial_phrases:
        docs = make_docs_adversarial(docs=docs, targets=targets, adversarial_phrases=adversarial_phrases)

    # Get input feature data
    logger.info("getting input features...")
    docs, inputs, in_features = get_feature_data(
        docs=docs,
        embeddings_path=embeddings_path
    )
    logger.info(f"number of features: \t {in_features}")

    # Get adjacency matrices
    if forward_weights and backward_weights:
        adjacencies = [
            make_custom_adjacency_matrix_for_doc(
                doc_length=features.shape[0],
                forward_weights=forward_weights,
                backward_weights=backward_weights
            )
            for features in inputs
        ]
    else:
        adjacencies = [
            make_adjacency_matrix_for_doc(doc_length=features.shape[0])
            for features in inputs
        ]

    # Convert data to tensors
    numpy_to_tensor = lambda array: torch.from_numpy(array).float()
    inputs = map(numpy_to_tensor, inputs)
    adjacencies = map(numpy_to_tensor, adjacencies)
    targets = map(torch.tensor, targets)

    # Zip data and return
    return docs, list(zip(inputs, adjacencies, targets)), in_features


def partition_data(
        data,
        val_size: float = 0.15,
        test_size: float = 0.15,
        seed: int = 0
) -> Tuple[
    List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
]:
    train_data, test_data = train_test_split(
        data,
        test_size=test_size,
        stratify=[target.item() for _, _, target in data],
        shuffle=True,
        random_state=seed
    )
    train_data, val_data = train_test_split(
        train_data,
        test_size=val_size/(1 - test_size),
        stratify=[target.item() for _, _, target in train_data],
        shuffle=True,
        random_state=seed
    )
    return train_data, val_data, test_data
