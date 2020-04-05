import io
import logging
from typing import List, Callable, Tuple, Optional

import numpy as np
import torch
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

from data_processing.adjacency_matrix import make_adjacency_matrix_for_doc

POSITIVE_REVIEWS_PATH = '../../data/movie_reviews/positive_reviews.txt'
NEGATIVE_REVIEWS_PATH = '../../data/movie_reviews/negative_reviews.txt'

# Set up logging
logger = logging.getLogger(__name__)


def load_docs(path: str) -> List[List[str]]:
    file = io.open(path, encoding='latin-1')
    docs = []
    stemmer = PorterStemmer()
    for line in file:
        docs.append(
            [
                stemmer.stem(token)
                for token in word_tokenize(line.replace("-", " ")) if token.isalpha()
            ]
        )
    file.close()
    return docs


def build_vocab(docs: List[List[str]], min_occurrences: int = 2, sort: bool = False) -> List[str]:
    vocab_counts = {}
    for doc in docs:
        for token in doc:
            if token in vocab_counts.keys():
                vocab_counts[token] += 1
            else:
                vocab_counts[token] = 1
    vocab = [token for token, count in vocab_counts.items() if count > min_occurrences]
    if sort:
        return sorted(vocab)
    else:
        return vocab


def make_token_to_id_lookup(vocab: List[str]) -> Callable:
    vocab_to_id = {token: i for i, token in enumerate(vocab)}

    def token_to_id_lookup(token: str) -> int:
        try:
            return vocab_to_id[token]
        except KeyError:
            return len(vocab)
    return token_to_id_lookup


def one_hot_encode_doc(
        doc: List[str],
        token_to_id_lookup: Callable,
        vocab_size: int
) -> np.ndarray:
    doc_length = len(doc)
    one_hot_encodings = np.zeros((doc_length, vocab_size + 1))
    for i, token in enumerate(doc):
        one_hot_encodings[i, token_to_id_lookup(token)] = 1
    return one_hot_encodings


def get_data(max_examples_per_class: Optional[int] = None) -> Tuple[
    List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    List[str]
]:

    # Load docs
    logger.info("Loading documents...")
    positive_docs = load_docs(POSITIVE_REVIEWS_PATH)
    negative_docs = load_docs(POSITIVE_REVIEWS_PATH)
    if max_examples_per_class:
        positive_docs = positive_docs[:max_examples_per_class]
        negative_docs = negative_docs[:max_examples_per_class]
    all_docs = positive_docs + negative_docs
    logger.info(f"Total documents: {len(all_docs)}")

    # Make vocab lookup
    logger.info("Building vocab...")
    vocab = build_vocab(all_docs)
    token_to_id_lookup = make_token_to_id_lookup(vocab)
    vocab_size = len(vocab)
    logger.info(f"Vocab size: {vocab_size}")

    # Get input data
    logger.info("Batching data...")
    inputs = [
        one_hot_encode_doc(
            doc=doc,
            token_to_id_lookup=token_to_id_lookup,
            vocab_size=vocab_size
        )
        for doc in all_docs
    ]

    # Get adjacency matrix
    adjacencies = [
        make_adjacency_matrix_for_doc(doc_length=len(doc))
        for doc in all_docs
    ]

    # Get target data
    targets = [0 for _ in range(len(positive_docs))] + [1 for _ in range(len(negative_docs))]

    # Partition data
    train_inputs, test_inputs, train_adjacencies, test_adjacencies, train_targets, test_targets = train_test_split(
        inputs,
        adjacencies,
        targets,
        test_size=0.3,
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
        list(zip(test_inputs, test_adjacencies, test_targets)), vocab







