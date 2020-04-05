import io
import logging
from typing import List, Callable, Tuple, Optional, Dict

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


def load_docs(path: str, stem_tokens: bool = False) -> List[List[str]]:
    file = io.open(path, encoding='latin-1')
    docs = []
    stemmer = PorterStemmer()
    for line in file:
        doc = [token for token in word_tokenize(line.replace("-", " "))]
        if stem_tokens:
            doc = [stemmer.stem(token) for token in doc if token.isalpha()]
        docs.append(doc)
    file.close()
    return docs


def load_embeddings(path: str) -> Dict[str, np.ndarray]:
    embeddings = {}
    file = open(path)
    for line in file:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float64")
        embeddings[word] = vector
    file.close()
    return embeddings


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


def make_token_to_id_lookup(docs: List[List[str]]) -> Tuple[Callable, int]:
    vocab = build_vocab(docs)
    vocab_to_id = {token: i for i, token in enumerate(vocab)}

    def token_to_id_lookup(token: str) -> int:
        try:
            return vocab_to_id[token]
        except KeyError:
            return len(vocab)
    return token_to_id_lookup, len(vocab)


def get_one_hot_encodings_for_doc(
        doc: List[str],
        token_to_id_lookup: Callable,
        vocab_size: int
) -> np.ndarray:
    doc_one_hot_encodings = np.zeros((len(doc), vocab_size + 1))
    for i, token in enumerate(doc):
        doc_one_hot_encodings[i, token_to_id_lookup(token)] = 1
    return doc_one_hot_encodings


def make_token_to_embedding_lookup(path: str) -> Tuple[Callable, int]:
    glove_embeddings = load_embeddings(path=path)
    embedding_size = len(list(glove_embeddings.values())[0])

    def token_to_embedding_lookup(token: str) -> Optional[np.ndarray]:
        try:
            return glove_embeddings[token]
        except KeyError:
            return None
    return token_to_embedding_lookup, embedding_size


def get_embeddings_for_doc(
        doc: List[str],
        token_to_embedding_lookup: Callable
) -> np.ndarray:
    doc_glove_embeddings = [token_to_embedding_lookup(token) for token in doc]
    return np.stack([embedding for embedding in doc_glove_embeddings if embedding is not None])


def get_feature_data(docs: List[List[str]], embeddings_path: Optional[str] = None):
    if embeddings_path:
        token_to_embedding_lookup, in_features = make_token_to_embedding_lookup(path=embeddings_path)
        features = [
            get_embeddings_for_doc(
                doc=doc,
                token_to_embedding_lookup=token_to_embedding_lookup
            )
            for doc in docs
        ]
    else:
        token_to_id_lookup, in_features = make_token_to_id_lookup(docs=docs)
        features = [
            get_one_hot_encodings_for_doc(
                doc=doc,
                token_to_id_lookup=token_to_id_lookup,
                vocab_size=in_features
            )
            for doc in docs
        ]
    return features, in_features


def get_data(
        embeddings_path: Optional[str] = None,
        stem_tokens: bool = False,
        max_examples_per_class: Optional[int] = None
) -> Tuple[
    List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    int
]:
    # Load docs
    logger.info("Loading data...")
    positive_docs = load_docs(POSITIVE_REVIEWS_PATH, stem_tokens=stem_tokens)
    negative_docs = load_docs(POSITIVE_REVIEWS_PATH, stem_tokens=stem_tokens)
    if max_examples_per_class:
        positive_docs = positive_docs[:max_examples_per_class]
        negative_docs = negative_docs[:max_examples_per_class]
    all_docs = positive_docs + negative_docs
    logger.info(f"Number of documents: \t {len(all_docs)}")

    # Get input feature data
    logger.info("Getting input features...")
    inputs, in_features = get_feature_data(docs=all_docs, embeddings_path=embeddings_path)
    logger.info(f"Number of features: \t {in_features}")

    # Get adjacency matrix
    adjacencies = [
        make_adjacency_matrix_for_doc(doc_length=features.shape[0])
        for features in inputs
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
        list(zip(test_inputs, test_adjacencies, test_targets)), in_features





