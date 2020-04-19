import logging
from typing import List, Tuple, Optional

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from data_processing.data_loading import load_docs_by_class
from data_processing.embedding import get_embedding_features
from data_processing.one_hot_encoding import get_one_hot_encoding_features
from data_processing.adjacency_matrix import make_adjacency_matrix_for_doc

# Set up logging
logger = logging.getLogger(__name__)


def get_pmi(docs, vocabulary, window_size=20):
    word_windows = {}
    window_id = 0

    # Create a dict with { word: [window_id1, window_id2, ...]}
    for doc in docs:
        windows = [doc[i:i + window_size] for i in range(0, len(doc), window_size)]

        for window in windows:
            for word in window:
                if word not in word_windows:
                    word_windows[word] = [window_id]
                else:
                    word_windows[word].append(window_id)
            window_id += 1

    # Use that dict to compute PMI
    pmis = np.zeros((len(vocabulary), len(vocabulary)))

    for i in range(len(vocabulary)):
        for j in range(i, len(vocabulary)):
            pij = wij / window_id
            pi = wi / window_id
            pj = wj / window_id
            pmis[i, j] = np.log(pij / (pi * pj))


def get_original_model_data(
        doc_paths: List[str],
        stem_tokens: bool = False,
        clean_tokens: bool = False,
        max_examples_per_class: Optional[int] = None
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
    docs_worded = [' '.join(doc) for doc in docs]
    logger.info("creating one-hot-encoding matrix for each word + doc")
    cv = CountVectorizer()
    cv_transform = cv.fit_transform(docs_worded)
    total_nodes = len(docs_worded) + len(cv.vocabulary_.keys())

    inputs = np.identity(total_nodes)
    doc_indices = range(len(docs_worded))
    vocab_indices = range(len(docs_worded), total_nodes)

    # Get adjacency matrix
    adjacencies = np.identity(total_nodes)
    # There are two types of adjacencies: word-to-word, and word-to-doc
    # 1. word-to-word, uses PMI

    # 2. word-to-doc, TF-IDF
    tfidf_transformer = TfidfTransformer()
    tfidfs = tfidf_transformer.fit_transform(cv_transform).todense()

    adjacencies[0:len(doc_indices), len(doc_indices):total_nodes] = tfidfs

    return adjacencies

    # return nodes, adjancencies, targets
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
