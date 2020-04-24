import logging
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from data_processing.data_loading import load_docs_by_class

# Set up logging
logger = logging.getLogger(__name__)


def get_pmi(docs: List[List[str]], vocabulary: List[str], window_size=20):
    word_windows: Dict[str, List[int]] = {}
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
    pmis = np.zeros((len(vocabulary), len(vocabulary)), "float64")

    for i in range(len(vocabulary)):
        i_word = vocabulary[i]
        if i_word in word_windows:
            wi = len(word_windows[i_word])
            pi = wi / window_id
            for j in range(i+1, len(vocabulary)):
                j_word = vocabulary[j]
                if j_word in word_windows:
                    wj = len(word_windows[j_word])
                    wij = len(set(word_windows[j_word]) & set(word_windows[i_word])) # change to indexed arrays
                    pij = wij / window_id
                    pj = wj / window_id
                    pmis[i, j] = max(np.log(pij / (pi * pj)), 0)

    # makes symmetric
    pmis = (pmis + pmis.T)

    return pmis


def get_original_model_data(
        doc_paths: List[str],
        stem_tokens: bool = False,
        clean_tokens: bool = False,
        max_examples_per_class: Optional[int] = None
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    List[int],
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
    vocabulary = list(cv.vocabulary_.keys())
    total_nodes = len(docs_worded) + len(vocabulary)
    doc_indices = list(range(len(docs_worded)))
    inputs = np.identity(total_nodes, "float64")

    logger.info("creating adjacency matrix")
    # Get adjacency matrix
    adjacency = np.identity(total_nodes, "float64")

    # There are two types of adjacencies: word-to-word, and word-to-doc
    # 1. word-to-word, uses PMI
    logger.info("calculating point-wise mutual information (PMI) of words")
    pmis = get_pmi(docs, vocabulary)

    adjacency[len(docs_worded):total_nodes, len(docs_worded):total_nodes] = pmis
    np.fill_diagonal(adjacency, 1)

    # 2. word-to-doc, TF-IDF
    logger.info("calculating TFIDF")
    tfidf_transformer = TfidfTransformer()
    tfidfs = tfidf_transformer.fit_transform(cv_transform).todense()

    adjacency[0:len(docs_worded), len(docs_worded):total_nodes] = tfidfs

    final_inputs = torch.from_numpy(inputs)
    final_targets = torch.tensor(targets)
    final_adj = torch.from_numpy(adjacency)

    return final_inputs, final_adj, final_targets, doc_indices, total_nodes
