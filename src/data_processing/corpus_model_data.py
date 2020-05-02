import logging
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from data_processing.data_loading import load_docs_by_class

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_pmi(docs: List[List[str]], vocabulary: List[str], window_size=20):
    word_windows = {}
    window_id = 0

    logger.info("Getting number of windows...")
    num_windows = 0
    for doc in docs:
        num_windows += int(np.ceil(len(doc) / window_size))

    logger.info("Getting words in each window...")
    # Create a dict with { word: [window_id1, window_id2, ...]}
    for doc in docs:
        windows = [doc[i:i + window_size] for i in range(0, len(doc), window_size)]

        for window in windows:
            for word in window:
                if word not in word_windows:
                    word_windows[word] = np.zeros(num_windows)

                word_windows[word][window_id] = 1
            window_id += 1

    # Use that dict to compute PMI
    pmis = np.zeros((len(vocabulary), len(vocabulary)), "float64")

    # Create matrices
    logger.info("Creating word window matrix...")
    vocab_mat = np.zeros((len(vocabulary), num_windows))
    for i in range(len(vocabulary)):
        curr_word = vocabulary[i]

        if curr_word in word_windows:
            vocab_mat[i] = word_windows[curr_word]

    logger.info("Calculating pijs...")
    pijs = np.matmul(vocab_mat, vocab_mat.T) / num_windows
    word_window_freqs = np.sum(pijs, axis=1)

    logger.info("Calculating PMIs...")
    for i in range(len(vocabulary)):
        pi = word_window_freqs[i]
        pmis[i, :] = pijs[i, :] / (pi * word_window_freqs)

    pmis = np.log(pmis + 1e-10)
    pmis = pmis.clip(min=0)
    return pmis


def get_corpus_model_data(
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
    pmis = get_pmi(docs, vocabulary)  # np.zeros((len(vocabulary), len(vocabulary)))

    adjacency[len(docs_worded):total_nodes, len(docs_worded):total_nodes] = pmis
    np.fill_diagonal(adjacency, 1)

    # 2. word-to-doc, TF-IDF
    logger.info("calculating TFIDF")
    tfidf_transformer = TfidfTransformer()
    tfidfs = tfidf_transformer.fit_transform(cv_transform).todense()

    adjacency[0:len(docs_worded), len(docs_worded):total_nodes] = tfidfs
    logger.info(adjacency.shape)
    logger.info(adjacency.mean())

    final_inputs = torch.from_numpy(inputs)
    final_targets = torch.tensor(targets)
    final_adj = torch.from_numpy(adjacency)

    return final_inputs, final_adj, final_targets, doc_indices, total_nodes


