from typing import List, Callable, Tuple

import numpy as np


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


def one_hot_encode_doc(
        doc: List[str],
        token_to_id_lookup: Callable,
        vocab_size: int
) -> np.ndarray:
    doc_one_hot_encodings = np.zeros((len(doc), vocab_size + 1))
    for i, token in enumerate(doc):
        doc_one_hot_encodings[i, token_to_id_lookup(token)] = 1
    return doc_one_hot_encodings


def get_one_hot_encoding_features(docs: List[List[str]]) -> Tuple[List[np.ndarray], int]:

    token_to_id_lookup, vocab_size = make_token_to_id_lookup(docs=docs)
    features = [
        one_hot_encode_doc(
            doc=doc,
            token_to_id_lookup=token_to_id_lookup,
            vocab_size=vocab_size
        )
        for doc in docs
    ]
    return features, vocab_size
