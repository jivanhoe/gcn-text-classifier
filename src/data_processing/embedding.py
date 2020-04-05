from typing import List, Callable, Tuple, Optional, Dict

import numpy as np


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


def make_token_to_embedding_lookup(path: str) -> Tuple[Callable, int]:
    glove_embeddings = load_embeddings(path=path)
    embedding_size = len(list(glove_embeddings.values())[0])

    def token_to_embedding_lookup(token: str) -> Optional[np.ndarray]:
        try:
            return glove_embeddings[token]
        except KeyError:
            return None
    return token_to_embedding_lookup, embedding_size


def embed_doc(
        doc: List[str],
        token_to_embedding_lookup: Callable
) -> np.ndarray:
    doc_glove_embeddings = [token_to_embedding_lookup(token) for token in doc]
    return np.stack([embedding for embedding in doc_glove_embeddings if embedding is not None])


def get_embedding_features(docs: List[List[str]], path: str) -> Tuple[List[np.ndarray], int]:
    token_to_embedding_lookup, embedding_size = make_token_to_embedding_lookup(path=path)
    features = [
        embed_doc(
            doc=doc,
            token_to_embedding_lookup=token_to_embedding_lookup
        )
        for doc in docs
    ]
    return features, embedding_size
