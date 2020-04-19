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
) -> Tuple[List[str], np.ndarray]:
    embedding = [token_to_embedding_lookup(token) for token in doc]
    filtered_doc = [doc[i] for i in range(len(doc)) if embedding[i] is not None]
    embedding = np.stack([embedding for embedding in embedding if embedding is not None])
    return filtered_doc, embedding


def get_embedding_features(docs: List[List[str]], path: str) -> Tuple[List[List[str]], List[np.ndarray], int]:
    token_to_embedding_lookup, embedding_size = make_token_to_embedding_lookup(path=path)
    filtered_docs = []
    embeddings = []
    for doc in docs:
        filtered_doc, embedding = embed_doc(doc=doc, token_to_embedding_lookup=token_to_embedding_lookup)
        filtered_docs.append(filtered_doc)
        embeddings.append(embedding)
    return filtered_docs, embeddings, embedding_size
