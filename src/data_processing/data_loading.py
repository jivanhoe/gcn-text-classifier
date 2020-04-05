import io
from typing import List, Tuple, Optional

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


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


def load_docs_by_class(
        paths: List[str],
        stem_tokens: bool = False,
        max_examples_per_class: Optional[int] = None
) -> Tuple[List[List[str]], List[int]]:
    docs = []
    targets = []
    class_id = 0
    for path in paths:
        class_docs = load_docs(path=path, stem_tokens=stem_tokens)
        if max_examples_per_class:
            class_docs = class_docs[:max_examples_per_class]
        docs += class_docs
        targets += [class_id for _ in range(len(class_docs))]
        class_id += 1
    return docs, targets
