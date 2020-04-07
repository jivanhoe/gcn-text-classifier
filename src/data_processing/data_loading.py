import io
import re
from typing import List, Tuple, Optional

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


# function is causing stuff to break; not sure why
def clean_token(token: str) -> str:
    return re.sub(r'^([^a-z0-9]+)([a-z0-9]+)', r'\g<2>', token)


def clean_line(line: str) -> str:
    line_without_joiners = re.sub(r'[-\/]', ' ', line)

    return line_without_joiners


def load_docs(path: str, stem_tokens: bool, clean_tokens: bool) -> List[List[str]]:
    file = io.open(path, encoding='latin-1')
    docs = []
    stemmer = PorterStemmer()
    for line in file:
        doc = [token for token in word_tokenize(clean_line(line))]
        if clean_tokens:
            doc = [clean_token(token) for token in doc if clean_token(token) != ""]
        if stem_tokens:
            doc = [stemmer.stem(token) for token in doc if token.isalpha()]
        docs.append(doc)
    file.close()
    return docs


def load_docs_by_class(
        paths: List[str],
        stem_tokens: bool = False,
        clean_tokens: bool = False,
        max_examples_per_class: Optional[int] = None
) -> Tuple[List[List[str]], List[int]]:
    docs = []
    targets = []
    class_id = 0
    for path in paths:
        class_docs = load_docs(path=path, stem_tokens=stem_tokens, clean_tokens=clean_tokens)
        if max_examples_per_class:
            class_docs = class_docs[:max_examples_per_class]
        docs += class_docs
        targets += [class_id for _ in range(len(class_docs))]
        class_id += 1
    return docs, targets
