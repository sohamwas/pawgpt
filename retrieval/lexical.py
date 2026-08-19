"""
BM25 search over the breed chunks, with no model and no torch.

The dense path needs sentence-transformers, which drags in torch: 395 MB of the
615 MB the service occupies, against a 512 MB free-tier ceiling. That single
dependency is the difference between deploying free and not.

Dropping it is defensible here for a reason specific to this system rather than a
general preference for BM25. Before the structured filter existed, semantic search
carried every question, including "a quiet dog for a small flat" where the wording
of the query and the wording of the answer have nothing in common. It no longer
does: constraint questions are answered by `filter_breeds` against real columns, and
what reaches prose search is the residue the columns cannot express - hip dysplasia,
hypoallergenic coats, getting on with cats. Those are largely *lexical* lookups, the
case BM25 is strongest on, because the user's word is usually the corpus's word.

Whether that holds is measurable, not assumed: `scripts/compare_retrieval.py` scores
this against the dense index on the queries that matter.

The chunks are the same ones the dense index was built from. `data/breed_index.npz`
stores its text alongside its vectors, and `np.load` reads members lazily, so the
text can be taken without ever materialising the 7 MB matrix.
"""
import functools
import math
import re
from collections import Counter

import numpy as np

from .passages import INDEX_PATH, IndexMissing

# Standard BM25 parameters. k1 controls how quickly repeated terms stop adding
# score; b controls how strongly long documents are penalised. These are the usual
# defaults and there is no tuning set here to justify moving them.
K1 = 1.5
B = 0.75

# Words carrying no discriminative signal in a corpus that is entirely about dogs.
# "dog", "breed" and "breeds" appear in nearly every chunk, so leaving them in makes
# every document look slightly relevant to every query.
STOPWORDS = frozenset("""
a an and are as at be been but by can could do does for from had has have he her
his how i if in into is it its of on or our she that the their them then there
these they this to was were what when where which who will with would you your
dog dogs breed breeds
""".split())

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return [t for t in TOKEN.findall(text.lower())
            if len(t) > 2 and t not in STOPWORDS]


@functools.lru_cache(maxsize=1)
def load_corpus():
    """Chunk text and breed labels, without touching the vector matrix.

    npz members are decompressed on access, so naming only these three keeps the
    384-dimension float matrix off the heap entirely.
    """
    if not INDEX_PATH.exists():
        raise IndexMissing(
            f"{INDEX_PATH.name} not found. Run `python scripts/build_index.py`."
        )
    data = np.load(INDEX_PATH, allow_pickle=True)
    return (
        [str(t) for t in data["texts"]],
        [str(b) for b in data["breeds"]],
        [int(i) for i in data["chunk_index"]],
    )


@functools.lru_cache(maxsize=1)
def build():
    """An inverted index plus the statistics BM25 needs.

    4,801 documents is small enough that a dict of postings is both simpler and
    faster than a sparse matrix, and it costs a few megabytes.
    """
    texts, breeds, chunk_ids = load_corpus()

    postings = {}          # term -> {doc index: term frequency}
    lengths = np.zeros(len(texts), dtype=np.float32)

    for i, text in enumerate(texts):
        tokens = tokenize(text)
        lengths[i] = len(tokens)
        for term, freq in Counter(tokens).items():
            postings.setdefault(term, {})[i] = freq

    n_docs = len(texts)
    avg_len = float(lengths.mean()) or 1.0

    # Inverse document frequency, computed once per term. The +1 inside the log is
    # the standard guard that keeps a term appearing in every document from scoring
    # negative rather than merely zero.
    idf = {
        term: math.log(1 + (n_docs - len(docs) + 0.5) / (len(docs) + 0.5))
        for term, docs in postings.items()
    }

    return {
        "postings": postings,
        "idf": idf,
        "lengths": lengths,
        "avg_len": avg_len,
        "texts": texts,
        "breeds": breeds,
        "chunk_ids": chunk_ids,
        "n_docs": n_docs,
    }


def score(query, allowed=None):
    """BM25 score for every chunk. `allowed` restricts to a set of breed names."""
    index = build()
    scores = np.zeros(index["n_docs"], dtype=np.float32)

    for term in tokenize(query):
        docs = index["postings"].get(term)
        if not docs:
            continue
        weight = index["idf"][term]
        for doc, freq in docs.items():
            norm = 1 - B + B * index["lengths"][doc] / index["avg_len"]
            scores[doc] += weight * (freq * (K1 + 1)) / (freq + K1 * norm)

    if allowed is not None:
        mask = np.array([b in allowed for b in index["breeds"]])
        scores = np.where(mask, scores, -np.inf)
    return scores


def search(query, breeds=None, k=8, max_per_breed=2):
    """Top chunks for a query, in the same shape the dense path returns."""
    index = build()
    allowed = set(breeds) if breeds is not None else None
    scores = score(query, allowed)

    picked, per_breed = [], {}
    for i in np.argsort(-scores):
        if not np.isfinite(scores[i]) or scores[i] <= 0:
            break
        breed = index["breeds"][i]
        if per_breed.get(breed, 0) >= max_per_breed:
            continue
        per_breed[breed] = per_breed.get(breed, 0) + 1
        picked.append({
            "breed": breed,
            "chunk_index": index["chunk_ids"][i],
            "score": float(scores[i]),
            "text": index["texts"][i],
        })
        if len(picked) >= k:
            break
    return picked
