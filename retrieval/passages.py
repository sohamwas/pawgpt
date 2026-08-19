"""
Turning a set of candidate breeds into text for the prompt.

There are three ways to do this and the right one depends on how many candidates
there are, so this module picks:

  full_prose      few enough candidates that their entire deduplicated write-ups
                  fit in the budget. No search, no ranking, no approximation - the
                  model gets everything known about those breeds.

  scoped_search   too many candidates to dump, so search their text and take the
                  passages that answer the question.

  global_search   no usable structured constraints at all ("which breeds are prone
                  to hip dysplasia?" - there is no hip-dysplasia column), so search
                  all 391. This is what the original pipeline did for everything.

The first mode only became possible after deduplication: a raw breed document is
~64,000 characters, of which ~82% is advice repeated across every breed. Stripped
of that it is ~10,000 characters, and two or three of those fit in one prompt.
"""
import functools
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
PROSE_PATH = REPO_ROOT / "data" / "breed_prose.json"
INDEX_PATH = REPO_ROOT / "data" / "breed_index.npz"

EMBED_MODEL = "all-MiniLM-L6-v2"

# Which retriever answers the prose questions: "lexical" (BM25) or "dense" (MiniLM).
#
# Dense is better on paraphrase, where the user's wording differs from the corpus's;
# measured over deliberately lay phrasings it scored 0.547 against BM25's 0.438.
# Lexical is better on literal topic lookup - 1.000 against 0.906 - and is 11x
# faster, needs no model, and does not pull in torch.
#
# torch is 395 MB of the service's 615 MB against a 512 MB free-tier ceiling, so this
# switch decides whether the project deploys for nothing or not at all. Lexical is
# the default for that reason, and it is a smaller concession than it looks: the
# paraphrase cases BM25 loses on - "fur everywhere on the sofa", "panics when nobody
# is home" - are shedding and tolerates-being-alone, both of which are columns the
# structured filter now answers exactly. They never reach prose search.
#
# Set RETRIEVAL_BACKEND=dense to restore the embedding path; nothing else changes.
RETRIEVAL_BACKEND = os.environ.get("RETRIEVAL_BACKEND", "lexical").strip().lower()

# Rough chars-per-token for English prose. Only used to decide which mode to take,
# so an approximation is fine - being 10% out changes nothing.
CHARS_PER_TOKEN = 4

# Dump full prose only while it stays under this. Above it, search instead.
#
# Dumping is strictly better than searching when it fits - the model gets a whole
# profile rather than four 220-token fragments - so the only reason not to is size.
# Measured against this account, Groq's free tier reports:
#
#     413: TPM Limit 8000, Requested 12088
#
# and that 8,000 is the cap on a single request as well as the per-minute budget.
# The agent's own overhead is roughly 1,700 tokens (tool schemas plus system prompt)
# and tool results accumulate across rounds, so the assembled context has to leave
# real headroom rather than spending the whole allowance. 4,000 admits one median
# profile (~2,560 tokens) with room for a filter result alongside it. Raise it on a
# paid tier, where the limit is far higher.
DUMP_TOKEN_BUDGET = 4_000

# When searching across many candidate breeds, stop any single breed from taking
# every slot. Without this a 22-breed candidate set routinely returns eight chunks
# from whichever breed happens to be worded most like the question.
MAX_CHUNKS_PER_BREED = 2

# ...but with only a handful of candidates the opposite risk applies: capping at 2
# chunks each returns almost nothing. Below this many breeds, allow more per breed.
FEW_BREEDS = 4
MAX_CHUNKS_PER_FEW_BREEDS = 4


class IndexMissing(FileNotFoundError):
    pass


@dataclass
class Context:
    """Assembled prompt context, plus enough provenance to explain it."""
    text: str
    mode: str
    breeds: list = field(default_factory=list)
    chunks: list = field(default_factory=list)   # dicts: breed, score, text

    @property
    def tokens(self):
        return len(self.text) // CHARS_PER_TOKEN

    def provenance(self):
        """One line per source, for a debug panel or a log."""
        if self.mode == "full_prose":
            return [f"{b}: full profile" for b in self.breeds]
        return [f"{c['breed']} (score {c['score']:.3f})" for c in self.chunks]


@functools.lru_cache(maxsize=1)
def load_prose():
    if not PROSE_PATH.exists():
        raise IndexMissing(
            f"{PROSE_PATH.name} not found. Run `python scripts/dedup_prose.py`."
        )
    return json.loads(PROSE_PATH.read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=1)
def load_index():
    if not INDEX_PATH.exists():
        raise IndexMissing(
            f"{INDEX_PATH.name} not found. Run `python scripts/build_index.py`."
        )
    data = np.load(INDEX_PATH, allow_pickle=True)
    return (
        data["vectors"],
        data["breeds"].astype(str),
        data["chunk_index"],
        data["texts"].astype(str),
    )


@functools.lru_cache(maxsize=1)
def _encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL, device="cpu")


def embed_query(query):
    """Unit-normalized query vector, so cosine similarity is a dot product."""
    return _encoder().encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)[0]


def full_prose(breeds, budget_tokens=DUMP_TOKEN_BUDGET):
    """Every deduplicated word about these breeds, or None if it will not fit."""
    prose = load_prose()
    missing = [b for b in breeds if b not in prose]
    if missing:
        raise KeyError(f"no prose for {missing}")

    blocks = [f"[{b}]\n{prose[b].strip()}" for b in breeds]
    text = "\n\n".join(blocks)
    if len(text) // CHARS_PER_TOKEN > budget_tokens:
        return None
    return Context(text=text, mode="full_prose", breeds=list(breeds))


def search(query, breeds=None, k=8, max_per_breed=MAX_CHUNKS_PER_BREED):
    """Search the prose, by whichever backend is configured.

    Both return the same shape, so everything above this is unaffected by the choice.
    """
    if RETRIEVAL_BACKEND == "lexical":
        # Imported here rather than at module scope so the dense path does not pay
        # to build an inverted index it will never consult, and vice versa.
        from . import lexical

        picked = lexical.search(query, breeds=breeds, k=k, max_per_breed=max_per_breed)
        blocks = [f"[{c['breed']}]\n{c['text']}" for c in picked]
        return Context(
            text="\n\n".join(blocks),
            mode="scoped_search" if breeds is not None else "global_search",
            breeds=sorted({c["breed"] for c in picked}),
            chunks=picked,
        )
    return _dense_search(query, breeds=breeds, k=k, max_per_breed=max_per_breed)


def _dense_search(query, breeds=None, k=8, max_per_breed=MAX_CHUNKS_PER_BREED):
    """Exhaustive cosine over the whole index.

    After deduplication that is 4,801 vectors - small enough that a full scan beats
    an approximate structure, and it removes the network call the original pipeline
    needed. Requires sentence-transformers, hence torch.
    """
    vectors, chunk_breeds, chunk_ids, texts = load_index()
    scores = vectors @ embed_query(query)

    if breeds is not None:
        allowed = np.isin(chunk_breeds, list(breeds))
        if not allowed.any():
            return Context(text="", mode="scoped_search", breeds=list(breeds))
        scores = np.where(allowed, scores, -np.inf)

    order = np.argsort(-scores)
    picked, per_breed = [], {}
    for i in order:
        if not np.isfinite(scores[i]):
            break
        # str() rather than the raw numpy scalar: these names travel out through
        # the public API into prompts, JSON and tests, and np.str_ repr leaks
        # everywhere it lands.
        breed = str(chunk_breeds[i])
        if per_breed.get(breed, 0) >= max_per_breed:
            continue
        per_breed[breed] = per_breed.get(breed, 0) + 1
        picked.append({
            "breed": breed,
            "chunk_index": int(chunk_ids[i]),
            "score": float(scores[i]),
            "text": str(texts[i]),
        })
        if len(picked) >= k:
            break

    blocks = [f"[{c['breed']}]\n{c['text']}" for c in picked]
    return Context(
        text="\n\n".join(blocks),
        mode="scoped_search" if breeds is not None else "global_search",
        breeds=sorted({c["breed"] for c in picked}),
        chunks=picked,
    )


def assemble(query, breeds=None, k=8, budget_tokens=DUMP_TOKEN_BUDGET):
    """Pick a retrieval mode for this query and return the assembled context.

    `breeds=None` means no structured constraints applied - search everything.
    """
    if breeds is None:
        return search(query, breeds=None, k=k)

    breeds = list(breeds)
    if not breeds:
        return Context(text="", mode="full_prose", breeds=[])

    dumped = full_prose(breeds, budget_tokens=budget_tokens)
    if dumped is not None:
        return dumped

    per_breed = (MAX_CHUNKS_PER_FEW_BREEDS if len(breeds) < FEW_BREEDS
                 else MAX_CHUNKS_PER_BREED)
    return search(query, breeds=breeds, k=k, max_per_breed=per_breed)
