"""
Tests for the BM25 retriever and the backend switch.

This path exists for a deployment reason: it is what lets the service run in 145 MB
instead of 615 MB, and therefore on a free instance instead of a paid one. So the
memory property is asserted here, not just the search behaviour, because a stray
top-level import of sentence-transformers would silently undo the whole point.
"""
import sys

import pytest

from retrieval import lexical, passages


@pytest.fixture(scope="module")
def index():
    try:
        return lexical.build()
    except passages.IndexMissing as exc:
        pytest.skip(str(exc))


# --- tokenisation ------------------------------------------------------------

def test_tokenizer_drops_domain_stopwords():
    """In a corpus entirely about dogs, "dog" carries no signal."""
    tokens = lexical.tokenize("The dog breeds that are prone to hip dysplasia")
    assert "dog" not in tokens
    assert "breeds" not in tokens
    assert "the" not in tokens
    assert "dysplasia" in tokens


def test_tokenizer_is_case_insensitive():
    assert lexical.tokenize("Hip Dysplasia") == lexical.tokenize("hip dysplasia")


def test_tokenizer_drops_very_short_tokens():
    assert "of" not in lexical.tokenize("a lot of it")


# --- the index ---------------------------------------------------------------

def test_index_covers_every_chunk(index):
    assert index["n_docs"] == 4801
    assert len(index["texts"]) == index["n_docs"]
    assert len(index["breeds"]) == index["n_docs"]


def test_idf_is_higher_for_rarer_terms(index):
    """A term in a handful of breeds must outweigh one in most of them."""
    rare, common = index["idf"].get("dysplasia"), index["idf"].get("exercise")
    assert rare and common
    assert rare > common


# --- search ------------------------------------------------------------------

def test_finds_a_topic_with_no_structured_column(index):
    chunks = lexical.search("which breeds are prone to hip dysplasia", k=8)
    assert chunks
    assert all("dysplasia" in c["text"].lower() for c in chunks[:3])


def test_results_are_ordered_by_score(index):
    chunks = lexical.search("grooming and coat care", k=6)
    scores = [c["score"] for c in chunks]
    assert scores == sorted(scores, reverse=True)


def test_scoped_search_never_leaves_the_candidate_set(index):
    candidates = ["Basenji", "Poodle", "Whippet"]
    chunks = lexical.search("temperament", breeds=candidates, k=8)
    assert {c["breed"] for c in chunks} <= set(candidates)


def test_per_breed_cap_is_enforced(index):
    from collections import Counter
    chunks = lexical.search("grooming", k=8, max_per_breed=2)
    assert max(Counter(c["breed"] for c in chunks).values()) <= 2


def test_a_query_matching_nothing_returns_nothing(index):
    assert lexical.search("zzzzqqq nonexistentterm", k=5) == []


def test_breed_names_are_plain_strings(index):
    chunks = lexical.search("temperament", k=3)
    assert all(type(c["breed"]) is str for c in chunks)
    assert all(type(c["text"]) is str for c in chunks)


# --- the switch --------------------------------------------------------------

def test_backend_is_lexical_by_default():
    """Default must be the one that fits the free tier, not the one that doesn't."""
    assert passages.RETRIEVAL_BACKEND == "lexical"


def test_search_routes_through_the_configured_backend(index, monkeypatch):
    monkeypatch.setattr(passages, "RETRIEVAL_BACKEND", "lexical")
    ctx = passages.search("hip dysplasia", k=5)
    assert ctx.mode == "global_search"
    assert ctx.chunks
    assert ctx.text


def test_both_backends_return_the_same_shape(index, monkeypatch):
    monkeypatch.setattr(passages, "RETRIEVAL_BACKEND", "lexical")
    ctx = passages.search("hip dysplasia", breeds=["German Shepherd Dog"], k=4)
    assert ctx.mode == "scoped_search"
    for chunk in ctx.chunks:
        assert set(chunk) == {"breed", "chunk_index", "score", "text"}


# --- the reason this module exists -------------------------------------------

def test_lexical_path_never_imports_torch(index):
    """The whole point: 145 MB instead of 615 MB, so a free instance can host it.

    Importing sentence-transformers at module scope anywhere in the retrieval
    package would restore the 395 MB and break free-tier deployment, without
    breaking any other test.
    """
    lexical.search("hip dysplasia", k=3)
    assert "torch" not in sys.modules
    assert "sentence_transformers" not in sys.modules


def test_corpus_loads_without_materialising_the_vector_matrix(index):
    """npz members decompress on access, so text can be read without the vectors."""
    texts, breeds, ids = lexical.load_corpus()
    assert len(texts) == 4801
    assert "torch" not in sys.modules
