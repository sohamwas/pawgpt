"""
Tests for prose assembly and search. Still no API keys and no network - the index
is local, so the whole retrieval path is verifiable offline.

Skips rather than fails when the generated artifacts are absent, so a fresh clone
does not report red before the two build scripts have been run.
"""
import pytest

from retrieval import passages

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _require(fn):
    try:
        return fn()
    except passages.IndexMissing as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def prose():
    return _require(passages.load_prose)


@pytest.fixture(scope="module")
def index():
    return _require(passages.load_index)


# --- the deduplicated corpus -------------------------------------------------

def test_every_breed_has_prose(prose):
    assert len(prose) == 391
    assert all(text.strip() for text in prose.values())


def test_boilerplate_is_gone(prose):
    """A sentence that survived in every breed would mean dedup did nothing."""
    marker = "Looking for the best dog for your apartment"
    hits = sum(marker.lower() in text.lower() for text in prose.values())
    assert hits < 20, f"{hits} breeds still carry the apartment boilerplate"


def test_documents_shrank_to_dumpable_size(prose):
    lengths = sorted(len(t) for t in prose.values())
    median = lengths[len(lengths) // 2]
    assert 5_000 < median < 20_000, f"median {median} chars is not dump-sized"


# --- full prose --------------------------------------------------------------

def test_full_prose_returns_whole_profile(prose):
    ctx = passages.full_prose(["Basenji"])
    assert ctx.mode == "full_prose"
    assert ctx.breeds == ["Basenji"]
    assert prose["Basenji"][:200] in ctx.text


def test_full_prose_declines_when_over_budget():
    many = ["Basenji", "Poodle", "Poochon", "Borzoi", "Azawakh"]
    assert passages.full_prose(many) is None


def test_full_prose_respects_an_explicit_budget():
    assert passages.full_prose(["Basenji"], budget_tokens=10) is None
    assert passages.full_prose(["Basenji"], budget_tokens=100_000) is not None


def test_unknown_breed_is_rejected():
    with pytest.raises(KeyError):
        passages.full_prose(["Direwolf"])


# --- search ------------------------------------------------------------------

def test_search_finds_a_topic_with_no_structured_column(index):
    """No hip-dysplasia column exists; only the prose can answer this.

    Asserts the passages are on topic, not which breeds come back. An earlier
    version required "German Shepherd Dog" in the top 5, which looked reasonable and
    was not: 159 of the 391 breeds mention dysplasia, so the query has no small set
    of correct answers, and the assertion was really pinning an incidental property
    of one ranking function. Under BM25 the German Shepherd sits at rank 17 of 4,801
    - found, just not top-5 - and the test failed for no defensible reason.
    """
    ctx = passages.search("which breeds are prone to hip dysplasia", k=5)
    assert ctx.mode == "global_search"
    assert ctx.chunks
    assert all("dysplasia" in c["text"].lower() for c in ctx.chunks[:3])


def test_search_results_are_ordered_by_score(index):
    ctx = passages.search("grooming and coat care", k=6)
    scores = [c["score"] for c in ctx.chunks]
    assert scores == sorted(scores, reverse=True)


def test_scoped_search_never_leaves_the_candidate_set(index):
    candidates = ["Basenji", "Poodle", "Whippet"]
    ctx = passages.search("temperament", breeds=candidates, k=8)
    assert ctx.mode == "scoped_search"
    assert set(ctx.breeds) <= set(candidates)


def test_scoped_search_on_unknown_breeds_returns_empty(index):
    ctx = passages.search("temperament", breeds=["Direwolf"])
    assert ctx.chunks == []
    assert ctx.text == ""


def test_per_breed_cap_is_enforced(index):
    from collections import Counter
    candidates = ["Basenji", "Poodle", "Whippet", "Borzoi", "Azawakh"]
    ctx = passages.search("grooming", breeds=candidates, k=8, max_per_breed=2)
    assert max(Counter(c["breed"] for c in ctx.chunks).values()) <= 2


def test_breed_names_are_plain_strings(index):
    """numpy scalars leaking out of the index would poison prompts and JSON."""
    ctx = passages.search("temperament", k=3)
    assert all(type(c["breed"]) is str for c in ctx.chunks)
    assert all(type(b) is str for b in ctx.breeds)


# --- routing -----------------------------------------------------------------

def test_no_constraints_searches_everything(index):
    assert passages.assemble("anything at all", breeds=None).mode == "global_search"


def test_one_breed_is_dumped_whole(index):
    assert passages.assemble("tell me about it", breeds=["Basenji"]).mode == "full_prose"


def test_large_candidate_set_falls_back_to_search(index):
    many = ["Basenji", "Poodle", "Poochon", "Borzoi", "Azawakh", "Whippet"]
    ctx = passages.assemble("what are they like", breeds=many)
    assert ctx.mode == "scoped_search"
    assert set(ctx.breeds) <= set(many)


def test_empty_candidate_set_produces_no_context():
    ctx = passages.assemble("anything", breeds=[])
    assert ctx.text == ""
    assert ctx.breeds == []


def test_context_reports_its_own_provenance(index):
    ctx = passages.assemble("hip dysplasia", breeds=None)
    lines = ctx.provenance()
    assert len(lines) == len(ctx.chunks)
    assert all("score" in line for line in lines)
