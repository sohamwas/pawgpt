"""
Score BM25 against the dense index, to decide whether torch can be dropped.

torch is 395 MB of the service's 615 MB, and the free hosting tier caps at 512 MB.
So this is not an academic comparison: if lexical search holds up on the queries
that actually reach prose search, the whole thing deploys free.

The queries below are deliberately the ones the structured filter *cannot* answer.
Questions like "a quiet dog for a flat" never get here any more, because they are
columns and a comparison. What is left is topic lookup, which is BM25's best case
and dense retrieval's least distinctive one.

Ground truth is the breed the passage must mention, derived from the dataset rather
than from either retriever, so neither is being graded on its own homework.

Usage:
    python scripts/compare_retrieval.py
"""
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from retrieval import lexical, passages  # noqa: E402

# (query, terms that must appear in a genuinely relevant passage)
QUERIES = [
    ("which breeds are prone to hip dysplasia", ["hip dysplasia"]),
    ("hypoallergenic dogs for allergy sufferers", ["allergy", "allergies", "hypoallergenic"]),
    ("breeds that get along well with cats", ["cat", "cats"]),
    ("dogs prone to separation anxiety", ["separation anxiety"]),
    ("which breeds suffer from epilepsy or seizures", ["epilepsy", "seizure"]),
    ("dogs with breathing problems in hot weather", ["breathing", "brachycephalic", "heat"]),
    ("breeds that need professional grooming regularly", ["groom"]),
    ("dogs prone to bloat or gastric torsion", ["bloat", "gastric"]),
    ("which breeds are good guard dogs", ["guard"]),
    ("dogs with a high prey drive that chase small animals", ["prey", "chase"]),
    ("breeds prone to eye problems like cataracts", ["cataract", "eye"]),
    ("dogs that drool a lot", ["drool"]),
]

K = 8


def relevant(chunk, terms):
    body = chunk["text"].lower()
    return any(t in body for t in terms)


def evaluate(name, search_fn):
    hits_at_k, reciprocal_ranks, latencies = [], [], []

    for query, terms in QUERIES:
        start = time.perf_counter()
        chunks = search_fn(query)
        latencies.append((time.perf_counter() - start) * 1000)

        flags = [relevant(c, terms) for c in chunks]
        hits_at_k.append(sum(flags) / max(len(flags), 1))
        rank = next((i + 1 for i, ok in enumerate(flags) if ok), None)
        reciprocal_ranks.append(1 / rank if rank else 0.0)

    return {
        "name": name,
        "precision@8": sum(hits_at_k) / len(hits_at_k),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "answered": sum(1 for r in reciprocal_ranks if r > 0),
        "median_ms": sorted(latencies)[len(latencies) // 2],
    }


def main():
    print("warming the dense encoder...")
    passages.search("warmup", k=1)
    lexical.build()

    dense = evaluate("dense (MiniLM + torch)",
                     lambda q: passages.search(q, k=K).chunks)
    bm25 = evaluate("lexical (BM25, no torch)",
                    lambda q: lexical.search(q, k=K))

    print(f"\n{'retriever':<28} {'P@8':>7} {'MRR':>7} {'answered':>9} {'median':>9}")
    print("-" * 64)
    for r in (dense, bm25):
        print(f"{r['name']:<28} {r['precision@8']:>7.3f} {r['mrr']:>7.3f} "
              f"{r['answered']:>6}/{len(QUERIES)} {r['median_ms']:>7.1f}ms")

    print("\nper query (relevant chunks in top 8):")
    print(f"{'query':<48} {'dense':>6} {'bm25':>6}")
    print("-" * 64)
    for query, terms in QUERIES:
        d = sum(relevant(c, terms) for c in passages.search(query, k=K).chunks)
        b = sum(relevant(c, terms) for c in lexical.search(query, k=K))
        flag = "  <-- worse" if b < d else ("  <-- better" if b > d else "")
        print(f"{query[:47]:<48} {d:>6} {b:>6}{flag}")

    print(json.dumps({"dense": dense, "bm25": bm25}, indent=2))


if __name__ == "__main__":
    main()
