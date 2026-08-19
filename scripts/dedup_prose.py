"""
Strip cross-breed boilerplate from the prose and write one clean document per breed.

Each breed's `Combined_Info` is ~64,000 characters, but measured across all 391
breeds only ~16% of it is actually about that breed. The rest is generic advice -
choosing a breeder, apartment living, health screening - repeated near-verbatim
everywhere. That boilerplate competes with breed-specific passages during
retrieval: four different breeds returning the same chunk is four wasted slots.

Removing it leaves ~10,000 chars per breed, which is small enough to hand to the
LLM whole when a filter has narrowed to a handful of candidates - no approximate
search required.

Method: split every document into sentences, count how many *breeds* each
sentence appears in (its document frequency), and drop the ones that appear
almost everywhere.

Usage (from anywhere):
    python scripts/dedup_prose.py
"""
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "data" / "dogs_final_for_rag.csv"
OUT_PATH = REPO_ROOT / "data" / "breed_prose.json"

# Drop a sentence if it appears in this many breeds or more.
#
# The document-frequency distribution is sharply bimodal: 32,671 sentences occur
# in exactly one breed, 425 occur in 301-391 breeds, and only ~1,000 sit anywhere
# in between. Retained text barely moves across T=5..T=50 (10,062 -> 10,834 chars
# median), so this threshold is not a tuned magic number - anything inside the gap
# gives the same answer. 20 is ~5% of the corpus: well above the handful of breeds
# that might legitimately share a sentence, well below the boilerplate cluster.
DF_THRESHOLD = 20

# Sentences shorter than this are fragments ("Height: 16 to 17 inches.") that add
# noise to the frequency counts without carrying content.
MIN_SENTENCE_CHARS = 40


def strip_attribute_header(text):
    """Drop the `Breed: X Size: Y ... Description:` prefix.

    That header is just the CSV's own columns concatenated into prose. We already
    have them as structured fields, so keeping them here would duplicate data the
    filter reads directly - and they are what makes every document look similar.
    """
    return text.split("Description:", 1)[-1]


def split_sentences(text):
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text)
        if len(s.strip()) >= MIN_SENTENCE_CHARS
    ]


def fingerprint(sentence, breed):
    """Hash a sentence so that templated boilerplate collides across breeds.

    Much of the filler is generated per breed ("Is the Basenji right for you?" /
    "Is the Chihuahua right for you?"). Those are the same sentence wearing
    different names, so the breed's own name is replaced before hashing -
    otherwise every one of them looks unique and survives the cut.
    """
    s = re.sub(re.escape(breed), "<BREED>", sentence, flags=re.IGNORECASE)
    return hashlib.md5(" ".join(s.lower().split()).encode()).hexdigest()


def main():
    if not CSV_PATH.exists():
        sys.exit(f"[X] Dataset not found at {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    for col in ("Breed Name", "Combined_Info"):
        if col not in df.columns:
            sys.exit(f"[X] '{col}' column not found in the dataset.")

    print(f"Reading {len(df)} breeds from {CSV_PATH.name}...")
    per_breed = {
        row["Breed Name"]: split_sentences(strip_attribute_header(str(row["Combined_Info"])))
        for _, row in df.iterrows()
    }
    total_sentences = sum(len(v) for v in per_breed.values())

    # Document frequency: how many breeds contain this sentence at all. Counting
    # the set per breed, not every occurrence, so a sentence repeated twice within
    # one document still counts once.
    doc_freq = Counter()
    for breed, sentences in per_breed.items():
        for h in {fingerprint(s, breed) for s in sentences}:
            doc_freq[h] += 1

    print(f"{total_sentences:,} sentences, {len(doc_freq):,} distinct")
    print(f"Dropping sentences present in >= {DF_THRESHOLD} breeds...")

    cleaned, before, after, dropped_sentences = {}, 0, 0, 0
    for breed, sentences in per_breed.items():
        kept = []
        for s in sentences:
            if doc_freq[fingerprint(s, breed)] < DF_THRESHOLD:
                kept.append(s)
            else:
                dropped_sentences += 1
        cleaned[breed] = " ".join(kept)
        before += len(" ".join(sentences))
        after += len(cleaned[breed])

    OUT_PATH.write_text(json.dumps(cleaned, indent=1), encoding="utf-8")

    lengths = pd.Series({b: len(t) for b, t in cleaned.items()})
    empty = lengths[lengths == 0]

    print(f"\n[OK] Wrote {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"     sentences dropped : {dropped_sentences:,} of {total_sentences:,} "
          f"({dropped_sentences / total_sentences:.0%})")
    print(f"     chars  {before:,} -> {after:,}  ({after / before:.0%} retained)")
    print(f"     per breed: median {lengths.median():,.0f}  "
          f"min {lengths.min():,.0f}  max {lengths.max():,.0f} chars")
    print(f"     ~{lengths.median() / 4:,.0f} tokens per breed "
          f"(~{lengths.median() * 3 / 4:,.0f} tokens for a 3-breed candidate set)")
    if len(empty):
        print(f"     [!] {len(empty)} breeds have no unique prose left: "
              f"{list(empty.index)[:5]}")


if __name__ == "__main__":
    main()
