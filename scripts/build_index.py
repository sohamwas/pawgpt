"""
Build a local vector index over the deduplicated breed prose.

Run `scripts/dedup_prose.py` first - this reads its output.

Why local rather than a hosted vector database: removing boilerplate cut the corpus
from 25,134 chunks to 4,801, which is 7.4 MB of float32. At that size an exhaustive
cosine scan in numpy is faster than a network round trip, needs no API key, and lets
the retrieval tests run offline and in CI.

The output is committed, so this only needs running when the dataset, the chunking
settings or the embedding model change. The deployed service reads the chunk text
out of it for BM25 and never materialises the vectors.

Usage (from anywhere):
    python scripts/dedup_prose.py
    python scripts/build_index.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
PROSE_PATH = REPO_ROOT / "data" / "breed_prose.json"
OUT_PATH = REPO_ROOT / "data" / "breed_index.npz"

EMBED_MODEL = "all-MiniLM-L6-v2"
HF_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Same settings as the Pinecone script, for the same reason: all-MiniLM-L6-v2 hard
# -truncates at 256 tokens, so chunks are split on the model's own tokenizer at 220
# to leave headroom for the breed-name prefix, with a 32-token overlap so sentences
# straddling a boundary stay retrievable.
CHUNK_TOKENS = 220
CHUNK_OVERLAP = 32
EMBED_BATCH = 128


def main():
    if not PROSE_PATH.exists():
        sys.exit(
            f"[X] {PROSE_PATH.name} not found.\n"
            "    Run `python scripts/dedup_prose.py` first."
        )

    prose = json.loads(PROSE_PATH.read_text(encoding="utf-8"))
    print(f"Loaded deduplicated prose for {len(prose)} breeds.")

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=CHUNK_TOKENS, chunk_overlap=CHUNK_OVERLAP
    )

    breeds, indices, texts = [], [], []
    for breed, text in prose.items():
        if not text.strip():
            continue
        for i, chunk in enumerate(splitter.split_text(text)):
            # A mid-document chunk rarely names the breed ("they need regular
            # grooming..."), which makes it both unretrievable by breed name and
            # ambiguous once handed to the LLM. The prefix fixes both.
            texts.append(f"Breed: {breed}. {chunk}")
            breeds.append(breed)
            indices.append(i)

    print(f"{len(texts):,} chunks (~{len(texts) / len(prose):.1f} per breed)")

    # sentence-transformers directly rather than the LangChain wrapper: this script
    # only needs to turn strings into vectors, and encode() handles batching,
    # progress and L2 normalization itself.
    print(f"Loading '{EMBED_MODEL}' (CPU)...")
    model = SentenceTransformer(EMBED_MODEL, device="cpu")

    # Unit vectors, so a cosine query is a plain dot product at search time.
    matrix = model.encode(
        texts,
        batch_size=EMBED_BATCH,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    np.savez_compressed(
        OUT_PATH,
        vectors=matrix,
        breeds=np.array(breeds, dtype=object),
        chunk_index=np.array(indices, dtype=np.int32),
        texts=np.array(texts, dtype=object),
    )

    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"\n[OK] Wrote {OUT_PATH.relative_to(REPO_ROOT)} "
          f"({matrix.shape[0]:,} x {matrix.shape[1]}, {size_mb:.1f} MB on disk)")


if __name__ == "__main__":
    main()
