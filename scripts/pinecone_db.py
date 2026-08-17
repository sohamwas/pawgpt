"""
Populate the Pinecone index with chunked dog-breed embeddings. One-time setup.

Each breed's `Combined_Info` is ~65,000 characters, while all-MiniLM-L6-v2 can
only embed 256 tokens at a time. Embedding the whole document therefore threw
away ~98% of it. This script splits each breed into token-sized chunks, embeds
each chunk separately, and stores the chunk text in the vector's metadata so the
LLM can actually read what was retrieved.

Usage (from anywhere):
    cp .env.example .env    # then fill in PINECONE_API_KEY
    python scripts/pinecone_db.py
"""
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent

# Load .env from the repo root; real environment variables take precedence.
load_dotenv(REPO_ROOT / ".env", override=False)

INDEX_NAME = "pawgpt"
EMBED_MODEL = "all-MiniLM-L6-v2"
HF_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384          # all-MiniLM-L6-v2 output dimension
MODEL_MAX_TOKENS = 256   # hard limit of the embedding model

# Chunk below the model limit so the "Breed: <name>." prefix we prepend to every
# chunk still fits without being silently truncated.
CHUNK_TOKENS = 220
CHUNK_OVERLAP = 32       # keeps sentences that straddle a boundary retrievable

EMBED_BATCH = 128        # embed_documents is far faster than one call per chunk
UPSERT_BATCH = 100       # Pinecone recommends batches of this order


def as_text(value):
    """Pinecone metadata must be str/number/bool/list-of-str - never NaN/None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


api_key = os.environ.get("PINECONE_API_KEY")
if not api_key:
    sys.exit(
        "❌ PINECONE_API_KEY is not set.\n"
        "   Get a key at https://app.pinecone.io, then either add it to .env\n"
        "   (copy .env.example to .env) or export it before running."
    )

csv_path = REPO_ROOT / "data" / "dogs_final_for_rag.csv"
if not csv_path.exists():
    sys.exit(f"❌ Dataset not found at {csv_path}")

df = pd.read_csv(csv_path)
if "Combined_Info" not in df.columns:
    sys.exit("❌ 'Combined_Info' column not found in the dataset.")

pc = Pinecone(api_key=api_key)

index_existed = pc.has_index(INDEX_NAME)
if not index_existed:
    print(f"Creating serverless index '{INDEX_NAME}' ({EMBED_DIM}-dim, cosine)...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(INDEX_NAME)

# Vector IDs changed from "<row>" to "<row>-<chunk>" when chunking was
# introduced, so stale one-vector-per-breed entries would otherwise survive
# forever and keep polluting results.
if index_existed:
    print("Clearing existing vectors (ID scheme is row-chunk, not row)...")
    try:
        index.delete(delete_all=True)
    except Exception as exc:  # empty index raises rather than no-opping
        print(f"   nothing to clear ({type(exc).__name__})")

print(f"Loading embedding model '{EMBED_MODEL}' (CPU)...")
embeddings_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
)

# Split on the model's own tokenizer so chunk_size means real tokens, not an
# approximation from character counts.
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer,
    chunk_size=CHUNK_TOKENS,
    chunk_overlap=CHUNK_OVERLAP,
)

print(f"Chunking {len(df)} breeds...")
records = []
for row_idx, row in df.iterrows():
    breed = as_text(row.get("Breed Name")) or f"row-{row_idx}"
    chunks = splitter.split_text(str(row["Combined_Info"]))
    for chunk_idx, chunk in enumerate(chunks):
        # Middle-of-document chunks rarely name the breed ("they need regular
        # grooming..."), which makes them both unretrievable by breed name and
        # ambiguous once handed to the LLM. The prefix fixes both.
        text = f"Breed: {breed}. {chunk}"
        records.append(
            {
                "id": f"{row_idx}-{chunk_idx}",
                "text": text,
                "metadata": {
                    "breed_name": breed,
                    "chunk_index": chunk_idx,
                    "chunk_count": len(chunks),
                    "dog_size": as_text(row.get("Dog Size")),
                    "breed_group": as_text(row.get("Dog Breed Group")),
                    "text": text,
                },
            }
        )

print(f"✅ {len(df)} breeds -> {len(records):,} chunks "
      f"(~{len(records) / max(len(df), 1):.0f} per breed)")

print("Embedding and upserting...")
upserted = 0
for start in range(0, len(records), EMBED_BATCH):
    batch = records[start:start + EMBED_BATCH]
    vectors = embeddings_model.embed_documents([r["text"] for r in batch])

    payload = [
        {"id": r["id"], "values": vec, "metadata": r["metadata"]}
        for r, vec in zip(batch, vectors)
    ]
    for i in range(0, len(payload), UPSERT_BATCH):
        index.upsert(vectors=payload[i:i + UPSERT_BATCH])

    upserted += len(payload)
    print(f"   {upserted:,}/{len(records):,} chunks", end="\r", flush=True)

print(f"\n✅ Uploaded {upserted:,} chunks to '{INDEX_NAME}'.")
print(f"Index stats: {index.describe_index_stats()}")
