"""
Populate the Pinecone index with dog-breed embeddings. One-time setup.

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
from langchain_community.embeddings import HuggingFaceEmbeddings

REPO_ROOT = Path(__file__).resolve().parent.parent

# Load .env from the repo root; real environment variables take precedence.
load_dotenv(REPO_ROOT / ".env", override=False)

# Read the key from the environment so this file never holds a secret.
api_key = os.environ.get("PINECONE_API_KEY")
if not api_key:
    sys.exit(
        "❌ PINECONE_API_KEY is not set.\n"
        "   Get a key at https://app.pinecone.io, then either add it to .env\n"
        "   (copy .env.example to .env) or export it before running."
    )

pc = Pinecone(api_key=api_key)

# Must match the index name queried by streamlit_app.py
index_name = "pawgpt"

# Create index if it doesn't exist (384 dimensions for all-MiniLM-L6-v2)
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

# Resolve the CSV relative to the repo root, so the script works from any cwd.
csv_path = REPO_ROOT / "data" / "dogs_final_for_rag.csv"
if not csv_path.exists():
    sys.exit(f"❌ Dataset not found at {csv_path}")

df = pd.read_csv(csv_path)

# Initialize MiniLM embeddings model
embeddings_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

batch_size = 100
vectors = []

print(f"Processing {len(df)} documents...")

for i, row in df.iterrows():
    doc_text = str(row['Combined_Info'])
    metadata = row.to_dict()
    metadata.pop('Combined_Info', None)

    # Generate embedding vector
    vector = embeddings_model.embed_query(doc_text)

    # Prepare for Pinecone upsert
    vectors.append({
        'id': str(i),
        'values': vector,
        'metadata': metadata
    })

    # Upsert in batches
    if (i + 1) % batch_size == 0 or i == len(df) - 1:
        index.upsert(vectors=vectors)
        print(f"✅ Upserted batch ending at row {i+1}")
        vectors = []

print("✅ Successfully uploaded all vectors to Pinecone!")
print(f"Total records in index: {index.describe_index_stats()}")
