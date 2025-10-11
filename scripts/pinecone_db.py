import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize Pinecone client with your API key
pc = Pinecone(api_key="YOUR_API_KEY_HERE")

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

# Load your CSV data
df = pd.read_csv("dogs_final_for_rag.csv")

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
