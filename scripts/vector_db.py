# ==============================================================================
# File: 1_create_vector_db.py
# Purpose: Creates the ChromaDB vector database from your final data file.
# Input: dogs_final_for_rag.csv
# Output: A folder named 'db_chroma' containing the vector store.
# ==============================================================================
import pandas as pd
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

print("--- Step 1: Load Data from the Final CSV using Pandas ---")
try:
    # Using pandas to read the CSV file
    file_path = 'dogs_final_for_rag.csv'
    df = pd.read_csv(file_path)

    # Ensure the required column exists
    if 'Combined_Info' not in df.columns:
        raise ValueError("'Combined_Info' column not found in the CSV.")

    # Convert the pandas DataFrame into a list of LangChain Document objects
    documents = []
    for index, row in df.iterrows():
        # The main content for the embedding is from the 'Combined_Info' column
        page_content = str(row.get('Combined_Info', ''))
        # We can store the rest of the row's data as metadata
        metadata = row.to_dict()
        # Remove the main content from metadata to avoid duplication
        metadata.pop('Combined_Info', None)
        
        documents.append(Document(page_content=page_content, metadata=metadata))

    print(f"✅ Successfully loaded and processed {len(documents)} documents using pandas.")
except FileNotFoundError:
    print(f"❌ Error: '{file_path}' not found. Please make sure the file path is correct.")
    exit()
except Exception as e:
    print(f"❌ An error occurred: {e}")
    exit()


print("\n--- Step 2: Initialize the Embedding Model ---")
model_name = 'all-MiniLM-L6-v2'
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings_model = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
print(f"✅ Embedding model '{model_name}' is ready.")


print("\n--- Step 3: Create and Persist the Vector Store ---")
persist_directory = 'db_chroma'
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings_model,
    persist_directory=persist_directory
)
print(f"✅ Successfully created and populated the ChromaDB vector store.")
print(f"Total documents in store: {vector_store._collection.count()}")
print(f"The database has been saved to the '{persist_directory}' directory.")
