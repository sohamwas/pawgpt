from flask import Flask, request, jsonify, render_template
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Global Variables (Load models once on startup) ---
vector_store = None
llm = None

def load_models():
    """Load the vector store and LLM once when the app starts."""
    global vector_store, llm
    if vector_store and llm: return

    print("--- Loading models... ---")
    
    # --- Load Vector Database ---
    model_name = 'all-MiniLM-L6-v2'
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})
    persist_directory = 'db_chroma'
    try:
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings_model)
        
        # Better way to check document count
        try:
            doc_count = len(vector_store.get()['ids'])
            print(f"✅ Vector store loaded with {doc_count} documents.")
            
            # If no documents, this is the problem
            if doc_count == 0:
                print("⚠️  WARNING: Vector store is empty! No documents found.")
                
        except Exception as count_error:
            print(f"⚠️  Could not get document count: {count_error}")
            
    except Exception as e:
        print(f"❌ FATAL ERROR: Could not load vector store: {e}")
        exit()

    # --- Set up Groq LLM ---
    try:
        # IMPORTANT: Replace with your actual Groq API key
        GROQ_API_KEY = "YOUR_GROQ_API_KEY" 
        os.environ["GROQ_API_KEY"] = GROQ_API_KEY
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=800)
        print("✅ LLM is ready via Groq API.")
    except Exception as e:
        print(f"❌ FATAL ERROR: Could not set up ChatGroq: {e}")
        exit()

def general_purpose_rag(query, vector_store, llm, max_docs=5, max_context_chars=2500):
    """
    General purpose RAG system with better error handling and debugging.
    """
    try:
        print(f"🔍 Searching for: '{query}'")
        
        # Try multiple search approaches
        # Method 1: Using similarity search directly
        try:
            docs = vector_store.similarity_search(query, k=max_docs)
            print(f"📄 Found {len(docs)} documents using similarity_search")
        except Exception as search_error:
            print(f"❌ similarity_search failed: {search_error}")
            # Method 2: Using retriever as fallback
            try:
                retriever = vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": max_docs}
                )
                docs = retriever.invoke(query)
                print(f"📄 Found {len(docs)} documents using retriever")
            except Exception as retriever_error:
                print(f"❌ Retriever also failed: {retriever_error}")
                return {"result": "Error: Could not search the database.", "source_documents": []}
        
        if not docs:
            print("⚠️  No documents found for the query")
            # Let's try a broader search
            try:
                # Get all documents to see what's available
                all_docs = vector_store.get()
                if all_docs and 'documents' in all_docs:
                    print(f"📊 Total documents in store: {len(all_docs['documents'])}")
                    if len(all_docs['documents']) > 0:
                        print(f"📝 Sample document preview: {all_docs['documents'][0][:200]}...")
                
                # Try a more relaxed search
                docs = vector_store.similarity_search(query, k=max_docs * 2)
                if not docs:
                    return {"result": "I couldn't find any relevant information for your question. The database might be empty or the search terms don't match any content.", "source_documents": []}
                    
            except Exception as broad_search_error:
                print(f"❌ Broad search failed: {broad_search_error}")
                return {"result": "Error: Database search failed.", "source_documents": []}
        
        # Debug: Print what we found
        print(f"✅ Retrieved {len(docs)} documents")
        for i, doc in enumerate(docs[:2]):  # Print first 2 docs
            preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
            print(f"📄 Doc {i+1} preview: {preview}")
        
        # Combine document contents, respecting the character limit
        combined_context = ""
        used_docs = []
        
        for doc in docs:
            doc_content = doc.page_content.strip()
            if len(combined_context) + len(doc_content) < max_context_chars:
                combined_context += doc_content + "\n\n"
                used_docs.append(doc_content)
            else:
                # Try to fit partial content
                remaining_chars = max_context_chars - len(combined_context)
                if remaining_chars > 100:  # Only if we have meaningful space left
                    combined_context += doc_content[:remaining_chars-10] + "...\n\n"
                    used_docs.append(doc_content[:remaining_chars-10] + "...")
                break
        
        if not combined_context.strip():
            return {"result": "Found documents but they appear to be empty.", "source_documents": []}
        
        print(f"📝 Context length: {len(combined_context)} characters")
        
        prompt_template = """You are a helpful assistant answering questions based on the provided information.

Information:
{context}

Question: {question}

Instructions:
- Answer using only the information provided above
- Be specific and helpful
- If the information doesn't contain what's needed to answer the question, say "I don't have enough information to answer that question based on the available data"
- Do not make up information

Answer:"""
        
        formatted_prompt = prompt_template.format(context=combined_context.strip(), question=query)
        
        print("🤖 Sending prompt to LLM...")
        response = llm.invoke(formatted_prompt)
        
        return {
            "result": response.content,
            "source_documents": used_docs
        }
        
    except Exception as e:
        print(f"❌ RAG Error: {str(e)}")
        return {"result": f"Error processing your question: {str(e)}", "source_documents": []}

@app.route('/')
def home():
    """Render the main HTML page."""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    """Handle the question submission from the frontend."""
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'error': 'No question provided'}), 400
            
        question = data.get('question').strip()
        if not question:
            return jsonify({'error': 'Empty question provided'}), 400
            
        print(f"🎯 Received question: {question}")
        
        result = general_purpose_rag(question, vector_store, llm)
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ API Error: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/debug', methods=['GET'])
def debug_info():
    """Debug endpoint to check the state of the vector store."""
    try:
        if not vector_store:
            return jsonify({'error': 'Vector store not loaded'})
            
        # Get basic info about the vector store
        all_data = vector_store.get()
        
        debug_info = {
            'total_documents': len(all_data.get('ids', [])),
            'sample_ids': all_data.get('ids', [])[:5],
            'sample_documents': [doc[:200] + "..." if len(doc) > 200 else doc 
                               for doc in all_data.get('documents', [])[:3]],
            'vector_store_type': type(vector_store).__name__
        }
        
        return jsonify(debug_info)
        
    except Exception as e:
        return jsonify({'error': f'Debug error: {str(e)}'}), 500

if __name__ == '__main__':
    load_models()
    app.run(host='0.0.0.0', port=5000, debug=True)