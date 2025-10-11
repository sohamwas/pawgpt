import os
import streamlit as st
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


# Page config
st.set_page_config(page_title="PawGPT", page_icon="🐾", layout="wide")


@st.cache_resource(show_spinner=True)
def load_pinecone_index():
    try:
        pinecone_api_key = st.secrets["pinecone_api_key"]
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index("pawgpt")
        st.success("✅ Pinecone index connected successfully.")
        return index
    except Exception as e:
        st.error(f"❌ Could not connect to Pinecone: {e}")
        st.stop()


@st.cache_resource(show_spinner=True)
def load_embeddings():
    embeddings_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2", 
        model_kwargs={'device': 'cpu'}
    )
    st.success("✅ Embedding model loaded.")
    return embeddings_model


@st.cache_resource(show_spinner=True)
def load_llm():
    try:
        groq_api_key = st.secrets["groq_api_key"]
        os.environ["GROQ_API_KEY"] = groq_api_key
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=800)
        st.success("✅ LLM is ready via Groq API.")
        return llm
    except Exception as e:
        st.error(f"❌ Could not set up ChatGroq: {e}")
        st.stop()


def query_pinecone_rag(query, index, embeddings_model, llm, top_k=5):
    try:
        # Generate query embedding
        query_vector = embeddings_model.embed_query(query)
        
        # Query Pinecone
        results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        
        if not results['matches']:
            return "I couldn't find relevant information. Please try rephrasing your question."
        
        # Build context from retrieved documents
        combined_context = ""
        for match in results['matches']:
            metadata = match.get('metadata', {})
            # Reconstruct document text from metadata or use specific fields
            doc_text = str(metadata)  # You can customize based on your metadata structure
            combined_context += doc_text + "\n\n"
        
        if not combined_context.strip():
            return "Found documents but they appear to be empty."
        
        # Create prompt for LLM
        prompt_template = """You are a helpful assistant answering questions about dog breeds based on the provided information.

Information:
{context}

Question: {question}

Instructions:
- Use only the information provided above
- Answer specifically and helpfully
- If information is insufficient, say "I don't have enough information to answer based on available data"
- Do not make up information

Answer:"""
        
        formatted_prompt = prompt_template.format(context=combined_context.strip(), question=query)
        response = llm.invoke(formatted_prompt)
        return response.content
        
    except Exception as e:
        return f"Error processing your question: {str(e)}"


# Sidebar UI
st.sidebar.header("🐾 PawGPT")
st.sidebar.markdown(
    """
    Welcome! Ask anything about dog breeds using natural language.
    
    Features:
    - Personalized breed recommendations
    - Powered by Retrieval-Augmented Generation (RAG)
    - Vector database hosted on Pinecone
    """
)
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("<h2 style='text-align: center; color:#8e43ed;'>🐾 Paws up! How can I assist you today?</h2>", unsafe_allow_html=True)


# Load resources
pinecone_index = load_pinecone_index()
embeddings_model = load_embeddings()
llm = load_llm()

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🙂" if msg["role"] == "user" else "🐾"):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about dog breeds...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🙂"):
        st.markdown(user_input)

    with st.spinner("PawGPT is thinking..."):
        answer = query_pinecone_rag(user_input, pinecone_index, embeddings_model, llm)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant", avatar="🐾"):
        st.markdown(answer)

st.divider()

# Download chat history
with st.sidebar:
    if st.button("Download Chat History"):
        chat_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
        st.download_button(
            label="Download conversation as .txt",
            data=chat_text,
            file_name="pawgpt_chat_history.txt",
            mime="text/plain"
        )

