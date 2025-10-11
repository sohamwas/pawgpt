import os
import streamlit as st
import pandas as pd
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq


# Page config
st.set_page_config(page_title="PawGPT", page_icon="🐾", layout="wide")


@st.cache_resource(show_spinner=True)
def build_or_load_vectorstore():
    persist_directory = 'db_chroma'
    if not os.path.exists(persist_directory):
        st.info("--- Step 1: Load Data from the Final CSV using Pandas ---")
        file_path = 'data/dogs_final_for_rag.csv'
        try:
            df = pd.read_csv(file_path)
            if 'Combined_Info' not in df.columns:
                raise ValueError("'Combined_Info' column not found in the CSV.")

            documents = []
            for _, row in df.iterrows():
                page_content = str(row.get('Combined_Info', ''))
                metadata = row.to_dict()
                metadata.pop('Combined_Info', None)
                documents.append(Document(page_content=page_content, metadata=metadata))

            st.success(f"✅ Successfully loaded and processed {len(documents)} documents using pandas.")

        except FileNotFoundError:
            st.error(f"❌ Error: '{file_path}' not found. Please make sure the file path is correct.")
            st.stop()
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
            st.stop()

        st.info("\n--- Step 2: Initialize the Embedding Model ---")
        model_name = 'all-MiniLM-L6-v2'
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings_model = HuggingFaceEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
        st.success(f"✅ Embedding model '{model_name}' is ready.")

        st.info("\n--- Step 3: Create and Persist the Vector Store ---")
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings_model,
            persist_directory=persist_directory
        )
        vector_store.persist()
        st.success(f"✅ Successfully created and populated the ChromaDB vector store.")
        st.write(f"Total documents in store: {vector_store._collection.count()}")
        st.write(f"The database has been saved to the '{persist_directory}' directory.")
    else:
        embeddings_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings_model)
        st.success("✅ Vector store loaded from persistence.")

    return vector_store


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


def general_purpose_rag(query, vector_store, llm, max_docs=5, max_context_chars=2500):
    try:
        docs = vector_store.similarity_search(query, k=max_docs)
        if not docs:
            return "I couldn't find relevant info. The database might be empty or no matches found."

        combined_context = ""
        for doc in docs:
            doc_content = doc.page_content.strip()
            if len(combined_context) + len(doc_content) < max_context_chars:
                combined_context += doc_content + "\n\n"
            else:
                remaining_chars = max_context_chars - len(combined_context)
                if remaining_chars > 100:
                    combined_context += doc_content[:remaining_chars-10] + "...\n\n"
                break

        if not combined_context.strip():
            return "Found documents appear to be empty."

        prompt_template = """You are a helpful assistant answering questions based on the provided information.

Information:
{context}

Question: {question}

Instructions:
- Use only the information above
- Answer specifically and helpfully
- If info is insufficient, say "I don't have enough information to answer based on available data"
- Do not make up info

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
    """
)
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("<h2 style='text-align: center; color:#8e43ed;'>🐾 Paws up! How can I assist you today?</h2>", unsafe_allow_html=True)


# Load vector store and LLM
vector_store = build_or_load_vectorstore()
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
        answer = general_purpose_rag(user_input, vector_store, llm)

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
