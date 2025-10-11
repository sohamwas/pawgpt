import streamlit as st
import requests

# Page config with title, favicon, and wide layout
st.set_page_config(
    page_title="PawGPT",
    page_icon="🐾",
    layout="wide"
)

# Sidebar for app info and controls
with st.sidebar:
    st.header("🐾 PawGPT")
    st.markdown(
        """
        Welcome! Ask anything about dog breeds using natural language.
        
        Features:
        - Personalized breed recommendations
        - Powered by Retrieval-Augmented Generation (RAG)
        """
    )
    if st.button("Clear Chat History"):
        st.session_state.messages = []

# Initialize chat history if not exists
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if not st.session_state["messages"]:
    st.markdown(
        "<h2 style='text-align: center; color:#8e43ed;'>🐾 Paws up! How can I assist you today?</h2>",
        unsafe_allow_html=True
    )

# Display the chat messages from history on the page
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"], avatar="🙂" if msg["role"] == "user" else "🐾"):
        st.markdown(msg["content"])

# User input box
user_input = st.chat_input("Ask about dog breeds...")

# When user submits a message
if user_input:
    # Append user message to chat history & render UI
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🙂"):
        st.markdown(user_input)

    # Prepare payload for backend Flask API
    payload = {"question": user_input}

    # Show spinner while waiting for response
    with st.spinner("PawGPT is thinking..."):
        try:
            response = requests.post("http://localhost:5000/ask", json=payload)
            response.raise_for_status()
            data = response.json()
            answer = data.get("result", "Sorry, no answer received.")
        except requests.exceptions.RequestException as e:
            answer = f"Error communicating with backend: {e}"

    # Append bot answer to chat history & render UI
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    with st.chat_message("assistant", avatar="🐾"):
        st.markdown(answer)

# Optional: Divider for cleaner layout separation
st.divider()

# Add a chat download button in sidebar
with st.sidebar:
    if st.button("Download Chat History"):
        chat_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state["messages"]])
        st.download_button(
            label="Download conversation as .txt",
            data=chat_text,
            file_name="pawgpt_chat_history.txt",
            mime="text/plain"
        )
