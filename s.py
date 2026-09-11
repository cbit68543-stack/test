import streamlit as st

from rag_local import (
    load_documents,
    build_chunk_db,
    ask_question
)


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Private RAG",
    page_icon="🤖",
    layout="wide"
)


# -----------------------------
# Title
# -----------------------------

st.title("🤖 Private RAG with Qwen2.5")
st.write("Ask questions about your local PDF and DOCX documents.")


# -----------------------------
# Load documents
# -----------------------------

@st.cache_resource
def initialize_rag():
    documents = load_documents()

    if not documents:
        return None

    chunk_db = build_chunk_db(documents)

    return chunk_db


chunk_db = initialize_rag()


# -----------------------------
# Check documents
# -----------------------------

if chunk_db is None:

    st.warning(
        "No documents found. Please put PDF or DOCX files inside the 'data' folder."
    )

    st.stop()


# -----------------------------
# Show document information
# -----------------------------

st.success(
    f"Loaded {len(chunk_db['doc_names'])} document(s)"
)

with st.expander("📚 View loaded documents"):

    for document in chunk_db["doc_names"]:
        st.write(f"• {document}")


# -----------------------------
# Question input
# -----------------------------

question = st.text_input(
    "📝 Ask a question",
    placeholder="Example: What is Python?"
)


# -----------------------------
# Ask button
# -----------------------------

if st.button("Ask 🤖"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching documents and asking Qwen2.5..."):

            answer = ask_question(
                chunk_db,
                question
            )

        st.subheader("💡 Answer")

        st.write(answer)