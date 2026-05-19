import os
import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─────────────────────────────────────────
# Load API key
# ─────────────────────────────────────────
def load_env():
    # On Streamlit Cloud — use st.secrets
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    else:
        # On local machine — use .env file
        try:
            with open(".env") as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        key, value = line.split("=", 1)
                        os.environ[key.strip()] = value.strip()
        except FileNotFoundError:
            pass

load_env()
# ─────────────────────────────────────────
# Load embedding model once
# ─────────────────────────────────────────
@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ─────────────────────────────────────────
# Process uploaded PDF
# ─────────────────────────────────────────
def process_pdf(uploaded_file):
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = splitter.split_text(full_text)

    embed_model = load_embed_model()
    embeddings = embed_model.encode(chunks).tolist()

    client = chromadb.PersistentClient(path="chroma_db")
    try:
        client.delete_collection("my_documents")
    except:
        pass
    collection = client.create_collection("my_documents")
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    return collection, len(chunks)

# ─────────────────────────────────────────
# Search — now returns chunks WITH their IDs
# ─────────────────────────────────────────
def search(question, collection, n_results=3):
    embed_model = load_embed_model()
    question_embedding = embed_model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=n_results
    )
    # Return both the text chunks and their IDs
    chunks = results["documents"][0]
    ids    = results["ids"][0]
    return chunks, ids

# ─────────────────────────────────────────
# Ask LLM
# ─────────────────────────────────────────
def ask_llm(question, context_chunks):
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant. Answer the user's question
using ONLY the context provided below. If the answer is not in the
context, say "I couldn't find that in the document."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ─────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Chat with your PDF")
st.caption("Powered by LLaMA 3.3 + ChromaDB + Sentence Transformers — 100% Free")

# ── Sidebar ──
with st.sidebar:
    st.header("Upload your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        if "processed_file" not in st.session_state or \
           st.session_state.processed_file != uploaded_file.name:
            with st.spinner("Reading and indexing your PDF..."):
                collection, num_chunks = process_pdf(uploaded_file)
                st.session_state.collection = collection
                st.session_state.processed_file = uploaded_file.name
                st.session_state.num_chunks = num_chunks

        st.success(f"✅ {uploaded_file.name}")
        st.caption(f"{st.session_state.num_chunks} chunks indexed")
        st.divider()
        st.caption("Model: LLaMA 3.3 70B")
        st.caption("Embeddings: all-MiniLM-L6-v2")
        st.caption("Vector DB: ChromaDB")

        if st.button("🗑️ Clear chat"):
            st.session_state.messages = []
            st.rerun()

# ── Main chat area ──
if "messages" not in st.session_state:
    st.session_state.messages = []

if "collection" not in st.session_state:
    st.info("👈 Upload a PDF in the sidebar to get started!")
    st.stop()

# Display full chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        # Show citations if they exist for this message
        if "sources" in message and message["sources"]:
            with st.expander("📚 View Sources"):
                for i, source in enumerate(message["sources"]):
                    st.caption(f"**Source {i+1}:**")
                    st.info(source[:300] + "..." if len(source) > 300 else source)

# Greeting on empty chat
if len(st.session_state.messages) == 0:
    with st.chat_message("assistant"):
        st.write("Hello! I've read your PDF. Ask me anything about it!")

# Chat input
if question := st.chat_input("Ask a question about your PDF..."):

    # Show user message
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Get answer + sources
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            relevant_chunks, chunk_ids = search(
                question,
                st.session_state.collection
            )
            answer = ask_llm(question, relevant_chunks)

        # Show the answer
        st.write(answer)

        # Show citations in an expandable section
        with st.expander("📚 View Sources"):
            for i, (source, chunk_id) in enumerate(
                zip(relevant_chunks, chunk_ids)
            ):
                st.caption(f"**Source {i+1}** — `{chunk_id}`")
                st.info(
                    source[:300] + "..."
                    if len(source) > 300
                    else source
                )

    # Save message with sources attached
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": relevant_chunks
    })