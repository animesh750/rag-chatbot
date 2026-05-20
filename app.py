import os
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fpdf import FPDF
from datetime import datetime

# ─────────────────────────────────────────
# Load API key
# ─────────────────────────────────────────
def load_env():
    try:
        if st.secrets and "GROQ_API_KEY" in st.secrets:
            os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
            return
    except Exception:
        pass
    try:
        with open(".env") as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()
    except FileNotFoundError:
        pass

# ─────────────────────────────────────────
# Load embedding model once
# ─────────────────────────────────────────
@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ─────────────────────────────────────────
# FAISS index stored in session state
# ─────────────────────────────────────────
def get_faiss_store():
    if "faiss_index" not in st.session_state:
        st.session_state.faiss_index    = None
        st.session_state.faiss_chunks   = []
        st.session_state.faiss_metas    = []
    return (
        st.session_state.faiss_index,
        st.session_state.faiss_chunks,
        st.session_state.faiss_metas,
    )

def reset_faiss_store():
    st.session_state.faiss_index  = None
    st.session_state.faiss_chunks = []
    st.session_state.faiss_metas  = []

# ─────────────────────────────────────────
# Process PDF and add to FAISS
# ─────────────────────────────────────────
def process_and_add_pdf(uploaded_file):
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, length_function=len
    )
    chunks = splitter.split_text(full_text)

    embed_model = load_embed_model()
    embeddings = embed_model.encode(chunks, show_progress_bar=False).astype("float32")
    faiss.normalize_L2(embeddings)

    _, existing_chunks, existing_metas = get_faiss_store()

    if st.session_state.faiss_index is None:
        dim = embeddings.shape[1]
        st.session_state.faiss_index = faiss.IndexFlatIP(dim)

    st.session_state.faiss_index.add(embeddings)
    st.session_state.faiss_chunks.extend(chunks)
    st.session_state.faiss_metas.extend(
        [{"source": uploaded_file.name}] * len(chunks)
    )
    return len(chunks)

# ─────────────────────────────────────────
# Remove PDF from FAISS store
# ─────────────────────────────────────────
def remove_pdf(filename):
    chunks = st.session_state.faiss_chunks
    metas  = st.session_state.faiss_metas

    keep_idx = [i for i, m in enumerate(metas) if m["source"] != filename]

    if not keep_idx:
        reset_faiss_store()
        return

    kept_chunks = [chunks[i] for i in keep_idx]
    kept_metas  = [metas[i]  for i in keep_idx]

    embed_model = load_embed_model()
    embeddings  = embed_model.encode(kept_chunks, show_progress_bar=False).astype("float32")
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    new_index = faiss.IndexFlatIP(dim)
    new_index.add(embeddings)

    st.session_state.faiss_index  = new_index
    st.session_state.faiss_chunks = kept_chunks
    st.session_state.faiss_metas  = kept_metas

# ─────────────────────────────────────────
# Search FAISS
# ─────────────────────────────────────────
def search(question, n_results=4):
    index, chunks, metas = get_faiss_store()
    if index is None or len(chunks) == 0:
        return [], []

    embed_model = load_embed_model()
    q_emb = embed_model.encode([question]).astype("float32")
    faiss.normalize_L2(q_emb)

    n = min(n_results, len(chunks))
    _, indices = index.search(q_emb, n)

    result_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
    result_metas  = [metas[i]  for i in indices[0] if i < len(metas)]
    return result_chunks, result_metas

# ─────────────────────────────────────────
# Ask LLM with memory
# ─────────────────────────────────────────
def ask_llm(question, context_chunks, context_metas, chat_history):
    if not context_chunks:
        return "I couldn't find any relevant information in the uploaded documents.", 0

    context_parts = []
    for chunk, meta in zip(context_chunks, context_metas):
        source = meta.get("source", "unknown")
        context_parts.append(f"[From: {source}]\n{chunk}")
    context = "\n\n---\n\n".join(context_parts)

    history_text = ""
    for msg in chat_history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    system_prompt = """You are a helpful assistant that answers questions about uploaded documents.
Rules:
- Answer using ONLY the provided context
- Mention which document information came from when relevant
- If the answer is not in the context, say "I couldn't find that in the uploaded documents"
- Handle follow-up questions using the conversation history
- Keep answers clear and well-structured"""

    user_prompt = f"""DOCUMENT CONTEXT:
{context}

CONVERSATION HISTORY:
{history_text}
User: {question}

Answer using the context and conversation history above."""

    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=1024
    )
    return response.choices[0].message.content, response.usage.total_tokens

# ─────────────────────────────────────────
# Export chat to PDF
# ─────────────────────────────────────────
def export_chat_to_pdf(messages, doc_names):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    def clean(text):
        return text.encode("latin-1", "ignore").decode("latin-1")

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, clean("RAG Chatbot - Conversation Export"), ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, clean(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}"), ln=True)
    pdf.cell(0, 6, clean(f"Documents: {', '.join(doc_names)}"), ln=True)
    pdf.ln(4)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    for msg in messages:
        if msg["role"] == "user":
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(60, 120, 220)
            pdf.cell(0, 8, "You:", ln=True)
        else:
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(40, 160, 80)
            pdf.cell(0, 8, "Assistant:", ln=True)

        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(30, 30, 30)
        pdf.set_x(10)
        pdf.multi_cell(190, 7, clean(msg["content"]))
        pdf.ln(3)

        if "sources" in msg and msg["sources"]:
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(100, 100, 100)
            for i, (chunk, meta) in enumerate(
                zip(msg["sources"], msg.get("source_metas", []))
            ):
                src = meta.get("source", "unknown") if meta else "unknown"
                snippet = chunk[:100].replace("\n", " ")
                pdf.set_x(10)
                pdf.multi_cell(190, 5, clean(f"Src {i+1} [{src[:30]}]: {snippet}"))
            pdf.ln(2)

        pdf.set_draw_color(230, 230, 230)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

    return bytes(pdf.output())

# ─────────────────────────────────────────
# PAGE CONFIG + CSS
# ─────────────────────────────────────────
st.set_page_config(page_title="RAG Chatbot", page_icon="📄", layout="wide")
load_env()

st.markdown("""
<style>
.source-card {
    background: rgba(79,139,249,0.08);
    border-left: 3px solid #4f8bf9;
    padding: 10px 14px;
    border-radius: 4px;
    margin: 6px 0;
    font-size: 0.85rem;
    color: #ccc;
}
.doc-badge {
    display: inline-block;
    background: rgba(79,139,249,0.15);
    border: 1px solid #4f8bf9;
    color: #4f8bf9;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.78rem;
    margin: 3px 2px;
}
.doc-badge-green {
    display: inline-block;
    background: rgba(40,167,69,0.15);
    border: 1px solid #28a745;
    color: #28a745;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.78rem;
    margin: 3px 2px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if "messages"         not in st.session_state: st.session_state.messages         = []
if "uploaded_docs"    not in st.session_state: st.session_state.uploaded_docs    = {}
if "total_tokens"     not in st.session_state: st.session_state.total_tokens     = 0
if "pending_question" not in st.session_state: st.session_state.pending_question = None
if "faiss_index"      not in st.session_state: st.session_state.faiss_index      = None
if "faiss_chunks"     not in st.session_state: st.session_state.faiss_chunks     = []
if "faiss_metas"      not in st.session_state: st.session_state.faiss_metas      = []

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.title("📄 RAG Chatbot")
    st.caption("Chat with multiple documents")
    st.divider()

    st.header("📤 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Add one or more PDFs", type="pdf", accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.uploaded_docs:
                with st.spinner(f"Indexing {uploaded_file.name}..."):
                    n_chunks = process_and_add_pdf(uploaded_file)
                    st.session_state.uploaded_docs[uploaded_file.name] = n_chunks
                st.success(f"✅ {uploaded_file.name} — {n_chunks} chunks")

    st.divider()

    if st.session_state.uploaded_docs:
        st.header("📚 Loaded Documents")
        for filename, n_chunks in list(st.session_state.uploaded_docs.items()):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(
                    f'<div class="doc-badge-green">📄 {filename[:22]}{"..." if len(filename)>22 else ""}</div>',
                    unsafe_allow_html=True
                )
                st.caption(f"{n_chunks} chunks")
            with col2:
                if st.button("🗑", key=f"del_{filename}"):
                    remove_pdf(filename)
                    del st.session_state.uploaded_docs[filename]
                    st.session_state.messages = []
                    st.rerun()
        st.divider()

    total_docs   = len(st.session_state.uploaded_docs)
    total_chunks = len(st.session_state.faiss_chunks)

    col1, col2 = st.columns(2)
    col1.metric("Documents", total_docs)
    col2.metric("Chunks", total_chunks)
    st.metric("Tokens used", f"{st.session_state.total_tokens:,}")

    st.divider()
    st.caption("🧠 LLaMA 3.3 70B via Groq")
    st.caption("📦 all-MiniLM-L6-v2 embeddings")
    st.caption("🗄️ FAISS vector store")
    st.caption("💬 Conversation memory: 3 turns")
    st.divider()

    if st.session_state.messages:
        pdf_bytes = export_chat_to_pdf(
            st.session_state.messages,
            list(st.session_state.uploaded_docs.keys())
        )
        st.download_button(
            label="📥 Export chat as PDF",
            data=pdf_bytes,
            file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        st.divider()

    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages     = []
        st.session_state.total_tokens = 0
        st.rerun()

    if st.button("🔄 Reset everything", use_container_width=True):
        reset_faiss_store()
        st.session_state.messages         = []
        st.session_state.uploaded_docs    = {}
        st.session_state.total_tokens     = 0
        st.session_state.pending_question = None
        st.rerun()

# ─────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────
st.title("📄 Chat with your PDFs")

if total_docs == 0:
    st.info("👈 Upload one or more PDFs in the sidebar to get started!")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📂 Multi-Document")
        st.caption("Upload several PDFs and ask questions across all of them.")
    with col2:
        st.markdown("### 🧠 Smart Memory")
        st.caption("Follow-up questions work — it remembers the full conversation.")
    with col3:
        st.markdown("### 📥 Export Chat")
        st.caption("Download the full conversation as a PDF at any time.")
    st.stop()

doc_badges = " ".join([
    f'<span class="doc-badge">📄 {name[:20]}{"..." if len(name)>20 else ""}</span>'
    for name in st.session_state.uploaded_docs.keys()
])
st.markdown(f"**Searching across:** {doc_badges}", unsafe_allow_html=True)
st.divider()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 View Sources"):
                for i, (chunk, meta) in enumerate(
                    zip(message["sources"], message.get("source_metas", []))
                ):
                    source_name = meta.get("source", "unknown") if meta else "unknown"
                    st.markdown(
                        f'<div class="source-card">'
                        f'<b>Source {i+1}</b> · <code>{source_name}</code><br><br>'
                        f'{chunk[:280]}{"..." if len(chunk)>280 else ""}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

if len(st.session_state.messages) == 0:
    with st.chat_message("assistant"):
        doc_list = ", ".join(f"**{n}**" for n in st.session_state.uploaded_docs)
        st.write(f"Hello! I've indexed {total_docs} document(s): {doc_list}. Ask me anything!")

    st.markdown("**💡 Try asking:**")
    suggestions = [
        "Summarise this document",
        "What are the main topics?",
        "What are the key findings?",
        "Who is the author?",
        "List the most important points",
    ]
    cols = st.columns(len(suggestions))
    for i, suggestion in enumerate(suggestions):
        if cols[i].button(suggestion, key=f"sug_{i}"):
            st.session_state.pending_question = suggestion
            st.rerun()

# Handle question
question = None
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None
elif prompt := st.chat_input("Ask anything across your PDFs..."):
    question = prompt

if question:
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            chunks, metas = search(question)
            answer, tokens = ask_llm(
                question, chunks, metas,
                st.session_state.messages[:-1]
            )
            st.session_state.total_tokens += tokens

        st.write(answer)

        if chunks:
            with st.expander("📚 View Sources"):
                for i, (chunk, meta) in enumerate(zip(chunks, metas)):
                    source_name = meta.get("source", "unknown")
                    st.markdown(
                        f'<div class="source-card">'
                        f'<b>Source {i+1}</b> · <code>{source_name}</code><br><br>'
                        f'{chunk[:280]}{"..." if len(chunk)>280 else ""}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

    st.session_state.messages.append({
        "role":         "assistant",
        "content":      answer,
        "sources":      chunks,
        "source_metas": metas
    })
