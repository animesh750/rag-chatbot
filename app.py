"""Streamlit UI for the RAG chatbot (hybrid search + reranking + streaming + citations)."""

import os
from datetime import datetime

import streamlit as st

from rag import config
from rag.embeddings import CrossEncoderReranker, SentenceTransformerEmbedder
from rag.export import export_chat_to_pdf
from rag.llm import GroqLLM, LLMError
from rag.loader import PDFError, load_pdf_bytes
from rag.pipeline import NOT_FOUND, RAGPipeline
from rag.retriever import HybridRetriever

st.set_page_config(page_title="RAG Chatbot", page_icon="📄", layout="wide")

# ─────────────────────────────────────────
# API key: Streamlit secrets first, then .env / environment (loaded by rag.config)
# ─────────────────────────────────────────
try:
    if st.secrets and "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

st.markdown(
    """
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
.score-line { font-size: 0.75rem; color: #8aa4d6; margin-top: 6px; }
.doc-badge {
    display: inline-block; background: rgba(79,139,249,0.15); border: 1px solid #4f8bf9;
    color: #4f8bf9; padding: 3px 10px; border-radius: 12px; font-size: 0.78rem; margin: 3px 2px;
}
.doc-badge-green {
    display: inline-block; background: rgba(40,167,69,0.15); border: 1px solid #28a745;
    color: #28a745; padding: 3px 10px; border-radius: 12px; font-size: 0.78rem; margin: 3px 2px;
}
</style>
""",
    unsafe_allow_html=True,
)

MODE_LABELS = {
    "hybrid": "Hybrid (semantic + keyword)",
    "dense": "Semantic only",
    "bm25": "Keyword only (BM25)",
}


# ─────────────────────────────────────────
# Models are loaded once per server, the index is per browser session
# ─────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder():
    return SentenceTransformerEmbedder()


@st.cache_resource
def get_reranker():
    return CrossEncoderReranker()


def get_pipeline() -> RAGPipeline:
    if "retriever" not in st.session_state:
        st.session_state.retriever = HybridRetriever(get_embedder(), get_reranker())
    return RAGPipeline(st.session_state.retriever, GroqLLM())


def render_sources(hits):
    for i, h in enumerate(hits, start=1):
        bits = [f"score {h.score:.3f}"]
        if h.dense_rank:
            bits.append(f"semantic #{h.dense_rank}")
        if h.bm25_rank:
            bits.append(f"keyword #{h.bm25_rank}")
        if h.rerank_score is not None:
            bits.append(f"rerank {h.rerank_score:.2f}")
        text = h.chunk.text
        st.markdown(
            f'<div class="source-card"><b>[{i}]</b> · <code>{h.chunk.source}</code> · page {h.chunk.page}'
            f"<br><br>{text[:280]}{'...' if len(text) > 280 else ''}"
            f'<div class="score-line">{" · ".join(bits)}</div></div>',
            unsafe_allow_html=True,
        )


def render_meta(meta):
    if not meta:
        return
    t = meta["timings"]
    stages = " · ".join(f"{k.removesuffix('_ms')} {v:.0f} ms" for k, v in t.items() if k != "total_ms")
    st.caption(f"⚡ {stages} · mode: {MODE_LABELS[meta['mode']]}{' + rerank' if meta['reranked'] else ''}")
    if meta["standalone"] != meta["question"]:
        st.caption(f"🔎 Searched for: _{meta['standalone']}_")


# ─────────────────────────────────────────
# Session state
# ─────────────────────────────────────────
defaults = {
    "messages": [],
    "uploaded_docs": {},
    "total_tokens": 0,
    "pending_question": None,
    "uploader_key": 0,
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)

pipeline = get_pipeline()
retriever = pipeline.retriever

# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    st.title("📄 RAG Chatbot")
    st.caption("Chat with multiple documents")
    st.divider()

    st.header("📤 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Add one or more PDFs",
        type="pdf",
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )
    for uploaded_file in uploaded_files or []:
        if uploaded_file.name in st.session_state.uploaded_docs:
            continue
        with st.spinner(f"Indexing {uploaded_file.name}..."):
            try:
                chunks = load_pdf_bytes(uploaded_file.getvalue(), uploaded_file.name)
                n_chunks = retriever.add_document(uploaded_file.name, chunks)
                st.session_state.uploaded_docs[uploaded_file.name] = n_chunks
                st.success(f"✅ {uploaded_file.name} — {n_chunks} chunks")
            except PDFError as exc:
                st.error(str(exc))

    st.divider()

    if st.session_state.uploaded_docs:
        st.header("📚 Loaded Documents")
        for filename, n_chunks in list(st.session_state.uploaded_docs.items()):
            col1, col2 = st.columns([3, 1])
            with col1:
                short = filename[:22] + ("..." if len(filename) > 22 else "")
                st.markdown(f'<div class="doc-badge-green">📄 {short}</div>', unsafe_allow_html=True)
                st.caption(f"{n_chunks} chunks")
            with col2:
                if st.button("🗑", key=f"del_{filename}"):
                    retriever.remove_source(filename)
                    del st.session_state.uploaded_docs[filename]
                    st.session_state.messages = []
                    st.session_state.uploader_key += 1  # clear the widget so the file isn't re-indexed
                    st.rerun()
        st.divider()

    st.header("⚙️ Retrieval settings")
    mode = st.radio(
        "Search mode",
        list(MODE_LABELS),
        format_func=MODE_LABELS.get,
        index=list(MODE_LABELS).index(config.DEFAULT_MODE),
    )
    rerank = st.toggle(
        "Cross-encoder reranking",
        value=config.DEFAULT_RERANK,
        help="Re-scores the top candidates with a more accurate (slower) model.",
    )
    k = st.slider("Passages sent to the LLM", 2, 8, config.DEFAULT_K)
    rewrite = st.toggle(
        "Rewrite follow-up questions",
        value=True,
        help="Turns 'what about its limits?' into a standalone search query.",
    )
    scope = st.multiselect(
        "Limit to documents",
        list(st.session_state.uploaded_docs),
        help="Leave empty to search all documents.",
    )
    st.divider()

    total_docs = len(st.session_state.uploaded_docs)
    col1, col2 = st.columns(2)
    col1.metric("Documents", total_docs)
    col2.metric("Chunks", len(retriever))
    st.metric("Tokens used", f"{st.session_state.total_tokens:,}")

    st.divider()
    st.caption(f"🧠 {config.LLM_MODEL} via Groq")
    st.caption(f"📦 {config.EMBED_MODEL} embeddings")
    st.caption("🗄️ FAISS + BM25 hybrid index")
    st.caption("💬 Conversation memory: 3 turns")
    st.divider()

    if st.session_state.messages:
        st.download_button(
            label="📥 Export chat as PDF",
            data=export_chat_to_pdf(st.session_state.messages, list(st.session_state.uploaded_docs)),
            file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        st.divider()

    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_tokens = 0
        st.rerun()

    if st.button("🔄 Reset everything", use_container_width=True):
        retriever.clear()
        st.session_state.messages = []
        st.session_state.uploaded_docs = {}
        st.session_state.total_tokens = 0
        st.session_state.pending_question = None
        st.session_state.uploader_key += 1
        st.rerun()

# ─────────────────────────────────────────
# Main area
# ─────────────────────────────────────────
st.title("📄 Chat with your PDFs")

if total_docs == 0:
    st.info("👈 Upload one or more PDFs in the sidebar to get started!")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🔀 Hybrid Search")
        st.caption("Semantic + keyword search fused with RRF, then reranked for precision.")
    with col2:
        st.markdown("### 📑 Page Citations")
        st.caption("Every answer cites the document and page it came from.")
    with col3:
        st.markdown("### 🧠 Smart Memory")
        st.caption("Follow-ups work: they're rewritten into standalone queries.")
    st.stop()

doc_badges = " ".join(
    f'<span class="doc-badge">📄 {name[:20]}{"..." if len(name) > 20 else ""}</span>'
    for name in st.session_state.uploaded_docs
)
st.markdown(f"**Searching across:** {doc_badges}", unsafe_allow_html=True)
st.divider()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("sources"):
            with st.expander("📚 View Sources"):
                render_sources(message["sources"])
        render_meta(message.get("meta"))

if not st.session_state.messages:
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

question = None
if st.session_state.pending_question:
    question, st.session_state.pending_question = st.session_state.pending_question, None
elif prompt := st.chat_input("Ask anything across your PDFs..."):
    question = prompt

if question:
    history = list(st.session_state.messages)
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents..."):
                prep = pipeline.prepare(
                    question, history, mode=mode, rerank=rerank, k=k, sources=scope or None, rewrite=rewrite
                )
            if prep.messages is None:
                answer, tokens = NOT_FOUND, prep.tokens
                st.write(answer)
            else:
                stream = pipeline.stream(prep)
                answer = st.write_stream(stream)
                tokens = prep.tokens + stream.total_tokens
        except LLMError as exc:
            st.error(f"LLM error: {exc}")
            st.session_state.messages.pop()  # don't keep an unanswered question in the history
            st.stop()

        st.session_state.total_tokens += tokens
        if prep.hits:
            with st.expander("📚 View Sources"):
                render_sources(prep.hits)
        meta = {
            "timings": prep.timings,
            "mode": mode,
            "reranked": prep.reranked,
            "standalone": prep.standalone_question,
            "question": question,
        }
        render_meta(meta)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": prep.hits, "meta": meta}
    )
