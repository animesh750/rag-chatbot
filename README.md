# 📄 RAG Chatbot

A production-ready Retrieval-Augmented Generation (RAG) chatbot that lets you chat with your PDF documents using free, open-source tools.

## 🚀 Live Demo

[View on Streamlit Cloud](https://your-app-url.streamlit.app)

## ✨ Features

- Upload multiple PDFs and search across all of them
- Semantic search using vector embeddings
- Conversation memory — follow-up questions work naturally
- Source citations for every answer
- Export full chat history as PDF
- 100% free — no OpenAI costs

## 🛠️ Tech Stack

| Layer       | Tool                                     |
| ----------- | ---------------------------------------- |
| LLM         | LLaMA 3.3 70B via Groq API               |
| Embeddings  | all-MiniLM-L6-v2 (sentence-transformers) |
| Vector DB   | ChromaDB                                 |
| PDF parsing | PyMuPDF (fitz)                           |
| Chunking    | LangChain RecursiveCharacterTextSplitter |
| Frontend    | Streamlit                                |

## 📦 Setup

```bash
git clone https://github.com/animesh750/rag-chatbot
cd rag-chatbot
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Create a `.env` file:
