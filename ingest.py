import os
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# ─────────────────────────────────────────
# Read PDF
# ─────────────────────────────────────────
def read_pdf(file_path):
    doc = fitz.open(file_path)
    full_text = ""
    for page_number in range(len(doc)):
        full_text += doc[page_number].get_text()
    doc.close()
    return full_text

# ─────────────────────────────────────────
# Split into chunks
# ─────────────────────────────────────────
def split_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    return splitter.split_text(text)

# ─────────────────────────────────────────
# Embed and store in ChromaDB
# ─────────────────────────────────────────
def store_in_chromadb(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Creating embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True)

    client = chromadb.PersistentClient(path="chroma_db")
    try:
        client.delete_collection("my_documents")
    except:
        pass

    collection = client.create_collection("my_documents")
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    print(f"✓ Stored {len(chunks)} chunks in ChromaDB")


# ─────────────────────────────────────────
# RUN INGESTION
# ─────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # Accept PDF path as argument or use default
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "docs/genai-principles.pdf"

    print(f"Reading PDF: {file_path}")
    text = read_pdf(file_path)
    print(f"✓ Extracted {len(text)} characters")

    print("Splitting into chunks...")
    chunks = split_into_chunks(text)
    print(f"✓ Created {len(chunks)} chunks")

    print("Storing in ChromaDB...")
    store_in_chromadb(chunks)

    print("\n✅ Ingestion complete! You can now run app.py")