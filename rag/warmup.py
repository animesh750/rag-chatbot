"""Download model weights ahead of time (used while building the Docker image)."""

from .embeddings import CrossEncoderReranker, SentenceTransformerEmbedder

if __name__ == "__main__":
    SentenceTransformerEmbedder()._load()
    CrossEncoderReranker()._load()
    print("models cached")
