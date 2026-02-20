# rag/embedder.py
from sentence_transformers import SentenceTransformer

class BGEEmbedder:
    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        print("Embedding model loaded.")

    def embed_query(self, query: str) -> list[float]:
        return self.model.encode(query).tolist()

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return self.model.encode(documents).tolist()
