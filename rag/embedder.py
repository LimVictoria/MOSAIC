# rag/embedder.py
from FlagEmbedding import FlagModel
import os

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")

class BGEEmbedder:
    def __init__(self):
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.model = FlagModel(
            EMBEDDING_MODEL,
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
            use_fp16=True
        )
        print("Embedding model loaded.")

    def embed_query(self, query: str) -> list[float]:
        return self.model.encode_queries([query])[0].tolist()

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return self.model.encode(documents).tolist()
