# rag/embedder.py
# BGE-large-en-v1.5 embedding model

from FlagEmbedding import FlagModel
from config import EMBEDDING_MODEL


class BGEEmbedder:
    """
    BGE-large-en-v1.5 embedding model.
    Used by all three agents to embed queries
    before searching ChromaDB.
    """

    def __init__(self):
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.model = FlagModel(
            EMBEDDING_MODEL,
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
            use_fp16=True
        )
        print("Embedding model loaded.")

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        embedding = self.model.encode_queries([query])
        return embedding[0].tolist()

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed a list of documents for storage."""
        embeddings = self.model.encode(documents)
        return embeddings.tolist()
