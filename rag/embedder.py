# rag/embedder.py
from sentence_transformers import SentenceTransformer


class BGEEmbedder:
    """
    Embedder using BAAI/bge-small-en-v1.5.

    BGE models require a query prefix for retrieval tasks:
      - Queries:    prefix with "Represent this sentence for searching relevant passages: "
      - Documents:  no prefix needed

    This distinction improves retrieval accuracy significantly for BGE models.
    """

    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        print("Embedding model loaded.")

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single search query.
        Applies BGE query prefix and L2 normalisation for cosine similarity.
        """
        prefixed = self.QUERY_PREFIX + query.strip()
        return self.model.encode(
            prefixed,
            normalize_embeddings=True   # unit vectors → cosine sim = dot product
        ).tolist()

    def embed_documents(self, documents: list[str], batch_size: int = 32) -> list[list[float]]:
        """
        Embed a list of documents in batches.
        No prefix for documents — BGE is asymmetric by design.
        Batching prevents OOM on large document sets.
        """
        if not documents:
            return []
        return self.model.encode(
            documents,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(documents) > 100
        ).tolist()
