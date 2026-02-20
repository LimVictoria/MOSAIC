# rag/retriever.py
# ChromaDB retrieval — used by all 3 teaching agents

import chromadb
from chromadb.config import Settings
from config import (
    CHROMA_PERSIST_DIR,
    CHROMA_KNOWLEDGE_COLLECTION,
    CHROMA_ASSESSMENT_COLLECTION,
    RAG_TOP_K
)
from rag.embedder import BGEEmbedder


class RAGRetriever:
    """
    ChromaDB retriever.
    All three agents (Solver, Assessment, Feedback)
    call different convenience methods based on their needs.
    """

    def __init__(self, embedder: BGEEmbedder):
        self.embedder = embedder
        self.client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        self.knowledge_collection = self.client.get_or_create_collection(
            name=CHROMA_KNOWLEDGE_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
        self.assessment_collection = self.client.get_or_create_collection(
            name=CHROMA_ASSESSMENT_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )

    def retrieve(
        self,
        query: str,
        collection: str = "knowledge_base",
        top_k: int = RAG_TOP_K,
        topic_filter: str = None
    ) -> list[dict]:
        """Base retrieval method."""
        query_embedding = self.embedder.embed_query(query)
        col = (
            self.knowledge_collection
            if collection == "knowledge_base"
            else self.assessment_collection
        )
        where = {"topic_area": topic_filter} if topic_filter else None
        results = col.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        documents = []
        for i, doc in enumerate(results["documents"][0]):
            documents.append({
                "text": doc,
                "source": results["metadatas"][0][i].get("source", "unknown"),
                "topic": results["metadatas"][0][i].get("topic_area", "general"),
                "score": 1 - results["distances"][0][i]
            })
        return documents

    def retrieve_for_solver(self, query: str, topic: str = None) -> list[dict]:
        """Solver Agent — heavy retrieval from knowledge base."""
        return self.retrieve(query, "knowledge_base", top_k=5, topic_filter=topic)

    def retrieve_for_assessment(self, concept: str) -> list[dict]:
        """Assessment Agent — gets misconceptions and question material."""
        return self.retrieve(
            f"common misconceptions about {concept}",
            "assessment_bank",
            top_k=3
        )

    def retrieve_for_feedback(self, misconception: str) -> list[dict]:
        """Feedback Agent — gets explanation strategies for gaps."""
        return self.retrieve(
            f"how to explain {misconception} clearly",
            "knowledge_base",
            top_k=3
        )
