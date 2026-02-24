# rag/retriever.py
# Pinecone vector store retrieval — used by all 3 teaching agents

import os
from pinecone import Pinecone, ServerlessSpec
from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    RAG_TOP_K
)
from rag.embedder import BGEEmbedder


class RAGRetriever:
    """
    Pinecone retriever.
    All three agents (Solver, Assessment, Feedback)
    call different convenience methods based on their needs.
    """

    def __init__(self, embedder: BGEEmbedder):
        self.embedder = embedder
        self.pc = Pinecone(api_key=PINECONE_API_KEY)

        # Create index if it doesn't exist
        existing = [i.name for i in self.pc.list_indexes()]
        if PINECONE_INDEX_NAME not in existing:
            print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
            self.pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=384,  # BGE-small-en-v1.5
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"Index created: {PINECONE_INDEX_NAME}")
        else:
            print(f"Pinecone index ready: {PINECONE_INDEX_NAME}")

        self.index = self.pc.Index(PINECONE_INDEX_NAME)

    def retrieve(
        self,
        query: str,
        top_k: int = RAG_TOP_K,
        topic_filter: str = None,
        namespace: str = "knowledge_base"
    ) -> list[dict]:
        """Base retrieval method."""
        query_embedding = self.embedder.embed_query(query)

        filter_dict = {}
        if topic_filter:
            filter_dict["topic_area"] = {"$eq": topic_filter}

        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            filter=filter_dict if filter_dict else None,
            include_metadata=True
        )

        documents = []
        for match in results["matches"]:
            documents.append({
                "text":   match["metadata"].get("text", ""),
                "source": match["metadata"].get("source", "unknown"),
                "topic":  match["metadata"].get("topic_area", "general"),
                "score":  match["score"]
            })
        return documents

    def retrieve_for_solver(self, query: str, topic: str = None) -> list[dict]:
        """Solver Agent — heavy retrieval from knowledge base."""
        return self.retrieve(query, top_k=5, topic_filter=topic, namespace="knowledge_base")

    def retrieve_for_assessment(self, concept: str) -> list[dict]:
        return self.retrieve(
            f"assessment questions about {concept}",
            top_k=3,
            namespace="knowledge_base"    # ← use existing namespace
        )

    def retrieve_for_feedback(self, misconception: str) -> list[dict]:
        """Feedback Agent — gets explanation strategies for gaps."""
        return self.retrieve(
            f"how to explain {misconception} clearly",
            top_k=3,
            namespace="knowledge_base"
        )

    def get_ingested_sources(self, namespace: str = "knowledge_base") -> set[str]:
        """
        Returns set of source filenames already ingested into Pinecone.
        Used by incremental ingestion to skip already-processed files.
        """
        try:
            # Fetch index stats to check if namespace exists
            stats = self.index.describe_index_stats()
            ns_stats = stats.get("namespaces", {}).get(namespace, {})
            if ns_stats.get("vector_count", 0) == 0:
                return set()

            # Query with a dummy vector to get sample metadata
            dummy = [0.0] * 384
            results = self.index.query(
                vector=dummy,
                top_k=10000,
                namespace=namespace,
                include_metadata=True
            )
            return {
                match["metadata"].get("source", "")
                for match in results["matches"]
                if match["metadata"].get("source")
            }
        except Exception as e:
            print(f"get_ingested_sources error: {e}")
            return set()
