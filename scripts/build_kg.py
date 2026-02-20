# scripts/build_kg.py
# Run after collect_documents.py to populate Neo4j
# python scripts/build_kg.py

from llm_client import LLMClient
from kg.neo4j_client import Neo4jClient
from rag.embedder import BGEEmbedder
from rag.retriever import RAGRetriever
from agents.kg_builder_agent import KGBuilderAgent

if __name__ == "__main__":
    print("Initialising components...")
    llm       = LLMClient()
    neo4j     = Neo4jClient()
    embedder  = BGEEmbedder()
    retriever = RAGRetriever(embedder)

    builder = KGBuilderAgent(llm, neo4j, retriever)
    builder.build_kg_from_all_documents()

    print(f"\nKG ready. {neo4j.get_node_count()} concepts in Neo4j.")
    print("KG will appear in Streamlit automatically (node_count > 1).")
