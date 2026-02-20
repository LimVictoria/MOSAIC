# agents/kg_builder_agent.py
# Builds the Knowledge Graph from RAG documents
# Background job — runs once at setup, again when new docs are added

import json
from llm_client import LLMClient
from kg.neo4j_client import Neo4jClient
from rag.retriever import RAGRetriever

KG_EXTRACTION_SYSTEM_PROMPT = """
You are a knowledge graph builder specialized in AI and data science education.

Your ONLY job is to read technical documents and extract structured knowledge.

Extract:
1. Key concepts as nodes
2. Relationships between concepts as directed edges
3. Prerequisites (what must be known BEFORE learning this concept)
4. Difficulty level of each concept
5. Topic area each concept belongs to

Relationship types:
- REQUIRES:    concept A requires knowing concept B first
- BUILDS_ON:   concept A extends or builds on concept B
- PART_OF:     concept A is a component of concept B
- USED_IN:     concept A is applied when doing concept B
- RELATED_TO:  concept A and B are related but neither requires the other

Difficulty levels: beginner / intermediate / advanced

Topic areas:
  python_fundamentals, mathematics, classical_ml, deep_learning,
  nlp, computer_vision, mlops, llm_engineering, data_engineering

CRITICAL: Return ONLY valid JSON. No explanation, no preamble, no markdown.
"""

KG_EXTRACTION_USER_PROMPT = """
Read this document and extract the knowledge graph components.

DOCUMENT:
{document_text}

Return ONLY this JSON:
{{
  "concepts": [
    {{
      "name": "Backpropagation",
      "description": "Algorithm for computing gradients in neural networks",
      "difficulty": "intermediate",
      "topic_area": "deep_learning"
    }}
  ],
  "relationships": [
    {{
      "from": "Backpropagation",
      "to": "Chain Rule",
      "type": "REQUIRES"
    }}
  ]
}}
"""


class KGBuilderAgent:
    """
    KG Builder Agent.

    Reads all document chunks from ChromaDB RAG knowledge base,
    extracts concepts and relationships using LLaMA,
    writes everything to Neo4j.

    Does NOT use Letta memory — it is a background data pipeline, not a tutor.
    Does NOT run in real time — runs once at setup or when new docs are added.

    Once Neo4j has > 1 node, the KG visual appears automatically in Streamlit.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        neo4j_client: Neo4jClient,
        retriever: RAGRetriever
    ):
        self.llm      = llm_client
        self.neo4j    = neo4j_client
        self.retriever = retriever

    def build_kg_from_all_documents(self):
        """
        Main entry point.
        Reads all chunks from ChromaDB and populates Neo4j.
        """
        print("KG Builder Agent starting...")

        all_docs  = self.retriever.knowledge_collection.get(
            include=["documents", "metadatas"]
        )
        documents = all_docs["documents"]
        metadatas = all_docs["metadatas"]

        print(f"Processing {len(documents)} document chunks...")

        total_concepts      = 0
        total_relationships = 0

        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            source = meta.get("source", "unknown")
            print(f"  [{i+1}/{len(documents)}] {source}")

            extraction = self._extract_from_document(doc)
            if extraction:
                counts               = self._write_to_neo4j(extraction)
                total_concepts      += counts["concepts"]
                total_relationships += counts["relationships"]

            node_count = self.neo4j.get_node_count()
            if node_count > 1:
                print(f"  KG now visible: {node_count} nodes indexed")

        print(f"\nKG Build complete.")
        print(f"  Total concept nodes:  {total_concepts}")
        print(f"  Total relationships:  {total_relationships}")
        print(f"  Final Neo4j count:    {self.neo4j.get_node_count()}")

    def update_kg_with_new_documents(self, new_doc_ids: list[str]):
        """Called when new documents are added to ChromaDB."""
        print(f"Updating KG with {len(new_doc_ids)} new documents...")
        docs = self.retriever.knowledge_collection.get(
            ids=new_doc_ids,
            include=["documents", "metadatas"]
        )
        for doc, meta in zip(docs["documents"], docs["metadatas"]):
            extraction = self._extract_from_document(doc)
            if extraction:
                self._write_to_neo4j(extraction)
        print("KG update complete.")

    def _extract_from_document(self, document_text: str) -> dict | None:
        """Use LLaMA to extract concepts and relationships from a document chunk."""
        prompt = KG_EXTRACTION_USER_PROMPT.format(
            document_text=document_text[:3000]
        )
        response = self.llm.generate(
            system_prompt=KG_EXTRACTION_SYSTEM_PROMPT,
            user_message=prompt,
            temperature=0.1
        )
        try:
            clean = response.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            return json.loads(clean)
        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}")
            return None

    def _write_to_neo4j(self, extraction: dict) -> dict:
        """Write extracted concepts and relationships to Neo4j."""
        concepts_written      = 0
        relationships_written = 0

        for concept in extraction.get("concepts", []):
            if concept.get("name"):
                self.neo4j.create_concept_node({
                    "name":        concept["name"],
                    "description": concept.get("description", ""),
                    "difficulty":  concept.get("difficulty", "intermediate"),
                    "topic_area":  concept.get("topic_area", "general"),
                    "status":      "grey"
                })
                concepts_written += 1

        for rel in extraction.get("relationships", []):
            if rel.get("from") and rel.get("to") and rel.get("type"):
                self.neo4j.create_relationship(
                    from_concept=rel["from"],
                    to_concept=rel["to"],
                    rel_type=rel["type"]
                )
                relationships_written += 1

        return {"concepts": concepts_written, "relationships": relationships_written}
