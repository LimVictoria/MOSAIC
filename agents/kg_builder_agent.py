# agents/kg_builder_agent.py
# Builds the Knowledge Graph from RAG documents
# Runs in background - not real time

import json
from llm_client import LLMClient
from kg.neo4j_client import Neo4jClient
from rag.retriever import RAGRetriever


KG_EXTRACTION_SYSTEM_PROMPT = """
You are a knowledge graph builder specialized in AI and data science education.

Your ONLY job is to read technical documents and extract structured knowledge.

You must extract:
1. Key concepts as nodes
2. Relationships between concepts as directed edges
3. Prerequisites (what must be known before learning this concept)
4. Difficulty level of each concept
5. Topic area each concept belongs to

Relationship types you can use:
- REQUIRES: concept A requires knowing concept B first
- BUILDS_ON: concept A extends or builds on concept B
- PART_OF: concept A is a component of concept B
- USED_IN: concept A is used when doing concept B
- RELATED_TO: concept A and B are related but neither requires the other

Difficulty levels:
- beginner
- intermediate
- advanced

Topic areas:
- python_fundamentals
- mathematics
- classical_ml
- deep_learning
- nlp
- computer_vision
- mlops
- llm_engineering
- data_engineering

CRITICAL: Return ONLY valid JSON. No explanation, no preamble, no markdown.
"""

KG_EXTRACTION_USER_PROMPT = """
Read this document and extract the knowledge graph components.

DOCUMENT:
{document_text}

Return ONLY this JSON structure:
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

    Reads all documents from ChromaDB RAG knowledge base,
    extracts concepts and relationships using LLaMA,
    writes everything to Neo4j.

    Runs once during setup.
    Runs again when new documents are added.
    Does NOT run in real time during teaching sessions.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        neo4j_client: Neo4jClient,
        retriever: RAGRetriever
    ):
        self.llm = llm_client
        self.neo4j = neo4j_client
        self.retriever = retriever

    def build_kg_from_all_documents(self):
        """
        Main entry point.
        Reads all chunks from ChromaDB and builds KG.
        """
        print("KG Builder Agent starting...")

        # Get all documents from ChromaDB
        all_docs = self.retriever.knowledge_collection.get(
            include=["documents", "metadatas"]
        )

        documents = all_docs["documents"]
        metadatas = all_docs["metadatas"]

        print(f"Processing {len(documents)} document chunks...")

        total_concepts = 0
        total_relationships = 0

        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            print(f"Processing chunk {i+1}/{len(documents)} from {meta.get('source', 'unknown')}")

            # Extract concepts and relationships from this chunk
            extraction = self._extract_from_document(doc)

            if extraction:
                concepts_added = self._write_to_neo4j(extraction)
                total_concepts += concepts_added["concepts"]
                total_relationships += concepts_added["relationships"]

            # Log KG visibility status
            node_count = self.neo4j.get_node_count()
            if node_count > 1:
                print(f"KG now visible: {node_count} nodes")

        print(f"KG Build complete.")
        print(f"Total concepts: {total_concepts}")
        print(f"Total relationships: {total_relationships}")
        print(f"Final node count in Neo4j: {self.neo4j.get_node_count()}")

    def _extract_from_document(self, document_text: str) -> dict:
        """
        Use LLaMA to extract concepts and relationships
        from a document chunk.
        Returns parsed JSON or None if extraction fails.
        """
        prompt = KG_EXTRACTION_USER_PROMPT.format(
            document_text=document_text[:3000]  # limit chunk size for LLM
        )

        response = self.llm.generate(
            system_prompt=KG_EXTRACTION_SYSTEM_PROMPT,
            user_message=prompt,
            temperature=0.1  # low temperature for structured extraction
        )

        # Parse JSON response
        try:
            # Clean response in case LLM adds markdown
            clean = response.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            return json.loads(clean)

        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Raw response: {response[:200]}")
            return None

    def _write_to_neo4j(self, extraction: dict) -> dict:
        """
        Write extracted concepts and relationships to Neo4j.
        Returns counts of what was written.
        """
        concepts_written = 0
        relationships_written = 0

        # Write concept nodes
        for concept in extraction.get("concepts", []):
            if concept.get("name"):
                self.neo4j.create_concept_node({
                    "name": concept["name"],
                    "description": concept.get("description", ""),
                    "difficulty": concept.get("difficulty", "intermediate"),
                    "topic_area": concept.get("topic_area", "general"),
                    "status": "grey"  # all nodes start grey
                })
                concepts_written += 1

        # Write relationships
        for rel in extraction.get("relationships", []):
            if rel.get("from") and rel.get("to") and rel.get("type"):
                self.neo4j.create_relationship(
                    from_concept=rel["from"],
                    to_concept=rel["to"],
                    rel_type=rel["type"]
                )
                relationships_written += 1

        return {
            "concepts": concepts_written,
            "relationships": relationships_written
        }

    def update_kg_with_new_documents(self, new_doc_ids: list[str]):
        """
        Update KG when new documents are added to ChromaDB.
        Only processes the new document chunks.
        """
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
