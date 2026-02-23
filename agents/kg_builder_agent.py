# agents/kg_builder_agent.py
# Maps student interactions to existing curriculum KG nodes
# Updates Topic/Technique status based on what student is engaging with
# Does NOT create new nodes — curriculum is fixed in Neo4j

from llm_client import LLMClient
from kg.neo4j_client import Neo4jClient
from rag.retriever import RAGRetriever

TOPIC_MAPPING_PROMPT = """
You are a curriculum mapping assistant.

Given a student's question or concept, identify which topic from the curriculum it belongs to.

Curriculum topics:
- Python for Data Science
- Reading Structured Files
- Structured Data Types
- Exploratory Data Analysis
- Data Visualization
- Imputation Techniques
- Data Augmentation
- Feature Reduction
- Business Metrics
- Preprocessing Summary
- ML Frameworks

Return ONLY the exact topic name from the list above that best matches.
If none match, return "Python for Data Science" as the default.
Return ONLY the topic name — no explanation, no punctuation.
"""


class KGBuilderAgent:
    """
    KG Builder Agent — updated for curriculum KG.

    NO longer creates new nodes. The curriculum is fixed in Neo4j.

    Instead:
    - Maps student questions to existing Topic nodes
    - Updates status on matched Topic nodes
    - Keeps the KG clean and structured

    Called by Solver Agent after every explanation.
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

    def map_and_update(self, concept: str, status: str = "blue") -> str | None:
        """
        Map a free-text concept to the nearest curriculum Topic node
        and update its status.

        Called by Solver Agent after every explanation.

        Returns the matched topic name, or None if no match found.
        """
        # First try direct match in Neo4j
        matched = self.neo4j.map_concept_to_topic(concept)

        # If no direct match, use LLM to map to nearest topic
        if not matched:
            matched = self._llm_map_to_topic(concept)

        if matched:
            self.neo4j.update_node_status(matched, status)
            print(f"KG Builder: '{concept}' mapped to '{matched}' → status={status}")
            return matched
        else:
            print(f"KG Builder: Could not map '{concept}' to any curriculum topic")
            return None

    def _llm_map_to_topic(self, concept: str) -> str | None:
        """
        Use LLM to map a concept to the nearest curriculum topic.
        Falls back to this when direct Neo4j string matching fails.
        """
        try:
            response = self.llm.generate(
                system_prompt=TOPIC_MAPPING_PROMPT,
                user_message=f"Map this to a curriculum topic: {concept}",
                temperature=0.0
            )
            matched = response.strip().strip(".")
            # Verify the returned topic actually exists in Neo4j
            verified = self.neo4j.map_concept_to_topic(matched)
            return verified if verified else None
        except Exception as e:
            print(f"KG Builder LLM mapping error: {e}")
            return None

    def update_technique_if_matched(self, concept: str, status: str = "blue") -> str | None:
        """
        Try to match concept to a Technique node specifically.
        Used for detailed technique-level tracking.
        """
        cypher = """
        MATCH (t:Technique)
        WHERE toLower(t.name) CONTAINS toLower($name)
           OR toLower($name) CONTAINS toLower(t.name)
        RETURN t.name as name
        LIMIT 1
        """
        results = self.neo4j.query(cypher, {"name": concept})
        if results:
            technique_name = results[0]["name"]
            self.neo4j.update_technique_status(technique_name, status)
            print(f"KG Builder: '{concept}' matched technique '{technique_name}' → status={status}")
            return technique_name
        return None
