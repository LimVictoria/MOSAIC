# agents/solver_agent.py
# Explains concepts step by step
# Also auto-builds the KG from every explanation

import json
from llm_client import LLMClient
from rag.retriever import RAGRetriever
from kg.neo4j_client import Neo4jClient
from memory.letta_client import LettaClient

SOLVER_SYSTEM_PROMPT = """
You are an expert AI and data science technical tutor.

Your ONLY job is to explain concepts clearly and accurately, step by step.

Rules:
- Always explain step by step
- Always include working code examples
- Match your explanation to the student's level (from memory)
- If student is beginner, use analogies before technical terms
- If student struggles with math, use intuitive explanations first
- Never test the student
- Never tell the student what they got wrong
- Just explain, clearly and completely

When you receive context about prerequisites and related concepts,
use them to build a connected explanation that shows how things fit together.
"""

KG_EXTRACT_PROMPT = """
You are a knowledge graph builder for AI and data science education.

Given a concept that was just explained to a student, extract:
1. The main concept
2. Its prerequisites (what must be known BEFORE this)
3. Related concepts (what connects to this)
4. Difficulty level

Return ONLY this JSON, nothing else:
{
  "main_concept": {
    "name": "Backpropagation",
    "description": "Algorithm for computing gradients in neural networks",
    "difficulty": "intermediate",
    "topic_area": "deep_learning"
  },
  "prerequisites": [
    {"name": "Chain Rule", "difficulty": "intermediate", "topic_area": "mathematics"},
    {"name": "Gradient Descent", "difficulty": "intermediate", "topic_area": "deep_learning"}
  ],
  "related": [
    {"name": "Neural Networks", "difficulty": "intermediate", "topic_area": "deep_learning"}
  ]
}

Topic areas: python_fundamentals, mathematics, classical_ml, deep_learning,
nlp, computer_vision, mlops, llm_engineering, data_engineering

Difficulty: beginner, intermediate, advanced
"""


class SolverAgent:
    """
    Solver Agent.

    Triggered when:
    - Student asks a question
    - Feedback Agent decides re-teaching is needed

    After every explanation, automatically extracts concepts
    and writes them to Neo4j — building the KG from conversations.
    """

    def __init__(
        self,
        llm: LLMClient,
        retriever: RAGRetriever,
        neo4j: Neo4jClient,
        letta: LettaClient
    ):
        self.llm = llm
        self.retriever = retriever
        self.neo4j = neo4j
        self.letta = letta

    def explain(self, student_id: str, concept: str, focus: str = None) -> str:
        """
        Explain a concept to the student.
        After explaining, automatically builds KG from the concept.
        """

        # 1. Read student profile from shared Letta memory
        student_memory = self.letta.read_core_memory(student_id)
        student_level  = student_memory.get("current_level", "intermediate")
        learning_style = student_memory.get("learning_style", "code_first")

        # 2. Get prerequisites and related concepts from KG (if any exist)
        prerequisites   = self.neo4j.get_prerequisites(concept)
        related         = self.neo4j.get_related_concepts(concept)

        # 3. Find which prerequisites the student is missing
        mastered        = self.letta.get_mastered_concepts(student_id)
        missing_prereqs = [p for p in prerequisites if p not in mastered]

        # 4. Query RAG for relevant documentation
        query    = focus if focus else concept
        rag_docs = self.retriever.retrieve_for_solver(query)
        rag_context = "\n\n".join([doc["text"] for doc in rag_docs])

        # 5. Build prompt
        user_message = f"""
Student Level:  {student_level}
Learning Style: {learning_style}

Concept to explain: {concept}
{f"Specific focus: {focus}" if focus else ""}

Missing prerequisites: {missing_prereqs if missing_prereqs else "None"}
Related concepts to connect: {related[:3] if related else "None"}

Relevant documentation:
{rag_context if rag_context else "No documentation available — use your knowledge."}

Provide a clear step by step explanation.
{"Start with code, then explain the theory." if learning_style == "code_first" else "Start with the concept, then show code."}
{"Briefly cover the missing prerequisites first." if missing_prereqs else ""}
"""

        # 6. Generate explanation
        explanation = self.llm.generate(
            system_prompt=SOLVER_SYSTEM_PROMPT,
            user_message=user_message
        )

        # 7. Write to shared Letta memory
        self.letta.write_archival_memory(student_id, {
            "type":          "explanation_given",
            "concept":       concept,
            "focus":         focus,
            "student_level": student_level,
            "approach":      learning_style
        })

        # 8. Update KG node to blue — currently studying
        self.neo4j.update_node_status(concept, "blue")

        # 9. AUTO-BUILD KG from this concept
        # Runs silently after every explanation
        self._build_kg_from_concept(concept)

        return explanation

    def _build_kg_from_concept(self, concept: str):
        """
        Automatically extract concept relationships using LLM
        and write them to Neo4j.

        This is called after every explanation so the KG
        grows organically as the student learns — no scripts needed.
        """
        try:
            # Ask LLM to extract concept structure
            response = self.llm.generate(
                system_prompt=KG_EXTRACT_PROMPT,
                user_message=f"Extract knowledge graph data for: {concept}",
                temperature=0.1
            )

            # Parse JSON response
            clean = response.strip()
            if "```" in clean:
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            
            data = json.loads(clean)

            # Write main concept node
            main = data.get("main_concept", {})
            if main.get("name"):
                self.neo4j.create_concept_node({
                    "name":        main["name"],
                    "description": main.get("description", ""),
                    "difficulty":  main.get("difficulty", "intermediate"),
                    "topic_area":  main.get("topic_area", "general"),
                    "status":      "blue"
                })

            # Write prerequisite nodes and relationships
            for prereq in data.get("prerequisites", []):
                if prereq.get("name"):
                    self.neo4j.create_concept_node({
                        "name":       prereq["name"],
                        "description": "",
                        "difficulty": prereq.get("difficulty", "intermediate"),
                        "topic_area": prereq.get("topic_area", "general"),
                        "status":     "grey"
                    })
                    self.neo4j.create_relationship(
                        from_concept=main["name"],
                        to_concept=prereq["name"],
                        rel_type="REQUIRES"
                    )

            # Write related concept nodes and relationships
            for rel in data.get("related", []):
                if rel.get("name"):
                    self.neo4j.create_concept_node({
                        "name":       rel["name"],
                        "description": "",
                        "difficulty": rel.get("difficulty", "intermediate"),
                        "topic_area": rel.get("topic_area", "general"),
                        "status":     "grey"
                    })
                    self.neo4j.create_relationship(
                        from_concept=main["name"],
                        to_concept=rel["name"],
                        rel_type="RELATED_TO"
                    )

            print(f"KG updated with: {main.get('name', concept)}")

        except Exception as e:
            # Never crash the explanation if KG building fails
            print(f"KG auto-build error (non-fatal): {e}")
