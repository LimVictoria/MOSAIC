# agents/solver_agent.py
# Explains concepts step by step

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


class SolverAgent:
    """
    Solver Agent.

    Triggered when:
    - Student asks a question
    - Feedback Agent decides re-teaching is needed

    Reads from Letta:  student level, learning style
    Reads from KG:     prerequisites, related concepts
    Reads from RAG:    documentation, papers, code examples
    Writes to Letta:   what was explained and how
    Writes to KG:      sets node status to blue (studying)
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

        Args:
            student_id: unique student identifier
            concept:    the concept to explain
            focus:      specific aspect to focus on (used when re-teaching)

        Returns:
            step by step explanation string
        """

        # 1. Read student profile from shared Letta memory
        student_memory = self.letta.read_core_memory(student_id)
        student_level  = student_memory.get("current_level", "intermediate")
        learning_style = student_memory.get("learning_style", "code_first")

        # 2. Get prerequisites and related concepts from KG
        prerequisites  = self.neo4j.get_prerequisites(concept)
        related        = self.neo4j.get_related_concepts(concept)

        # 3. Find which prerequisites the student is missing
        mastered       = self.letta.get_mastered_concepts(student_id)
        missing_prereqs = [p for p in prerequisites if p not in mastered]

        # 4. Query RAG for relevant documentation and code examples
        query    = focus if focus else concept
        rag_docs = self.retriever.retrieve_for_solver(query)
        rag_context = "\n\n".join([doc["text"] for doc in rag_docs])

        # 5. Build prompt
        user_message = f"""
Student Level:  {student_level}
Learning Style: {learning_style}

Concept to explain: {concept}
{f"Specific focus: {focus}" if focus else ""}

Missing prerequisites: {missing_prereqs if missing_prereqs else "None — student knows all prerequisites"}
Related concepts to connect: {related[:3] if related else "None"}

Relevant documentation and examples:
{rag_context}

Provide a clear step by step explanation.
{"Start with code, then explain the theory." if learning_style == "code_first" else "Start with the concept, then show code."}
{"Briefly cover the missing prerequisites first before explaining the main concept." if missing_prereqs else ""}
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

        return explanation
