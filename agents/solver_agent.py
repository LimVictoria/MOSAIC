# agents/solver_agent.py
# Explains concepts step by step
# Maps explained concepts to curriculum KG nodes via KG Builder

import json
from llm_client import LLMClient
from rag.retriever import RAGRetriever
from kg.neo4j_client import Neo4jClient
from memory.letta_client import LettaClient

SOLVER_SYSTEM_PROMPT = """
You are an expert data science technical tutor.

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

You will be given the curriculum structure — use it to show students
how the current topic connects to what they have already learned
and what they will learn next.
"""


class SolverAgent:
    """
    Solver Agent.

    Triggered when:
    - Student asks a question
    - Feedback Agent decides re-teaching is needed

    After every explanation:
    - Maps the concept to nearest curriculum Topic node
    - Updates that node status to blue (currently studying)
    - No new nodes are created — curriculum is fixed
    """

    def __init__(
        self,
        llm: LLMClient,
        retriever: RAGRetriever,
        neo4j: Neo4jClient,
        letta: LettaClient
    ):
        self.llm      = llm
        self.retriever = retriever
        self.neo4j    = neo4j
        self.letta    = letta

    def explain(self, student_id: str, concept: str, focus: str = None, message: str = None) -> str:
        """
        Explain a concept to the student.
        After explaining, maps concept to curriculum node and updates status to blue.
        """

        # 1. Read student profile from Letta memory
        student_memory = self.letta.read_core_memory(student_id)
        student_level  = student_memory.get("current_level", "intermediate")
        learning_style = student_memory.get("learning_style", "code_first")

        # 2. Get curriculum structure from Neo4j — passed to LLM as context
        curriculum = self.neo4j.get_curriculum_structure()
        curriculum_text = "\n".join([
            f"- {c['topic']} (status: {c['status']}) "
            f"{'← requires: ' + ', '.join(c['prerequisites']) if c['prerequisites'] else ''}"
            for c in curriculum
        ])

        # 3. Get prerequisites for this concept from curriculum KG
        prerequisites   = self.neo4j.get_prerequisites(concept)
        mastered        = self.letta.get_mastered_concepts(student_id)
        missing_prereqs = [p for p in prerequisites if p not in mastered]

        # 4. Check if prerequisites are unmastered — warn student
        unmastered = self.neo4j.get_unmastered_prerequisites(concept)

        # 5. Get related topics from KG
        related = self.neo4j.get_related_topics(concept)

        # 6. Query RAG for relevant documentation
        query       = focus if focus else (message if message else concept)
        rag_docs    = self.retriever.retrieve_for_solver(query)
        rag_context = "\n\n".join([doc["text"] for doc in rag_docs])

        # 7. Build prompt
        user_message = f"""
Student Level:  {student_level}
Learning Style: {learning_style}

Concept to explain: {concept}
{f"Specific focus: {focus}" if focus else ""}
{f"Student message: {message}" if message else ""}

Curriculum structure (for context):
{curriculum_text}

{"⚠️ Unmastered prerequisites: " + str([p for p in unmastered]) if unmastered else "Prerequisites: all covered"}
Related topics to connect: {related[:3] if related else "None"}

Relevant documentation:
{rag_context if rag_context else "No documentation available — use your knowledge."}

Provide a clear step by step explanation.
{"Start with code, then explain the theory." if learning_style == "code_first" else "Start with the concept, then show code."}
{"Briefly mention the student should cover " + str(missing_prereqs) + " first as they are prerequisites." if missing_prereqs else ""}
"""

        # 8. Generate explanation
        explanation = self.llm.generate(
            system_prompt=SOLVER_SYSTEM_PROMPT,
            user_message=user_message
        )

        # 9. Write to Letta memory
        self.letta.write_archival_memory(student_id, {
            "type":          "explanation_given",
            "concept":       concept,
            "focus":         focus,
            "student_level": student_level,
            "approach":      learning_style
        })

        # 10. Map concept to curriculum Topic node and set status to blue
        self._update_curriculum_node(concept, "blue")

        return explanation

    def _update_curriculum_node(self, concept: str, status: str):
        """
        Map the concept to the nearest curriculum Topic node
        and update its status. No new nodes are created.
        """
        try:
            matched = self.neo4j.map_concept_to_topic(concept)
            if matched:
                self.neo4j.update_node_status(matched, status)
                print(f"Solver: '{concept}' → '{matched}' status={status}")
            else:
                print(f"Solver: '{concept}' could not be mapped to curriculum topic")
        except Exception as e:
            print(f"Solver KG update error (non-fatal): {e}")
