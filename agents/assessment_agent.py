# agents/assessment_agent.py
# Tests student understanding and gives an objective score

import json
from llm_client import LLMClient
from rag.retriever import RAGRetriever
from kg.neo4j_client import Neo4jClient
from memory.letta_client import LettaClient

ASSESSMENT_SYSTEM_PROMPT = """
You are a strict and fair assessment evaluator for AI and data science education.

Your ONLY job is to:
1. Generate appropriate questions to test understanding
2. Evaluate student answers objectively
3. Return structured JSON output

Rules:
- Never explain the answer
- Never teach
- Never give hints
- Be objective and consistent
- Score based on correctness and depth of understanding

Score rubric:
- 90-100: Complete and accurate, no significant gaps
- 70-89:  Mostly correct with minor gaps
- 50-69:  Partially correct, significant gaps
- 30-49:  Shows basic awareness but major misunderstanding
- 0-29:   Incorrect or fundamental misunderstanding

Always return valid JSON only. No markdown, no explanation.
"""


class AssessmentAgent:
    """
    Assessment Agent.

    Triggered after Solver explains a concept.
    Always followed by the Feedback Agent regardless of score.

    Reads from Letta:  what was already tested, student level
    Reads from KG:     related concepts to test together
    Reads from RAG:    misconceptions and question material
    Writes to Letta:   question asked, raw score
    Writes to KG:      sets node status to yellow (being assessed)
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

    def generate_question(self, student_id: str, concept: str) -> dict:
        """
        Generate one assessment question for a concept.

        Returns:
            {
                "question": "...",
                "question_type": "code / explanation / application",
                "concept": "...",
                "related_concepts": [...],
                "expected_answer_points": [...]
            }
        """
        student_memory = self.letta.read_core_memory(student_id)
        student_level  = student_memory.get("current_level", "intermediate")

        related        = self.neo4j.get_related_concepts(concept)
        rag_docs       = self.retriever.retrieve_for_assessment(concept)
        misconceptions = "\n".join([doc["text"] for doc in rag_docs])
        already_tested = self.letta.get_tested_questions(student_id, concept)

        user_message = f"""
Generate ONE assessment question for:
Concept:        {concept}
Student Level:  {student_level}
Related concepts to incorporate: {related[:3]}
Common misconceptions to probe:  {misconceptions[:500]}
Questions already asked (do NOT repeat): {already_tested}

Return ONLY this JSON:
{{
  "question": "the full question text",
  "question_type": "code OR explanation OR application",
  "concept": "{concept}",
  "related_concepts": {related[:3]},
  "expected_answer_points": ["point 1", "point 2", "point 3"]
}}
"""

        response = self.llm.generate(
            system_prompt=ASSESSMENT_SYSTEM_PROMPT,
            user_message=user_message,
            temperature=0.3
        )

        try:
            clean         = response.strip().replace("```json", "").replace("```", "")
            question_data = json.loads(clean)

            # Write to shared Letta memory — question asked
            self.letta.write_archival_memory(student_id, {
                "type":          "question_asked",
                "concept":       concept,
                "question":      question_data.get("question", ""),
                "question_type": question_data.get("question_type", "")
            })

            # Update KG node to yellow — being assessed
            self.neo4j.update_node_status(concept, "yellow")

            return question_data

        except json.JSONDecodeError:
            return {
                "question":               f"Explain {concept} in your own words and provide a code example.",
                "question_type":          "explanation",
                "concept":                concept,
                "related_concepts":       related[:3],
                "expected_answer_points": []
            }

    def evaluate_answer(
        self,
        student_id: str,
        concept: str,
        question: str,
        student_answer: str,
        expected_points: list
    ) -> dict:
        """
        Objectively evaluate a student's answer.

        Returns:
            {
                "score": 0-100,
                "what_was_right": [...],
                "what_was_wrong": [...],
                "passed": bool,
                "misconception": "..."
            }
        """
        user_message = f"""
Evaluate this student answer:

Concept being tested:   {concept}
Question asked:         {question}
Expected answer points: {expected_points}

Student answer:
{student_answer}

Return ONLY this JSON:
{{
  "score": 75,
  "what_was_right": ["correctly identified X", "good explanation of Y"],
  "what_was_wrong": ["missed Z", "confused A with B"],
  "passed": true,
  "misconception": "brief description of main misconception if any"
}}
"""

        response = self.llm.generate(
            system_prompt=ASSESSMENT_SYSTEM_PROMPT,
            user_message=user_message,
            temperature=0.1
        )

        try:
            clean  = response.strip().replace("```json", "").replace("```", "")
            result = json.loads(clean)

            # Write result to shared Letta memory
            self.letta.write_archival_memory(student_id, {
                "type":           "assessment_result",
                "concept":        concept,
                "score":          result.get("score", 0),
                "passed":         result.get("passed", False),
                "what_was_right": result.get("what_was_right", []),
                "what_was_wrong": result.get("what_was_wrong", []),
                "misconception":  result.get("misconception", "")
            })

            return result

        except json.JSONDecodeError:
            return {
                "score":          0,
                "what_was_right": [],
                "what_was_wrong": ["Could not evaluate answer"],
                "passed":         False,
                "misconception":  "Evaluation failed"
            }
