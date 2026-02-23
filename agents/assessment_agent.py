# agents/assessment_agent.py
# Tests student understanding and gives an objective score
# Updates curriculum KG Topic node status during assessment

import json
from llm_client import LLMClient
from rag.retriever import RAGRetriever
from kg.neo4j_client import Neo4jClient
from memory.letta_client import LettaClient

ASSESSMENT_SYSTEM_PROMPT = """
You are a strict and fair assessment evaluator for data science education.

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

    Triggered from Assessment Tab.
    Always followed by Feedback Agent regardless of score.

    KG color updates:
    - yellow = topic is being assessed right now

    Reads from Letta:  what was already tested, student level
    Reads from KG:     related topics and techniques to test
    Reads from RAG:    question material
    Writes to Letta:   question asked, raw score
    Writes to KG:      sets Topic node to yellow (being assessed)
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

    def generate_question(self, student_id: str, concept: str) -> dict:
        """
        Generate one assessment question for a concept.
        Maps concept to nearest curriculum Topic.
        Sets that Topic node to yellow while being assessed.
        """
        student_memory = self.letta.read_core_memory(student_id)
        student_level  = student_memory.get("current_level", "intermediate")

        # Map to curriculum topic first
        matched_topic = self.neo4j.map_concept_to_topic(concept)
        topic_to_use  = matched_topic if matched_topic else concept

        # Get techniques under this topic for detailed questions
        techniques     = self.neo4j.get_topic_techniques(topic_to_use)
        technique_names = [t["name"] for t in techniques]

        # Get related topics from curriculum KG
        related = self.neo4j.get_related_topics(topic_to_use)

        # Check unmastered prerequisites
        unmastered_prereqs = self.neo4j.get_unmastered_prerequisites(topic_to_use)

        # Query RAG for question material
        rag_docs       = self.retriever.retrieve_for_assessment(concept)
        misconceptions = "\n".join([doc["text"] for doc in rag_docs])
        already_tested = self.letta.get_tested_questions(student_id, concept)

        user_message = f"""
Generate ONE assessment question for:
Concept:                  {concept}
Matched curriculum topic: {topic_to_use}
Student Level:            {student_level}
Techniques to probe:      {technique_names[:5]}
Related topics:           {related[:3]}
Unmastered prerequisites: {[p for p in unmastered_prereqs]}
Common misconceptions:    {misconceptions[:500] if misconceptions else "None available"}
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

            # Write to Letta memory
            self.letta.write_archival_memory(student_id, {
                "type":          "question_asked",
                "concept":       concept,
                "topic":         topic_to_use,
                "question":      question_data.get("question", ""),
                "question_type": question_data.get("question_type", "")
            })

            # Set curriculum Topic node to yellow — being assessed
            self.neo4j.update_node_status(topic_to_use, "yellow")

            return question_data

        except json.JSONDecodeError:
            self.neo4j.update_node_status(topic_to_use, "yellow")
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
        KG color update happens in Feedback Agent after this.
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

            # Write result to Letta memory
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
