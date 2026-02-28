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
    - Student is stuck on a problem

    Connects to:
    - RAG (heavy) — for docs, papers, code examples
    - Letta (read medium) — for student level and learning style
    - Neo4j KG (read) — for prerequisites and related concepts
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

    def explain(self, student_id: str, concept: str, focus: str = None, message: str = None, history: list = None, kg: str = "fods") -> str:
        """
        Explain a concept to the student.

        Args:
            student_id: unique student identifier
            concept: the concept to explain
            focus: specific aspect to focus on (for re-teaching)

        Returns:
            step by step explanation string
        """

        # 1. Read student profile from Letta memory
        student_memory = self.letta.read_core_memory(student_id)
        student_level = student_memory.get("current_level", "intermediate")
        learning_style = student_memory.get("learning_style", "code_first")

        # 2. Get prerequisites from KG
        prerequisites = self.neo4j.get_prerequisites(concept)
        related = self.neo4j.get_related_concepts(concept)

        # 3. Check which prerequisites student knows (from Letta archival)
        mastered = self.letta.get_mastered_concepts(student_id)
        missing_prereqs = [p for p in prerequisites if p not in mastered]

        # 4. Query RAG for relevant content
        query = focus if focus else concept
        rag_docs = self.retriever.retrieve_for_solver(query, topic=None)
        rag_context = "\n\n".join([doc["text"] for doc in rag_docs])

        # 5. Build explanation prompt
        user_message = f"""
Student Level: {student_level}
Learning Style: {learning_style}

Concept to explain: {concept}
{f"Specific focus: {focus}" if focus else ""}

Prerequisites student is missing: {missing_prereqs if missing_prereqs else "None - student knows all prerequisites"}
Related concepts to connect: {related[:3] if related else "None"}

Relevant documentation and examples:
{rag_context}

Provide a clear step by step explanation.
{"Start with code, then explain the theory." if learning_style == "code_first" else "Start with the concept, then show code."}
{"Since student is missing prerequisites, briefly cover them first." if missing_prereqs else ""}
"""

        # 6. Generate explanation
        explanation = self.llm.generate(
            system_prompt=SOLVER_SYSTEM_PROMPT,
            user_message=user_message
        )

        # 7. Write to Letta memory — what was explained
        self.letta.write_archival_memory(student_id, {
            "type": "explanation_given",
            "concept": concept,
            "focus": focus,
            "student_level": student_level,
            "approach": learning_style
        })

        # 8. Update KG node to blue (currently studying)
        self.neo4j.update_node_status(concept, "blue", kg=kg)

        return explanation


# ─────────────────────────────────────────────────────
# agents/assessment_agent.py
# Tests understanding and gives a score
# ─────────────────────────────────────────────────────

import json

ASSESSMENT_SYSTEM_PROMPT = """
You are a strict and fair assessment evaluator for AI and data science education.

Your ONLY job is to:
1. Generate appropriate questions to test understanding
2. Evaluate student answers objectively
3. Return a structured score

Rules:
- Never explain the answer
- Never teach
- Never give hints
- Be objective and consistent
- Score based on correctness and depth of understanding

Score rubric:
- 90-100: Complete and accurate answer
- 70-89:  Mostly correct with minor gaps
- 50-69:  Partially correct, significant gaps
- 30-49:  Shows basic awareness but major misunderstanding
- 0-29:   Incorrect or shows fundamental misunderstanding

Always return valid JSON only.
"""


class AssessmentAgent:
    """
    Assessment Agent.

    Triggered after Solver explains a concept.
    Generates questions and objectively scores answers.
    Always passes results to Feedback Agent.

    Connects to:
    - RAG (medium) — for question material
    - Letta (read) — for what was already tested
    - Neo4j KG (read) — for related concepts to test
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

    def generate_question(self, student_id: str, concept: str, kg: str = "fods") -> dict:
        """
        Generate an assessment question for a concept.

        Returns:
            {
                "question": "...",
                "question_type": "code/explanation/multiple_choice",
                "concept": "...",
                "related_concepts": [...]
            }
        """

        # Read student level from Letta
        student_memory = self.letta.read_core_memory(student_id)
        student_level = student_memory.get("current_level", "intermediate")

        # Get related concepts from KG
        related = self.neo4j.get_related_concepts(concept)

        # Get assessment material from RAG
        rag_docs = self.retriever.retrieve_for_assessment(concept)
        misconceptions = "\n".join([doc["text"] for doc in rag_docs])

        # Check what was already tested in Letta
        already_tested = self.letta.get_tested_questions(student_id, concept)

        user_message = f"""
Generate ONE assessment question for:
Concept: {concept}
Student Level: {student_level}
Related concepts to incorporate: {related[:3]}
Common misconceptions to probe: {misconceptions[:500]}
Questions already asked (avoid repeating): {already_tested}

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
            clean = response.strip().replace("```json", "").replace("```", "")
            question_data = json.loads(clean)

            # Write to Letta — question asked
            self.letta.write_archival_memory(student_id, {
                "type": "question_asked",
                "concept": concept,
                "question": question_data.get("question", ""),
                "question_type": question_data.get("question_type", "")
            })

            # Update KG node to yellow (being assessed)
            self.neo4j.update_node_status(concept, "yellow", kg=kg)

            return question_data

        except json.JSONDecodeError:
            return {
                "question": f"Explain {concept} in your own words and provide a code example.",
                "question_type": "explanation",
                "concept": concept,
                "related_concepts": related[:3],
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
        Evaluate a student's answer objectively.

        Returns:
            {
                "score": 0-100,
                "what_was_right": [...],
                "what_was_wrong": [...],
                "passed": bool
            }
        """

        user_message = f"""
Evaluate this student answer:

Concept being tested: {concept}
Question asked: {question}
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
            temperature=0.1  # very low — need consistent scoring
        )

        try:
            clean = response.strip().replace("```json", "").replace("```", "")
            result = json.loads(clean)

            # Write score to Letta
            self.letta.write_archival_memory(student_id, {
                "type": "assessment_result",
                "concept": concept,
                "score": result.get("score", 0),
                "passed": result.get("passed", False),
                "what_was_right": result.get("what_was_right", []),
                "what_was_wrong": result.get("what_was_wrong", []),
                "misconception": result.get("misconception", "")
            })

            return result

        except json.JSONDecodeError:
            return {
                "score": 0,
                "what_was_right": [],
                "what_was_wrong": ["Could not evaluate answer"],
                "passed": False,
                "misconception": "Evaluation failed"
            }


# ─────────────────────────────────────────────────────
# agents/feedback_agent.py
# Tells student exactly what is right and wrong
# Most consequential agent — makes key decisions
# ─────────────────────────────────────────────────────

FEEDBACK_SYSTEM_PROMPT = """
You are a precise and encouraging diagnostic feedback expert for AI education.

Your ONLY job is to tell the student:
1. Exactly what they got RIGHT (acknowledge this first)
2. Exactly what they got WRONG
3. WHY it was wrong (the specific misconception)
4. What the correct understanding should be

Rules:
- ALWAYS acknowledge what was right first, even if small
- Be precise about what was wrong — not vague
- Explain WHY the mistake happened (the root misconception)
- Be encouraging but honest
- Never re-teach the full concept — just pinpoint the gap
- Keep feedback concise and actionable
- Give feedback for BOTH passing and failing answers
"""


class FeedbackAgent:
    """
    Feedback Agent.

    Triggered ALWAYS after Assessment — pass or fail.
    Most powerful agent — makes key decisions about what happens next.

    Connects to:
    - RAG (medium) — for explanation strategies
    - Letta (heaviest) — reads full history, writes richest data
    - Neo4j KG (read heavy) — traces misconception to root cause
    - Neo4j KG (write) — updates node colors
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

    def give_feedback(
        self,
        student_id: str,
        concept: str,
        question: str,
        student_answer: str,
        assessment_result: dict,
        kg: str = "fods"
    ) -> dict:
        """
        Give detailed feedback on assessment result.
        Always called regardless of pass or fail.

        Returns:
            {
                "feedback_text": "...",
                "what_was_right": [...],
                "what_was_wrong": [...],
                "root_cause": "...",
                "next_action": "re_teach / advance / practice_more",
                "re_teach_focus": "specific aspect to re-teach if needed"
            }
        """

        score = assessment_result.get("score", 0)
        passed = assessment_result.get("passed", False)
        what_was_right = assessment_result.get("what_was_right", [])
        what_was_wrong = assessment_result.get("what_was_wrong", [])
        misconception = assessment_result.get("misconception", "")

        # 1. Read full mistake history from Letta
        mistake_history = self.letta.get_mistake_history(student_id, concept)
        attempt_count = len(mistake_history) + 1

        # 2. Get prerequisite chain from KG to trace root cause
        prereq_chain = self.neo4j.get_prerequisite_chain_for_feedback(concept)
        weak_prereqs = [
            p for p in prereq_chain
            if p.get("status") in ["red", "orange", "grey"]
        ]

        # 3. Get explanation strategy from RAG
        if misconception:
            rag_docs = self.retriever.retrieve_for_feedback(misconception)
            strategy_context = "\n".join([doc["text"] for doc in rag_docs[:2]])
        else:
            strategy_context = ""

        # 4. Build feedback prompt
        user_message = f"""
Generate precise feedback for this assessment result:

Concept tested: {concept}
Question: {question}
Student answer: {student_answer}
Score: {score}/100
Passed: {passed}
What was right: {what_was_right}
What was wrong: {what_was_wrong}
Main misconception: {misconception}

Student history on this concept:
Previous attempts: {attempt_count - 1}
Previous mistakes: {[m.get("misconception") for m in mistake_history]}

Weak prerequisites from KG:
{[p["name"] for p in weak_prereqs[:3]] if weak_prereqs else "None identified"}

Strategy context from documentation:
{strategy_context[:500] if strategy_context else "Not available"}

Return feedback that:
1. Acknowledges what was right (even if score is 0, find something)
2. Pinpoints exactly what was wrong
3. Explains WHY it was wrong
4. {"Since student failed {attempt_count} times, be especially clear about the root cause" if attempt_count >= 2 else ""}
"""

        # 5. Generate feedback
        feedback_text = self.llm.generate(
            system_prompt=FEEDBACK_SYSTEM_PROMPT,
            user_message=user_message
        )

        # 6. Decide next action
        next_action = self._decide_next_action(
            score=score,
            passed=passed,
            attempt_count=attempt_count,
            weak_prereqs=weak_prereqs
        )

        # 7. Determine re-teach focus if needed
        re_teach_focus = None
        if next_action == "re_teach":
            re_teach_focus = (
                weak_prereqs[0]["name"] if weak_prereqs
                else misconception if misconception
                else concept
            )

        # 8. Write detailed diagnosis to Letta memory
        self.letta.write_archival_memory(student_id, {
            "type": "feedback_given",
            "concept": concept,
            "score": score,
            "passed": passed,
            "what_was_right": what_was_right,
            "what_was_wrong": what_was_wrong,
            "misconception": misconception,
            "root_cause": re_teach_focus,
            "attempt_number": attempt_count,
            "next_action": next_action
        })

        # 9. Update KG node color based on result
        self._update_kg_node(concept, passed, attempt_count, weak_prereqs, kg=kg)

        return {
            "feedback_text": feedback_text,
            "what_was_right": what_was_right,
            "what_was_wrong": what_was_wrong,
            "root_cause": re_teach_focus,
            "next_action": next_action,
            "re_teach_focus": re_teach_focus,
            "score": score,
            "passed": passed
        }

    def _decide_next_action(
        self,
        score: int,
        passed: bool,
        attempt_count: int,
        weak_prereqs: list
    ) -> str:
        """
        Decide what happens after feedback.

        Returns:
            "advance"      — student passed, move to next concept
            "practice_more"— student passed but shaky, do more practice
            "re_teach"     — student failed, call Solver to re-explain
        """

        if passed and score >= 90:
            return "advance"

        elif passed and score >= 70:
            return "practice_more"

        elif not passed and attempt_count >= 3:
            # Failed 3 times — definitely re-teach
            return "re_teach"

        elif not passed and weak_prereqs:
            # Failed and has weak prerequisites — re-teach prerequisite
            return "re_teach"

        else:
            # Failed but first or second attempt — try again
            return "re_teach"

    def _update_kg_node(
        self,
        concept: str,
        passed: bool,
        attempt_count: int,
        weak_prereqs: list,
        kg: str = "fods"
    ):
        """Update the KG node color based on assessment result."""

        if passed:
            self.neo4j.update_node_status(concept, "green", kg=kg)

        elif attempt_count >= 3:
            self.neo4j.update_node_status(concept, "red", kg=kg)

        elif weak_prereqs:
            # Mark concept orange — prerequisite gap
            self.neo4j.update_node_status(concept, "orange", kg=kg)
            # Also mark the weak prerequisite red
            if weak_prereqs:
                self.neo4j.update_node_status(
                    weak_prereqs[0]["name"], "red", kg=kg
                )

        else:
            # Failed but not critical yet
            self.neo4j.update_node_status(concept, "yellow", kg=kg)
