# agents/feedback_agent.py
# Diagnoses what was right and wrong
# Makes key decisions about what happens next
# Updates curriculum KG Topic node colors based on assessment result

from llm_client import LLMClient
from rag.retriever import RAGRetriever
from kg.neo4j_client import Neo4jClient
from memory.letta_client import LettaClient

FEEDBACK_SYSTEM_PROMPT = """
You are a precise and encouraging diagnostic feedback expert for data science education.

Your ONLY job is to tell the student:
1. Exactly what they got RIGHT — acknowledge this first, always
2. Exactly what they got WRONG — be specific, not vague
3. WHY it was wrong — the specific misconception or gap
4. What the correct understanding should be

Rules:
- ALWAYS acknowledge what was right first, even if the score is 0
- Be precise about what was wrong — name the exact gap
- Explain WHY the mistake happened (the root misconception)
- Be encouraging but completely honest
- Never re-teach the full concept — just pinpoint the gap
- Keep feedback concise and actionable
- Give feedback for BOTH passing and failing answers
"""


class FeedbackAgent:
    """
    Feedback Agent.

    Triggered ALWAYS after Assessment — pass or fail.
    Most powerful agent — makes key decisions about what happens next.

    KG color updates on curriculum Topic nodes:
    - green  = passed (score >= 70) — topic mastered
    - red    = failed 3+ times — serious weak area
    - orange = failed with weak prerequisites — prereq gap
    - yellow = failed but first/second attempt — still learning

    Reads from Letta:  full mistake history
    Reads from KG:     prerequisite chain to trace root cause
    Reads from RAG:    explanation strategies for misconceptions
    Writes to Letta:   richest diagnosis data + mastered topics
    Writes to KG:      updates Topic node colors based on result
    Decides:           re_teach / advance / practice_more
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
        Give detailed feedback on an assessment result.
        Always called regardless of pass or fail.
        Updates curriculum Topic node color based on result.

        KG color logic:
          score >= 70 (passed)        → GREEN  ✓ mastered
          failed, attempt >= 3        → RED    serious weak area
          failed, weak prerequisites  → ORANGE prereq gap
          failed, attempt 1 or 2      → YELLOW still learning
        """
        score          = assessment_result.get("score", 0)
        passed         = assessment_result.get("passed", False)
        what_was_right = assessment_result.get("what_was_right", [])
        what_was_wrong = assessment_result.get("what_was_wrong", [])
        misconception  = assessment_result.get("misconception", "")

        # Map concept to curriculum Topic node
        matched_topic = self.neo4j.map_concept_to_topic(concept, kg=kg)
        topic_to_use  = matched_topic if matched_topic else concept

        # 1. Read mistake history from Letta
        mistake_history = self.letta.get_mistake_history(student_id, concept)
        attempt_count   = len(mistake_history) + 1

        # 2. Get prerequisite chain from curriculum KG
        prereq_chain = self.neo4j.get_prerequisite_chain_for_feedback(topic_to_use)
        weak_prereqs = [
            p for p in prereq_chain
            if p.get("status") in ["red", "orange", "grey", None]
        ]

        # 3. Get explanation strategy from RAG
        strategy_context = ""
        if misconception:
            rag_docs         = self.retriever.retrieve_for_feedback(misconception)
            strategy_context = "\n".join([doc["text"] for doc in rag_docs[:2]])

        # 4. Build feedback prompt
        user_message = f"""
Generate precise feedback for this assessment result:

Concept tested:          {concept}
Matched curriculum topic: {topic_to_use}
Question:                {question}
Student answer:          {student_answer}
Score:                   {score}/100
Passed:                  {passed}
What was right:          {what_was_right}
What was wrong:          {what_was_wrong}
Main misconception:      {misconception}

Student history on this concept:
  Previous attempts: {attempt_count - 1}
  Previous mistakes: {[m.get("misconception") for m in mistake_history]}

Weak prerequisites found in curriculum KG:
  {[p["name"] for p in weak_prereqs[:3]] if weak_prereqs else "None identified"}

Strategy context:
  {strategy_context[:500] if strategy_context else "Not available"}

Write feedback that:
1. Acknowledges what was right (find something positive even at score 0)
2. Pinpoints exactly what was wrong
3. Explains WHY it was wrong
{"4. This is attempt " + str(attempt_count) + " — be especially clear about root cause" if attempt_count >= 2 else ""}
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

        # 7. Determine re-teach focus
        re_teach_focus = None
        if next_action == "re_teach":
            re_teach_focus = (
                weak_prereqs[0]["name"] if weak_prereqs
                else misconception       if misconception
                else concept
            )

        # 8. Write diagnosis to Letta memory
        self.letta.write_archival_memory(student_id, {
            "type":           "feedback_given",
            "concept":        concept,
            "topic":          topic_to_use,
            "score":          score,
            "passed":         passed,
            "what_was_right": what_was_right,
            "what_was_wrong": what_was_wrong,
            "misconception":  misconception,
            "root_cause":     re_teach_focus,
            "attempt_number": attempt_count,
            "next_action":    next_action,
            "kg":             kg
        })

        # 9. Update curriculum Topic node color
        self._update_kg_node(topic_to_use, passed, score, attempt_count, weak_prereqs, kg=kg)

        # 10. If passed, update Letta mastered concepts with Topic name
        if passed:
            core     = self.letta.read_core_memory(student_id)
            kg_key   = f"mastered_concepts_{kg}"
            mastered = core.get(kg_key, [])
            if topic_to_use not in mastered:
                mastered.append(topic_to_use)
                self.letta.update_core_memory(student_id, {
                    kg_key:          mastered,
                    "current_topic": topic_to_use
                })

            # Check what topic to recommend next
            next_topic = self.neo4j.get_next_recommended_topic()
            if next_topic:
                self.letta.update_core_memory(student_id, {
                    "current_topic": next_topic
                })

        return {
            "feedback_text":  feedback_text,
            "what_was_right": what_was_right,
            "what_was_wrong": what_was_wrong,
            "root_cause":     re_teach_focus,
            "next_action":    next_action,
            "re_teach_focus": re_teach_focus,
            "score":          score,
            "passed":         passed
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
            "advance"       — passed cleanly (score >= 90), move on
            "practice_more" — passed but shaky (70-89), more practice
            "re_teach"      — failed, call Solver to re-explain
        """
        if passed and score >= 90:
            return "advance"
        elif passed and score >= 70:
            return "practice_more"
        else:
            return "re_teach"

    def _update_kg_node(
        self,
        topic: str,
        passed: bool,
        score: int,
        attempt_count: int,
        weak_prereqs: list,
        kg: str = "fods"
    ):
        """
        Update curriculum Topic node color to reflect student understanding.

        Color meanings:
          GREEN  — passed, topic mastered
          RED    — failed 3+ times, serious weak area
          ORANGE — failed with prereq gap detected
          YELLOW — failed but still early attempts
        """
        if passed:
            # GREEN — student understands this topic
            self.neo4j.update_node_status(topic, "green", kg=kg)

        elif attempt_count >= 3:
            # RED — failed 3+ times
            self.neo4j.update_node_status(topic, "red", kg=kg)

        elif weak_prereqs:
            # ORANGE — prerequisite gap is root cause
            self.neo4j.update_node_status(topic, "orange", kg=kg)
            # Also mark the weak prerequisite red
            self.neo4j.update_node_status(weak_prereqs[0]["name"], "red", kg=kg)

        else:
            # YELLOW — failed but still early attempts
            self.neo4j.update_node_status(topic, "yellow", kg=kg)
