# agents/orchestrator.py
# Chat-first orchestrator: always brief answer first, then offer deeper

from langgraph.graph import StateGraph, END
from typing import TypedDict


class TutorState(TypedDict):
    student_id:        str
    message:           str
    intent:            str
    concept:           str
    response:          str
    agent_used:        str
    question_data:     dict
    assessment_result: dict
    next_action:       str
    re_teach_focus:    str


# ── Prompts ───────────────────────────────────────────────────────────────────

IS_TECHNICAL_PROMPT = """
You are a message classifier for an AI tutoring system.

Is the student's message related to a technical or educational topic (even loosely)?

Answer YES if it's about any concept, tool, technology, science, math, or anything educational.
Answer NO if it's pure casual chat (greetings, small talk, feelings, opinions about non-technical things).

Reply with ONLY one word: YES or NO.
"""

IS_ASSESSMENT_PROMPT = """
You are a message classifier.

Is the student explicitly asking to be TESTED or QUIZZED?

Answer YES only if they say things like: "test me", "quiz me", "give me a question", "assess me", "practice questions"
Answer NO for everything else including normal questions and explanations.

Reply with ONLY one word: YES or NO.
"""

CONCEPT_EXTRACT_PROMPT = """
Extract the main technical concept from the student's message.

Reply with ONLY the concept name in title case (e.g. "Transformer", "Gradient Descent", "BERT").
If you cannot identify a specific concept, reply with: NONE
"""

BRIEF_ANSWER_PROMPT = """
You are MOSAIC, a friendly AI tutor specialising in AI engineering and data science.

Give a SHORT, conversational answer to the student's question.

Rules:
- Answer in 2-5 sentences maximum
- Be warm, clear, and direct
- No bullet points, headers, or code blocks
- At the very end, ask: "Would you like a more detailed explanation?"
"""

FOLLOWUP_PROMPT = """
You are a message classifier.

The student was just given a brief answer and asked "Would you like a more detailed explanation?"

Did they say YES?

Answer YES for: yes, sure, please, go ahead, yeah, yep, definitely, of course, elaborate, more, tell me more, explain more, deeper, full explanation, why not
Answer NO for: no, nope, thanks, that's enough, i'm good, ok thanks, no thanks, not now

Reply with ONLY one word: YES or NO.
"""

CASUAL_CHAT_PROMPT = """
You are MOSAIC, a friendly AI tutor specialising in AI engineering and data science.

Have a natural, warm, brief conversation. Keep it short and friendly.
If they ask what you can do, explain you teach AI/ML concepts and can test understanding.
"""


class Orchestrator:
    """
    Chat-first orchestrator.

    ALL technical questions go through Chat first (brief answer + offer deeper).
    Only if user says yes → Solver (full lesson).
    Explicit test requests → Assessment directly.
    Pure casual chat → casual reply.
    """

    def __init__(self, solver, assessment, feedback, neo4j, letta):
        self.solver          = solver
        self.assessment      = assessment
        self.feedback        = feedback
        self.neo4j           = neo4j
        self.letta           = letta
        self.llm             = solver.llm
        self.last_agent_used = None
        self.pending_concept: str | None = None
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(TutorState)

        workflow.add_node("classify",   self._classify)
        workflow.add_node("chat",       self._run_casual_chat)
        workflow.add_node("brief",      self._run_brief_answer)
        workflow.add_node("solver",     self._run_solver)
        workflow.add_node("assessment", self._run_assessment)
        workflow.add_node("feedback",   self._run_feedback)

        workflow.set_entry_point("classify")

        workflow.add_conditional_edges(
            "classify",
            self._route,
            {
                "chat":       "chat",
                "brief":      "brief",
                "solver":     "solver",
                "assessment": "assessment",
            }
        )

        workflow.add_edge("chat",       END)
        workflow.add_edge("brief",      END)
        workflow.add_edge("solver",     END)
        workflow.add_edge("assessment", "feedback")

        workflow.add_conditional_edges(
            "feedback",
            lambda s: "solver" if s.get("next_action") == "re_teach" else END,
            {"solver": "solver", END: END}
        )

        return workflow.compile()

    def route(self, student_id: str, message: str) -> str:
        state = {
            "student_id":        student_id,
            "message":           message,
            "intent":            "",
            "concept":           "",
            "response":          "",
            "agent_used":        "",
            "question_data":     {},
            "assessment_result": {},
            "next_action":       "",
            "re_teach_focus":    ""
        }
        result = self.graph.invoke(state)
        self.last_agent_used = result.get("agent_used", "Solver")
        return result.get("response", "I could not process that request.")

    def _classify(self, state: TutorState) -> TutorState:
        message = state["message"].strip().lower()

        # 1. Explicit assessment — always takes priority over everything
        assessment_words = ["test me", "quiz me", "assess me", "give me a question", "practice question"]
        if any(w in message for w in assessment_words):
            concept = self.pending_concept or self._extract_concept(state["message"])
            self.pending_concept = None
            return {**state, "intent": "assessment", "concept": concept}

        # 2. Check if this is a follow-up "yes/no" to a pending concept
        if self.pending_concept:
            try:
                answer = self.llm.generate(
                    system_prompt=FOLLOWUP_PROMPT,
                    user_message=state["message"]
                ).strip().upper()

                if answer.startswith("YES"):
                    concept = self.pending_concept
                    self.pending_concept = None
                    return {**state, "intent": "solver", "concept": concept}
                else:
                    self.pending_concept = None
                    # Fall through to normal classification
            except Exception:
                self.pending_concept = None

        # 2. Pure casual chat keywords
        casual_words = [
            "hi", "hello", "hey", "how are you", "what do you do",
            "who are you", "thanks", "thank you", "good morning",
            "good evening", "bye", "goodbye", "what's up", "whats up",
            "ok", "okay", "cool", "nice", "great", "awesome",
            "that makes sense", "got it", "i see", "what can you do",
        ]
        if any(message == w or message.startswith(w + " ") for w in casual_words):
            return {**state, "intent": "chat"}

        # 3. Everything else — check if technical, then route to brief answer
        try:
            answer = self.llm.generate(
                system_prompt=IS_TECHNICAL_PROMPT,
                user_message=state["message"]
            ).strip().upper()
            intent = "brief" if answer.startswith("YES") else "chat"
        except Exception:
            intent = "brief"

        return {**state, "intent": intent}

    def _route(self, state: TutorState) -> str:
        return state["intent"]

    def _extract_concept(self, message: str) -> str:
        try:
            concept = self.llm.generate(
                system_prompt=CONCEPT_EXTRACT_PROMPT,
                user_message=message
            ).strip()
            if concept.upper() == "NONE" or not concept:
                return message[:60]
            return concept
        except Exception:
            return message[:60]

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _run_casual_chat(self, state: TutorState) -> TutorState:
        try:
            response = self.llm.generate(
                system_prompt=CASUAL_CHAT_PROMPT,
                user_message=state["message"]
            )
        except Exception as e:
            response = f"Hey! Something went wrong: {e}"
        return {**state, "response": response, "agent_used": "Solver"}

    def _run_brief_answer(self, state: TutorState) -> TutorState:
        """Give a short answer and store pending concept for follow-up."""
        try:
            response = self.llm.generate(
                system_prompt=BRIEF_ANSWER_PROMPT,
                user_message=state["message"]
            )
            # Extract and store concept for potential follow-up
            concept = self._extract_concept(state["message"])
            if concept:
                self.pending_concept = concept
        except Exception as e:
            response = f"Something went wrong: {e}"

        return {**state, "response": response, "agent_used": "Solver"}

    def _run_solver(self, state: TutorState) -> TutorState:
        response = self.solver.explain(
            student_id=state["student_id"],
            concept=state["concept"],
            focus=state.get("re_teach_focus") or None
        )
        return {**state, "response": response, "agent_used": "Solver"}

    def _run_assessment(self, state: TutorState) -> TutorState:
        q_data   = self.assessment.generate_question(
            student_id=state["student_id"],
            concept=state["concept"]
        )
        response = f"Assessment Question:\n\n{q_data['question']}"
        return {**state, "response": response, "agent_used": "Assessment",
                "question_data": q_data}

    def _run_feedback(self, state: TutorState) -> TutorState:
        fb = self.feedback.give_feedback(
            student_id=state["student_id"],
            concept=state["concept"],
            question=state.get("question_data", {}).get("question", ""),
            student_answer=state["message"],
            assessment_result=state.get("assessment_result", {})
        )
        return {
            **state,
            "response":       fb["feedback_text"],
            "agent_used":     "Feedback",
            "next_action":    fb["next_action"],
            "re_teach_focus": fb.get("re_teach_focus", "")
        }
