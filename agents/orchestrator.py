# agents/orchestrator.py
# Two-stage routing orchestrator

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


# ── Stage 1: Is this a learning request? ──────────────────────────────────────
STAGE1_PROMPT = """
You are a message classifier for an AI tutoring system.

Answer ONE question: Is the student asking to LEARN or be TESTED on a specific technical concept?

Answer YES if:
- They want something explained ("explain X", "what is X", "how does X work", "teach me X")
- They want to be tested or quizzed ("test me on X", "quiz me", "give me a question about X")
- They are asking a technical question that requires a structured explanation

Answer NO if:
- It is casual conversation ("hi", "how are you", "thanks", "what do you do")
- It is a general question about the system ("what can you help with", "who are you")
- It is a simple yes/no or factual question not requiring a lesson
- It is feedback or a reaction ("that makes sense", "ok got it", "interesting")

Reply with ONLY one word: YES or NO.
"""

# ── Stage 2: What is the concept and mode? ───────────────────────────────────
STAGE2_PROMPT = """
You are a concept extractor for an AI tutoring system.

The student wants to learn something. Extract:
1. The specific technical concept they are asking about
2. Whether they want to be TAUGHT or TESTED

Reply in this exact format (two lines only):
CONCEPT: <concept name, title case, e.g. "Gradient Descent">
MODE: <teach or test>

Rules:
- CONCEPT must be a real technical concept, never "current topic" or vague terms
- If multiple concepts, pick the main one
- MODE is "test" only if they explicitly ask to be quizzed or tested
- MODE is "teach" for everything else
"""

# ── Chat handler ─────────────────────────────────────────────────────────────
CHAT_PROMPT = """
You are MOSAIC, a friendly AI tutor specialising in AI engineering and data science.

Have a natural, warm conversation. Rules:
- Keep replies concise and friendly
- If they ask what you can do, briefly explain you can teach AI/ML concepts and test understanding
- Never launch into a full lesson unprompted
- If they seem curious about a topic, you can offer to explain it in depth
"""


class Orchestrator:
    """
    Two-stage routing orchestrator.

    Stage 1: Is this a learning request? (YES / NO)
      NO  → Chat handler
      YES → Stage 2

    Stage 2: Extract concept + mode (teach / test)
      teach → Solver Agent
      test  → Assessment Agent → Feedback Agent
      Feedback re_teach → Solver Agent
    """

    def __init__(self, solver, assessment, feedback, neo4j, letta):
        self.solver     = solver
        self.assessment = assessment
        self.feedback   = feedback
        self.neo4j      = neo4j
        self.letta      = letta
        self.llm        = solver.llm
        self.last_agent_used = None
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(TutorState)

        workflow.add_node("stage1",     self._stage1_classify)
        workflow.add_node("stage2",     self._stage2_extract)
        workflow.add_node("chat",       self._run_chat)
        workflow.add_node("solver",     self._run_solver)
        workflow.add_node("assessment", self._run_assessment)
        workflow.add_node("feedback",   self._run_feedback)

        workflow.set_entry_point("stage1")

        # Stage 1 → chat or stage2
        workflow.add_conditional_edges(
            "stage1",
            lambda s: "stage2" if s["intent"] == "learn" else "chat",
            {"chat": "chat", "stage2": "stage2"}
        )

        # Stage 2 → solver or assessment
        workflow.add_conditional_edges(
            "stage2",
            lambda s: "solver" if s["intent"] == "teach" else "assessment",
            {"solver": "solver", "assessment": "assessment"}
        )

        workflow.add_edge("chat",       END)
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

    # ── Stage 1: Learning request or not? ────────────────────────────────────
    def _stage1_classify(self, state: TutorState) -> TutorState:
        """Binary: is this a learning request?"""
        # Fast keyword shortcuts — no LLM needed
        message = state["message"].strip().lower()

        definite_chat = [
            "hi", "hello", "hey", "how are you", "what do you do",
            "who are you", "thanks", "thank you", "good morning",
            "good evening", "bye", "goodbye", "what's up", "whats up",
            "ok", "okay", "cool", "nice", "great", "awesome",
            "that makes sense", "got it", "i see", "interesting",
            "what can you do", "what can you help",
        ]
        if any(message == w or message.startswith(w) for w in definite_chat):
            return {**state, "intent": "chat"}

        definite_learn = [
            "explain", "what is ", "what are ", "how does", "how do",
            "teach me", "tell me about", "describe", "define",
            "help me understand", "test me", "quiz me", "assess me",
            "give me a question", "i want to learn", "i want to understand",
        ]
        if any(w in message for w in definite_learn):
            return {**state, "intent": "learn"}

        # Ambiguous — ask the LLM
        try:
            answer = self.llm.generate(
                system_prompt=STAGE1_PROMPT,
                user_message=state["message"]
            ).strip().upper()
            intent = "learn" if answer.startswith("YES") else "chat"
        except Exception:
            intent = "chat"

        return {**state, "intent": intent}

    # ── Stage 2: Extract concept + teach vs test ──────────────────────────────
    def _stage2_extract(self, state: TutorState) -> TutorState:
        """Extract concept and whether to teach or test."""
        try:
            raw = self.llm.generate(
                system_prompt=STAGE2_PROMPT,
                user_message=state["message"]
            ).strip()

            concept = "Unknown Concept"
            mode    = "teach"

            for line in raw.splitlines():
                line = line.strip()
                if line.upper().startswith("CONCEPT:"):
                    concept = line.split(":", 1)[1].strip()
                elif line.upper().startswith("MODE:"):
                    mode_raw = line.split(":", 1)[1].strip().lower()
                    mode = "test" if "test" in mode_raw else "teach"

            # Safety: if concept is vague, use the raw message as fallback
            if not concept or concept.lower() in ("current topic", "unknown", "unknown concept", ""):
                concept = state["message"][:60]

        except Exception:
            concept = state["message"][:60]
            mode    = "teach"

        return {**state, "concept": concept, "intent": mode}

    # ── Handlers ──────────────────────────────────────────────────────────────
    def _run_chat(self, state: TutorState) -> TutorState:
        try:
            response = self.llm.generate(
                system_prompt=CHAT_PROMPT,
                user_message=state["message"]
            )
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
