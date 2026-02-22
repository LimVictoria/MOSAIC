# agents/orchestrator.py
# Two-stage routing orchestrator with "quick answer + offer deeper" flow

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
- It is a simple yes/no or follow-up ("yes please", "no thanks", "ok", "sure", "go ahead")
- It is a reaction or feedback ("that makes sense", "ok got it", "interesting")
- It is a conversational technical question that could be answered briefly

Reply with ONLY one word: YES or NO.
"""

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

CHAT_PROMPT = """
You are MOSAIC, a friendly AI tutor specialising in AI engineering and data science.

Your job here is to give a SHORT, conversational answer to the student's question — NOT a full lesson.

Rules:
- Answer in 2-4 sentences maximum
- Be warm and clear
- At the end, ALWAYS ask: "Would you like me to go deeper on this?"
- Do NOT use bullet points, headers, or code blocks
- Do NOT teach a full lesson — just give a brief, friendly answer
"""

FOLLOWUP_PROMPT = """
You are a message classifier.

The student just received a brief answer and was asked "Would you like me to go deeper on this?"

Did they say YES (they want more detail / a full explanation)?

Answer YES if they said: yes, sure, please, go ahead, yeah, yep, definitely, of course, elaborate, more, tell me more, explain more, deeper, full explanation
Answer NO if they said: no, nope, thanks, that's enough, i'm good, ok thanks, no thanks

Reply with ONLY one word: YES or NO.
"""


class Orchestrator:
    """
    Two-stage routing with conversational bridge.

    Flow:
      message → Stage 1 (learning request?)
        NO  → check if follow-up "yes" to pending concept
                yes pending + user said yes → Solver (full lesson)
                otherwise → Chat (short answer + offer deeper)
        YES → Stage 2 (extract concept + mode)
                teach → Solver
                test  → Assessment → Feedback
    """

    def __init__(self, solver, assessment, feedback, neo4j, letta):
        self.solver          = solver
        self.assessment      = assessment
        self.feedback        = feedback
        self.neo4j           = neo4j
        self.letta           = letta
        self.llm             = solver.llm
        self.last_agent_used = None
        # Stores concept from last chat answer, waiting for "yes go deeper"
        self.pending_concept: str | None = None
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

        workflow.add_conditional_edges(
            "stage1",
            self._route_stage1,
            {"chat": "chat", "stage2": "stage2", "solver": "solver"}
        )

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

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    def _stage1_classify(self, state: TutorState) -> TutorState:
        message = state["message"].strip().lower()

        # Check if this is a follow-up "yes" to a pending concept
        if self.pending_concept:
            try:
                answer = self.llm.generate(
                    system_prompt=FOLLOWUP_PROMPT,
                    user_message=state["message"]
                ).strip().upper()
                if answer.startswith("YES"):
                    # User wants deeper — route to solver with pending concept
                    concept = self.pending_concept
                    self.pending_concept = None
                    return {**state, "intent": "solver", "concept": concept}
                else:
                    self.pending_concept = None
            except Exception:
                self.pending_concept = None

        # Fast keyword shortcuts
        definite_chat = [
            "hi", "hello", "hey", "how are you", "what do you do",
            "who are you", "thanks", "thank you", "good morning",
            "good evening", "bye", "goodbye", "what's up", "whats up",
            "ok", "okay", "cool", "nice", "great", "awesome",
            "that makes sense", "got it", "i see", "interesting",
            "what can you do", "what can you help",
        ]
        definite_learn = [
            "explain ", "teach me", "tell me about", "describe ",
            "help me understand", "test me", "quiz me", "assess me",
            "give me a question", "i want to learn", "i want to understand",
        ]

        if any(message == w or message.startswith(w) for w in definite_chat):
            return {**state, "intent": "chat"}

        if any(w in message for w in definite_learn):
            return {**state, "intent": "learn"}

        # LLM for ambiguous messages
        try:
            answer = self.llm.generate(
                system_prompt=STAGE1_PROMPT,
                user_message=state["message"]
            ).strip().upper()
            intent = "learn" if answer.startswith("YES") else "chat"
        except Exception:
            intent = "chat"

        return {**state, "intent": intent}

    def _route_stage1(self, state: TutorState) -> str:
        intent = state["intent"]
        if intent == "solver":
            return "solver"
        elif intent == "learn":
            return "stage2"
        else:
            return "chat"

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    def _stage2_extract(self, state: TutorState) -> TutorState:
        try:
            raw = self.llm.generate(
                system_prompt=STAGE2_PROMPT,
                user_message=state["message"]
            ).strip()

            concept = ""
            mode    = "teach"

            for line in raw.splitlines():
                line = line.strip()
                if line.upper().startswith("CONCEPT:"):
                    concept = line.split(":", 1)[1].strip()
                elif line.upper().startswith("MODE:"):
                    mode_raw = line.split(":", 1)[1].strip().lower()
                    mode = "test" if "test" in mode_raw else "teach"

            if not concept or concept.lower() in ("current topic", "unknown", ""):
                concept = state["message"][:60]

        except Exception:
            concept = state["message"][:60]
            mode    = "teach"

        return {**state, "concept": concept, "intent": mode}

    # ── Handlers ──────────────────────────────────────────────────────────────
    def _run_chat(self, state: TutorState) -> TutorState:
        """Short conversational answer + extract concept for pending follow-up."""
        try:
            response = self.llm.generate(
                system_prompt=CHAT_PROMPT,
                user_message=state["message"]
            )

            # Extract concept in background so we can go deeper if user says yes
            try:
                raw = self.llm.generate(
                    system_prompt=STAGE2_PROMPT,
                    user_message=state["message"]
                ).strip()
                for line in raw.splitlines():
                    if line.strip().upper().startswith("CONCEPT:"):
                        concept = line.split(":", 1)[1].strip()
                        if concept and concept.lower() not in ("current topic", "unknown", ""):
                            self.pending_concept = concept
                            break
            except Exception:
                pass

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
