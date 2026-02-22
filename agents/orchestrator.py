# agents/orchestrator.py
# LangGraph orchestrator — routes student messages to the right agent

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


CHAT_SYSTEM_PROMPT = """
You are MOSAIC, a friendly AI tutor specialising in AI engineering and data science.

You can have casual conversations AND teach technical concepts.

Rules:
- If the student is just chatting, respond conversationally and naturally
- If they ask a technical question, answer it clearly but conversationally (not in full lesson format)
- Keep casual replies short and warm
- Never force a lesson when someone just wants to chat
- You can mention that you can teach concepts in depth if they want
"""

INTENT_SYSTEM_PROMPT = """
Classify the student's message into exactly one of these intents:

chat     — casual conversation, greetings, small talk, non-technical questions
           ("hi", "how are you", "thanks", "what can you do", "can you chat")
explain  — asking to learn or understand a technical concept
           ("explain X", "what is X", "how does X work", "teach me X")
assess   — asking to be tested or quizzed
           ("test me", "quiz me", "give me a question", "practice")

Reply with ONLY one word: chat, explain, or assess.
"""


class Orchestrator:
    """
    LangGraph orchestrator.

    Routing logic:
      Casual chat               → Chat handler (friendly response)
      Student question          → Solver Agent (full lesson)
      "test me / quiz me"       → Assessment Agent
      After Assessment          → Feedback Agent
      Feedback says re_teach    → Solver Agent
      Feedback says advance     → END
    """

    def __init__(self, solver, assessment, feedback, neo4j, letta):
        self.solver     = solver
        self.assessment = assessment
        self.feedback   = feedback
        self.neo4j      = neo4j
        self.letta      = letta
        self.llm        = solver.llm   # reuse the same LLM client
        self.last_agent_used = None
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(TutorState)

        workflow.add_node("detect_intent", self._detect_intent)
        workflow.add_node("chat",          self._run_chat)
        workflow.add_node("solver",        self._run_solver)
        workflow.add_node("assessment",    self._run_assessment)
        workflow.add_node("feedback",      self._run_feedback)

        workflow.set_entry_point("detect_intent")

        workflow.add_conditional_edges(
            "detect_intent",
            self._route_by_intent,
            {"chat": "chat", "explain": "solver", "assess": "assessment"}
        )

        workflow.add_edge("chat",       END)
        workflow.add_edge("assessment", "feedback")

        workflow.add_conditional_edges(
            "feedback",
            self._route_after_feedback,
            {"re_teach": "solver", "done": END}
        )

        workflow.add_edge("solver", END)

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
        self.last_agent_used = result.get("agent_used", "unknown")
        return result.get("response", "I could not process that request.")

    def _detect_intent(self, state: TutorState) -> TutorState:
        """Detect intent: keyword-first, LLM fallback."""
        message = state["message"].strip().lower()

        # 1. Hard keyword rules — fast and reliable
        assess_words = ["test me", "quiz me", "assess me", "give me a question", "practice"]
        chat_words   = [
            "hi", "hello", "hey", "how are you", "what do you do", "what can you do",
            "who are you", "thanks", "thank you", "good morning", "good evening",
            "what's up", "whats up", "can you chat", "just chatting", "nice",
            "cool", "okay", "ok", "great", "awesome", "bye", "goodbye",
        ]
        explain_words = [
            "explain", "what is", "what are", "how does", "how do", "teach me",
            "tell me about", "describe", "define", "help me understand",
        ]

        if any(w in message for w in assess_words):
            intent = "assess"
        elif any(w in message for w in chat_words):
            intent = "chat"
        elif any(w in message for w in explain_words):
            intent = "explain"
        else:
            # 2. LLM fallback for ambiguous messages
            try:
                intent_raw = self.llm.generate(
                    system_prompt=INTENT_SYSTEM_PROMPT,
                    user_message=state["message"]
                ).strip().lower()
                # Only accept first word in case LLM adds extra text
                intent_raw = intent_raw.split()[0] if intent_raw else "chat"
                intent = intent_raw if intent_raw in ("chat", "explain", "assess") else "chat"
            except Exception:
                intent = "chat"  # Default to chat, not explain

        concept = self._extract_concept(state["message"])
        return {**state, "intent": intent, "concept": concept}

    def _route_by_intent(self, state: TutorState) -> str:
        return state["intent"]

    def _route_after_feedback(self, state: TutorState) -> str:
        return "re_teach" if state.get("next_action") == "re_teach" else "done"

    def _run_chat(self, state: TutorState) -> TutorState:
        """Handle casual conversation naturally."""
        try:
            response = self.llm.generate(
                system_prompt=CHAT_SYSTEM_PROMPT,
                user_message=state["message"],
                
            )
        except Exception as e:
            response = f"Hey! Something went wrong on my end ({e}). Try again?"
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

    def _extract_concept(self, message: str) -> str:
        """Extract concept name from student message."""
        concepts = [
            "backpropagation", "gradient descent", "neural network",
            "transformer", "attention mechanism", "embeddings",
            "rag", "fine-tuning", "llm", "pytorch", "tensorflow",
            "overfitting", "regularization", "batch normalization",
            "convolutional neural network", "lstm", "bert", "gpt",
            "chain rule", "calculus", "linear algebra", "probability"
        ]
        message_lower = message.lower()
        for concept in concepts:
            if concept in message_lower:
                return concept.title()
        return "current topic"
