# agents/orchestrator.py
# LangGraph orchestrator — routes student messages to the right agent

from langgraph.graph import StateGraph, END
from typing import TypedDict


class TutorState(TypedDict):
    student_id:     str
    message:        str
    intent:         str
    concept:        str
    response:       str
    agent_used:     str
    question_data:  dict
    assessment_result: dict
    next_action:    str
    re_teach_focus: str


class Orchestrator:
    """
    LangGraph orchestrator.

    Routing logic:
      Student question          → Solver Agent
      "test me / quiz me"       → Assessment Agent
      After Assessment          → Feedback Agent (always)
      Feedback says re_teach    → Solver Agent
      Feedback says advance     → END
    """

    def __init__(self, solver, assessment, feedback, neo4j, letta):
        self.solver     = solver
        self.assessment = assessment
        self.feedback   = feedback
        self.neo4j      = neo4j
        self.letta      = letta
        self.last_agent_used = None
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(TutorState)

        workflow.add_node("detect_intent", self._detect_intent)
        workflow.add_node("solver",        self._run_solver)
        workflow.add_node("assessment",    self._run_assessment)
        workflow.add_node("feedback",      self._run_feedback)

        workflow.set_entry_point("detect_intent")

        workflow.add_conditional_edges(
            "detect_intent",
            self._route_by_intent,
            {"explain": "solver", "assess": "assessment"}
        )

        workflow.add_edge("assessment", "feedback")

        workflow.add_conditional_edges(
            "feedback",
            self._route_after_feedback,
            {"re_teach": "solver", "done": END}
        )

        workflow.add_edge("solver", END)

        return workflow.compile()

    def route(self, student_id: str, message: str) -> str:
        """Main entry point — route message and return response."""
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
        message = state["message"].lower()
        intent  = "assess" if any(
            w in message for w in ["test", "assess", "quiz", "practice", "question"]
        ) else "explain"
        concept = self._extract_concept(state["message"])
        return {**state, "intent": intent, "concept": concept}

    def _route_by_intent(self, state: TutorState) -> str:
        return state["intent"]

    def _route_after_feedback(self, state: TutorState) -> str:
        return "re_teach" if state.get("next_action") == "re_teach" else "done"

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
