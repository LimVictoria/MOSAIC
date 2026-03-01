# agents/orchestrator.py
# Chat-first orchestrator: always brief answer first, then offer deeper
# Routes to Solver, Recommender, Assessment, or Feedback based on intent

from langgraph.graph import StateGraph, END
from typing import TypedDict


class TutorState(TypedDict):
    student_id:        str
    message:           str
    history:           list   # recent conversation turns for context
    intent:            str
    concept:           str
    response:          str
    agent_used:        str
    question_data:     dict
    assessment_result: dict
    next_action:       str
    re_teach_focus:    str
    kg:                str    # active KG: 'fods' or 'timeseries'


# ── Prompts ───────────────────────────────────────────────────────────────────

SINGLE_CLASSIFIER_PROMPT = """
You are a message router for an AI tutoring system.

Classify the student message into EXACTLY ONE of these categories:

CHAT     — pure casual chat, greetings, small talk (hi, thanks, how are you, bye)
ASSESS   — student explicitly wants to be tested/quizzed (test me, quiz me, give me a question)
COMPARE  — comparing methods, pros/cons, trade-offs, when to use, mathematical differences,
           what methods exist for a problem, project suggestions, recommendations for a goal
TEACH    — everything else: what is X, how does X work, explain X, how do I code X, show me X

Reply with ONLY one word: CHAT, ASSESS, COMPARE, or TEACH.
"""

FOLLOWUP_KEYWORDS_YES = {
    "yes", "sure", "please", "go ahead", "yeah", "yep", "definitely",
    "of course", "elaborate", "more", "tell me more", "explain more",
    "deeper", "full explanation", "why not", "show me", "give me more",
    "ok", "okay", "sounds good", "do it", "let's go", "great"
}
FOLLOWUP_KEYWORDS_NO = {
    "no", "nope", "thanks", "that's enough", "i'm good", "ok thanks",
    "no thanks", "not now", "skip", "never mind", "nevermind", "pass"
}

BRIEF_ANSWER_PROMPT = """
You are MOSAIC, an expert AI tutor teaching data science and machine learning.

Give a SHORT, direct answer to the student's question using YOUR OWN expert knowledge.

IDENTITY RULES — who you are:
- You are a tutor, not a researcher. Never say "the authors", "the paper", "the study", "the document", "according to the text", "the proposed approach". Speak as an expert teacher.
- Speak in first person or directly: "There are several techniques...", "The key idea is...", "You can think of it as..."

CONTENT RULES:
- Answer ONLY what the student asked — match their exact question
- The "Retrieved documentation" below is background context ONLY — it may be irrelevant or academic. If it doesn't directly answer the question, IGNORE it and answer from your own expert knowledge
- Never copy, quote, or paraphrase the retrieved docs — synthesise the answer yourself
- If the student asks for "types", "strategies", "methods" — name them specifically, don't give a vague summary
- 2-4 sentences maximum for the brief answer
- No bullet points, headers, or code blocks in the brief answer
- At the very end, ask: "Would you like a full breakdown with code examples?"
"""

BRIEF_RECOMMENDER_PROMPT = """
You are MOSAIC, an expert AI tutor teaching data science and machine learning.

Give a SHORT, direct answer to the student's comparison or recommendation question using YOUR OWN expert knowledge.

IDENTITY RULES — who you are:
- You are a tutor, not a researcher. Never say "the authors", "the paper", "the study", "the document", "according to the text". Speak as an expert teacher giving advice.
- Speak directly: "For time series, I'd recommend...", "The key difference is...", "SMOTE works by..."

CONTENT RULES:
- The "Retrieved documentation" is background context only — if it doesn't match the question, ignore it entirely and answer from your expert knowledge
- Never copy or paraphrase the retrieved docs
- Give a direct verdict or recommendation — don't hedge with "it depends" alone
- 2-4 sentences maximum
- No bullet points, headers, or code blocks
- At the very end, ask ONE specific follow-up:
  - Comparison: "Would you like a full side-by-side breakdown with code?"
  - Recommendation: "Want me to show you how to implement this?"
  - Project: "Want a full project roadmap with code snippets?"
  - Default: "Would you like a full breakdown with code examples?"
"""

FOLLOWUP_PROMPT = """
You are a message classifier.

The student was just given a brief answer and asked if they want more detail.

Did they say YES?

Answer YES for: yes, sure, please, go ahead, yeah, yep, definitely, of course, elaborate, more,
                tell me more, explain more, deeper, full explanation, why not, show me, give me more
Answer NO for: no, nope, thanks, that's enough, i'm good, ok thanks, no thanks, not now

Reply with ONLY one word: YES or NO.
"""

CASUAL_CHAT_PROMPT = """
You are MOSAIC, a friendly AI tutor specialising in data science.

Have a natural, warm, brief conversation. Keep it short and friendly.
If they ask what you can do, explain you teach data science concepts, compare methods,
recommend techniques for specific goals, and can test understanding.
"""


class Orchestrator:
    def __init__(self, solver, recommender, assessment, feedback, neo4j, letta):
        self.solver          = solver
        self.recommender     = recommender
        self.assessment      = assessment
        self.feedback        = feedback
        self.neo4j           = neo4j
        self.letta           = letta
        self.llm             = solver.llm
        self.last_agent_used = None
        self.pending_concept:  str | None = None
        self.pending_message:  str | None = None
        self.pending_intent:   str | None = None  # "solver" or "recommender"
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(TutorState)

        workflow.add_node("classify",         self._classify)
        workflow.add_node("chat",             self._run_casual_chat)
        workflow.add_node("brief",            self._run_brief_answer)
        workflow.add_node("brief_recommend",  self._run_brief_recommend)
        workflow.add_node("solver",           self._run_solver)
        workflow.add_node("recommender",      self._run_recommender)
        workflow.add_node("assessment",       self._run_assessment)

        workflow.set_entry_point("classify")

        workflow.add_conditional_edges(
            "classify",
            self._route,
            {
                "chat":            "chat",
                "brief":           "brief",
                "brief_recommend": "brief_recommend",
                "solver":          "solver",
                "recommender":     "recommender",
                "assessment":      "assessment",
            }
        )

        workflow.add_edge("chat",            END)
        workflow.add_edge("brief",           END)
        workflow.add_edge("brief_recommend", END)
        workflow.add_edge("solver",          END)
        workflow.add_edge("recommender",     END)
        workflow.add_edge("assessment",      END)

        return workflow.compile()

    def route(self, student_id: str, message: str, history: list = None, kg: str = "fods") -> str:
        state = {
            "student_id":        student_id,
            "message":           message,
            "history":           history or [],
            "intent":            "",
            "concept":           "",
            "response":          "",
            "agent_used":        "",
            "question_data":     {},
            "assessment_result": {},
            "next_action":       "",
            "re_teach_focus":    "",
            "kg":                kg,
        }
        result = self.graph.invoke(state)
        self.last_agent_used = result.get("agent_used", "Solver")
        return result.get("response", "I could not process that request.")

    def _classify(self, state: TutorState) -> TutorState:
        message     = state["message"].strip()
        msg_lower   = message.lower()
        msg_words   = set(msg_lower.replace(",", "").replace(".", "").split())

        # 1. Check if this is a follow-up yes/no to a pending deeper-explanation offer
        #    Use keywords only — no LLM call needed
        if self.pending_concept and self.pending_intent:
            # If the student is asking a NEW question (long message or contains
            # question words), treat it as a new question and clear pending
            is_new_question = (
                len(message.split()) > 8
                or any(w in msg_lower for w in ["what", "how", "why", "when", "which", "explain", "show"])
            )
            if is_new_question:
                # New question — clear pending and classify normally below
                self.pending_concept = None
                self.pending_message = None
                self.pending_intent  = None
            elif msg_words & FOLLOWUP_KEYWORDS_YES:
                concept          = self.pending_concept
                original_message = self.pending_message or message
                intent           = self.pending_intent
                self.pending_concept = None
                self.pending_message = None
                self.pending_intent  = None
                return {**state, "intent": intent, "concept": concept, "message": original_message}
            elif msg_words & FOLLOWUP_KEYWORDS_NO:
                self.pending_concept = None
                self.pending_message = None
                self.pending_intent  = None
                return {**state, "intent": "chat"}
            # else ambiguous short message — fall through to normal classification

        # 2. Single LLM call to classify intent
        try:
            label = self.llm.generate(
                system_prompt=SINGLE_CLASSIFIER_PROMPT,
                user_message=message
            ).strip().upper()
        except Exception:
            label = "TEACH"

        if label == "CHAT":
            return {**state, "intent": "chat"}
        elif label == "ASSESS":
            self.pending_concept = None
            self.pending_message = None
            self.pending_intent  = None
            return {**state, "intent": "assessment"}
        elif label == "COMPARE":
            return {**state, "intent": "brief_recommend"}
        else:  # TEACH
            return {**state, "intent": "brief"}

    def _route(self, state: TutorState) -> str:
        return state["intent"]

    def _extract_concept(self, message: str) -> str:
        try:
            concept = self.llm.generate(
                system_prompt=CONCEPT_EXTRACT_PROMPT,
                user_message=message
            ).strip()
            return concept if concept else message[:60]
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

    def _build_brief_context(self, message: str, history: list, retriever_fn) -> str:
        """Build user_message with RAG context + history for brief answers.
        Only includes history block if there are actual messages — prevents hallucination.
        """
        # Fetch RAG
        try:
            rag_docs    = retriever_fn(message)
            rag_context = "\n\n".join([d["text"] for d in rag_docs[:3]]) if rag_docs else ""
        except Exception:
            rag_context = ""

        # Format history — only if non-empty
        history_text = ""
        if history:
            turns = []
            for m in history[-6:]:
                role = "Student" if m["role"] == "user" else "Tutor"
                turns.append(f"{role}: {m['content'][:200]}")
            if turns:
                history_text = "Recent conversation:\n" + "\n".join(turns) + "\n"

        # Build context block
        parts = []
        if history_text:
            parts.append(history_text)
        else:
            parts.append("Recent conversation: (none — this is the student's first message)\n")

        parts.append(f"Retrieved documentation:\n{rag_context if rag_context else 'No documentation retrieved.'}")
        parts.append(f"\nStudent question: {message}")

        return "\n".join(parts)

    def _run_brief_answer(self, state: TutorState) -> TutorState:
        try:
            user_message = self._build_brief_context(
                state["message"],
                state.get("history", []),
                self.solver.retriever.retrieve_for_solver
            )
            response = self.llm.generate(
                system_prompt=BRIEF_ANSWER_PROMPT,
                user_message=user_message
            )
            # Use raw message as pending — no extra LLM call needed
            self.pending_concept = state["message"][:80]
            self.pending_message = state["message"] + " — give a complete explanation with detailed code examples and step by step breakdown"
            self.pending_intent  = "solver"
        except Exception as e:
            response = f"Something went wrong: {e}"
        return {**state, "response": response, "agent_used": "Solver"}

    def _run_brief_recommend(self, state: TutorState) -> TutorState:
        try:
            user_message = self._build_brief_context(
                state["message"],
                state.get("history", []),
                self.recommender.retriever.retrieve_for_recommender
            )
            response = self.llm.generate(
                system_prompt=BRIEF_RECOMMENDER_PROMPT,
                user_message=user_message
            )
            # Use raw message as pending — no extra LLM call needed
            self.pending_concept = state["message"][:80]
            self.pending_message = state["message"] + " — give a complete detailed comparison with code examples"
            self.pending_intent  = "recommender"
        except Exception as e:
            response = f"Something went wrong: {e}"
        return {**state, "response": response, "agent_used": "Recommender"}

    def _run_solver(self, state: TutorState) -> TutorState:
        response = self.solver.explain(
            student_id=state["student_id"],
            concept=state["concept"],
            focus=state.get("re_teach_focus") or None,
            message=state["message"],
            history=state.get("history", []),
            kg=state.get("kg", "fods")
        )
        return {**state, "response": response, "agent_used": "Solver"}

    def _run_recommender(self, state: TutorState) -> TutorState:
        response = self.recommender.recommend(
            student_id=state["student_id"],
            message=state["message"],
            mode="auto",
            history=state.get("history", []),
            kg=state.get("kg", "fods")
        )
        return {**state, "response": response, "agent_used": "Recommender"}

    def _run_assessment(self, state: TutorState) -> TutorState:
        response = (
            "Sure! Head over to the 📝 **Assessment** tab on the left to get tested. "
            "Type in the concept you want to practice and hit **Get Question →**."
        )
        return {**state, "response": response, "agent_used": "Solver"}

    def _run_feedback(self, state: TutorState) -> TutorState:
        fb = self.feedback.give_feedback(
            student_id=state["student_id"],
            concept=state["concept"],
            question=state.get("question_data", {}).get("question", ""),
            student_answer=state["message"],
            assessment_result=state.get("assessment_result", {}),
            kg=state.get("kg", "fods")
        )
        return {
            **state,
            "response":       fb["feedback_text"],
            "agent_used":     "Feedback",
            "next_action":    fb["next_action"],
            "re_teach_focus": fb.get("re_teach_focus", "")
        }
