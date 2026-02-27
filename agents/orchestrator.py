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

IS_RECOMMENDER_PROMPT = """
You are a message classifier for an AI tutoring system.

Should this message be routed to the RECOMMENDER agent?

Route to RECOMMENDER (answer YES) when the student is:
- Comparing two or more methods (vs, versus, compare, difference between, better than)
- Asking which technique/method to use for a goal (should I use, what method, recommend, suggest)
- Asking about pros/cons/trade-offs of methods (pros, cons, advantages, disadvantages, trade-off)
- Asking why one approach is better than another (why is X better than Y)
- Asking what methods/techniques exist for a problem (what are the different methods for)
- Asking mathematical or theoretical differences between approaches (mathematically, math behind)
- Asking when to use a technique (when should I use, when to use)
- Asking for a project suggestion (suggest a project, project idea, build a system)
- Asking for a recommendation for their specific goal (what should I use for fraud detection)

Route to SOLVER (answer NO) when the student is:
- Asking how to code or implement something (how do I code, show me how to implement)
- Asking for a step-by-step tutorial on one concept
- Asking what a single concept means (what is PCA, explain SMOTE)
- Asking for a code example of one specific thing

Examples:
"why is TFT better than 1D CNN-LSTM" → YES (comparison)
"what are the different methods to do resampling" → YES (what methods exist)
"mathematically, should we use T-SMOTE or ADASYN" → YES (mathematical comparison)
"how do I code SMOTE in Python" → NO (implementation)
"explain PCA step by step" → NO (explain one concept)
"compare SimpleImputer vs KNNImputer" → YES (comparison)
"what method should I use for imbalanced data" → YES (recommendation)
"show me how to read a CSV file" → NO (implementation)
"which is better for fraud detection, SMOTE or random oversampling" → YES (comparison + recommendation)
"what are the pros and cons of mean imputation" → YES (pros/cons)
"how do I implement KNN imputer in sklearn" → NO (implementation)
"when should I use PCA vs t-SNE" → YES (when to use + comparison)
"walk me through the math behind PCA" → YES (mathematical)
"how do I drop null values in pandas" → NO (implementation)
"is feature scaling necessary before PCA" → YES (recommendation)
"what should I use if my dataset has 95% missing values" → YES (recommendation)
"suggest a project for churn prediction" → YES (project)

Reply with ONLY one word: YES or NO.
"""

CONCEPT_EXTRACT_PROMPT = """
You are a curriculum topic classifier for a data science course.

Map the student's message to the most relevant curriculum topic from this list:
- Python for Data Science
- Reading Structured Files
- Structured Data Types
- Exploratory Data Analysis
- Data Visualization
- Imputation Techniques
- Data Augmentation
- Feature Reduction
- Business Metrics
- Preprocessing Summary
- ML Frameworks

Reply with ONLY the exact topic name from the list above.
If none match clearly, reply with the most relevant topic.
"""

BRIEF_ANSWER_PROMPT = """
You are MOSAIC, a friendly AI tutor specialising in data science.

Give a SHORT, conversational answer to the student's question.

CRITICAL: You will be given recent conversation history and retrieved documentation.
- Use the conversation history to resolve abbreviations (e.g. if "TFT" was mentioned as
  "Temporal Fusion Transformer" earlier, treat it as such — never guess from training knowledge)
- Use the retrieved documentation as your primary source of truth
- If the documentation says something different from your training knowledge, trust the documentation

Rules:
- Answer in 2-5 sentences maximum
- Be warm, clear, and direct
- No bullet points, headers, or code blocks
- At the very end, ask: "Would you like a more detailed explanation?"
"""

BRIEF_RECOMMENDER_PROMPT = """
You are MOSAIC, a friendly AI tutor specialising in data science.

Give a SHORT, conversational answer that directly addresses the comparison or recommendation.

CRITICAL: You will be given recent conversation history and retrieved documentation.
- Use the conversation history to resolve abbreviations (e.g. if "TFT" was mentioned as
  "Temporal Fusion Transformer" earlier, treat it as such — never guess from training knowledge)
- Use the retrieved documentation as your primary source of truth
- If the documentation says something different from your training knowledge, trust the documentation

Rules:
- Answer in 2-5 sentences maximum — give the key insight immediately
- Be direct — give a verdict or clear recommendation, don't hedge
- No bullet points, headers, or code blocks
- At the very end, ask ONE specific question based on what would help most:
  - If it's a comparison: "Would you like a deeper comparison with code examples for both?"
  - If it's a recommendation: "Would you like me to show you how to implement this with code?"
  - If it's a project: "Would you like a full project roadmap with code snippets?"
  - Default: "Would you like a deeper explanation with code examples?"
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
    """
    Chat-first orchestrator.

    Routing logic:
      Recommender keywords (compare/vs/recommend/project/pros-cons/math) → brief_recommend → Recommender
      Technical questions (explain/how/what is)                          → brief → Solver
      Explicit test requests                                              → Assessment
      Pure casual chat                                                    → casual reply

    For both Solver and Recommender:
      First gives a brief answer + "want more detail?"
      If student says yes → full response from the appropriate agent
    """

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

    def route(self, student_id: str, message: str, history: list = None) -> str:
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
            "re_teach_focus":    ""
        }
        result = self.graph.invoke(state)
        self.last_agent_used = result.get("agent_used", "Solver")
        return result.get("response", "I could not process that request.")

    def _classify(self, state: TutorState) -> TutorState:
        message = state["message"].strip().lower()

        # 1. Explicit assessment — always takes priority
        assessment_words = [
            "test me", "quiz me", "assess me", "give me a question",
            "practice question", "test my", "test on", "quiz on",
            "want to test", "want to be tested", "test my understanding",
            "test on my", "i want to test", "assess my", "check my understanding",
            "check my knowledge", "practice on", "want to practice",
        ]
        if any(w in message for w in assessment_words):
            self.pending_concept = None
            self.pending_message = None
            self.pending_intent  = None
            return {**state, "intent": "assessment"}

        # 2. Check if this is a follow-up yes/no to a pending concept
        if self.pending_concept and self.pending_intent:
            try:
                answer = self.llm.generate(
                    system_prompt=FOLLOWUP_PROMPT,
                    user_message=state["message"]
                ).strip().upper()

                if answer.startswith("YES"):
                    concept         = self.pending_concept
                    original_message = self.pending_message or state["message"]
                    intent          = self.pending_intent  # "solver" or "recommender"
                    self.pending_concept = None
                    self.pending_message = None
                    self.pending_intent  = None
                    return {
                        **state,
                        "intent":  intent,
                        "concept": concept,
                        "message": original_message
                    }
                else:
                    self.pending_concept = None
                    self.pending_message = None
                    self.pending_intent  = None
                    # Fall through to normal classification
            except Exception:
                self.pending_concept = None
                self.pending_message = None
                self.pending_intent  = None

        # 3. Pure casual chat keywords
        casual_words = [
            "hi", "hello", "hey", "how are you", "what do you do",
            "who are you", "thanks", "thank you", "good morning",
            "good evening", "bye", "goodbye", "what's up", "whats up",
            "ok", "okay", "cool", "nice", "great", "awesome",
            "that makes sense", "got it", "i see", "what can you do",
        ]
        if any(message == w or message.startswith(w + " ") for w in casual_words):
            return {**state, "intent": "chat"}

        # 4. LLM classifier — is it technical at all?
        try:
            is_technical = self.llm.generate(
                system_prompt=IS_TECHNICAL_PROMPT,
                user_message=state["message"]
            ).strip().upper()
        except Exception:
            is_technical = "YES"

        if not is_technical.startswith("YES"):
            return {**state, "intent": "chat"}

        # 5. LLM classifier — should it go to Recommender?
        try:
            is_recommender = self.llm.generate(
                system_prompt=IS_RECOMMENDER_PROMPT,
                user_message=state["message"]
            ).strip().upper()
        except Exception:
            is_recommender = "NO"

        if is_recommender.startswith("YES"):
            return {**state, "intent": "brief_recommend"}

        # 6. Default — technical question goes to brief → Solver
        return {**state, "intent": "brief"}

    def _route(self, state: TutorState) -> str:
        return state["intent"]

    def _extract_concept(self, message: str) -> str:
        """Map student message to nearest curriculum Topic."""
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
        """Build user_message with RAG context + history for brief answers."""
        # Fetch RAG
        try:
            rag_docs    = retriever_fn(message)
            rag_context = "\n\n".join([d["text"] for d in rag_docs[:3]]) if rag_docs else ""
        except Exception:
            rag_context = ""

        # Format history
        history_text = ""
        if history:
            turns = []
            for m in history[-6:]:
                role = "Student" if m["role"] == "user" else "Tutor"
                turns.append(f"{role}: {m['content'][:200]}")
            history_text = "Recent conversation:\n" + "\n".join(turns)

        return f"""{history_text}

Retrieved documentation:
{rag_context if rag_context else "No documentation retrieved."}

Student question: {message}"""

    def _run_brief_answer(self, state: TutorState) -> TutorState:
        """Brief answer with RAG + history → store pending for Solver follow-up."""
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
            concept = self._extract_concept(state["message"])
            if concept:
                self.pending_concept = concept
                # Store original message + explicit instruction to include code
                self.pending_message = state["message"] + " — include detailed code examples and step by step explanation"
                self.pending_intent  = "solver"
        except Exception as e:
            response = f"Something went wrong: {e}"
        return {**state, "response": response, "agent_used": "Solver"}

    def _run_brief_recommend(self, state: TutorState) -> TutorState:
        """Brief recommendation with RAG + history → store pending for Recommender follow-up."""
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
            concept = self._extract_concept(state["message"])
            if concept:
                self.pending_concept = concept
                # Store original message + explicit instruction to include code
                self.pending_message = state["message"] + " — include detailed code examples"
                self.pending_intent  = "recommender"
        except Exception as e:
            response = f"Something went wrong: {e}"
        return {**state, "response": response, "agent_used": "Recommender"}

    def _run_solver(self, state: TutorState) -> TutorState:
        """Full explanation from Solver."""
        response = self.solver.explain(
            student_id=state["student_id"],
            concept=state["concept"],
            focus=state.get("re_teach_focus") or None,
            message=state["message"],
            history=state.get("history", [])
        )
        return {**state, "response": response, "agent_used": "Solver"}

    def _run_recommender(self, state: TutorState) -> TutorState:
        """Full recommendation from Recommender."""
        response = self.recommender.recommend(
            student_id=state["student_id"],
            message=state["message"],
            mode="auto",
            history=state.get("history", [])
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
            assessment_result=state.get("assessment_result", {})
        )
        return {
            **state,
            "response":       fb["feedback_text"],
            "agent_used":     "Feedback",
            "next_action":    fb["next_action"],
            "re_teach_focus": fb.get("re_teach_focus", "")
        }
