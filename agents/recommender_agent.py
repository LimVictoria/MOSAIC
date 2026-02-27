# agents/recommender_agent.py
# Recommender Agent — compares methods, recommends techniques, suggests projects
# Triggered by: comparisons, "which is better", "what should I use", "pros/cons",
#               "when to use", "mathematically", "what methods exist"
# NOT triggered by: "how do I code", "show me the implementation" → those go to Solver

from llm_client import LLMClient
from rag.retriever import RAGRetriever
from kg.neo4j_client import Neo4jClient
from memory.letta_client import LettaClient

RECOMMENDER_SYSTEM_PROMPT = """
You are MOSAIC's Recommender Agent — an expert at comparing data science methods,
recommending the right technique for a given goal, and suggesting projects.

You are part of the FODS (Foundations of Data Science) curriculum which covers:
1. Python for Data Science
2. Reading Structured Files
3. Structured Data Types
4. Exploratory Data Analysis
5. Data Visualization
6. Imputation Techniques
7. Data Augmentation
8. Feature Reduction
9. Business Metrics
10. Preprocessing Summary
11. ML Frameworks

Your THREE modes:

── MODE 1: COMPARE ──
When asked to compare two or more methods (e.g. PCA vs t-SNE, SMOTE vs ADASYN):
- Give a structured comparison: what each does, when to use each, trade-offs
- Include mathematical intuition if relevant (don't avoid math)
- Show a short code snippet for each
- Give a clear recommendation based on the scenario
- Format as: Overview → Key Differences → When to use each → Code → Verdict

── MODE 2: RECOMMEND ──
When asked which method to use for a goal (e.g. "what should I use for fraud detection"):
- Ask clarifying questions only if truly necessary — otherwise just recommend
- Recommend the best technique for the goal with clear reasoning
- If the student hasn't learned it yet, ENCOURAGE them to learn it — never block them
- Show what they already know that's relevant
- Suggest the learning path: "You know X, next learn Y, then Z"

── MODE 3: PROJECT ──
When asked to suggest a project:
- Propose a project relevant to the student's goal or curriculum topic
- Use techniques the student has already mastered as the foundation
- Introduce new techniques as stretch goals — encourage, never gatekeep
- Give a clear project roadmap: dataset → steps → techniques → outcome
- Connect to real-world business metrics from the curriculum (fraud, churn, forecasting)

Rules that apply to ALL modes:
- Never gatekeep — if a technique is the right one, recommend it even if student hasn't learned it yet
- Always be direct — give a verdict, not just "it depends"
- Use the student's mastery level to calibrate depth
- If a topic is outside FODS curriculum, still answer — this agent is not curriculum-bound
- Be precise about mathematical differences when asked
- Always connect back to curriculum topics where possible
"""


class RecommenderAgent:
    """
    Recommender Agent.

    Three modes:
      compare   — side-by-side method comparison with verdict
      recommend — best technique for a given goal
      project   — project suggestion tied to mastery + stretch goals

    Never gatekeeps — if a technique is right for the goal, recommend it
    even if student hasn't learned it yet. Encourage, never block.

    Reads from:  Letta (mastery, level), Neo4j KG (curriculum structure, mastery)
                 RAG (method details, documentation)
    Writes to:   Letta (archival — recommendation given)
    Updates KG:  No status updates — read only
    """

    def __init__(
        self,
        llm: LLMClient,
        retriever: RAGRetriever,
        neo4j: Neo4jClient,
        letta: LettaClient
    ):
        self.llm       = llm
        self.retriever = retriever
        self.neo4j     = neo4j
        self.letta     = letta

    def recommend(self, student_id: str, message: str, mode: str = "auto", history: list = None) -> str:
        """
        Main entry point. Mode is auto-detected if not specified.

        Args:
            student_id: student session ID
            message:    student's original message
            mode:       "compare" | "recommend" | "project" | "auto"

        Returns:
            Formatted recommendation string
        """

        # 1. Read student profile
        student_memory = self.letta.read_core_memory(student_id)
        student_level  = student_memory.get("current_level", "intermediate")
        learning_style = student_memory.get("learning_style", "code_first")

        # 2. Get mastered concepts
        mastered = self.letta.get_mastered_concepts(student_id)

        # 3. Get full curriculum structure from KG
        curriculum = self.neo4j.get_curriculum_structure()
        curriculum_text = "\n".join([
            f"- {c['topic']} (status: {c['status']})"
            for c in curriculum
        ])

        # 4. Get mastery timeline if available
        try:
            timeline = self.neo4j.get_mastery_timeline()
            mastery_text = ", ".join([t["name"] for t in timeline]) if timeline else "None yet"
        except Exception:
            mastery_text = ", ".join(mastered) if mastered else "None yet"

        # 5. Auto-detect mode
        if mode == "auto":
            mode = self._detect_mode(message)

        # 6. Retrieve RAG context — method docs, papers, examples
        rag_docs    = self.retriever.retrieve_for_recommender(message)
        rag_context = "\n\n".join([doc["text"] for doc in rag_docs[:4]])

        # 7. Build mode-specific instructions
        mode_instruction = self._get_mode_instruction(mode, message)

        # 8. Format recent conversation history
        history_text = ""
        if history:
            turns = []
            for m in (history or [])[-6:]:
                role    = "Student" if m["role"] == "user" else "Tutor"
                snippet = m["content"][:200]
                turns.append(f"{role}: {snippet}")
            history_text = "Recent conversation (use this to resolve abbreviations and understand context):\n" + "\n".join(turns)

        # 9. Build prompt
        user_message = f"""
Student Level:    {student_level}
Learning Style:   {learning_style}
Mastered Topics:  {mastery_text}

{history_text}

Student message: {message}

{mode_instruction}

Curriculum structure:
{curriculum_text}

Relevant documentation from knowledge base:
{rag_context if rag_context else "Not available — use your expert knowledge."}
"""

        # 9. Generate response
        response = self.llm.generate(
            system_prompt=RECOMMENDER_SYSTEM_PROMPT,
            user_message=user_message
        )

        # 10. Write to Letta archival memory
        self.letta.write_archival_memory(student_id, {
            "type":          "recommendation_given",
            "mode":          mode,
            "message":       message,
            "student_level": student_level,
        })

        return response

    def _detect_mode(self, message: str) -> str:
        """
        Detect which mode to use based on message content.

        compare   — comparison keywords
        recommend — goal/selection keywords
        project   — project keywords
        """
        msg = message.lower()

        compare_keywords = [
            "vs", "versus", "compare", "difference between", "better than",
            "pros and cons", "trade-off", "tradeoff", "which is better",
            "mathematically", "math behind", "when to use", "why is",
            "what are the different", "methods for", "techniques for",
            "pros", "cons", "advantages", "disadvantages",
        ]

        project_keywords = [
            "project", "build", "create", "design", "implement a system",
            "end to end", "end-to-end", "capstone", "portfolio",
            "suggest a project", "give me a project", "project idea",
        ]

        recommend_keywords = [
            "should i use", "what should i", "recommend", "suggest",
            "which method", "what method", "best method", "best technique",
            "what to use", "which to use", "for my", "for this",
            "suitable for", "appropriate for", "good for",
        ]

        # Check project first (most specific)
        if any(k in msg for k in project_keywords):
            return "project"

        # Then compare
        if any(k in msg for k in compare_keywords):
            return "compare"

        # Then recommend
        if any(k in msg for k in recommend_keywords):
            return "recommend"

        # Default to recommend
        return "recommend"

    def _get_mode_instruction(self, mode: str, message: str) -> str:
        """Return mode-specific instruction injected into the prompt."""

        if mode == "compare":
            return f"""
MODE: COMPARE
The student wants a comparison. Structure your response as:
1. Brief overview of each method
2. Key differences (use a markdown table if there are 3+ dimensions)
3. Mathematical intuition if relevant — don't shy away from math
4. When to use each (with concrete scenarios)
5. Short code example for each
6. Clear verdict — pick one for the most common scenario

Student question: {message}
"""

        elif mode == "project":
            return f"""
MODE: PROJECT
The student wants a project suggestion. Structure your response as:
1. Project title and one-line description
2. Real-world relevance (connect to business metrics: fraud, churn, forecasting, etc.)
3. Dataset suggestion (public dataset they can use)
4. Step-by-step project roadmap
5. Techniques used — mark each as:
   ✅ Already mastered  (from their mastery list)
   📚 Needs to learn first  (encourage, don't block)
   🔥 Stretch goal  (optional advanced extension)
6. Expected outcome and what they'll be able to show

Student question: {message}
"""

        else:  # recommend
            return f"""
MODE: RECOMMEND
The student needs a recommendation. Structure your response as:
1. Understand their goal (state it back briefly)
2. Recommended technique(s) — be direct, give a clear answer
3. Why this technique fits their goal
4. If they haven't learned it yet — encourage them and point to the curriculum topic
5. Alternative techniques and when to consider them instead
6. Next step: what to do right now

Student question: {message}
"""
