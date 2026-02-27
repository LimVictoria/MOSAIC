# agents/solver_agent.py
# Explains concepts step by step
# Maps explained concepts to curriculum KG nodes via KG Builder

import json
from llm_client import LLMClient
from rag.retriever import RAGRetriever
from kg.neo4j_client import Neo4jClient
from memory.letta_client import LettaClient

CURRICULUM_TOPICS = [
    "Python for Data Science",
    "Reading Structured Files",
    "Structured Data Types",
    "Exploratory Data Analysis",
    "Data Visualization",
    "Imputation Techniques",
    "Data Augmentation",
    "Feature Reduction",
    "Business Metrics",
    "Preprocessing Summary",
    "ML Frameworks",
]

SOLVER_SYSTEM_PROMPT = """
You are MOSAIC, an expert data science tutor for the Foundations of Data Science (FODS) curriculum.

Your ONLY job is to explain concepts clearly and accurately, step by step.

The FODS curriculum covers EXACTLY these 11 topics:
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

Rules:
- Always explain step by step
- Always include working code examples
- Match your explanation to the student's level (from memory)
- If student is beginner, use analogies before technical terms
- If student struggles with math, use intuitive explanations first
- Never test the student
- Never tell the student what they got wrong
- Just explain, clearly and completely

IMPORTANT — curriculum boundary rule:
If the student asks about a topic NOT in the curriculum above (e.g. time series,
deep learning, NLP, reinforcement learning, statistics), do NOT just answer generically.
Instead:
1. Acknowledge their interest briefly
2. Tell them it is not covered in this FODS curriculum
3. Redirect them to the most relevant curriculum topic that IS covered
4. Offer to explain that topic instead

Example: "Time series analysis isn't part of this FODS curriculum, but the closest
topic we cover is Imputation Techniques, which deals with handling missing and
irregular data. Would you like to start there?"

You will be given the curriculum structure — use it to show students
how the current topic connects to what they have already learned
and what they will learn next.
"""


class SolverAgent:
    """
    Solver Agent.

    Triggered when:
    - Student asks a question
    - Feedback Agent decides re-teaching is needed

    After every explanation:
    - Maps the concept to nearest curriculum Topic node
    - Updates that node status to blue (currently studying)
    - No new nodes are created — curriculum is fixed

    Out-of-curriculum handling:
    - Detects when concept is outside the 11 FODS topics
    - Redirects to nearest relevant curriculum topic
    - Does not waste tokens explaining off-curriculum content
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

    def explain(self, student_id: str, concept: str, focus: str = None, message: str = None, history: list = None) -> str:
        """
        Explain a concept to the student.
        After explaining, maps concept to curriculum node and updates status to blue.
        If concept is outside curriculum, redirects to nearest relevant topic.
        """

        # 1. Read student profile from Letta memory
        student_memory = self.letta.read_core_memory(student_id)
        student_level  = student_memory.get("current_level", "intermediate")
        learning_style = student_memory.get("learning_style", "code_first")

        # 2. Get curriculum structure from Neo4j — passed to LLM as context
        curriculum = self.neo4j.get_curriculum_structure()
        curriculum_text = "\n".join([
            f"- {c['topic']} (status: {c['status']}) "
            f"{'← requires: ' + ', '.join(c['prerequisites']) if c['prerequisites'] else ''}"
            for c in curriculum
        ])

        # 3. Check if concept maps to a curriculum topic
        matched_topic    = self.neo4j.map_concept_to_topic(concept)
        is_in_curriculum = matched_topic is not None

        # 4. Get prerequisites for this concept from curriculum KG
        prerequisites   = self.neo4j.get_prerequisites(concept) if is_in_curriculum else []
        mastered        = self.letta.get_mastered_concepts(student_id)
        missing_prereqs = [p for p in prerequisites if p not in mastered]

        # 5. Check if prerequisites are unmastered — warn student
        unmastered = self.neo4j.get_unmastered_prerequisites(concept) if is_in_curriculum else []

        # 6. Get related topics from KG
        related = self.neo4j.get_related_topics(concept) if is_in_curriculum else []

        # 7. Query RAG for relevant documentation
        query       = focus if focus else (message if message else concept)
        rag_docs    = self.retriever.retrieve_for_solver(query)
        rag_context = "\n\n".join([doc["text"] for doc in rag_docs])

        # 8. Format recent conversation history for context
        history_text = ""
        if history:
            turns = []
            for m in (history or [])[-6:]:
                role    = "Student" if m["role"] == "user" else "Tutor"
                snippet = m["content"][:200]
                turns.append(f"{role}: {snippet}")
            history_text = "Recent conversation (for context):\n" + "\n".join(turns)

        # 9. Build prompt — include out-of-curriculum flag for LLM awareness
        out_of_curriculum_note = ""
        if not is_in_curriculum:
            closest = self._find_closest_curriculum_topic(concept)
            out_of_curriculum_note = f"""
\u26a0\ufe0f OUT-OF-CURRICULUM TOPIC DETECTED
The concept "{concept}" is NOT part of the FODS curriculum.
Do NOT provide a full explanation of this topic.
Instead: acknowledge their interest, say it's not in this curriculum,
and redirect them to the closest relevant topic: "{closest}".
"""

        user_message = f"""
Student Level:  {student_level}
Learning Style: {learning_style}

{history_text}

Concept to explain: {concept}
{f"Specific focus: {focus}" if focus else ""}
{f"Student message: {message}" if message else ""}

{out_of_curriculum_note}

Curriculum structure (for context):
{curriculum_text}

{"⚠️ Unmastered prerequisites: " + str([p for p in unmastered]) if unmastered else "Prerequisites: all covered"}
Related topics to connect: {related[:3] if related else "None"}

Relevant documentation:
{rag_context if rag_context else "No documentation available — use your knowledge."}

{"Provide a clear step by step explanation." if is_in_curriculum else "Redirect the student as instructed above."}
{"Start with code, then explain the theory." if learning_style == "code_first" and is_in_curriculum else ""}
{"Briefly mention the student should cover " + str(missing_prereqs) + " first as they are prerequisites." if missing_prereqs else ""}
"""

        # 9. Generate explanation
        explanation = self.llm.generate(
            system_prompt=SOLVER_SYSTEM_PROMPT,
            user_message=user_message
        )

        # 10. Write to Letta memory
        self.letta.write_archival_memory(student_id, {
            "type":          "explanation_given",
            "concept":       concept,
            "focus":         focus,
            "student_level": student_level,
            "approach":      learning_style,
            "in_curriculum": is_in_curriculum,
            "matched_topic": matched_topic,
        })

        # 11. Map concept to curriculum Topic node and set status to blue
        # Only update KG if concept is actually in curriculum
        if is_in_curriculum:
            self._update_curriculum_node(concept, "blue")

        return explanation

    def _find_closest_curriculum_topic(self, concept: str) -> str:
        """
        Find the most relevant curriculum topic for an out-of-curriculum concept.
        Used to redirect students to what we DO cover.
        """
        concept_lower = concept.lower()

        redirects = {
            "time series":      "Imputation Techniques",
            "forecasting":      "Business Metrics",
            "nlp":              "ML Frameworks",
            "natural language": "ML Frameworks",
            "deep learning":    "ML Frameworks",
            "neural network":   "ML Frameworks",
            "reinforcement":    "ML Frameworks",
            "computer vision":  "ML Frameworks",
            "statistics":       "Exploratory Data Analysis",
            "hypothesis":       "Exploratory Data Analysis",
            "regression":       "Feature Reduction",
            "classification":   "Data Augmentation",
            "clustering":       "Feature Reduction",
            "database":         "Reading Structured Files",
            "sql":              "Reading Structured Files",
            "api":              "Python for Data Science",
            "web scraping":     "Python for Data Science",
            "cloud":            "ML Frameworks",
            "deployment":       "ML Frameworks",
        }

        for keyword, topic in redirects.items():
            if keyword in concept_lower:
                return topic

        # Fall back to next recommended topic in curriculum
        next_topic = self.neo4j.get_next_recommended_topic()
        return next_topic or "Python for Data Science"

    def _update_curriculum_node(self, concept: str, status: str):
        """
        Map the concept to the nearest curriculum Topic node
        and update its status. No new nodes are created.
        """
        try:
            matched = self.neo4j.map_concept_to_topic(concept)
            if matched:
                self.neo4j.update_node_status(matched, status)
                print(f"Solver: '{concept}' → '{matched}' status={status}")
            else:
                print(f"Solver: '{concept}' could not be mapped to curriculum topic")
        except Exception as e:
            print(f"Solver KG update error (non-fatal): {e}")
