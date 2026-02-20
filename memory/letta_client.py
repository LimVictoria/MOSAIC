# memory/letta_client.py
# Letta memory framework — one agent per student, shared across all teaching agents

from letta import create_client
from letta.schemas.memory import ChatMemory
from letta.schemas.llm_config import LLMConfig
from config import LETTA_BASE_URL, OLLAMA_BASE_URL, LLM_MODEL
import json


MEMORY_PERSONA = """
You are the memory system for one student learning AI engineering.

You track everything about their learning journey across all teaching agents.

Things worth updating in CORE memory (always in context):
- Student skill level changes (beginner → intermediate → advanced)
- Recurring struggles seen 2+ times — these are priority weak areas
- Learning style preferences (code_first / theory_first / visual)
- Current topic and learning goal

Things worth storing in ARCHIVAL memory (long term history):
- Every assessment result with score and concept
- Every misconception identified by Feedback Agent
- Every explanation approach tried by Solver Agent
- Patterns you notice across multiple sessions

Things NOT worth storing:
- Casual greetings or one-word replies
- Repeated identical questions
- Simple yes/no exchanges

When Solver, Assessment, or Feedback agents send you an event,
read it carefully and decide autonomously what to remember and how.
Consolidate patterns — do not just append forever.
"""


class LettaClient:
    """
    Letta memory client.

    One Letta agent per student.
    All three teaching agents (Solver, Assessment, Feedback)
    share the same Letta agent for their student.

    The LLM inside Letta decides what to remember —
    we guide it via the persona, not hardcoded rules.

    Core Memory  = always in context window (like RAM)
    Archival Memory = outside context, searched on demand (like disk)
    """

    def __init__(self):
        self.client = create_client(base_url=LETTA_BASE_URL)
        self._agent_ids = {}  # cache: student_id → letta agent_id

    def get_or_create_agent(self, student_id: str) -> str:
        """
        Get or create one Letta agent for this student.
        All three teaching agents share this same Letta agent.
        """
        if student_id in self._agent_ids:
            return self._agent_ids[student_id]

        # Create a new Letta agent with LLaMA inside
        agent = self.client.create_agent(
            name=f"student_{student_id}_memory",

            # LLM config — same LLaMA model, running via Ollama
            llm_config=LLMConfig(
                model=LLM_MODEL,
                model_endpoint=OLLAMA_BASE_URL,
                model_endpoint_type="ollama",
                context_window=8192
            ),

            # Initial memory state
            memory=ChatMemory(
                human=json.dumps({
                    "student_id": student_id,
                    "current_level": "beginner",
                    "current_topic": "",
                    "learning_style": "code_first",
                    "goal": "ai_engineer",
                    "weak_areas": [],
                    "mastered_concepts": []
                }),
                persona=MEMORY_PERSONA
            )
        )

        self._agent_ids[student_id] = agent.id
        print(f"Created Letta agent for student: {student_id} → {agent.id}")
        return agent.id

    # ─────────────────────────────────────────────
    # Core Memory — always in context window
    # ─────────────────────────────────────────────

    def read_core_memory(self, student_id: str) -> dict:
        """
        Read student profile from core memory.
        Called by Solver to get student level and learning style.
        """
        agent_id = self.get_or_create_agent(student_id)
        memory = self.client.get_in_context_memory(agent_id)
        human_block = memory.get_block("human")

        if human_block:
            try:
                return json.loads(human_block.value)
            except json.JSONDecodeError:
                return {"raw": human_block.value}
        return {}

    def update_core_memory(self, student_id: str, updates: dict):
        """
        Update student profile in core memory.
        Called when student level or preferences change.
        """
        agent_id = self.get_or_create_agent(student_id)
        current = self.read_core_memory(student_id)
        current.update(updates)
        self.client.update_in_context_memory(
            agent_id,
            section="human",
            value=json.dumps(current)
        )

    # ─────────────────────────────────────────────
    # Archival Memory — long term storage
    # ─────────────────────────────────────────────

    def notify_memory(self, student_id: str, event: dict):
        """
        Send an event to the Letta agent and let the LLM
        decide what to remember and how.

        This is the CORRECT MemGPT approach —
        the LLM reads the event and autonomously decides
        whether to call core_memory_append, core_memory_replace,
        archival_memory_insert, or do nothing.

        Called by all three teaching agents after each interaction.
        """
        agent_id = self.get_or_create_agent(student_id)

        # Send event as a message — LLM decides what to store
        self.client.send_message(
            agent_id=agent_id,
            message=f"TUTOR EVENT: {json.dumps(event)}",
            role="system"
        )

    def write_archival_memory(self, student_id: str, data: dict):
        """
        Directly insert a record into archival memory.
        Used for guaranteed storage of critical records
        (assessment scores, misconceptions).
        """
        agent_id = self.get_or_create_agent(student_id)
        self.client.insert_archival_memory(
            agent_id,
            memory=json.dumps(data)
        )

    def search_archival_memory(self, student_id: str, query: str) -> list[dict]:
        """
        Search archival memory by semantic similarity.
        Returns relevant past records.
        """
        agent_id = self.get_or_create_agent(student_id)
        results = self.client.get_archival_memory(
            agent_id,
            query=query,
            limit=10
        )

        parsed = []
        for r in results:
            try:
                parsed.append(json.loads(r.text))
            except json.JSONDecodeError:
                parsed.append({"raw": r.text})
        return parsed

    # ─────────────────────────────────────────────
    # Convenience methods used by teaching agents
    # ─────────────────────────────────────────────

    def get_mastered_concepts(self, student_id: str) -> list[str]:
        """
        Get list of mastered concepts.
        Used by Solver to check which prerequisites student knows.
        """
        core = self.read_core_memory(student_id)
        mastered = core.get("mastered_concepts", [])

        # Also search archival for passed assessments
        records = self.search_archival_memory(
            student_id, "assessment passed mastered concept"
        )
        for r in records:
            if r.get("type") == "feedback_given" and r.get("passed"):
                concept = r.get("concept", "")
                if concept and concept not in mastered:
                    mastered.append(concept)

        return mastered

    def get_mistake_history(self, student_id: str, concept: str) -> list[dict]:
        """
        Get all past failed attempts for a concept.
        Used by Feedback Agent to count attempts and spot patterns.
        """
        records = self.search_archival_memory(
            student_id, f"failed wrong misconception {concept}"
        )
        return [
            r for r in records
            if r.get("concept") == concept and not r.get("passed", True)
        ]

    def get_tested_questions(self, student_id: str, concept: str) -> list[str]:
        """
        Get questions already asked about a concept.
        Used by Assessment Agent to avoid repeating questions.
        """
        records = self.search_archival_memory(
            student_id, f"question asked {concept}"
        )
        return [
            r.get("question", "")
            for r in records
            if r.get("type") == "question_asked" and r.get("concept") == concept
        ]
