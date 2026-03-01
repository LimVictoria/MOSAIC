# memory/letta_client.py
# Letta Cloud memory — no local server needed
# Uses Letta Cloud API at app.letta.com

import json
import os
from letta_client import Letta

# Read from environment / Streamlit secrets
LETTA_API_KEY  = os.getenv("LETTA_API_KEY", "")
LETTA_BASE_URL = os.getenv("LETTA_BASE_URL", "https://inference.letta.com")

MEMORY_PERSONA = """
You are the memory system for an AI engineering tutor.
You track everything about one student's learning journey.

Things worth storing in core memory (always visible):
- student skill level (beginner/intermediate/advanced)
- current topic being studied
- learning style (code_first/theory_first/visual)
- recurring weak areas (if seen 2+ times)
- goal (ai_engineer/data_scientist)

Things worth storing in archival memory (searched when needed):
- every assessment result with score
- every misconception identified
- every explanation approach tried
- every concept mastered

Things NOT worth storing:
- casual greetings
- simple yes/no exchanges
- duplicate records of same event

When you receive information from a teaching agent,
decide autonomously what to remember and how to update existing memories.
"""


class LettaClient:
    """
    Letta Cloud memory client.
    ONE Letta agent per student — shared by all three teaching agents.
    Connects to Letta Cloud — no local server needed.
    """

    def __init__(self):
        if not LETTA_API_KEY:
            raise ValueError(
                "LETTA_API_KEY not set. "
                "Add it to Streamlit secrets or your .env file. "
                "Get a free key at app.letta.com"
            )

        # Connect to Letta Cloud
        self.client = Letta(api_key=LETTA_API_KEY)
        self._agents = {}  # cache agent_id per student
        print("Connected to Letta Cloud")

    def get_or_create_agent(self, student_id: str) -> str:
        """
        Get or create ONE Letta Cloud agent per student.
        Shared by all three teaching agents.
        """
        if student_id in self._agents:
            return self._agents[student_id]

        # Check if agent already exists on Letta Cloud
        try:
            existing_agents = self.client.agents.list()
            for agent in existing_agents:
                if agent.name == f"tutor_memory_{student_id}":
                    self._agents[student_id] = agent.id
                    print(f"Found existing Letta agent for: {student_id}")
                    return agent.id
        except Exception:
            pass

        # Create new agent on Letta Cloud
        agent = self.client.agents.create(
            name=f"tutor_memory_{student_id}",
            system=MEMORY_PERSONA,
            memory_blocks=[
                {
                    "label": "human",
                    "value": json.dumps({
                        "student_id":        student_id,
                        "current_level":     "beginner",
                        "current_topic":     "",
                        "learning_style":    "code_first",
                        "goal":              "ai_engineer",
                        "weak_areas":        [],
                        "mastered_concepts": []
                    })
                },
                {
                    "label": "persona",
                    "value": MEMORY_PERSONA
                }
            ],
            model="letta-free",  # free tier model on Letta Cloud
        )

        self._agents[student_id] = agent.id
        print(f"Created Letta Cloud agent for student: {student_id}")
        return agent.id

    # ─────────────────────────────────────────────
    # Core Memory (RAM) — always in context
    # ─────────────────────────────────────────────

    def read_core_memory(self, student_id: str) -> dict:
        """Read student profile from core memory."""
        try:
            agent_id = self.get_or_create_agent(student_id)
            blocks   = self.client.agents.core_memory.retrieve(agent_id=agent_id)

            for block in blocks:
                if block.label == "human":
                    try:
                        return json.loads(block.value)
                    except json.JSONDecodeError:
                        return {"raw": block.value}
            return {}
        except Exception as e:
            print(f"read_core_memory error: {e}")
            return {}

    def update_core_memory(self, student_id: str, updates: dict):
        """Update student profile in core memory."""
        try:
            agent_id = self.get_or_create_agent(student_id)
            current  = self.read_core_memory(student_id)
            current.update(updates)

            self.client.agents.core_memory.modify(
                agent_id=agent_id,
                label="human",
                value=json.dumps(current)
            )
        except Exception as e:
            print(f"update_core_memory error: {e}")

    # ─────────────────────────────────────────────
    # Archival Memory (Disk) — searched on demand
    # ─────────────────────────────────────────────

    def write_archival_memory(self, student_id: str, data: dict):
        """Write a memory record to archival storage."""
        try:
            agent_id = self.get_or_create_agent(student_id)
            self.client.agents.archival_memory.create(
                agent_id=agent_id,
                text=json.dumps(data)
            )
        except Exception as e:
            print(f"write_archival_memory error: {e}")

    def search_archival_memory(self, student_id: str, query: str) -> list[dict]:
        """Search archival memory by semantic similarity."""
        try:
            agent_id = self.get_or_create_agent(student_id)
            results  = self.client.agents.archival_memory.list(
                agent_id=agent_id,
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
        except Exception as e:
            print(f"search_archival_memory error: {e}")
            return []

    # ─────────────────────────────────────────────
    # Convenience methods for teaching agents
    # ─────────────────────────────────────────────

    def get_mastered_concepts(self, student_id: str, kg: str = None) -> list[str]:
        """Get mastered concepts filtered by kg. Used by Solver for prerequisite check."""
        try:
            records = self.search_archival_memory(
                student_id, "mastered concept assessment passed"
            )
            mastered = [
                r.get("topic") or r.get("concept", "")
                for r in records
                if r.get("type") == "feedback_given"
                and r.get("passed")
                and (kg is None or r.get("kg", "fods") == kg)
            ]
            core          = self.read_core_memory(student_id)
            # Per-KG key: mastered_concepts_fods / mastered_concepts_timeseries
            kg_key        = f"mastered_concepts_{kg}" if kg else "mastered_concepts"
            core_mastered = core.get(kg_key, [])
            # Backwards compatibility — legacy key for fods
            if not core_mastered and kg == "fods":
                core_mastered = core.get("mastered_concepts", [])
            return list(set([c for c in mastered + core_mastered if c]))
        except Exception:
            return []

    def get_mistake_history(self, student_id: str, concept: str) -> list[dict]:
        """Get past mistakes for a concept. Used by Feedback Agent."""
        try:
            records = self.search_archival_memory(
                student_id, f"mistake wrong failed {concept}"
            )
            return [
                r for r in records
                if r.get("concept") == concept and not r.get("passed", True)
            ]
        except Exception:
            return []

    def get_tested_questions(self, student_id: str, concept: str) -> list[str]:
        """Get questions already asked. Used by Assessment Agent."""
        try:
            records = self.search_archival_memory(
                student_id, f"question asked {concept}"
            )
            return [
                r.get("question", "")
                for r in records
                if r.get("type") == "question_asked"
                and r.get("concept") == concept
            ]
        except Exception:
            return []
