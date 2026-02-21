# memory/letta_client.py
import json
import os
from letta_client import Letta

def _get_secret(key, default=""):
    try:
        import streamlit as st
        return st.secrets.get(key, default) or os.getenv(key, default)
    except Exception:
        return os.getenv(key, default)

LETTA_API_KEY  = _get_secret("LETTA_API_KEY", "")
LETTA_BASE_URL = "https://api.letta.com"
print(f"Letta base URL: {LETTA_BASE_URL}")

class LettaClient:
    def __init__(self):
        if not LETTA_API_KEY:
            raise ValueError("LETTA_API_KEY not set.")
        self.client = Letta(api_key=LETTA_API_KEY, base_url=LETTA_BASE_URL)
        self._agents = {}
        print("Connected to Letta Cloud")

    def get_or_create_agent(self, student_id: str) -> str:
        if student_id in self._agents:
            return self._agents[student_id]
        try:
            existing_agents = self.client.agents.list()
            for agent in existing_agents:
                if agent.name == f"tutor_memory_{student_id}":
                    self._agents[student_id] = agent.id
                    print(f"Found existing Letta agent for: {student_id}")
                    return agent.id
        except Exception as e:
            print(f"Letta list agents error: {e}")
        try:
            agent = self.client.agents.create(
                name=f"tutor_memory_{student_id}",
                model="letta-free",
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
                        "value": "I am a student memory tracker for an AI tutor."
                    }
                ],
            )
            self._agents[student_id] = agent.id
            print(f"Created Letta Cloud agent: {agent.id}")
            return agent.id
        except Exception as e:
            print(f"Failed to create Letta agent: {e}")
            raise

    def read_core_memory(self, student_id: str) -> dict:
        try:
            agent_id = self.get_or_create_agent(student_id)
            blocks = self.client.agents.core_memory.retrieve(agent_id=agent_id)
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
        try:
            agent_id = self.get_or_create_agent(student_id)
            current = self.read_core_memory(student_id)
            current.update(updates)
            self.client.agents.core_memory.modify(
                agent_id=agent_id, label="human", value=json.dumps(current)
            )
        except Exception as e:
            print(f"update_core_memory error: {e}")

    def write_archival_memory(self, student_id: str, data: dict):
        try:
            agent_id = self.get_or_create_agent(student_id)
            self.client.agents.archival_memory.create(
                agent_id=agent_id, text=json.dumps(data)
            )
        except Exception as e:
            print(f"write_archival_memory error: {e}")

    def search_archival_memory(self, student_id: str, query: str) -> list[dict]:
        try:
            agent_id = self.get_or_create_agent(student_id)
            results = self.client.agents.archival_memory.list(
                agent_id=agent_id, query=query, limit=10
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

    def get_mastered_concepts(self, student_id: str) -> list[str]:
        try:
            records = self.search_archival_memory(student_id, "mastered concept assessment passed")
            mastered = [r.get("concept", "") for r in records if r.get("type") == "feedback_given" and r.get("passed")]
            core = self.read_core_memory(student_id)
            core_mastered = core.get("mastered_concepts", [])
            return list(set([c for c in mastered + core_mastered if c]))
        except Exception:
            return []

    def get_mistake_history(self, student_id: str, concept: str) -> list[dict]:
        try:
            records = self.search_archival_memory(student_id, f"mistake wrong failed {concept}")
            return [r for r in records if r.get("concept") == concept and not r.get("passed", True)]
        except Exception:
            return []

    def get_tested_questions(self, student_id: str, concept: str) -> list[str]:
        try:
            records = self.search_archival_memory(student_id, f"question asked {concept}")
            return [r.get("question", "") for r in records if r.get("type") == "question_asked" and r.get("concept") == concept]
        except Exception:
            return []
