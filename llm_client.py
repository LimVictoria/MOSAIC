# llm_client.py
# Single LLM client — all 4 agents use the same LLaMA model

import requests
from groq import Groq
from config import LLM_PROVIDER, LLM_MODEL, GROQ_API_KEY, OLLAMA_BASE_URL


class LLMClient:
    """
    Single LLM client for all agents.
    All 4 agents (Solver, Assessment, Feedback, KG Builder)
    use the same LLaMA 3.1 70B model.
    What differs is the system prompt passed by each agent.

    Supports:
    - Ollama (local, needs GPU)
    - Groq API (remote, fast, free tier available)
    """

    def __init__(self):
        self.provider = LLM_PROVIDER
        self.model = LLM_MODEL

        if self.provider == "groq":
            self.groq_client = Groq(api_key=GROQ_API_KEY)
            print(f"LLM: Using Groq API with {self.model}")
        else:
            print(f"LLM: Using Ollama at {OLLAMA_BASE_URL} with {self.model}")

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """
        Generate a response from LLaMA.
        Each agent passes its own system_prompt —
        this is what makes the same model behave differently.
        """
        if self.provider == "groq":
            return self._generate_groq(system_prompt, user_message, temperature, max_tokens)
        else:
            return self._generate_ollama(system_prompt, user_message, temperature, max_tokens)

    def _generate_groq(self, system_prompt, user_message, temperature, max_tokens) -> str:
        """Generate via Groq API."""
        response = self.groq_client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

    def _generate_ollama(self, system_prompt, user_message, temperature, max_tokens) -> str:
        """Generate via local Ollama."""
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                },
                "stream": False
            }
        )
        return response.json()["message"]["content"]
