"""Fixed Groq Responses API client for baseline experiments."""

from __future__ import annotations

import os
from typing import Protocol

from dotenv import load_dotenv


DEFAULT_MODEL = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class LLMClient(Protocol):
    """Minimal interface to make baseline testing independent of a live API."""

    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


class GroqResponsesLLM:
    """Groq's OpenAI-compatible Responses API adapter for the baseline."""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to .env before running the baseline RAG CLI.")
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError("The openai package is not installed. Run: python -m pip install openai") from error
        self._client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.responses.create(
            model=self.model,
            instructions=system_prompt,
            input=user_prompt,
            max_output_tokens=1_000,
            store=False,
        )
        return response.output_text.strip()
