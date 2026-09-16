"""Fixed OpenAI Responses API client for baseline experiments."""

from __future__ import annotations

import os
from typing import Protocol

from dotenv import load_dotenv


DEFAULT_MODEL = "gpt-5.2"


class LLMClient(Protocol):
    """Minimal interface to make baseline testing independent of a live API."""

    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


class OpenAIResponsesLLM:
    """OpenAI Responses API adapter using one fixed model for the experiment."""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env before running the baseline RAG CLI.")
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError("The openai package is not installed. Run: python -m pip install openai") from error
        self._client = OpenAI()
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.responses.create(
            model=self.model,
            instructions=system_prompt,
            input=user_prompt,
            reasoning={"effort": "none"},
            text={"verbosity": "low"},
            max_output_tokens=1_000,
            store=False,
        )
        return response.output_text.strip()
