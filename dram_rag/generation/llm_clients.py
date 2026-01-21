from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    provider: str = "none"  # "none" | "openai"
    model: str = "gpt-4.1-mini"
    temperature: float = 0.2


class LLMClient:
    def generate(self, system: str, user: str) -> str:
        raise NotImplementedError


class NoopLLMClient(LLMClient):
    """Fallback generator for offline runs. It does not call any external model."""

    def generate(self, system: str, user: str) -> str:
        # The caller should provide an extractive fallback; this is a guard.
        return "[LLM disabled]"


class OpenAILLMClient(LLMClient):
    """OpenAI Chat Completions client (optional).

    Requires:
      - openai>=1
      - OPENAI_API_KEY in environment
    """

    def __init__(self, model: str, temperature: float = 0.2):
        self.model = model
        self.temperature = temperature
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("openai package not installed. pip install openai") from e

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:  # pragma: no cover
            raise EnvironmentError("OPENAI_API_KEY is not set")

        self._client = OpenAI(api_key=api_key)

    def generate(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


def make_llm_client(cfg: LLMConfig) -> LLMClient:
    provider = (cfg.provider or "none").lower()
    if provider == "openai":
        return OpenAILLMClient(model=cfg.model, temperature=cfg.temperature)
    return NoopLLMClient()
