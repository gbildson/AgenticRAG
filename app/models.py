from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from openai import OpenAI

from app.config import Settings


class ChatModel(Protocol):
    def chat(self, **kwargs): ...


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class OpenAICompatibleClient:
    """Adapter for endpoints implementing chat completions and embeddings."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)

    def chat(self, **kwargs):
        if not self.settings.chat_model:
            raise RuntimeError("CHAT_MODEL is not set")
        return self.client.chat.completions.create(model=self.settings.chat_model, **kwargs)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.settings.embedding_model:
            raise RuntimeError("EMBEDDING_MODEL is not set")
        response = self.client.embeddings.create(
            model=self.settings.embedding_model, input=list(texts)
        )
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
