from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return a list of vectors, one per input text."""
        ...


def get_provider(name: str) -> "EmbeddingProvider":
    """Return an EmbeddingProvider instance by name.

    name: "bge-m3" | "openai" | "qwen3"
    """
    if name == "bge-m3":
        from layer_c.providers.bge_m3 import BGEm3Provider
        return BGEm3Provider()
    elif name == "openai":
        from layer_c.providers.openai_embed import OpenAIEmbedProvider
        return OpenAIEmbedProvider()
    elif name == "qwen3":
        from layer_c.providers.qwen3_embed import Qwen3EmbedProvider
        return Qwen3EmbedProvider()
    else:
        raise ValueError(f"Unknown provider: {name!r}. Choose: bge-m3, openai, qwen3")
