from __future__ import annotations
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for pluggable embedding backends.

    Implementors must provide embed_texts() which takes a list of strings
    and returns a list of float vectors (one per input string).

    Structure-aware Layer does NOT call this — it only stores embedding_text
    and row_texts as the text payloads. The Chunk + Embed Layer (lead-C)
    calls embed_texts() to convert these payloads into vectors.

    This Protocol lives here so that lead-C can import and implement it
    without depending on any specific embedding library.
    """

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per input string."""
        ...


class NullEmbeddingProvider:
    """No-op provider: returns empty vectors. Used in tests and as default.

    Lead-C replaces this with a real provider (bge-m3, Qwen3-Embedding,
    text-embedding-3-large) in the Chunk + Embed Layer.
    """

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[] for _ in texts]
