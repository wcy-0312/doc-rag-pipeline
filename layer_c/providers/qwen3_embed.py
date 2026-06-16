from __future__ import annotations

import os
from typing import List

from layer_c.providers import EmbeddingProvider


class Qwen3EmbedProvider(EmbeddingProvider):
    """Qwen3-Embedding provider via OpenAI-compatible API.

    Required environment variables:
        QWEN3_API_KEY   — API key for the endpoint
        QWEN3_BASE_URL  — Base URL of the OpenAI-compatible server

    Optional:
        QWEN3_EMBED_MODEL — model name (default: Qwen/Qwen3-Embedding)
    """

    DEFAULT_MODEL = "Qwen/Qwen3-Embedding"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or os.environ["QWEN3_API_KEY"]
        self.base_url = base_url or os.environ["QWEN3_BASE_URL"]
        self.model = model or os.environ.get("QWEN3_EMBED_MODEL", self.DEFAULT_MODEL)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI  # noqa: PLC0415
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def embed(self, texts: List[str]) -> List[List[float]]:
        client = self._get_client()
        response = client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]
