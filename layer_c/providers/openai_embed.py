from __future__ import annotations

import os
from typing import List

from layer_c.providers import EmbeddingProvider


class OpenAIEmbedProvider(EmbeddingProvider):
    """OpenAI embedding provider (text-embedding-3-small by default).

    Required environment variables:
        OPENAI_API_KEY  — API key

    Optional:
        OPENAI_BASE_URL     — custom base URL (default: OpenAI official endpoint)
        OPENAI_EMBED_MODEL  — model name (default: text-embedding-3-small)
    """

    DEFAULT_MODEL = "text-embedding-3-small"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or os.environ["OPENAI_API_KEY"]
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.model = model or os.environ.get("OPENAI_EMBED_MODEL", self.DEFAULT_MODEL)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI  # noqa: PLC0415
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def embed(self, texts: List[str]) -> List[List[float]]:
        client = self._get_client()
        response = client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]
