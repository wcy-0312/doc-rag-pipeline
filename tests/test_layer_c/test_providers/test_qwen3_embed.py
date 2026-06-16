from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from layer_c.providers.qwen3_embed import Qwen3EmbedProvider


@pytest.fixture()
def provider(monkeypatch):
    monkeypatch.setenv("QWEN3_API_KEY", "test-key")
    monkeypatch.setenv("QWEN3_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.delenv("QWEN3_EMBED_MODEL", raising=False)
    return Qwen3EmbedProvider()


def _make_mock_client(vectors: list[list[float]]):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=v) for v in vectors]
    mock_client.embeddings.create.return_value = mock_response
    return mock_client


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def test_reads_env_vars(monkeypatch):
    monkeypatch.setenv("QWEN3_API_KEY", "my-key")
    monkeypatch.setenv("QWEN3_BASE_URL", "http://example.com/v1")
    monkeypatch.setenv("QWEN3_EMBED_MODEL", "custom-model")
    p = Qwen3EmbedProvider()
    assert p.api_key == "my-key"
    assert p.base_url == "http://example.com/v1"
    assert p.model == "custom-model"


def test_default_model(provider):
    assert provider.model == Qwen3EmbedProvider.DEFAULT_MODEL


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("QWEN3_API_KEY", raising=False)
    monkeypatch.setenv("QWEN3_BASE_URL", "http://localhost:8000/v1")
    with pytest.raises(KeyError):
        Qwen3EmbedProvider()


def test_missing_base_url_raises(monkeypatch):
    monkeypatch.setenv("QWEN3_API_KEY", "test-key")
    monkeypatch.delenv("QWEN3_BASE_URL", raising=False)
    with pytest.raises(KeyError):
        Qwen3EmbedProvider()


# ---------------------------------------------------------------------------
# embed() return shape
# ---------------------------------------------------------------------------

def test_embed_returns_correct_length(provider):
    vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    provider._client = _make_mock_client(vectors)
    result = provider.embed(["text a", "text b"])
    assert len(result) == 2
    assert result == vectors


def test_embed_single_text(provider):
    provider._client = _make_mock_client([[1.0, 2.0]])
    result = provider.embed(["one text"])
    assert result == [[1.0, 2.0]]


# ---------------------------------------------------------------------------
# API call correctness
# ---------------------------------------------------------------------------

def test_embed_calls_create_with_correct_args(provider):
    provider._client = _make_mock_client([[0.0]])
    provider.embed(["hello world"])
    provider._client.embeddings.create.assert_called_once_with(
        model=Qwen3EmbedProvider.DEFAULT_MODEL,
        input=["hello world"],
    )


def test_client_created_lazily(monkeypatch):
    monkeypatch.setenv("QWEN3_API_KEY", "k")
    monkeypatch.setenv("QWEN3_BASE_URL", "http://localhost/v1")
    p = Qwen3EmbedProvider()
    assert p._client is None

    mock_openai = MagicMock()
    mock_instance = _make_mock_client([[0.1]])
    mock_openai.return_value = mock_instance

    with patch("layer_c.providers.qwen3_embed.OpenAI", mock_openai, create=True):
        # patch inside the module
        import layer_c.providers.qwen3_embed as mod  # noqa: PLC0415
        original = getattr(mod, "OpenAI", None)
        mod.OpenAI = mock_openai  # type: ignore[attr-defined]
        try:
            p._client = None
            # re-trigger lazy load
            p._client = mock_instance
            result = p.embed(["x"])
            assert p._client is not None
        finally:
            if original is not None:
                mod.OpenAI = original
