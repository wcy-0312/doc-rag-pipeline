from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from layer_c.providers.openai_embed import OpenAIEmbedProvider


@pytest.fixture()
def provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_EMBED_MODEL", raising=False)
    return OpenAIEmbedProvider()


def _make_mock_client(vectors: list[list[float]]):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=v) for v in vectors]
    mock_client.embeddings.create.return_value = mock_response
    return mock_client


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def test_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "my-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    p = OpenAIEmbedProvider()
    assert p.api_key == "my-key"


def test_reads_all_env_vars(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://proxy/v1")
    monkeypatch.setenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
    p = OpenAIEmbedProvider()
    assert p.base_url == "http://proxy/v1"
    assert p.model == "text-embedding-3-large"


def test_default_model(provider):
    assert provider.model == OpenAIEmbedProvider.DEFAULT_MODEL


def test_base_url_defaults_to_none(provider):
    assert provider.base_url is None


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(KeyError):
        OpenAIEmbedProvider()


# ---------------------------------------------------------------------------
# embed() return shape
# ---------------------------------------------------------------------------

def test_embed_returns_correct_length(provider):
    vectors = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    provider._client = _make_mock_client(vectors)
    result = provider.embed(["a", "b", "c"])
    assert len(result) == 3
    assert result == vectors


def test_embed_single_text(provider):
    provider._client = _make_mock_client([[1.0, 2.0]])
    result = provider.embed(["hello"])
    assert result == [[1.0, 2.0]]


# ---------------------------------------------------------------------------
# API call correctness
# ---------------------------------------------------------------------------

def test_embed_calls_create_with_correct_args(provider):
    provider._client = _make_mock_client([[0.0]])
    provider.embed(["test input"])
    provider._client.embeddings.create.assert_called_once_with(
        model=OpenAIEmbedProvider.DEFAULT_MODEL,
        input=["test input"],
    )


def test_client_is_none_before_embed(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    p = OpenAIEmbedProvider()
    assert p._client is None


def test_client_set_after_embed(provider):
    provider._client = _make_mock_client([[0.0]])
    provider.embed(["x"])
    assert provider._client is not None
