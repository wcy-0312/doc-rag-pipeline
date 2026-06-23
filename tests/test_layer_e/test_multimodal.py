import base64
import pytest
from unittest.mock import MagicMock, patch
from layer_e.llm_client import _StubLLMClient, LLMClient


FAKE_JPEG = b'\xff\xd8\xff\xe0' + b'\x00' * 100  # minimal fake JPEG header


def test_base_class_fallback_ignores_images():
    """Base class generate_text_multimodal falls back to generate_text."""
    stub = _StubLLMClient()
    result = stub.generate_text_multimodal("question", [FAKE_JPEG], system="sys")
    assert result == "stub summary"  # _StubLLMClient.generate_text returns this


def test_gpt41_sends_image_in_content_array():
    """GPT41Client formats images as OpenAI content array."""
    from layer_e.llm_client import GPT41Client

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "vision answer"
    mock_response.choices[0].message.tool_calls = None

    with patch.object(GPT41Client, '_make_client') as mock_make:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_make.return_value = mock_client

        client = GPT41Client.__new__(GPT41Client)
        client._token_provider = lambda: "fake-token"
        client._endpoint = "https://fake.endpoint"
        client._deployment = "gpt-4.1"

        result = client.generate_text_multimodal("query text", [FAKE_JPEG], system="sys")

    assert result == "vision answer"
    call_args = mock_client.chat.completions.create.call_args
    messages = call_args.kwargs.get("messages") or call_args.args[0] if call_args.args else call_args.kwargs["messages"]
    user_msg = next(m for m in messages if m["role"] == "user")
    assert isinstance(user_msg["content"], list)
    types = [part["type"] for part in user_msg["content"]]
    assert "text" in types
    assert "image_url" in types
    image_part = next(p for p in user_msg["content"] if p["type"] == "image_url")
    assert "data:image/jpeg;base64," in image_part["image_url"]["url"]


def test_gemma3_sends_image_in_content_array():
    """Gemma3Client formats images as OpenAI-compatible content array."""
    from layer_e.llm_client import Gemma3Client

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "gemma vision answer"

    with patch('layer_e.llm_client.OpenAI') as mock_openai_cls:
        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client_instance

        client = Gemma3Client.__new__(Gemma3Client)
        client._client = mock_client_instance

        result = client.generate_text_multimodal("query text", [FAKE_JPEG], system="sys")

    assert result == "gemma vision answer"
    call_args = mock_client_instance.chat.completions.create.call_args
    messages = call_args.kwargs.get("messages") or call_args.kwargs["messages"]
    user_msg = next(m for m in messages if m["role"] == "user")
    assert isinstance(user_msg["content"], list)
    types = [part["type"] for part in user_msg["content"]]
    assert "image_url" in types


def test_empty_images_falls_back_to_text():
    """generate_text_multimodal with empty images should still return a result."""
    stub = _StubLLMClient()
    result = stub.generate_text_multimodal("question", [], system="sys")
    assert isinstance(result, str)
