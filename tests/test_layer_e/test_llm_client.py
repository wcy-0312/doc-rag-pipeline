import os
import pytest
from layer_e.llm_client import _StubLLMClient, get_llm_client


@pytest.fixture
def stub():
    return _StubLLMClient()


@pytest.fixture
def result(stub):
    return stub.generate(system="sys", user="usr")


def test_stub_returns_required_keys(result):
    for key in ("answer", "claims", "abstain", "abstain_reason"):
        assert key in result


def test_stub_answer_is_stub(result):
    assert result["answer"] == "stub"


def test_stub_claims_is_nonempty_list(result):
    assert isinstance(result["claims"], list)
    assert len(result["claims"]) >= 1


def test_stub_abstain_is_false(result):
    assert result["abstain"] is False


def test_stub_abstain_reason_is_none(result):
    assert result["abstain_reason"] is None


def test_get_llm_client_stub(monkeypatch):
    monkeypatch.setenv("GENERATION_LLM_BACKEND", "stub")
    client = get_llm_client()
    assert isinstance(client, _StubLLMClient)


def test_get_llm_client_unknown_backend_raises(monkeypatch):
    monkeypatch.setenv("GENERATION_LLM_BACKEND", "unknown_backend_xyz")
    with pytest.raises(ValueError):
        get_llm_client()


def test_generate_multimodal_fallback_text_only(stub):
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]
    result = stub.generate_multimodal(messages)
    assert result["answer"] == "stub"


def test_generate_multimodal_fallback_content_list(stub):
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": [{"type": "text", "text": "query"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]},
    ]
    result = stub.generate_multimodal(messages)
    assert result["answer"] == "stub"
