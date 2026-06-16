import pytest
from dataclasses import dataclass, field
from typing import List, Optional

from layer_e.pipeline import generate, GenerationPipeline
from layer_e.llm_client import _StubLLMClient


@dataclass
class _FakeRankedResult:
    chunk_id: str
    chunk_type: str = "paragraph"
    parent_chunk_id: Optional[str] = None
    retrieval_unit_id: str = "unit_1"
    final_score: float = 0.9
    rrf_score: float = 0.9
    retrieval_weight: float = 0.95
    display_markdown: str = "fake content for testing purposes"
    metadata: dict = field(default_factory=dict)
    source_tool: str = "azure_cu"
    source_pages: List[int] = field(default_factory=lambda: [1])
    embedding_text: str = ""
    rerank_score: float = 0.8


def _normal_result():
    return _FakeRankedResult(chunk_id="chunk_1")


def test_abstain_when_empty_results():
    result = generate("query", [], _StubLLMClient())
    assert result.abstain is True
    assert result.safety_verdict == "abstained"
    assert result.answer == ""


def test_abstain_when_low_rerank_score():
    r = _FakeRankedResult(chunk_id="chunk_low", rerank_score=0.05)
    result = generate("query", [r], _StubLLMClient())
    assert result.abstain is True


def test_abstain_threshold_normalized_range():
    # rerank_score=0.05 is below default threshold 0.10 → should abstain
    r = _FakeRankedResult(chunk_id="chunk_low", rerank_score=0.05)
    result = generate("query", [r], _StubLLMClient())
    assert result.abstain is True

    # rerank_score=0.55 is above threshold → should not abstain
    r2 = _FakeRankedResult(chunk_id="chunk_ok", rerank_score=0.55)
    result2 = generate("query", [r2], _StubLLMClient())
    assert result2.abstain is False


def test_abstention_threshold_parameter():
    # custom threshold passed via generate()
    r = _FakeRankedResult(chunk_id="chunk_1", rerank_score=0.30)
    result_abstain = generate("query", [r], _StubLLMClient(), abstention_threshold=0.50)
    assert result_abstain.abstain is True

    result_pass = generate("query", [r], _StubLLMClient(), abstention_threshold=0.20)
    assert result_pass.abstain is False


def test_generation_pipeline_abstention_threshold():
    r = _FakeRankedResult(chunk_id="chunk_1", rerank_score=0.30)
    pipeline = GenerationPipeline(
        llm_client=_StubLLMClient(),
        abstention_threshold=0.20,
    )
    result = pipeline.run("query", [r])
    assert result.abstain is False


def test_normal_flow():
    result = generate("query", [_normal_result()], _StubLLMClient())
    assert result.abstain is False
    assert result.answer == "stub"
    assert len(result.evidence_map) >= 1


def test_claims_parsed():
    result = generate("query", [_normal_result()], _StubLLMClient())
    assert len(result.claims) > 0
    assert result.claims[0].text == "stub claim"


def test_safety_verdict_safe():
    result = generate("query", [_normal_result()], _StubLLMClient())
    assert result.safety_verdict == "safe"


def test_evidence_map_structure():
    result = generate("query", [_normal_result()], _StubLLMClient())
    e1 = result.evidence_map["E1"]
    assert "chunk_id" in e1
    assert "source_pages" in e1
    assert "source_tool" in e1
    assert "retrieval_weight" in e1


def test_llm_abstain_propagation():
    class _AbstainStub(_StubLLMClient):
        def generate(self, system: str, user: str) -> dict:
            return {
                "answer": "我沒有足夠資訊",
                "claims": [],
                "abstain": True,
                "abstain_reason": "資料不足",
            }

    result = generate("query", [_normal_result()], _AbstainStub())
    assert result.abstain is True
    assert result.abstain_reason == "資料不足"
    assert result.safety_verdict == "abstained"


def test_unsupported_marker_filtered_from_citations():
    # [unsupported] should be stripped from citations, not treated as invalid evidence ID
    class _UnsupportedStub(_StubLLMClient):
        def generate(self, system: str, user: str) -> dict:
            return {
                "answer": "部分答案。[E1] 另一部分無支持。[unsupported]",
                "claims": [
                    {"text": "有支持的 claim", "citations": ["E1"]},
                    {"text": "無支持的 claim", "citations": ["unsupported"]},
                ],
                "abstain": False,
                "abstain_reason": None,
            }

    result = generate("query", [_normal_result()], _UnsupportedStub(), skip_unsupported_check=True)
    assert result.abstain is False
    # [unsupported] must not appear in any claim's citations
    for claim in result.claims:
        assert "unsupported" not in claim.citations
    # the claim carrying [unsupported] should be in unsupported_claims
    assert any("無支持的 claim" in c for c in result.unsupported_claims)
    # safety_verdict should reflect unsupported claims exist
    assert result.safety_verdict == "needs_review"


def test_pipeline_backward_compat_no_images():
    """When ranked results have no page_image_refs, generate() must use text-only path."""
    calls = []

    class _TrackingStub(_StubLLMClient):
        def generate(self, system: str, user: str) -> dict:
            calls.append("text")
            return super().generate(system, user)

        def generate_multimodal(self, messages: list) -> dict:
            calls.append("multimodal")
            return super().generate_multimodal(messages)

    result = generate("query", [_normal_result()], _TrackingStub())
    assert result.abstain is False
    assert "text" in calls
    assert "multimodal" not in calls


def test_pipeline_uses_multimodal_when_images_present(tmp_path):
    img = tmp_path / "p1.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)

    @dataclass
    class _ResultWithImage(_FakeRankedResult):
        page_image_refs: dict = field(default_factory=dict)

    r = _ResultWithImage(chunk_id="chunk_img")
    r.page_image_refs = {"1": str(img)}

    calls = []

    class _VisionStub(_StubLLMClient):
        def generate_multimodal(self, messages: list) -> dict:
            calls.append(messages)
            return {
                "answer": "vision answer",
                "claims": [{"text": "vision claim", "citations": ["E1"]}],
                "abstain": False,
                "abstain_reason": None,
            }

    result = generate("query", [r], _VisionStub(), skip_unsupported_check=True)
    assert result.answer == "vision answer"
    assert len(calls) == 1
    user_content = calls[0][1]["content"]
    assert any(b.get("type") == "image_url" for b in user_content)


def test_pipeline_fallback_when_llm_no_vision(tmp_path):
    """LLM that doesn't override generate_multimodal falls back to text via base class."""
    img = tmp_path / "p1.png"
    img.write_bytes(b"\x89PNG")

    @dataclass
    class _ResultWithImage(_FakeRankedResult):
        page_image_refs: dict = field(default_factory=dict)

    r = _ResultWithImage(chunk_id="chunk_img2")
    r.page_image_refs = {"1": str(img)}

    result = generate("query", [r], _StubLLMClient(), skip_unsupported_check=True)
    assert result.answer == "stub"


def test_unsupported_marker_no_warning_logged(caplog):
    import logging
    class _UnsupportedStub(_StubLLMClient):
        def generate(self, system: str, user: str) -> dict:
            return {
                "answer": "答案。[unsupported]",
                "claims": [{"text": "無支持 claim", "citations": ["unsupported"]}],
                "abstain": False,
                "abstain_reason": None,
            }

    with caplog.at_level(logging.WARNING, logger="layer_e.pipeline"):
        generate("query", [_normal_result()], _UnsupportedStub(), skip_unsupported_check=True)
    assert "Invalid citation IDs" not in caplog.text
