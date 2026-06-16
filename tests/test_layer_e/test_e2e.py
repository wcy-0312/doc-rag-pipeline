from dataclasses import dataclass, field
from typing import List, Optional

from layer_e.pipeline import generate
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
    display_markdown: str = "建議換藥每 8 小時一次，並觀察傷口狀況。"
    metadata: dict = field(default_factory=dict)
    source_tool: str = "azure_cu"
    source_pages: List[int] = field(default_factory=lambda: [1])
    embedding_text: str = ""
    rerank_score: float = 0.5


def test_e2e_normal_flow():
    results = [
        _FakeRankedResult(chunk_id="chunk_001", rerank_score=0.5),
        _FakeRankedResult(chunk_id="chunk_002", rerank_score=0.3),
    ]
    result = generate("換藥頻率", results, _StubLLMClient())
    assert result.abstain is False
    assert result.answer == "stub"
    assert result.safety_verdict == "safe"
    assert len(result.evidence_map) == 2


def test_e2e_abstain_insufficient_evidence():
    result = generate("任何查詢", [], _StubLLMClient())
    assert result.abstain is True
    assert result.safety_verdict == "abstained"
    assert result.answer == ""


def test_e2e_abstain_low_rerank_score():
    results = [_FakeRankedResult(chunk_id="chunk_low", rerank_score=0.05)]
    result = generate("任何查詢", results, _StubLLMClient())
    assert result.abstain is True
    assert result.safety_verdict == "abstained"


def test_e2e_needs_review_unsupported_claim():
    class _UnsupportedStub:
        def generate(self, system, user):
            return {
                "answer": "某個無引用的聲明",
                "claims": [{"text": "某個無引用的聲明", "citations": []}],
                "abstain": False,
                "abstain_reason": None,
            }

    results = [_FakeRankedResult(chunk_id="chunk_001", rerank_score=0.5)]
    result = generate("查詢", results, _UnsupportedStub())
    assert result.safety_verdict == "needs_review"
    assert len(result.unsupported_claims) == 1


def test_e2e_evidence_map_complete():
    results = [
        _FakeRankedResult(chunk_id="chunk_001", source_pages=[1], rerank_score=0.9),
        _FakeRankedResult(chunk_id="chunk_002", source_pages=[2, 3], rerank_score=0.7),
        _FakeRankedResult(chunk_id="chunk_003", source_pages=[4], rerank_score=0.5),
    ]
    result = generate("測試", results, _StubLLMClient())
    assert len(result.evidence_map) == 3
    for entry in result.evidence_map.values():
        assert "chunk_id" in entry
        assert "source_pages" in entry
        assert "source_tool" in entry
        assert "retrieval_weight" in entry


def test_e2e_llm_abstain():
    class _AbstainStub:
        def generate(self, system, user):
            return {
                "answer": "",
                "claims": [],
                "abstain": True,
                "abstain_reason": "資料不足以回答",
            }

    results = [_FakeRankedResult(chunk_id="chunk_001", rerank_score=0.5)]
    result = generate("難以回答的問題", results, _AbstainStub())
    assert result.abstain is True
    assert result.abstain_reason == "資料不足以回答"
    assert result.safety_verdict == "abstained"
