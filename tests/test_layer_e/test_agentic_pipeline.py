import json
from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import MagicMock

from layer_e.agentic_pipeline import AgenticPipeline
from layer_e.llm_client import _StubLLMClient
from layer_e.models import GenerationResult


@dataclass
class _FakeRankedResult:
    chunk_id: str
    chunk_type: str = "paragraph"
    parent_chunk_id: Optional[str] = None
    retrieval_unit_id: str = "unit_1"
    final_score: float = 0.9
    rrf_score: float = 0.9
    retrieval_weight: float = 1.0
    display_markdown: str = "cT2N1M0 建議新輔助化療後手術"
    metadata: dict = field(default_factory=dict)
    source_tool: str = "azure_cu"
    source_pages: List[int] = field(default_factory=lambda: [5])
    embedding_text: str = "cT2N1M0"
    rerank_score: float = 0.8
    page_image_refs: dict = field(default_factory=dict)


def _make_pipeline(llm_client=None, retriever=None):
    return AgenticPipeline(
        llm_client=llm_client or _StubLLMClient(),
        retriever=retriever or MagicMock(),
        pdf_path="/fake/path.pdf",
        doc_stem="乳癌診療指引-2026年",
    )


def test_abstain_when_no_results():
    pipeline = _make_pipeline()
    result = pipeline.run("query", [])
    assert result.abstain is True
    assert result.safety_verdict == "abstained"


def test_abstain_when_low_rerank_score():
    pipeline = _make_pipeline()
    r = _FakeRankedResult(chunk_id="c1", rerank_score=0.01)
    result = pipeline.run("query", [r])
    assert result.abstain is True


def test_normal_flow_returns_generation_result():
    pipeline = _make_pipeline()
    r = _FakeRankedResult(chunk_id="c1")
    result = pipeline.run("cT2N1M0 治療建議", [r])
    assert isinstance(result, GenerationResult)
    assert result.abstain is False
    assert result.answer == "stub"


def test_steps_log_is_recorded():
    call_count = [0]

    class _ToolCallingStub(_StubLLMClient):
        def generate_with_tools(self, messages, tools):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: request a page image
                return (
                    [{"id": "call_1", "name": "get_page_image", "arguments": {"page_no": 5, "reason": "確認流程圖"}}],
                    None,
                )
            # Second call: final answer
            return ([], json.dumps({
                "answer": "cT2N1M0 建議先做新輔助化療",
                "claims": [{"text": "cT2N1M0 建議先做新輔助化療", "citations": ["E1"]}],
                "abstain": False,
                "abstain_reason": None,
            }))

    # pdf_path is fake → get_full_page_image would raise; we patch it to avoid real PDF access
    # This test validates that tool was CALLED (step log), not that image succeeds

    # Use a retriever mock and a patched pdf_tools to avoid real PDF access
    from unittest.mock import patch
    with patch("layer_e.agentic_tools.get_full_page_image", return_value="base64fakepng"):
        pipeline = AgenticPipeline(
            llm_client=_ToolCallingStub(),
            retriever=MagicMock(),
            pdf_path="/fake/path.pdf",
            doc_stem="doc",
        )
        r = _FakeRankedResult(chunk_id="c1")
        result = pipeline.run("query", [r])

    assert result.answer == "cT2N1M0 建議先做新輔助化療"
    assert len(result.steps_log) == 1
    assert result.steps_log[0]["tool"] == "get_page_image"
    assert result.steps_log[0]["step_no"] == 1


def test_hard_limit_prevents_infinite_loop():
    """LLM that always calls tools must be stopped by hard_limit."""
    class _AlwaysToolStub(_StubLLMClient):
        def generate_with_tools(self, messages, tools):
            return (
                [{"id": "call_x", "name": "retrieve_more", "arguments": {"query": "x", "reason": "x"}}],
                None,
            )

    mock_retriever = MagicMock()
    mock_retriever.search_text.return_value = []

    pipeline = AgenticPipeline(
        llm_client=_AlwaysToolStub(),
        retriever=mock_retriever,
        pdf_path="/fake/path.pdf",
        doc_stem="doc",
        hard_limit=3,
    )
    r = _FakeRankedResult(chunk_id="c1")
    result = pipeline.run("query", [r])
    # Must return some result (abstain or answer), never loop forever
    assert isinstance(result, GenerationResult)
