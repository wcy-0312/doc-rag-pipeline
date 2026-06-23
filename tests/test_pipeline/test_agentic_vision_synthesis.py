import pytest
from unittest.mock import MagicMock, patch
from layer_e.models import GenerationResult
from layer_f.tree_models import TreeNode


def _make_leaf(node_id: str, title: str, page: int, content: str) -> TreeNode:
    return TreeNode(
        node_id=node_id, title=title,
        start_page=page, end_page=page,
        summary="", content=content, children=[],
    )


class _VisionLLM:
    def __init__(self):
        self.multimodal_calls: list[tuple[str, list[bytes]]] = []

    def generate_text(self, user: str, system: str = "") -> str:
        # TreeRouter: need_patient_context=false, select first guideline
        return '{"need_patient_context": false, "relevant_guidelines": [0]}'

    def generate_text_multimodal(self, user_text: str, images: list[bytes], system: str = "") -> str:
        self.multimodal_calls.append((user_text, images))
        return "agentic vision answer"


def _make_agentic_pipeline(pdf_path: str | None = None):
    from pipeline.runner import RAGPipeline

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline._tree_pdf_paths = {}

    leaf = _make_leaf("sec_1", "TNBC 治療", 23, "neoadjuvant therapy content")
    root = TreeNode(
        node_id="root", title="root", start_page=1, end_page=50,
        summary="", content="", children=[leaf],
    )
    stem = "乳癌指引"
    if pdf_path:
        pipeline._tree_pdf_paths[stem] = pdf_path

    mock_store = MagicMock()
    mock_store.get_static_summaries.return_value = {stem: "乳癌診療指引 2026"}
    mock_store._static_cache = {stem: root}
    mock_store._dynamic = {}
    pipeline._tree_store = mock_store
    pipeline._gen = MagicMock()
    pipeline._gen._llm_client = MagicMock()
    pipeline._gen.run.return_value = GenerationResult(
        answer="text fallback", claims=[], evidence_map={},
        unsupported_claims=[], abstain=False, abstain_reason=None,
        safety_verdict="safe", steps_log=[],
    )
    return pipeline, stem, leaf


def test_query_tree_agentic_uses_vision_when_pdf_registered(tmp_path):
    """query_tree_agentic() should call multimodal LLM when PDF is registered."""
    import fitz
    doc = fitz.open()
    for _ in range(30):
        doc.new_page()
    pdf_path = tmp_path / "guideline.pdf"
    pdf_path.write_bytes(doc.tobytes())
    doc.close()

    pipeline, stem, leaf = _make_agentic_pipeline(pdf_path=str(pdf_path))
    llm = _VisionLLM()

    mock_search_result = MagicMock()
    mock_search_result.matched_nodes = [leaf]

    with patch("layer_f.tree_search.TreeSearcher.search", return_value=mock_search_result):
        result = pipeline.query_tree_agentic("TNBC 治療方案", llm_client=llm)

    assert not result.abstain
    assert result.answer == "agentic vision answer"
    assert len(llm.multimodal_calls) == 1
    _, images = llm.multimodal_calls[0]
    assert len(images) >= 1


def test_query_tree_agentic_vision_includes_patient_context(tmp_path):
    """Vision prompt should include patient_context when provided via dynamic tree."""
    import fitz
    doc = fitz.open()
    for _ in range(30):
        doc.new_page()
    pdf_path = tmp_path / "guideline.pdf"
    pdf_path.write_bytes(doc.tobytes())
    doc.close()

    pipeline, stem, leaf = _make_agentic_pipeline(pdf_path=str(pdf_path))
    llm = _VisionLLM()

    # Override router to request patient context
    class _RouterLLM:
        def generate_text(self, user, system=""):
            return '{"need_patient_context": true, "relevant_guidelines": [0]}'
        def generate_text_multimodal(self, user_text, images, system=""):
            llm.multimodal_calls.append((user_text, images))
            return "agentic vision with patient"

    patient_leaf = _make_leaf("p1", "病人資料", 1, "cT2N1M0 乳癌")
    patient_root = TreeNode(
        node_id="patient_root", title="patient_root", start_page=1, end_page=1,
        summary="", content="", children=[patient_leaf],
    )
    session_id = "sess_001"
    pipeline._tree_store._dynamic = {session_id: {"patient": patient_root}}
    pipeline._tree_store.load_dynamic = MagicMock(return_value=patient_root)

    guideline_search_result = MagicMock()
    guideline_search_result.matched_nodes = [leaf]
    patient_search_result = MagicMock()
    patient_search_result.matched_nodes = [patient_leaf]

    router_llm = _RouterLLM()

    with patch("layer_f.tree_search.TreeSearcher.search") as mock_search:
        mock_search.side_effect = [patient_search_result, guideline_search_result]
        result = pipeline.query_tree_agentic(
            "cT2N1M0 病人適合什麼治療？", session_id=session_id, llm_client=router_llm
        )

    assert result.answer == "agentic vision with patient"
    assert len(llm.multimodal_calls) == 1
    vision_prompt, _ = llm.multimodal_calls[0]
    assert "病人資料摘要" in vision_prompt
