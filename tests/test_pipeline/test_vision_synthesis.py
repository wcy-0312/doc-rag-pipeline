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
    """LLM stub that records multimodal calls."""
    def __init__(self):
        self.multimodal_calls: list[tuple[str, list[bytes]]] = []

    def generate_text(self, user: str, system: str = "") -> str:
        return '{"relevant": [0]}'

    def generate_text_multimodal(self, user_text: str, images: list[bytes], system: str = "") -> str:
        self.multimodal_calls.append((user_text, images))
        return "vision synthesis answer"


def _make_pipeline(pdf_path: str | None = None):
    """Build a minimal RAGPipeline with mocked Qdrant and generation."""
    from pipeline.runner import RAGPipeline
    from layer_e.llm_client import _StubLLMClient

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline._tree_pdf_paths = {}
    if pdf_path:
        pipeline._tree_pdf_paths["乳癌診療指引-2026年_pdf"] = pdf_path

    mock_store = MagicMock()
    leaf = _make_leaf("sec_1", "TNBC 治療", 23, "neoadjuvant therapy content")
    mock_result = MagicMock()
    mock_result.matched_nodes = [leaf]
    mock_store.load_static.return_value = TreeNode(
        node_id="root", title="root", start_page=1, end_page=50,
        summary="", content="", children=[leaf],
    )
    pipeline._tree_store = mock_store
    pipeline._qdrant_client = MagicMock()
    pipeline._collection_name_ref = "test"
    pipeline._gen = MagicMock()
    pipeline._gen._llm_client = _StubLLMClient()
    return pipeline


def test_build_tree_registers_pdf_path():
    """build_tree() with pdf_path param registers the mapping."""
    from pipeline.runner import RAGPipeline
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline._tree_pdf_paths = {}
    pipeline._ingester = MagicMock()
    pipeline._ingester.create_collection_if_not_exists = MagicMock()
    pipeline._tree_store = MagicMock()
    pipeline._tree_store.store_static = MagicMock()
    pipeline._qdrant_client = MagicMock()
    pipeline._collection_name_ref = "test"

    raw_doc = {
        "metadata": {"file_name": "乳癌診療指引-2026年.pdf"},
        "data": {"sections": [], "paragraphs": []},
    }
    with patch('pipeline.runner._build_tree_from_raw') as mock_build:
        mock_build.return_value = TreeNode(
            node_id="root", title="root", start_page=1, end_page=1,
            summary="", content="test", children=[],
        )
        pipeline.build_tree(raw_doc, "乳癌診療指引-2026年.pdf",
                            pdf_path="/path/to/doc.pdf")

    assert pipeline._tree_pdf_paths.get("乳癌診療指引-2026年_pdf") == "/path/to/doc.pdf"
