"""Tests that _synthesise_nodes() passes full node content (up to 2000 chars) to the LLM."""
from unittest.mock import MagicMock, patch
from layer_f.tree_models import TreeNode


def _make_leaf(node_id, title, page, content):
    return TreeNode(
        node_id=node_id, title=title,
        start_page=page, end_page=page,
        summary="", content=content, children=[],
    )


class _CaptureLLM:
    """Multimodal LLM stub that records the user_text it receives."""
    def __init__(self):
        self.captured_prompt: str = ""

    def generate_text(self, user: str, system: str = "") -> str:
        return '{"relevant": [0]}'

    def generate_text_multimodal(self, user_text: str, images, system: str = "") -> str:
        self.captured_prompt = user_text
        return "answer"


_STEM = "test_doc"


def _make_pipeline(pdf_path: str):
    from pipeline.runner import RAGPipeline
    from layer_e.llm_client import _StubLLMClient

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline._tree_pdf_paths = {_STEM: pdf_path}
    pipeline._tree_store = MagicMock()
    pipeline._qdrant_client = MagicMock()
    pipeline._collection_name_ref = "test"
    pipeline._gen = MagicMock()
    pipeline._gen._llm_client = _StubLLMClient()
    return pipeline


def test_node_content_up_to_2000_chars_included(tmp_path):
    """Content shorter than 2000 chars must appear in full in the synthesis prompt."""
    import fitz  # PyMuPDF
    pdf = fitz.open()
    pdf.new_page()
    pdf_file = str(tmp_path / "test.pdf")
    pdf.save(pdf_file)

    long_content = "乳房切除後輔助性放射線治療建議 [I,A]。" * 60  # ~840 chars, well within 2000
    assert len(long_content) < 2000

    leaf = _make_leaf("sec_1", "放射線治療", 1, long_content)
    pipeline = _make_pipeline(pdf_file)
    llm = _CaptureLLM()

    with patch("layer_f.page_renderer.render_pages", return_value=[b"fake_jpeg"]):
        pipeline._synthesise_nodes("test query", [leaf], [_STEM], llm)

    assert long_content in llm.captured_prompt


def test_node_content_over_500_chars_not_truncated_at_500(tmp_path):
    """Content over 500 chars must NOT be cut at the old 500-char boundary."""
    import fitz
    pdf = fitz.open()
    pdf.new_page()
    pdf_file = str(tmp_path / "test.pdf")
    pdf.save(pdf_file)

    # Craft content where evidence grades appear after the 500-char mark
    prefix = "一般內容文字。" * 80          # 7 chars × 80 = 560 chars padding
    suffix = "強烈建議術後放射治療 [I,A]。"     # evidence grade at the end
    content = prefix + suffix
    assert len(content) > 500
    assert len(content) < 2000

    leaf = _make_leaf("sec_1", "治療建議", 1, content)
    pipeline = _make_pipeline(pdf_file)
    llm = _CaptureLLM()

    with patch("layer_f.page_renderer.render_pages", return_value=[b"fake_jpeg"]):
        pipeline._synthesise_nodes("test query", [leaf], [_STEM], llm)

    assert "[I,A]" in llm.captured_prompt


def test_node_content_over_2000_chars_truncated_at_2000(tmp_path):
    """Content over 2000 chars is truncated at 2000, not beyond."""
    import fitz
    pdf = fitz.open()
    pdf.new_page()
    pdf_file = str(tmp_path / "test.pdf")
    pdf.save(pdf_file)

    content = "參考文獻內容。" * 300          # ~2400 chars
    assert len(content) > 2000
    marker_at_2100 = "SHOULD_NOT_APPEAR"
    content_with_marker = content[:2100] + marker_at_2100 + content[2100:]

    leaf = _make_leaf("sec_1", "參考文獻", 1, content_with_marker)
    pipeline = _make_pipeline(pdf_file)
    llm = _CaptureLLM()

    with patch("layer_f.page_renderer.render_pages", return_value=[b"fake_jpeg"]):
        pipeline._synthesise_nodes("test query", [leaf], [_STEM], llm)

    assert marker_at_2100 not in llm.captured_prompt
    assert content_with_marker[:2000] in llm.captured_prompt
