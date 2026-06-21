import json
import pytest
from unittest.mock import MagicMock, patch
from layer_e.agentic_tools import TOOL_DEFINITIONS, execute_tool


def test_tool_definitions_valid_openai_format():
    assert isinstance(TOOL_DEFINITIONS, list)
    assert len(TOOL_DEFINITIONS) == 2
    names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
    assert names == {"get_page_image", "retrieve_more"}
    for t in TOOL_DEFINITIONS:
        assert t["type"] == "function"
        fn = t["function"]
        assert "name" in fn
        assert "description" in fn
        assert "parameters" in fn
        assert "required" in fn["parameters"]


def test_execute_get_page_image(tmp_path):
    import fitz
    doc = fitz.open()
    doc.new_page()
    pdf_path = str(tmp_path / "test.pdf")
    doc.save(pdf_path)
    doc.close()

    tool_call = {"id": "call_1", "name": "get_page_image", "arguments": {"page_no": 1, "reason": "test"}}
    text, b64 = execute_tool(tool_call, pdf_path=pdf_path, retriever=None, doc_stem="test")
    assert "1" in text
    assert b64 is not None
    import base64
    decoded = base64.b64decode(b64)
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


def test_execute_retrieve_more():
    mock_result = MagicMock()
    mock_result.source_pages = [5]
    mock_result.display_markdown = "相關段落內容"

    mock_retriever = MagicMock()
    mock_retriever.search_text.return_value = [mock_result]

    tool_call = {"id": "call_2", "name": "retrieve_more", "arguments": {"query": "T2N1 分期", "reason": "test"}}
    text, b64 = execute_tool(tool_call, pdf_path="", retriever=mock_retriever, doc_stem="doc")
    assert "相關段落內容" in text
    assert b64 is None
    mock_retriever.search_text.assert_called_once_with(
        "T2N1 分期", top_k=3, doc_ids=["doc"], rerank=False
    )


def test_execute_retrieve_more_empty():
    mock_retriever = MagicMock()
    mock_retriever.search_text.return_value = []

    tool_call = {"id": "call_3", "name": "retrieve_more", "arguments": {"query": "不存在的內容", "reason": "test"}}
    text, b64 = execute_tool(tool_call, pdf_path="", retriever=mock_retriever, doc_stem="doc")
    assert "未找到" in text
    assert b64 is None


def test_execute_unknown_tool():
    tool_call = {"id": "call_x", "name": "nonexistent", "arguments": {}}
    text, b64 = execute_tool(tool_call, pdf_path="", retriever=None, doc_stem="doc")
    assert "未知工具" in text
    assert b64 is None
