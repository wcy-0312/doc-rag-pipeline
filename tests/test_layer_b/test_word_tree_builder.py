import pytest
from layer_b.word_tree_builder import build_word_tree


# ── 測試資料輔助函式 ──────────────────────────────────────────────────────────

def _raw(texts: list[dict], file_name: str = "test.docx") -> dict:
    """建立最小 docling raw document。"""
    return {
        "metadata": {"file_name": file_name},
        "data": {"texts": texts},
    }


def _heading(text: str, level: int, page: int | None = 1) -> dict:
    prov = [{"page_no": page}] if page is not None else []
    return {"label": "section_header", "text": text, "level": level, "prov": prov}


def _body(text: str, page: int | None = 1) -> dict:
    prov = [{"page_no": page}] if page is not None else []
    return {"label": "text", "text": text, "prov": prov}


def _list_item(text: str, page: int | None = 1) -> dict:
    prov = [{"page_no": page}] if page is not None else []
    return {"label": "list_item", "text": text, "prov": prov}


def test_returns_none_when_no_texts():
    assert build_word_tree({"metadata": {}, "data": {}}) is None


def test_returns_none_when_texts_empty():
    assert build_word_tree(_raw([])) is None
