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


# ── 測試資料 ──────────────────────────────────────────────────────────────────

_FLAT = _raw([
    _heading("第一章", level=1, page=1),
    _body("第一章內容文字。", page=1),
    _heading("第二章", level=1, page=5),
    _body("第二章內容文字。", page=5),
])

_NESTED = _raw([
    _heading("第一章", level=1, page=1),
    _heading("1.1 節", level=2, page=2),
    _body("1.1 節內容文字。", page=2),
    _heading("1.2 節", level=2, page=4),
    _body("1.2 節內容文字。", page=4),
    _heading("第二章", level=1, page=8),
    _body("第二章直接內容。", page=8),
])


def test_flat_multiple_h1_creates_virtual_root():
    tree = build_word_tree(_FLAT)
    assert tree is not None
    assert tree.node_id == "root"
    assert len(tree.children) == 2


def test_flat_children_titles():
    tree = build_word_tree(_FLAT)
    titles = [c.title for c in tree.children]
    assert "第一章" in titles
    assert "第二章" in titles


def test_flat_leaf_content():
    tree = build_word_tree(_FLAT)
    ch1 = next(c for c in tree.children if c.title == "第一章")
    assert "第一章內容文字" in ch1.content


def test_nested_h1_has_h2_children():
    tree = build_word_tree(_NESTED)
    ch1 = next(c for c in tree.children if c.title == "第一章")
    assert len(ch1.children) == 2
    subtitles = [c.title for c in ch1.children]
    assert "1.1 節" in subtitles
    assert "1.2 節" in subtitles


def test_nested_nonleaf_content_empty():
    tree = build_word_tree(_NESTED)
    ch1 = next(c for c in tree.children if c.title == "第一章")
    assert ch1.content == ""


def test_nested_leaf_content_aggregated():
    tree = build_word_tree(_NESTED)
    ch1 = next(c for c in tree.children if c.title == "第一章")
    sec11 = next(c for c in ch1.children if c.title == "1.1 節")
    assert "1.1 節內容文字" in sec11.content


def test_nested_h2_after_h3_resets_correctly():
    """H3 → H2 時，H2 成為 H1 的新子節點（H3 不殘留）。"""
    raw = _raw([
        _heading("Chapter", level=1, page=1),
        _heading("Section", level=2, page=2),
        _heading("Subsection", level=3, page=3),
        _body("Sub content", page=3),
        _heading("Next Section", level=2, page=5),
        _body("Next content", page=5),
    ])
    tree = build_word_tree(raw)
    # Single H1 → return directly (no virtual root)
    assert tree.title == "Chapter"
    assert len(tree.children) == 2
    titles = [c.title for c in tree.children]
    assert "Section" in titles
    assert "Next Section" in titles


# ── 頁碼範圍與 prov=[] 測試 ────────────────────────────────────────────────────

def test_leaf_page_range():
    tree = build_word_tree(_FLAT)
    ch1 = next(c for c in tree.children if c.title == "第一章")
    assert ch1.start_page == 1
    assert ch1.end_page == 1


def test_nonleaf_page_range_spans_children():
    tree = build_word_tree(_NESTED)
    ch1 = next(c for c in tree.children if c.title == "第一章")
    # H2 children are at pages 2 and 4
    assert ch1.start_page == 1  # includes heading page
    assert ch1.end_page == 4


def test_virtual_root_page_range_spans_all():
    tree = build_word_tree(_FLAT)
    assert tree.start_page == 1
    assert tree.end_page == 5


def test_prov_empty_page_is_none():
    """DOCX native 模式：prov=[] → start_page=None, end_page=None。"""
    raw = _raw([
        _heading("無頁碼章節", level=1, page=None),
        _body("內容", page=None),
    ])
    tree = build_word_tree(raw)
    assert tree is not None
    assert tree.start_page is None
    assert tree.end_page is None


def test_prov_empty_does_not_fallback_to_1():
    """prov=[] 時 page 絕對不能 fallback 為 1。"""
    raw = _raw([_heading("A", level=1, page=None)])
    tree = build_word_tree(raw)
    assert tree.start_page is None
    assert tree.end_page is None
