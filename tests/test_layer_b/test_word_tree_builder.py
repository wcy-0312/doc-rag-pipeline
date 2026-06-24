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


# ── LLM summary 與 list_item 測試 ────────────────────────────────────────────────

class _MockLLM:
    def __init__(self):
        self.call_count = 0

    def generate_text(self, user: str, system: str = "") -> str:
        self.call_count += 1
        return f"摘要{self.call_count}"


def test_summary_generated_for_nonleaf():
    llm = _MockLLM()
    tree = build_word_tree(_NESTED, llm_client=llm)
    ch1 = next(c for c in tree.children if c.title == "第一章")
    assert ch1.summary != ""
    assert llm.call_count >= 1


def test_no_summary_when_llm_is_none():
    tree = build_word_tree(_NESTED, llm_client=None)
    ch1 = next(c for c in tree.children if c.title == "第一章")
    assert ch1.summary == ""


def test_leaf_summary_always_empty():
    tree = build_word_tree(_FLAT)
    ch1 = next(c for c in tree.children if c.title == "第一章")
    assert ch1.is_leaf
    assert ch1.summary == ""


def test_list_item_treated_as_body():
    raw = _raw([
        _heading("清單章節", level=1, page=1),
        _list_item("第一項", page=1),
        _list_item("第二項", page=1),
    ])
    tree = build_word_tree(raw)
    assert tree is not None
    assert "第一項" in tree.content
    assert "第二項" in tree.content


def test_heading_page_preserved_when_body_prov_empty():
    """Heading's page must survive even when body items have prov=[]."""
    raw = _raw([
        _heading("Chapter", level=1, page=3),
        _body("Content with no page", page=None),
    ])
    tree = build_word_tree(raw)
    assert tree is not None
    assert tree.start_page == 3
    assert tree.end_page == 3


def test_body_before_first_heading_is_ignored():
    """Body text before any heading is silently dropped (no heading to attach to)."""
    raw = _raw([
        _body("Preamble text", page=1),
        _heading("Chapter 1", level=1, page=2),
        _body("Chapter content", page=2),
    ])
    tree = build_word_tree(raw)
    assert tree is not None
    assert "Preamble" not in tree.content


# ── Table classifier ──────────────────────────────────────────────────────────

from layer_b.word_tree_builder import _TableType, _classify_table

def _cell(text: str, col_hdr: bool = False) -> dict:
    return {"text": text, "column_header": col_hdr}

def _hdr(*texts):
    return [_cell(t, col_hdr=True) for t in texts]

def _row(*texts):
    return [_cell(t, col_hdr=False) for t in texts]


class TestClassifyTable:
    def test_matrix_unique_nav_labels(self):
        grid = [
            _hdr("類別", "2月", "4月"),
            _row("急救護理用品基數查核", "正確", "不正確"),
            _row("急救護理用品有效期限", "期限內", "期限2個月內"),
            _row("備用量適當性", "適當", "增加"),
        ]
        assert _classify_table(grid) == _TableType.MATRIX

    def test_longitudinal_repeated_col0(self):
        grid = [
            _hdr("類別", "項目"),
            _row("外滲皮膚外觀", "大小"),
            _row("外滲皮膚外觀", "腫脹"),
            _row("外滲皮膚外觀", "水泡"),
            _row("傷口照護", "Tegaderm"),
            _row("傷口照護", "Duoderm"),
        ]
        assert _classify_table(grid) == _TableType.LONGITUDINAL

    def test_record_slash_placeholders(self):
        grid = [
            _hdr("日期", "1", "2"),
            _row("/", "", ""),
            _row("：", "", ""),
            _row("診斷：", "", ""),
        ]
        assert _classify_table(grid) == _TableType.RECORD

    def test_record_field_labels_transposed(self):
        # B21: rows are field names, records are in columns
        grid = [
            _hdr("單位", "", ""),
            _row("日期/班別/", "", ""),
            _row("不符合項目", "", ""),
            _row("處理對策", "", ""),
            _row("再確認", "", ""),
        ]
        # col0 majority are nav-looking but table has < 3 unique meaningful rows
        # with a header "單位" and field rows that end with / or are action nouns;
        # since non-empty col0 ratio of nav_labels is ≥ 0.5 and all unique → MATRIX
        # BUT wait — this should be RECORD because the RECORDS are in columns.
        # Heuristic: if rows ≤ 6 AND unique meaningful labels AND each label
        # is a "field descriptor" (contains /, ends with 項目/對策/確認), treat as RECORD.
        # Implementation detail: the classifier uses a field-label regex to detect this.
        assert _classify_table(grid) == _TableType.RECORD

    def test_chart_numeric_col0(self):
        grid = [
            _hdr("溫度", "1", "2"),
            _row("14", "", ""),
            _row("12", "", ""),
            _row("8", "", ""),
            _row("-2", "", ""),
            _row("-4", "", ""),
        ]
        assert _classify_table(grid) == _TableType.CHART

    def test_index_sequential_integers(self):
        grid = [_hdr("序號", "編號", "名稱")] + [
            _row(str(i), f"A{i:02d}", f"表單{i}") for i in range(1, 8)
        ]
        assert _classify_table(grid) == _TableType.INDEX

    def test_empty_grid_returns_record(self):
        assert _classify_table([]) == _TableType.RECORD

    def test_single_header_row_returns_record(self):
        grid = [_hdr("類別", "內容")]
        assert _classify_table(grid) == _TableType.RECORD


# ── Table → TreeNode strategies ───────────────────────────────────────────────

from layer_b.word_tree_builder import _table_to_nodes, _table_to_markdown


class TestTableToNodes:
    def test_matrix_one_leaf_per_data_row(self):
        grid = [
            _hdr("類別", "2月", "4月"),
            _row("基數查核", "正確", "不正確"),
            _row("有效期限", "期限內", "2個月內"),
            _row("備用量適當性", "適當", "增加"),
        ]
        nodes = _table_to_nodes(grid, "查核表")
        assert len(nodes) == 3
        assert nodes[0].title == "基數查核"
        assert "正確" in nodes[0].content
        assert "不正確" in nodes[0].content
        assert nodes[1].title == "有效期限"
        assert nodes[2].title == "備用量適當性"
        for n in nodes:
            assert n.is_leaf
            assert n.summary == ""
            assert n.start_page is None

    def test_matrix_skips_empty_col0_rows(self):
        grid = [
            _hdr("類別", "值"),
            _row("有意義", "abc"),
            _row("", "xyz"),  # empty col=0 → skip
        ]
        nodes = _table_to_nodes(grid, "表")
        assert len(nodes) == 1
        assert nodes[0].title == "有意義"

    def test_longitudinal_groups_by_category(self):
        grid = [
            _hdr("類別", "項目"),
            _row("皮膚外觀", "大小"),
            _row("皮膚外觀", "腫脹"),
            _row("皮膚外觀", "水泡"),
            _row("傷口照護", "Tegaderm"),
            _row("傷口照護", "Duoderm"),
        ]
        nodes = _table_to_nodes(grid, "追蹤表")
        assert len(nodes) == 2
        assert nodes[0].title == "皮膚外觀"
        assert "大小" in nodes[0].content
        assert "腫脹" in nodes[0].content
        assert "水泡" in nodes[0].content
        assert nodes[1].title == "傷口照護"
        assert "Tegaderm" in nodes[1].content

    def test_record_single_leaf_with_full_table(self):
        grid = [
            _hdr("日期", "1", "2"),
            _row("/", "", ""),
            _row("診斷：", "", ""),
        ]
        nodes = _table_to_nodes(grid, "記錄單")
        assert len(nodes) == 1
        assert nodes[0].title == "記錄單"
        assert nodes[0].is_leaf
        assert "|" in nodes[0].content  # markdown table

    def test_chart_returns_empty_list(self):
        grid = [
            _hdr("溫度", "1", "2"),
            _row("14", "", ""),
            _row("8", "", ""),
            _row("-2", "", ""),
        ]
        nodes = _table_to_nodes(grid, "溫度記錄")
        assert nodes == []

    def test_index_chunks_into_groups_of_30(self):
        data = [_row(str(i), f"A{i:02d}", f"表單{i}") for i in range(1, 65)]
        grid = [_hdr("序號", "編號", "名稱")] + data
        nodes = _table_to_nodes(grid, "表單目錄")
        assert len(nodes) == 3         # ceil(64/30) = 3
        assert "1" in nodes[0].title
        assert "30" in nodes[0].title
        assert "31" in nodes[1].title
        assert "60" in nodes[1].title
        assert "|" in nodes[0].content  # markdown table

    def test_table_to_markdown_produces_pipe_table(self):
        grid = [
            _hdr("A", "B"),
            _row("x", "y"),
        ]
        md = _table_to_markdown(grid)
        assert "| A | B |" in md
        assert "|---|---|" in md
        assert "| x | y |" in md
