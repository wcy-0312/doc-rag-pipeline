import sys, os

from layer_b.models import IRCell, IRTable, QC
from layer_b.normalizers.header_path import build_header_paths


def _cell(row, col, content, row_span=1, col_span=1,
          is_col_header=False, is_row_header=False, header_source="flag"):
    return IRCell(
        row_index=row, col_index=col,
        row_span=row_span, col_span=col_span,
        content=content,
        is_col_header=is_col_header,
        is_row_header=is_row_header,
        header_source=header_source,
    )


def _table(cells):
    return IRTable("t_000", "azure_cu", [1], cells, QC(), {})


def _at(labelled_cells, row, col):
    return next((c for c in labelled_cells if c.row_index == row and c.col_index == col), None)


# ── 簡單表格（單層 header）────────────────────────────────────────────────────

def test_simple_col_header():
    """
    | 護理問題 | 措施     |  ← col headers (row 0)
    | 疼痛管理 | 給藥 PRN |  ← body (row 1)
    """
    cells = [
        _cell(0, 0, "護理問題", is_col_header=True),
        _cell(0, 1, "措施", is_col_header=True),
        _cell(1, 0, "疼痛管理"),
        _cell(1, 1, "給藥PRN"),
    ]
    result = build_header_paths(_table(cells))
    c = _at(result.cells, 1, 1)
    assert c.col_header_path == ["措施"]
    assert c.row_header_path == []


def test_simple_row_header():
    """
    | 疼痛管理 | 給藥PRN  |  row header at col 0
    | 跌倒風險 | 床欄升起 |
    """
    cells = [
        _cell(0, 0, "疼痛管理", is_row_header=True),
        _cell(0, 1, "給藥PRN"),
        _cell(1, 0, "跌倒風險", is_row_header=True),
        _cell(1, 1, "床欄升起"),
    ]
    result = build_header_paths(_table(cells))
    c = _at(result.cells, 0, 1)
    assert c.row_header_path == ["疼痛管理"]
    assert c.col_header_path == []


def test_col_and_row_header():
    """
    |          | 措施     | 衛教      |  ← col headers
    | 疼痛管理 | 給藥PRN  | 解釋副作用|  ← row header at col 0
    | 跌倒風險 | 床欄升起 | 教導呼叫  |
    """
    cells = [
        _cell(0, 1, "措施", is_col_header=True),
        _cell(0, 2, "衛教", is_col_header=True),
        _cell(1, 0, "疼痛管理", is_row_header=True),
        _cell(1, 1, "給藥PRN"),
        _cell(1, 2, "解釋副作用"),
        _cell(2, 0, "跌倒風險", is_row_header=True),
        _cell(2, 1, "床欄升起"),
        _cell(2, 2, "教導呼叫"),
    ]
    result = build_header_paths(_table(cells))
    c = _at(result.cells, 1, 1)
    assert c.col_header_path == ["措施"]
    assert c.row_header_path == ["疼痛管理"]

    c2 = _at(result.cells, 2, 2)
    assert c2.col_header_path == ["衛教"]
    assert c2.row_header_path == ["跌倒風險"]


# ── 階層式欄位頭（多層 col header）──────────────────────────────────────────

def test_hierarchical_col_headers():
    """
    | 適應症   | 第一線治療       | 第二線治療       |  ← row 0 col headers
    |          | 劑量  | 療程    | 劑量  | 療程    |  ← row 1 col headers
    | 輕度     | 100mg | 7天     | 200mg | 14天    |
    """
    cells = [
        # row 0: 階層外層（展平後每個被覆蓋欄都有複製）
        _cell(0, 0, "適應症", is_col_header=True),
        _cell(0, 1, "第一線治療", is_col_header=True),
        _cell(0, 2, "第一線治療", is_col_header=True),  # colspan 展開後的複製
        _cell(0, 3, "第二線治療", is_col_header=True),
        _cell(0, 4, "第二線治療", is_col_header=True),  # colspan 展開後的複製
        # row 1: 階層內層
        _cell(1, 0, "適應症", is_col_header=True),      # rowspan 展開後的複製
        _cell(1, 1, "劑量", is_col_header=True),
        _cell(1, 2, "療程", is_col_header=True),
        _cell(1, 3, "劑量", is_col_header=True),
        _cell(1, 4, "療程", is_col_header=True),
        # row 2: body
        _cell(2, 0, "輕度", is_row_header=True),
        _cell(2, 1, "100mg"),
        _cell(2, 2, "7天"),
        _cell(2, 3, "200mg"),
        _cell(2, 4, "14天"),
    ]
    result = build_header_paths(_table(cells))

    c = _at(result.cells, 2, 1)  # 100mg
    assert c.col_header_path == ["第一線治療", "劑量"]
    assert c.row_header_path == ["輕度"]

    c2 = _at(result.cells, 2, 4)  # 14天
    assert c2.col_header_path == ["第二線治療", "療程"]


# ── heuristic header source ──────────────────────────────────────────────────

def test_heuristic_header_source_preserved():
    """header_source 欄位應傳遞到 LabelledCell。"""
    cells = [
        _cell(0, 0, "欄A", is_col_header=True, header_source="heuristic"),
        _cell(1, 0, "資料"),
    ]
    result = build_header_paths(_table(cells))
    header = _at(result.cells, 0, 0)
    assert header.header_source == "heuristic"


# ── header cells 本身的 path ─────────────────────────────────────────────────

def test_header_cells_have_empty_own_axis_path():
    """col header cells 的 col_header_path 應為空（它們自己就是路徑節點）。"""
    cells = [
        _cell(0, 0, "護理問題", is_col_header=True),
        _cell(1, 0, "疼痛管理"),
    ]
    result = build_header_paths(_table(cells))
    hdr = _at(result.cells, 0, 0)
    assert hdr.col_header_path == []


def test_empty_table():
    result = build_header_paths(_table([]))
    assert result.cells == []


if __name__ == "__main__":
    test_simple_col_header()
    test_simple_row_header()
    test_col_and_row_header()
    test_hierarchical_col_headers()
    test_heuristic_header_source_preserved()
    test_header_cells_have_empty_own_axis_path()
    test_empty_table()
    print("All header_path tests passed.")
