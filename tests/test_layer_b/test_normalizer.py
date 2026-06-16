import sys, os

from layer_b.models import IRCell, IRTable, QC
from layer_b.normalizers.normalizer import expand_spans


def _make_cell(row, col, content, row_span=1, col_span=1,
               is_col_header=False, is_row_header=False):
    return IRCell(
        row_index=row, col_index=col,
        row_span=row_span, col_span=col_span,
        content=content,
        is_col_header=is_col_header,
        is_row_header=is_row_header,
    )


def _make_table(cells):
    return IRTable(
        table_id="t_000", source_tool="azure_cu",
        source_pages=[1], cells=cells,
        qc=QC(), page_image_refs={},
    )


def _cell_at(cells, row, col):
    return next((c for c in cells if c.row_index == row and c.col_index == col), None)


# ── rowspan ──────────────────────────────────────────────────────────────────

def test_rowspan_2_copies_to_next_row():
    """
    原始表格（rowspan=2 的疼痛管理）：
      列0: [疼痛管理(rowspan=2), 措施header]
      列1: [null,                給藥PRN]
    展平後：
      列1 的 col_index=0 應有「疼痛管理」的複製
    """
    cells = [
        _make_cell(0, 0, "疼痛管理", row_span=2, is_row_header=True),
        _make_cell(0, 1, "措施", is_col_header=True),
        _make_cell(1, 1, "給藥PRN"),
    ]
    result = expand_spans(_make_table(cells))
    c = _cell_at(result.cells, 1, 0)
    assert c is not None
    assert c.content == "疼痛管理"
    assert c.row_span == 1  # 展平後 span=1
    assert c.is_row_header is True


def test_rowspan_3_copies_to_two_rows():
    cells = [
        _make_cell(0, 0, "A", row_span=3),
        _make_cell(0, 1, "B"),
        _make_cell(1, 1, "C"),
        _make_cell(2, 1, "D"),
    ]
    result = expand_spans(_make_table(cells))
    assert _cell_at(result.cells, 0, 0).content == "A"
    assert _cell_at(result.cells, 1, 0).content == "A"
    assert _cell_at(result.cells, 2, 0).content == "A"


# ── colspan ──────────────────────────────────────────────────────────────────

def test_colspan_2_copies_to_next_col():
    cells = [
        _make_cell(0, 0, "合併標題", col_span=2, is_col_header=True),
        _make_cell(1, 0, "資料A"),
        _make_cell(1, 1, "資料B"),
    ]
    result = expand_spans(_make_table(cells))
    c = _cell_at(result.cells, 0, 1)
    assert c is not None
    assert c.content == "合併標題"
    assert c.col_span == 1
    assert c.is_col_header is True


# ── rowspan + colspan 同時 ───────────────────────────────────────────────────

def test_rowspan_and_colspan():
    """2×2 合併儲存格應展開為 4 個 cells。"""
    cells = [
        _make_cell(0, 0, "大標題", row_span=2, col_span=2, is_col_header=True),
        _make_cell(0, 2, "欄C", is_col_header=True),
        _make_cell(1, 2, "資料C"),
    ]
    result = expand_spans(_make_table(cells))
    positions = {(c.row_index, c.col_index) for c in result.cells if c.content == "大標題"}
    assert positions == {(0, 0), (0, 1), (1, 0), (1, 1)}


# ── no-op cases ──────────────────────────────────────────────────────────────

def test_no_span_unchanged():
    """無合併儲存格的表格，輸出 cell 數量不變。"""
    cells = [
        _make_cell(0, 0, "欄A", is_col_header=True),
        _make_cell(0, 1, "欄B", is_col_header=True),
        _make_cell(1, 0, "資料1"),
        _make_cell(1, 1, "資料2"),
    ]
    result = expand_spans(_make_table(cells))
    assert len(result.cells) == 4


def test_empty_table():
    result = expand_spans(_make_table([]))
    assert result.cells == []


def test_metadata_preserved():
    """table_id、source_pages、page_image_refs 應保持不變。"""
    t = IRTable(
        table_id="t_007", source_tool="docling",
        source_pages=[3, 4],
        cells=[_make_cell(0, 0, "X")],
        qc=QC(empty_cell_rate=0.1),
        page_image_refs={"3": "p3.png", "4": "p4.png"},
    )
    result = expand_spans(t)
    assert result.table_id == "t_007"
    assert result.source_pages == [3, 4]
    assert result.page_image_refs == {"3": "p3.png", "4": "p4.png"}
    assert result.qc.empty_cell_rate == 0.1


def test_sorted_by_row_col():
    """展平後 cells 應依 (row_index, col_index) 排序。"""
    cells = [
        _make_cell(1, 0, "B"),
        _make_cell(0, 0, "A", row_span=2),
    ]
    result = expand_spans(_make_table(cells))
    indices = [(c.row_index, c.col_index) for c in result.cells]
    assert indices == sorted(indices)


def test_origin_cell_span_reset_to_one():
    """主格（origin cell）展平後 row_span/col_span 也應重置為 1。"""
    cells = [_make_cell(0, 0, "疼痛管理", row_span=2, is_row_header=True)]
    result = expand_spans(_make_table(cells))
    origin = next(c for c in result.cells if c.row_index == 0 and c.col_index == 0)
    assert origin.row_span == 1
    assert origin.col_span == 1


def test_azure_cu_no_placeholder_cells():
    """Azure CU 不輸出空 cell 佔位，展平後被覆蓋位置應由演算法填充。

    模擬洗手法評核表：
      [1,0] rSpan=4 → 被覆蓋 (2,0)(3,0)(4,0) 不存在於輸入
      [1,1] rSpan=2 → 被覆蓋 (2,1) 不存在於輸入
    """
    cells = [
        _make_cell(0, 0, "步驟", is_col_header=True),
        _make_cell(0, 1, "評核項目", is_col_header=True),
        _make_cell(1, 0, "洗手步驟", row_span=4, is_row_header=True),
        _make_cell(1, 1, "濕手", row_span=2),
        _make_cell(2, 1, "抹皂"),
        _make_cell(3, 1, "搓洗"),
        _make_cell(4, 1, "沖水"),
    ]
    result = expand_spans(_make_table(cells))

    def contents_at(row, col):
        return {c.content for c in result.cells if c.row_index == row and c.col_index == col}

    # 被覆蓋位置應存在（由演算法填充）
    assert contents_at(2, 0) == {"洗手步驟"}
    assert contents_at(3, 0) == {"洗手步驟"}
    assert contents_at(4, 0) == {"洗手步驟"}
    # (2,1) 有「濕手」的複製（rSpan=2 的展開），以及原本的「抹皂」
    assert "濕手" in contents_at(2, 1)

    # 所有 cells 的 span 都應是 1
    assert all(c.row_span == 1 and c.col_span == 1 for c in result.cells)

    # header cell（columnHeader）的 colspan 展平
def test_column_header_with_colspan():
    """columnHeader cell 也可以有 colSpan，展平後被覆蓋欄也是 col_header。"""
    cells = [
        _make_cell(0, 0, "大分類", col_span=3, is_col_header=True),
        _make_cell(1, 0, "A"),
        _make_cell(1, 1, "B"),
        _make_cell(1, 2, "C"),
    ]
    result = expand_spans(_make_table(cells))
    header_positions = {(c.row_index, c.col_index) for c in result.cells if c.is_col_header}
    assert (0, 0) in header_positions
    assert (0, 1) in header_positions
    assert (0, 2) in header_positions


if __name__ == "__main__":
    test_rowspan_2_copies_to_next_row()
    test_rowspan_3_copies_to_two_rows()
    test_colspan_2_copies_to_next_col()
    test_rowspan_and_colspan()
    test_no_span_unchanged()
    test_empty_table()
    test_metadata_preserved()
    test_sorted_by_row_col()
    test_origin_cell_span_reset_to_one()
    test_azure_cu_no_placeholder_cells()
    test_column_header_with_colspan()
    print("All normalizer tests passed.")
