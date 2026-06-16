import sys, os

from layer_b.models import IRCell, IRTable, QC
from layer_b.normalizers.merger import merge_cross_page


def _make_cell(row, col, content, is_col_header=False, is_row_header=False):
    return IRCell(
        row_index=row, col_index=col,
        row_span=1, col_span=1,
        content=content,
        is_col_header=is_col_header,
        is_row_header=is_row_header,
    )


def _make_table(table_id, pages, cells, caption=""):
    t = IRTable(
        table_id=table_id,
        source_tool="azure_cu",
        source_pages=pages,
        cells=cells,
        qc=QC(),
        page_image_refs={str(p): f"img/p{p}.png" for p in pages},
    )
    t.caption = caption
    return t


# ── 三頁連續跨頁表格（模擬真實醫院文件）────────────────────────────────────

def test_three_page_merge_header_match():
    """三頁 header 完全相同，應合併為一張表格。"""
    def page_table(page_num, table_id, body_content):
        return _make_table(table_id, [page_num], [
            _make_cell(0, 0, "主題名稱", is_col_header=True),
            _make_cell(0, 1, "病人入院作業標準", is_col_header=True),
            _make_cell(1, 0, body_content),
            _make_cell(1, 1, f"內容{page_num}"),
        ])

    tables = [
        page_table(2, "t_000", "A"),
        page_table(3, "t_001", "B"),
        page_table(4, "t_002", "C"),
    ]
    result = merge_cross_page(tables)
    assert len(result) == 1
    assert result[0].source_pages == [2, 3, 4]
    assert result[0].page_image_refs == {"2": "img/p2.png", "3": "img/p3.png", "4": "img/p4.png"}
    # header 只留一份
    headers = [c for c in result[0].cells if c.is_col_header]
    assert len(headers) == 2
    # body cells 應有 3 頁各一列 × 2 欄 = 6 cells
    body = [c for c in result[0].cells if not c.is_col_header]
    assert len(body) == 6
    # row_index 應連續不重複
    body_rows = sorted({c.row_index for c in body})
    assert body_rows == [1, 2, 3]


def test_merge_no_header_second_table():
    """第二個表格無 header，視為延續。"""
    t1 = _make_table("t_000", [1], [
        _make_cell(0, 0, "欄A", is_col_header=True),
        _make_cell(0, 1, "欄B", is_col_header=True),
        _make_cell(1, 0, "資料1"),
        _make_cell(1, 1, "資料2"),
    ])
    t2 = _make_table("t_001", [2], [
        _make_cell(0, 0, "資料3"),
        _make_cell(0, 1, "資料4"),
    ])
    result = merge_cross_page([t1, t2])
    assert len(result) == 1
    assert result[0].source_pages == [1, 2]
    body = [c for c in result[0].cells if not c.is_col_header]
    assert len(body) == 4


def test_different_header_no_merge():
    """Header 不同，不合併。"""
    t1 = _make_table("t_000", [1], [
        _make_cell(0, 0, "欄A", is_col_header=True),
        _make_cell(1, 0, "資料1"),
    ])
    t2 = _make_table("t_001", [2], [
        _make_cell(0, 0, "欄X", is_col_header=True),
        _make_cell(1, 0, "資料2"),
    ])
    result = merge_cross_page([t1, t2])
    assert len(result) == 2


def test_non_consecutive_pages_no_merge():
    """頁碼不連續，不合併。"""
    t1 = _make_table("t_000", [1], [
        _make_cell(0, 0, "欄A", is_col_header=True),
        _make_cell(1, 0, "資料1"),
    ])
    t2 = _make_table("t_001", [3], [  # 跳過 page 2
        _make_cell(0, 0, "欄A", is_col_header=True),
        _make_cell(1, 0, "資料2"),
    ])
    result = merge_cross_page([t1, t2])
    assert len(result) == 2


def test_different_column_count_no_merge():
    """列數不同，不合併（即使 header 文字相同）。"""
    t1 = _make_table("t_000", [1], [
        _make_cell(0, 0, "欄A", is_col_header=True),
        _make_cell(0, 1, "欄B", is_col_header=True),
        _make_cell(1, 0, "資料1"),
        _make_cell(1, 1, "資料2"),
    ])
    t2 = _make_table("t_001", [2], [
        _make_cell(0, 0, "欄A", is_col_header=True),
        _make_cell(1, 0, "資料3"),
    ])
    result = merge_cross_page([t1, t2])
    assert len(result) == 2


def test_caption_match_merge():
    """Caption 相同時優先以 caption 判斷合併。"""
    t1 = _make_table("t_000", [1], [
        _make_cell(0, 0, "欄A", is_col_header=True),
        _make_cell(1, 0, "資料1"),
    ], caption="表一、各癌症分期存活率")
    t2 = _make_table("t_001", [2], [
        _make_cell(0, 0, "欄A", is_col_header=True),
        _make_cell(1, 0, "資料2"),
    ], caption="表一、各癌症分期存活率")
    result = merge_cross_page([t1, t2])
    assert len(result) == 1


def test_empty_input():
    assert merge_cross_page([]) == []


def test_single_table_unchanged():
    t = _make_table("t_000", [1], [_make_cell(0, 0, "A", is_col_header=True)])
    result = merge_cross_page([t])
    assert len(result) == 1
    assert result[0].table_id == "t_000"


if __name__ == "__main__":
    test_three_page_merge_header_match()
    test_merge_no_header_second_table()
    test_different_header_no_merge()
    test_non_consecutive_pages_no_merge()
    test_different_column_count_no_merge()
    test_caption_match_merge()
    test_empty_input()
    test_single_table_unchanged()
    print("All merger tests passed.")
