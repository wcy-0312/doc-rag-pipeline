import sys, os

from layer_b.models import IRCell, IRTable, QC
from layer_b.table import build_header_paths, linearize_kv, to_json


def _cell(row, col, content, is_col_header=False, is_row_header=False):
    return IRCell(
        row_index=row, col_index=col, row_span=1, col_span=1,
        content=content, is_col_header=is_col_header, is_row_header=is_row_header,
    )


def _labelled(cells):
    t = IRTable("t_000", "azure_cu", [1], cells, QC())
    return build_header_paths(t)


# ── 計畫書範例：護理問題表格 ──────────────────────────────────────────────────

def test_nursing_table_kv():
    """
    計畫書範例輸出：
      護理問題: 疼痛管理 | 措施: 給藥PRN | 衛教: 解釋副作用
      護理問題: 疼痛管理 | 措施: 更換姿勢 | 衛教: 示範技巧
      護理問題: 跌倒風險 | 措施: 床欄升起 | 衛教: 教導呼叫
    """
    cells = [
        _cell(0, 0, "護理問題", is_col_header=True),
        _cell(0, 1, "措施", is_col_header=True),
        _cell(0, 2, "衛教", is_col_header=True),
        _cell(1, 0, "疼痛管理", is_row_header=True),
        _cell(1, 1, "給藥PRN"),
        _cell(1, 2, "解釋副作用"),
        _cell(2, 0, "疼痛管理", is_row_header=True),
        _cell(2, 1, "更換姿勢"),
        _cell(2, 2, "示範技巧"),
        _cell(3, 0, "跌倒風險", is_row_header=True),
        _cell(3, 1, "床欄升起"),
        _cell(3, 2, "教導呼叫"),
    ]
    result = linearize_kv(_labelled(cells))
    lines = result.strip().split("\n")
    assert lines[0] == "護理問題: 疼痛管理 | 措施: 給藥PRN | 衛教: 解釋副作用"
    assert lines[1] == "護理問題: 疼痛管理 | 措施: 更換姿勢 | 衛教: 示範技巧"
    assert lines[2] == "護理問題: 跌倒風險 | 措施: 床欄升起 | 衛教: 教導呼叫"


# ── 階層式 col header（> 分隔）───────────────────────────────────────────────

def test_hierarchical_col_header_kv():
    """
    計畫書範例：
      適應症: 輕度 | 第一線治療 > 劑量: 100mg | 第一線治療 > 療程: 7天
    """
    cells = [
        _cell(0, 0, "適應症", is_col_header=True),
        _cell(0, 1, "第一線治療", is_col_header=True),
        _cell(0, 2, "第一線治療", is_col_header=True),
        _cell(1, 0, "適應症", is_col_header=True),   # rowspan 展開
        _cell(1, 1, "劑量", is_col_header=True),
        _cell(1, 2, "療程", is_col_header=True),
        _cell(2, 0, "輕度", is_row_header=True),
        _cell(2, 1, "100mg"),
        _cell(2, 2, "7天"),
    ]
    result = linearize_kv(_labelled(cells))
    assert "第一線治療 > 劑量: 100mg" in result
    assert "第一線治療 > 療程: 7天" in result
    assert "適應症: 輕度" in result


# ── col header only（無 row header）─────────────────────────────────────────

def test_no_row_header():
    cells = [
        _cell(0, 0, "姓名", is_col_header=True),
        _cell(0, 1, "年齡", is_col_header=True),
        _cell(1, 0, "王小明"),
        _cell(1, 1, "30"),
    ]
    result = linearize_kv(_labelled(cells))
    assert result == "姓名: 王小明 | 年齡: 30"


# ── JSON 輸出 ────────────────────────────────────────────────────────────────

def test_json_structure():
    cells = [
        _cell(0, 0, "護理問題", is_col_header=True),
        _cell(0, 1, "措施", is_col_header=True),
        _cell(1, 0, "疼痛管理", is_row_header=True),
        _cell(1, 1, "給藥PRN"),
    ]
    result = to_json(_labelled(cells))
    assert result["table_id"] == "t_000"
    assert result["source_pages"] == [1]
    assert len(result["rows"]) == 1
    row = result["rows"][0]
    assert row["row_header_path"] == ["疼痛管理"]
    assert row["cells"][0]["col_header_path"] == ["措施"]
    assert row["cells"][0]["value"] == "給藥PRN"


def test_json_skips_header_rows():
    """純 header row 不應出現在 JSON rows 中。"""
    cells = [
        _cell(0, 0, "欄A", is_col_header=True),
        _cell(1, 0, "資料"),
    ]
    result = to_json(_labelled(cells))
    assert len(result["rows"]) == 1


def test_empty_table():
    result = linearize_kv(_labelled([]))
    assert result == ""


if __name__ == "__main__":
    test_nursing_table_kv()
    test_hierarchical_col_header_kv()
    test_no_row_header()
    test_json_structure()
    test_json_skips_header_rows()
    test_empty_table()
    print("All formatter tests passed.")
