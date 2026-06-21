
from layer_b.adapters import azure_cu_adapter, docling_adapter, azure_di_adapter
from layer_b.adapters import adapt as unified_adapt


# ── Azure CU adapter ────────────────────────────────────────────────────────

def _azure_raw(cells: list[dict], page_images: dict = None) -> dict:
    return {
        "metadata": {"qc": {"empty_cell_rate": 0.0, "qc_level": "ok", "warnings": []}},
        "data": {
            "tables": [{"cells": cells}],
            "page_images": page_images or {"1": "img/p1.png"},
        },
    }


def test_azure_cu_simple_table():
    cells = [
        {"rowIndex": 0, "columnIndex": 0, "rowSpan": 1, "columnSpan": 1,
         "content": "護理問題", "kind": "columnHeader",
         "confidence": 0.99, "boundingRegions": [{"pageNumber": 1, "polygon": [0,0,1,0,1,1,0,1]}]},
        {"rowIndex": 1, "columnIndex": 0, "rowSpan": 1, "columnSpan": 1,
         "content": "疼痛管理", "kind": "content",
         "confidence": 0.95, "boundingRegions": [{"pageNumber": 1, "polygon": [0,1,1,1,1,2,0,2]}]},
    ]
    tables = azure_cu_adapter.adapt(_azure_raw(cells))
    assert len(tables) == 1
    t = tables[0]
    assert t.source_tool == "azure_cu"
    assert t.cells[0].is_col_header is True
    assert t.cells[0].header_source == "flag"
    assert t.cells[0].confidence == 0.99
    assert t.cells[1].is_col_header is False


def test_azure_cu_rowspan():
    cells = [
        {"rowIndex": 0, "columnIndex": 0, "rowSpan": 2, "columnSpan": 1,
         "content": "疼痛管理", "kind": "rowHeader", "confidence": 0.9,
         "boundingRegions": [{"pageNumber": 1, "polygon": [0,0,1,0,1,2,0,2]}]},
        {"rowIndex": 0, "columnIndex": 1, "rowSpan": 1, "columnSpan": 1,
         "content": "措施", "kind": "columnHeader", "confidence": 0.95,
         "boundingRegions": [{"pageNumber": 1, "polygon": [1,0,2,0,2,1,1,1]}]},
        {"rowIndex": 1, "columnIndex": 1, "rowSpan": 1, "columnSpan": 1,
         "content": "給藥 PRN", "kind": "content", "confidence": 0.88,
         "boundingRegions": [{"pageNumber": 1, "polygon": [1,1,2,1,2,2,1,2]}]},
    ]
    tables = azure_cu_adapter.adapt(_azure_raw(cells))
    t = tables[0]
    row_header = next(c for c in t.cells if c.content == "疼痛管理")
    assert row_header.is_row_header is True
    assert row_header.row_span == 2


def test_azure_cu_heuristic_fallback():
    """無任何 kind 旗標時，應 fallback 為 heuristic。"""
    cells = [
        {"rowIndex": 0, "columnIndex": 0, "rowSpan": 1, "columnSpan": 1,
         "content": "護理問題", "confidence": 0.9,
         "boundingRegions": [{"pageNumber": 1, "polygon": [0,0,1,0,1,1,0,1]}]},
        {"rowIndex": 1, "columnIndex": 0, "rowSpan": 1, "columnSpan": 1,
         "content": "疼痛管理", "confidence": 0.85,
         "boundingRegions": [{"pageNumber": 1, "polygon": [0,1,1,1,1,2,0,2]}]},
    ]
    tables = azure_cu_adapter.adapt(_azure_raw(cells))
    t = tables[0]
    header = next(c for c in t.cells if c.row_index == 0)
    assert header.is_col_header is True
    assert header.header_source == "heuristic"


def test_azure_cu_qc_passthrough():
    raw = {
        "metadata": {"qc": {"empty_cell_rate": 0.35, "qc_level": "warning", "warnings": ["scan_detected"]}},
        "data": {"tables": [{"cells": []}], "page_images": {}},
    }
    tables = azure_cu_adapter.adapt(raw)
    assert tables[0].qc.empty_cell_rate == 0.35
    assert "scan_detected" in tables[0].qc.warnings


# ── Docling adapter ──────────────────────────────────────────────────────────

def _docling_raw(cells: list[dict], page_images: dict = None) -> dict:
    return {
        "metadata": {"qc": {"empty_cell_rate": 0.0, "qc_level": "ok", "warnings": []}},
        "data": {
            "tables": [{"data": {"table_cells": cells}, "prov": [{"page_no": 1}]}],
            "page_images": page_images or {"1": "img/p1.png"},
        },
    }


def test_docling_simple_table():
    cells = [
        {"start_row_offset_idx": 0, "start_col_offset_idx": 0,
         "row_span": 1, "col_span": 1, "text": "護理問題",
         "column_header": True, "row_header": False,
         "bbox": {"page_no": 1, "l": 0.0, "t": 0.0, "r": 1.0, "b": 0.5}},
        {"start_row_offset_idx": 1, "start_col_offset_idx": 0,
         "row_span": 1, "col_span": 1, "text": "疼痛管理",
         "column_header": False, "row_header": True,
         "bbox": {"page_no": 1, "l": 0.0, "t": 0.5, "r": 1.0, "b": 1.0}},
    ]
    tables = docling_adapter.adapt(_docling_raw(cells))
    assert len(tables) == 1
    t = tables[0]
    assert t.source_tool == "docling"
    assert t.cells[0].is_col_header is True
    assert t.cells[0].header_source == "flag"
    assert t.cells[0].confidence is None
    assert t.cells[1].is_row_header is True


def test_docling_heuristic_fallback():
    """無 column_header/row_header 旗標時，應 fallback 為 heuristic。"""
    cells = [
        {"start_row_offset_idx": 0, "start_col_offset_idx": 0,
         "row_span": 1, "col_span": 1, "text": "項目",
         "column_header": False, "row_header": False, "bbox": None},
        {"start_row_offset_idx": 1, "start_col_offset_idx": 0,
         "row_span": 1, "col_span": 1, "text": "數值",
         "column_header": False, "row_header": False, "bbox": None},
    ]
    tables = docling_adapter.adapt(_docling_raw(cells))
    t = tables[0]
    header = next(c for c in t.cells if c.row_index == 0)
    assert header.is_col_header is True
    assert header.header_source == "heuristic"


def test_docling_colspan():
    cells = [
        {"start_row_offset_idx": 0, "start_col_offset_idx": 0,
         "row_span": 1, "col_span": 2, "text": "合併標題",
         "column_header": True, "row_header": False, "bbox": None},
        {"start_row_offset_idx": 1, "start_col_offset_idx": 0,
         "row_span": 1, "col_span": 1, "text": "A",
         "column_header": False, "row_header": False, "bbox": None},
        {"start_row_offset_idx": 1, "start_col_offset_idx": 1,
         "row_span": 1, "col_span": 1, "text": "B",
         "column_header": False, "row_header": False, "bbox": None},
    ]
    tables = docling_adapter.adapt(_docling_raw(cells))
    t = tables[0]
    merged = next(c for c in t.cells if c.content == "合併標題")
    assert merged.col_span == 2


def test_docling_empty_prov_source_pages():
    """prov=[] 時（DOCX xml_native 模式）source_pages 應為 []，不應 hardcode 為 [1]。

    與 paragraph path 的 page=None 策略一致：誠實回報無頁碼資訊。
    """
    raw = {
        "metadata": {"qc": {"empty_cell_rate": 0.0, "qc_level": "ok", "warnings": []}},
        "data": {
            "tables": [{
                "data": {"table_cells": [
                    {"start_row_offset_idx": 0, "start_col_offset_idx": 0,
                     "row_span": 1, "col_span": 1, "text": "項目",
                     "column_header": True, "row_header": False, "bbox": None},
                ]},
                "prov": [],
            }],
            "page_images": {},
        },
    }
    tables = docling_adapter.adapt(raw)
    assert len(tables) == 1
    assert tables[0].source_pages == [], (
        "prov=[] 時 source_pages 應為 []，不應 hardcode 為 [1]"
    )


# ── Azure DI adapter ─────────────────────────────────────────────────────────

def _azure_di_raw(cells: list[dict], warnings: list[str] = None) -> dict:
    return {
        "metadata": {"qc": {"empty_cell_rate": 0.0, "qc_level": "ok",
                             "warnings": warnings or []}},
        "data": {
            "tables": [{"cells": cells}],
            "page_images": {"1": "img/p1.png"},
        },
    }


def test_azure_di_heuristic_always():
    """Azure DI 無 header 旗標，全部走 heuristic。"""
    cells = [
        {"rowIndex": 0, "columnIndex": 0, "rowSpan": 1, "columnSpan": 1,
         "content": "護理問題",
         "boundingRegions": [{"pageNumber": 1, "polygon": [0,0,1,0,1,1,0,1]}]},
        {"rowIndex": 1, "columnIndex": 0, "rowSpan": 1, "columnSpan": 1,
         "content": "疼痛管理",
         "boundingRegions": [{"pageNumber": 1, "polygon": [0,1,1,1,1,2,0,2]}]},
    ]
    tables = azure_di_adapter.adapt(_azure_di_raw(cells))
    t = tables[0]
    assert t.source_tool == "azure_di"
    header = next(c for c in t.cells if c.row_index == 0)
    assert header.is_col_header is True
    assert header.header_source == "heuristic"
    assert header.confidence is None


def test_azure_di_no_per_cell_confidence():
    """confidence 欄位必須是 None，不可為 0 或其他值。"""
    cells = [
        {"rowIndex": 0, "columnIndex": 0, "rowSpan": 1, "columnSpan": 1,
         "content": "Header", "boundingRegions": []},
    ]
    tables = azure_di_adapter.adapt(_azure_di_raw(cells))
    assert all(c.confidence is None for c in tables[0].cells)


def test_azure_di_fallback_routes_to_docling():
    """qc.warnings 含 AZURE_DI_FALLBACK_TO_DOCLING 時，統一入口應路由至 Docling adapter。"""
    cells = [
        {"start_row_offset_idx": 0, "start_col_offset_idx": 0,
         "row_span": 1, "col_span": 1, "text": "項目",
         "column_header": True, "row_header": False, "bbox": None},
    ]
    raw = {
        "metadata": {"qc": {"empty_cell_rate": 0.0, "qc_level": "ok",
                             "warnings": ["AZURE_DI_FALLBACK_TO_DOCLING"]}},
        "data": {
            "tables": [{"data": {"table_cells": cells}, "prov": [{"page_no": 1}]}],
            "page_images": {},
        },
    }
    tables = unified_adapt(raw, "azure_document_intelligence")
    assert tables[0].source_tool == "docling"
    assert tables[0].cells[0].header_source == "flag"


if __name__ == "__main__":
    test_azure_cu_simple_table()
    test_azure_cu_rowspan()
    test_azure_cu_heuristic_fallback()
    test_azure_cu_qc_passthrough()
    test_docling_simple_table()
    test_docling_heuristic_fallback()
    test_docling_colspan()
    test_azure_di_heuristic_always()
    test_azure_di_no_per_cell_confidence()
    test_azure_di_fallback_routes_to_docling()
    print("All tests passed.")
