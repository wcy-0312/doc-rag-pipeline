from __future__ import annotations
from typing import Any
from layer_b.models import BoundingBox, IRCell, IRTable, QC


def _parse_bounding_box(bbox: dict | None) -> BoundingBox | None:
    if not bbox:
        return None
    return BoundingBox(
        page=bbox.get("page_no", 1),
        x0=bbox.get("l", 0.0),
        y0=bbox.get("t", 0.0),
        x1=bbox.get("r", 0.0),
        y1=bbox.get("b", 0.0),
    )


def _parse_cell(cell: dict) -> IRCell:
    is_col_header = bool(cell.get("column_header", False))
    is_row_header = bool(cell.get("row_header", False))
    return IRCell(
        row_index=cell.get("start_row_offset_idx", 0),
        col_index=cell.get("start_col_offset_idx", 0),
        row_span=cell.get("row_span", 1),
        col_span=cell.get("col_span", 1),
        content=cell.get("text", ""),
        is_row_header=is_row_header,
        is_col_header=is_col_header,
        header_source="flag",
        confidence=None,
        bounding_box=_parse_bounding_box(cell.get("bbox")),
    )


def _apply_header_heuristics(cells: list[IRCell]) -> list[IRCell]:
    """若無任何 header 旗標，以第一列為 column header、第一欄為 row header。"""
    has_any_header = any(c.is_row_header or c.is_col_header for c in cells)
    if has_any_header:
        return cells
    for c in cells:
        if c.row_index == 0:
            c.is_col_header = True
            c.header_source = "heuristic"
        if c.col_index == 0 and c.row_index != 0:
            c.is_row_header = True
            c.header_source = "heuristic"
    return cells


def _parse_qc(metadata: dict) -> QC:
    qc_data = metadata.get("qc", {})
    return QC(
        empty_cell_rate=qc_data.get("empty_cell_rate", 0.0),
        qc_level=qc_data.get("qc_level", "ok"),
        warnings=metadata.get("extractor_metadata", {}).get("warnings", qc_data.get("warnings", [])),
        word_avg=None,
        low_confidence_rate=None,
        estimated_info_loss_rate=qc_data.get("estimated_info_loss_rate"),
    )


def adapt(raw: dict) -> list[IRTable]:
    """Convert Docling native output to list of IRTable."""
    metadata = raw.get("metadata", {})
    qc = _parse_qc(metadata)

    tables = raw.get("data", {}).get("tables", [])
    result: list[IRTable] = []

    for i, table in enumerate(tables):
        cells = [_parse_cell(c) for c in table.get("data", {}).get("table_cells", [])]
        cells = _apply_header_heuristics(cells)

        prov = table.get("prov", [])
        # Docling DOCX (xml_native) has prov=[] — return [] rather than faking page=1.
        pages = sorted({p.get("page_no") for p in prov if p.get("page_no") is not None}) or []

        result.append(IRTable(
            table_id=f"t_{i:03d}",
            source_tool="docling",
            source_pages=pages,
            cells=cells,
            qc=qc,
        ))

    return result
