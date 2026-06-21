from __future__ import annotations
import re as _re
from typing import Any
from layer_b.models import BoundingBox, IRCell, IRTable, QC

_SOURCE_PAGE_RE = _re.compile(r'^D\((\d+),')


def _parse_source_page(source_str: str) -> int | None:
    m = _SOURCE_PAGE_RE.match(str(source_str or ""))
    return int(m.group(1)) if m else None


def _parse_bounding_box(regions: list[dict]) -> BoundingBox | None:
    if not regions:
        return None
    r = regions[0]
    polygon = r.get("polygon", [])
    if len(polygon) < 8:
        return None
    xs = polygon[0::2]
    ys = polygon[1::2]
    return BoundingBox(
        page=r.get("pageNumber", 1),
        x0=min(xs),
        y0=min(ys),
        x1=max(xs),
        y1=max(ys),
    )


def _parse_confidence(cell: dict) -> float | None:
    return cell.get("confidence")


def _parse_cell(cell: dict) -> IRCell:
    kind = cell.get("kind", "content")
    is_row_header = kind == "rowHeader"
    is_col_header = kind == "columnHeader"
    return IRCell(
        row_index=cell.get("rowIndex", 0),
        col_index=cell.get("columnIndex", 0),
        row_span=cell.get("rowSpan", 1),
        col_span=cell.get("columnSpan", 1),
        content=cell.get("content", ""),
        is_row_header=is_row_header,
        is_col_header=is_col_header,
        header_source="flag",
        confidence=_parse_confidence(cell),
        bounding_box=_parse_bounding_box(cell.get("boundingRegions", [])),
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
    """Convert Azure CU native output to list of IRTable."""
    metadata = raw.get("metadata", {})
    qc = _parse_qc(metadata)

    tables = raw.get("data", {}).get("tables", [])
    result: list[IRTable] = []

    for i, table in enumerate(tables):
        cells = [_parse_cell(c) for c in table.get("cells", [])]
        cells = _apply_header_heuristics(cells)

        pages = sorted({
            _parse_source_page(c.get("source", ""))
            for c in table.get("cells", [])
            if _parse_source_page(c.get("source", "")) is not None
        }) or [1]

        result.append(IRTable(
            table_id=f"t_{i:03d}",
            source_tool="azure_cu",
            source_pages=pages,
            cells=cells,
            qc=qc,
        ))

    return result
