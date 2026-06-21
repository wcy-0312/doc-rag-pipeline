from __future__ import annotations
from layer_b.models import IRCell, IRTable, QC


def _parse_cell(cell: dict) -> IRCell:
    # Azure DI has no column_header/row_header flags; always heuristic
    return IRCell(
        row_index=cell.get("rowIndex", 0),
        col_index=cell.get("columnIndex", 0),
        row_span=cell.get("rowSpan", 1),
        col_span=cell.get("columnSpan", 1),
        content=cell.get("content", ""),
        is_row_header=False,
        is_col_header=False,
        header_source="heuristic",
        confidence=None,  # SDK confirmed: no per-cell confidence
    )


def _apply_header_heuristics(cells: list[IRCell]) -> list[IRCell]:
    for c in cells:
        if c.row_index == 0:
            c.is_col_header = True
        elif c.col_index == 0:
            c.is_row_header = True
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
    """Convert Azure DI v4.0 native output to list of IRTable.

    If qc.warnings contains AZURE_DI_FALLBACK_TO_DOCLING, the caller should
    route to docling_adapter instead. This adapter handles the Azure DI path.
    """
    metadata = raw.get("metadata", {})
    qc = _parse_qc(metadata)

    tables = raw.get("data", {}).get("tables", [])
    result: list[IRTable] = []

    for i, table in enumerate(tables):
        cells = [_parse_cell(c) for c in table.get("cells", [])]
        cells = _apply_header_heuristics(cells)

        pages = sorted({
            r.get("pageNumber", 1)
            for c in table.get("cells", [])
            for r in c.get("boundingRegions", [])
        }) or [1]

        result.append(IRTable(
            table_id=f"t_{i:03d}",
            source_tool="azure_di",
            source_pages=pages,
            cells=cells,
            qc=qc,
        ))

    return result
