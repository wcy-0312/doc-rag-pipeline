from __future__ import annotations
from dataclasses import replace
from layer_b.models import IRCell, IRTable


def expand_spans(table: IRTable) -> IRTable:
    """Row-major linearization: expand spanning cells so every row is self-contained.

    For a cell with row_span=2, col_span=1:
      - original cell stays at (row_index, col_index)
      - a copy is inserted at (row_index+1, col_index)

    For a cell with row_span=1, col_span=2:
      - original cell stays at (row_index, col_index)
      - a copy is inserted at (row_index, col_index+1)

    Cells with row_span=1, col_span=1 are unchanged.
    The expanded cells have row_span=1, col_span=1 to mark them as virtual copies.

    Reference: arXiv:2204.03357 (ACL 2022)
    """
    expanded: list[IRCell] = []

    for cell in table.cells:
        # Emit all positions covered by this cell (including the origin)
        for dr in range(cell.row_span):
            for dc in range(cell.col_span):
                expanded.append(IRCell(
                    row_index=cell.row_index + dr,
                    col_index=cell.col_index + dc,
                    row_span=1,
                    col_span=1,
                    content=cell.content,
                    is_row_header=cell.is_row_header,
                    is_col_header=cell.is_col_header,
                    header_source=cell.header_source,
                    confidence=cell.confidence,
                    bounding_box=cell.bounding_box,
                ))

    # Sort by (row_index, col_index) for predictable downstream ordering
    expanded.sort(key=lambda c: (c.row_index, c.col_index))

    return IRTable(
        table_id=table.table_id,
        source_tool=table.source_tool,
        source_pages=table.source_pages,
        cells=expanded,
        qc=table.qc,
        page_image_refs=table.page_image_refs,
    )
