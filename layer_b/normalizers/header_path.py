from __future__ import annotations
from dataclasses import dataclass, field
from layer_b.models import IRCell, IRTable


@dataclass
class LabelledCell:
    row_index: int
    col_index: int
    content: str
    col_header_path: list[str]
    row_header_path: list[str]
    header_source: str
    confidence: float | None = None


@dataclass
class LabelledTable:
    table_id: str
    source_tool: str
    source_pages: list[int]
    cells: list[LabelledCell]
    page_image_refs: dict[str, str] = field(default_factory=dict)


def build_header_paths(table: IRTable) -> LabelledTable:
    """Assign col_header_path and row_header_path to every content cell.

    Precondition: table has been processed by expand_spans() — all cells
    have row_span=col_span=1.

    Algorithm:
    - col_header_path: for cell at (r, c), collect all cells where
      is_col_header=True and col_index==c, ordered by row_index ascending
      (outermost header first).
    - row_header_path: for cell at (r, c), collect all cells where
      is_row_header=True and row_index==r, ordered by col_index ascending
      (outermost header first).
    - Header cells themselves get empty paths (they are the path nodes).
    """
    # Build lookup: (row, col) -> IRCell
    grid: dict[tuple[int, int], IRCell] = {
        (c.row_index, c.col_index): c for c in table.cells
    }

    # Pre-index header cells by column and by row
    col_headers_by_col: dict[int, list[IRCell]] = {}
    row_headers_by_row: dict[int, list[IRCell]] = {}
    for cell in table.cells:
        if cell.is_col_header:
            col_headers_by_col.setdefault(cell.col_index, []).append(cell)
        if cell.is_row_header:
            row_headers_by_row.setdefault(cell.row_index, []).append(cell)

    # Sort once: col headers by row_index asc, row headers by col_index asc
    for col in col_headers_by_col:
        col_headers_by_col[col].sort(key=lambda c: c.row_index)
    for row in row_headers_by_row:
        row_headers_by_row[row].sort(key=lambda c: c.col_index)

    labelled: list[LabelledCell] = []
    for cell in table.cells:
        col_path = [
            h.content
            for h in col_headers_by_col.get(cell.col_index, [])
            if h.row_index < cell.row_index or cell.is_col_header
        ]
        row_path = [
            h.content
            for h in row_headers_by_row.get(cell.row_index, [])
            if h.col_index < cell.col_index or cell.is_row_header
        ]

        # Header cells that are also in the other axis get empty own-axis path
        if cell.is_col_header:
            col_path = []
        if cell.is_row_header:
            row_path = []

        labelled.append(LabelledCell(
            row_index=cell.row_index,
            col_index=cell.col_index,
            content=cell.content,
            col_header_path=col_path,
            row_header_path=row_path,
            header_source=cell.header_source,
            confidence=cell.confidence,
        ))

    labelled.sort(key=lambda c: (c.row_index, c.col_index))

    return LabelledTable(
        table_id=table.table_id,
        source_tool=table.source_tool,
        source_pages=table.source_pages,
        cells=labelled,
        page_image_refs=table.page_image_refs,
    )
