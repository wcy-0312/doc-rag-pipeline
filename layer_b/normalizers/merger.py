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
    )


def _col_count(table: IRTable) -> int:
    if not table.cells:
        return 0
    return max(c.col_index + c.col_span for c in table.cells)


def _header_cells(table: IRTable) -> list[IRCell]:
    return [c for c in table.cells if c.is_col_header]


def _has_header(table: IRTable) -> bool:
    return bool(_header_cells(table))


def _header_key(table: IRTable) -> tuple[str, ...]:
    """Sorted by (row_index, col_index) for stable comparison."""
    return tuple(
        c.content
        for c in sorted(_header_cells(table), key=lambda c: (c.row_index, c.col_index))
    )


def _pages_consecutive(a: IRTable, b: IRTable) -> bool:
    if not a.source_pages or not b.source_pages:
        return False
    return max(a.source_pages) + 1 == min(b.source_pages)


def _can_merge(a: IRTable, b: IRTable) -> bool:
    if not _pages_consecutive(a, b):
        return False
    if _col_count(a) != _col_count(b):
        return False

    # caption match (when both non-empty)
    cap_a = getattr(a, "caption", "") or ""
    cap_b = getattr(b, "caption", "") or ""
    if cap_a and cap_b:
        return cap_a == cap_b

    # header exact match
    if _has_header(a) and _has_header(b):
        return _header_key(a) == _header_key(b)

    # second table has no header — treat as continuation
    if _has_header(a) and not _has_header(b):
        return True

    return False


def _merge(a: IRTable, b: IRTable) -> IRTable:
    """Merge b into a, discarding b's header rows and reindexing b's body."""
    header_row_count = (
        max((c.row_index for c in a.cells if c.is_col_header), default=-1) + 1
    )
    body_row_count = (
        max((c.row_index for c in a.cells if not c.is_col_header), default=header_row_count - 1)
        - header_row_count
        + 1
    )
    a_body_rows = body_row_count if body_row_count > 0 else 0
    a_row_offset = header_row_count + a_body_rows

    b_body_cells: list[IRCell] = []
    for c in b.cells:
        if c.is_col_header:
            continue  # discard repeated header
        new_row = a_row_offset + (c.row_index - (
            max((x.row_index for x in b.cells if x.is_col_header), default=-1) + 1
        ))
        b_body_cells.append(IRCell(
            row_index=new_row,
            col_index=c.col_index,
            row_span=c.row_span,
            col_span=c.col_span,
            content=c.content,
            is_row_header=c.is_row_header,
            is_col_header=False,
            header_source=c.header_source,
            confidence=c.confidence,
            bounding_box=c.bounding_box,
        ))

    merged_pages = sorted(set(a.source_pages + b.source_pages))

    return IRTable(
        table_id=a.table_id,
        source_tool=a.source_tool,
        source_pages=merged_pages,
        cells=a.cells + b_body_cells,
        qc=a.qc,
    )


def merge_cross_page(tables: list[IRTable]) -> list[IRTable]:
    """Merge consecutive cross-page tables in a list of IRTable.

    Tables must be ordered by page (as produced by Input Adapter).
    Returns a new list; input is not mutated.
    """
    if not tables:
        return []

    result: list[IRTable] = [tables[0]]
    for current in tables[1:]:
        if _can_merge(result[-1], current):
            result[-1] = _merge(result[-1], current)
        else:
            result.append(current)

    return result
