from __future__ import annotations
from collections import defaultdict
from layer_b.normalizers.header_path import LabelledCell, LabelledTable
from layer_b.models import IRDocument


def _kv_key(col_header_path: list[str]) -> str:
    return " > ".join(col_header_path) if col_header_path else ""


def _row_groups(table: LabelledTable) -> dict[int, list[LabelledCell]]:
    groups: dict[int, list[LabelledCell]] = defaultdict(list)
    for cell in table.cells:
        groups[cell.row_index].append(cell)
    for row in groups:
        groups[row].sort(key=lambda c: c.col_index)
    return dict(sorted(groups.items()))


def linearize_kv(table: LabelledTable) -> str:
    """Linearized Key-Value output (arXiv:2305.13062v4, ACL 2022).

    For each body row (has at least one cell with col_header_path):
      row_header_key: row_header_val | col_key: val | col_key: val ...

    Row header key: the col header label of the row-header column.
    Hierarchical col headers joined with ' > ': '第一線治療 > 劑量: 100mg'.
    Header-only rows are skipped.
    """
    lines: list[str] = []
    groups = _row_groups(table)

    # Derive innermost col header label per col_index from body cells
    col_label: dict[int, str] = {}
    for cell in table.cells:
        if cell.col_header_path:
            col_label[cell.col_index] = cell.col_header_path[-1]

    for _row_idx, cells in groups.items():
        body_cells = [c for c in cells if c.col_header_path]
        if not body_cells:
            # Pure row-header table: cells have row_header_path but no col_header_path
            value_cells = [c for c in cells if c.row_header_path]
            if value_cells:
                pairs = [f"{c.row_header_path[-1]}: {c.content}" for c in value_cells]
                lines.append(" | ".join(pairs))
            continue  # pure header row

        pairs: list[str] = []

        # Find row_header_path from the first body cell that has one
        row_header_path: list[str] = []
        rh_carrier: LabelledCell | None = None
        for c in body_cells:
            if c.row_header_path:
                row_header_path = c.row_header_path
                # The row-header cell is in the same row, has the rh value as content
                # It's a body cell whose content == row_header_path[-1] and col is the rh col
                # Find it: the cell whose content is the row header value
                for candidate in body_cells:
                    if candidate.content == row_header_path[-1] and not candidate.row_header_path:
                        rh_carrier = candidate
                        break
                break

        if row_header_path:
            if rh_carrier:
                rh_key = col_label.get(rh_carrier.col_index, row_header_path[0])
            else:
                rh_key = col_label.get(body_cells[0].col_index, row_header_path[0])
            pairs.append(f"{rh_key}: {row_header_path[-1]}")

        for cell in body_cells:
            # Skip the cell that IS the row header value (already emitted above)
            if rh_carrier is not None and cell is rh_carrier:
                continue
            pairs.append(f"{_kv_key(cell.col_header_path)}: {cell.content}")

        lines.append(" | ".join(pairs))

    return "\n".join(lines)


def to_json(table: LabelledTable) -> dict:
    """Structured JSON output for storage and downstream rendering."""
    groups = _row_groups(table)
    rows = []

    for _row_idx, cells in groups.items():
        body_cells = [c for c in cells if c.col_header_path]
        if not body_cells:
            value_cells = [c for c in cells if c.row_header_path]
            if value_cells:
                rows.append({
                    "row_header_path": [],
                    "cells": [
                        {
                            "col_header_path": c.row_header_path,
                            "value": c.content,
                            "confidence": c.confidence,
                        }
                        for c in value_cells
                    ],
                })
            continue

        # row_header_path and rh_carrier identification (mirrors linearize_kv logic)
        rh_path: list[str] = []
        rh_carrier: LabelledCell | None = None
        for c in body_cells:
            if c.row_header_path:
                rh_path = c.row_header_path
                for candidate in body_cells:
                    if candidate.content == rh_path[-1] and not candidate.row_header_path:
                        rh_carrier = candidate
                        break
                break

        rows.append({
            "row_header_path": rh_path,
            "cells": [
                {
                    "col_header_path": cell.col_header_path,
                    "value": cell.content,
                    "confidence": cell.confidence,
                }
                for cell in body_cells
                if cell is not rh_carrier  # exclude the row-header value cell
            ],
        })

    return {
        "table_id": table.table_id,
        "source_tool": table.source_tool,
        "source_pages": table.source_pages,
        "page_image_refs": table.page_image_refs,
        "rows": rows,
    }


def to_markdown(table: LabelledTable) -> str:
    """Markdown table output for LLM reading and UI display."""
    if not table.cells:
        return ""

    groups = _row_groups(table)
    if not groups:
        return ""

    # Determine number of columns
    max_col = max(c.col_index for c in table.cells)
    num_cols = max_col + 1

    # Header rows: all cells in the row have col_header_path == []
    # (they ARE the col header cells themselves)
    header_row_indices = set()
    body_row_indices = set()
    for row_idx, cells in groups.items():
        if all(c.col_header_path == [] for c in cells):
            header_row_indices.add(row_idx)
        if any(c.col_header_path != [] for c in cells):
            body_row_indices.add(row_idx)

    def _row_to_md(cells: list[LabelledCell]) -> str:
        col_map: dict[int, str] = {c.col_index: c.content for c in cells}
        parts = [col_map.get(i, "") for i in range(num_cols)]
        return "| " + " | ".join(parts) + " |"

    lines: list[str] = []

    # Output header rows
    for row_idx in sorted(header_row_indices):
        lines.append(_row_to_md(groups[row_idx]))

    # Separator (only when there are header rows)
    if header_row_indices:
        lines.append("|" + "|".join(["---"] * num_cols) + "|")

    # Output body rows
    for row_idx in sorted(body_row_indices):
        lines.append(_row_to_md(groups[row_idx]))

    return "\n".join(lines)


def document_to_retrieval_units(doc: IRDocument) -> list[dict]:
    """將 IRDocument 的每個 element 轉為 retrieval unit。

    每個 element 輸出一個 retrieval unit，包含：
    - retrieval_unit_id
    - source_tool
    - doc_id
    - section_id, section_title, semantic_type
    - page_no, reading_order
    - content（主要檢索文字）
    - entities（從 element 提取）
    - document_signals
    """
    units = []
    for section in doc.sections:
        for elem in section.elements:
            unit = {
                "retrieval_unit_id": f"{doc.doc_id}_{elem.get('element_id', '')}",
                "source_tool": doc.source_tool,
                "doc_id": doc.doc_id,
                "section_id": section.section_id,
                "section_title": section.title,
                "semantic_type": section.semantic_type,
                "page_no": elem.get("page_no"),
                "reading_order": elem.get("reading_order"),
                "element_type": elem.get("type", "text"),
                "content": elem.get("content") or "",
                "entities": elem.get("entities", {}),
                "document_signals": elem.get("document_signals", []),
            }
            units.append(unit)
    return units
