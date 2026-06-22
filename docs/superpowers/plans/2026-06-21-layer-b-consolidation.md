# Layer B 結構整合計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 將 `normalizers/`（merger.py + header_path.py）和 `formatters/`（formatter.py）三個子套件合併為單一的 `layer_b/table.py`，消除兩個空 `__init__.py` 和多層目錄，讓 Layer B 的表格處理流程一目了然。

**Architecture:** `merger.py → header_path.py → formatter.py` 是嚴格的線性 pipeline：IRTable → span 展開 → header 標記 → 文字格式化。三者永遠一起被使用、一起被修改，合在一個檔案中符合「一起變動的程式住在一起」原則。`adapters/` 子套件有四個後端，保持不動。

**Tech Stack:** Python 3.10, dataclasses (stdlib), pytest

## Global Constraints

- 純 refactor：所有對外行為完全相同，無邏輯變更
- `process_document(raw: dict) -> list[RetrievalUnit]` 簽章不變
- `adapters/` 目錄及其所有內容一字不動
- 測試命令：`conda run -n hospital-rag pytest tests/test_layer_b/ -q --ignore=tests/test_layer_b/test_integration_real_docs.py --ignore=tests/test_layer_b/test_integration_v3_real_docs.py`
- 期望結果：101 passed（與 refactor 前相同）
- **不新增任何測試**（純 refactor；101 個既有測試即為驗收標準）

---

## 修改檔案總覽

| 動作 | 路徑 | 說明 |
|------|------|------|
| Create | `layer_b/table.py` | merger + header_path + formatter 合併 |
| Modify | `layer_b/pipeline.py:6-8` | 3 行 import → 1 行 |
| Modify | `tests/test_layer_b/test_pipeline.py:4,6` | import 路徑更新 |
| Modify | `tests/test_layer_b/test_formatter.py:4-5` | import 路徑更新 |
| Modify | `tests/test_layer_b/test_header_path.py:4` | import 路徑更新 |
| Modify | `tests/test_layer_b/test_merger.py:4` | import 路徑更新 |
| Modify | `tests/test_layer_b/test_normalizer.py:4` | import 路徑更新 |
| Delete | `layer_b/normalizers/merger.py` | 內容已移入 table.py |
| Delete | `layer_b/normalizers/header_path.py` | 內容已移入 table.py |
| Delete | `layer_b/normalizers/__init__.py` | 空檔，目錄將被刪除 |
| Delete | `layer_b/formatters/formatter.py` | 內容已移入 table.py |
| Delete | `layer_b/formatters/__init__.py` | 空檔，目錄將被刪除 |

---

## Task 1：建立 `layer_b/table.py` 並更新所有 import

**Files:**
- Create: `layer_b/table.py`
- Modify: `layer_b/pipeline.py:6-8`
- Modify: `tests/test_layer_b/test_pipeline.py:4,6`
- Modify: `tests/test_layer_b/test_formatter.py:4-5`
- Modify: `tests/test_layer_b/test_header_path.py:4`
- Modify: `tests/test_layer_b/test_merger.py:4`
- Modify: `tests/test_layer_b/test_normalizer.py:4`

**Interfaces:**
- Produces: `layer_b/table.py` 匯出以下所有符號（被 pipeline.py 和各 test 使用）：
  - `expand_spans(table: IRTable) -> IRTable`
  - `merge_cross_page(tables: list[IRTable]) -> list[IRTable]`
  - `LabelledCell` dataclass
  - `LabelledTable` dataclass
  - `build_header_paths(table: IRTable) -> LabelledTable`
  - `linearize_kv(table: LabelledTable) -> str`
  - `to_json(table: LabelledTable) -> dict`
  - `to_markdown(table: LabelledTable) -> str`

- [ ] **Step 1：確認測試基線為 101 passed**

```bash
conda run -n hospital-rag pytest tests/test_layer_b/ -q \
  --ignore=tests/test_layer_b/test_integration_real_docs.py \
  --ignore=tests/test_layer_b/test_integration_v3_real_docs.py
```

期望：`101 passed`

- [ ] **Step 2：建立 `layer_b/table.py`**

完整建立以下檔案（內容是三個舊模組的合併，順序即 pipeline 執行順序）：

```python
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from layer_b.models import IRCell, IRTable


# ─── Span expansion and cross-page merging ────────────────────────────────────

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
                ))

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

    cap_a = getattr(a, "caption", "") or ""
    cap_b = getattr(b, "caption", "") or ""
    if cap_a and cap_b:
        return cap_a == cap_b

    if _has_header(a) and _has_header(b):
        return _header_key(a) == _header_key(b)

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
            continue
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


# ─── Header path labelling ────────────────────────────────────────────────────

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
    col_headers_by_col: dict[int, list[IRCell]] = {}
    row_headers_by_row: dict[int, list[IRCell]] = {}
    for cell in table.cells:
        if cell.is_col_header:
            col_headers_by_col.setdefault(cell.col_index, []).append(cell)
        if cell.is_row_header:
            row_headers_by_row.setdefault(cell.row_index, []).append(cell)

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
    )


# ─── Text formatters ─────────────────────────────────────────────────────────

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

    col_label: dict[int, str] = {}
    for cell in table.cells:
        if cell.col_header_path:
            col_label[cell.col_index] = cell.col_header_path[-1]

    for _row_idx, cells in groups.items():
        body_cells = [c for c in cells if c.col_header_path]
        if not body_cells:
            value_cells = [c for c in cells if c.row_header_path]
            if value_cells:
                pairs = [f"{c.row_header_path[-1]}: {c.content}" for c in value_cells]
                lines.append(" | ".join(pairs))
            continue

        pairs: list[str] = []

        row_header_path: list[str] = []
        rh_carrier: LabelledCell | None = None
        for c in body_cells:
            if c.row_header_path:
                row_header_path = c.row_header_path
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
                if cell is not rh_carrier
            ],
        })

    return {
        "table_id": table.table_id,
        "source_tool": table.source_tool,
        "source_pages": table.source_pages,
        "rows": rows,
    }


def to_markdown(table: LabelledTable) -> str:
    """Markdown table output for LLM reading and UI display."""
    if not table.cells:
        return ""

    groups = _row_groups(table)
    if not groups:
        return ""

    max_col = max(c.col_index for c in table.cells)
    num_cols = max_col + 1

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

    for row_idx in sorted(header_row_indices):
        lines.append(_row_to_md(groups[row_idx]))

    if header_row_indices:
        lines.append("|" + "|".join(["---"] * num_cols) + "|")

    for row_idx in sorted(body_row_indices):
        lines.append(_row_to_md(groups[row_idx]))

    return "\n".join(lines)
```

- [ ] **Step 3：更新 `layer_b/pipeline.py` 的 import（第 6-8 行）**

找到：

```python
from layer_b.normalizers.merger import merge_cross_page, expand_spans
from layer_b.normalizers.header_path import build_header_paths
from layer_b.formatters.formatter import linearize_kv, to_json, to_markdown
```

替換為：

```python
from layer_b.table import merge_cross_page, expand_spans, build_header_paths, linearize_kv, to_json, to_markdown
```

- [ ] **Step 4：更新 `tests/test_layer_b/test_pipeline.py` 的 import（第 4、6 行）**

找到：

```python
from layer_b.normalizers.header_path import build_header_paths
from layer_b.pipeline import assess
from layer_b.formatters.formatter import to_markdown
```

替換為：

```python
from layer_b.table import build_header_paths, to_markdown
from layer_b.pipeline import assess
```

- [ ] **Step 5：更新 `tests/test_layer_b/test_formatter.py` 的 import（第 4-5 行）**

找到：

```python
from layer_b.normalizers.header_path import build_header_paths
from layer_b.formatters.formatter import linearize_kv, to_json
```

替換為：

```python
from layer_b.table import build_header_paths, linearize_kv, to_json
```

- [ ] **Step 6：更新 `tests/test_layer_b/test_header_path.py` 的 import（第 4 行）**

找到：

```python
from layer_b.normalizers.header_path import build_header_paths
```

替換為：

```python
from layer_b.table import build_header_paths
```

- [ ] **Step 7：更新 `tests/test_layer_b/test_merger.py` 的 import（第 4 行）**

找到：

```python
from layer_b.normalizers.merger import merge_cross_page
```

替換為：

```python
from layer_b.table import merge_cross_page
```

- [ ] **Step 8：更新 `tests/test_layer_b/test_normalizer.py` 的 import（第 4 行）**

找到：

```python
from layer_b.normalizers.merger import expand_spans
```

替換為：

```python
from layer_b.table import expand_spans
```

- [ ] **Step 9：執行測試確認仍為 101 passed**

```bash
conda run -n hospital-rag pytest tests/test_layer_b/ -q \
  --ignore=tests/test_layer_b/test_integration_real_docs.py \
  --ignore=tests/test_layer_b/test_integration_v3_real_docs.py
```

期望：`101 passed`

- [ ] **Step 10：刪除舊目錄**

```bash
rm layer_b/normalizers/merger.py
rm layer_b/normalizers/header_path.py
rm layer_b/normalizers/__init__.py
rmdir layer_b/normalizers
rm layer_b/formatters/formatter.py
rm layer_b/formatters/__init__.py
rmdir layer_b/formatters
```

- [ ] **Step 11：再次執行測試確認刪除後仍為 101 passed**

```bash
conda run -n hospital-rag pytest tests/test_layer_b/ -q \
  --ignore=tests/test_layer_b/test_integration_real_docs.py \
  --ignore=tests/test_layer_b/test_integration_v3_real_docs.py
```

期望：`101 passed`

- [ ] **Step 12：Commit**

```bash
git add layer_b/table.py layer_b/pipeline.py
git add tests/test_layer_b/test_pipeline.py tests/test_layer_b/test_formatter.py
git add tests/test_layer_b/test_header_path.py tests/test_layer_b/test_merger.py
git add tests/test_layer_b/test_normalizer.py
git rm layer_b/normalizers/merger.py layer_b/normalizers/header_path.py layer_b/normalizers/__init__.py
git rm layer_b/formatters/formatter.py layer_b/formatters/__init__.py
git commit -m "refactor(layer_b): merge normalizers/ + formatters/ into table.py

The three modules (merger, header_path, formatter) form a strict linear
pipeline — IRTable → span expansion → header labelling → text formatting.
They always change together and are always used together.

Consolidating into table.py eliminates 2 sub-packages, 2 empty __init__.py
files, and 3 levels of import nesting. Layer B now has a flat structure:
  models.py, table.py, pipeline.py, enrichment.py, adapters/

101 tests pass; no logic changed."
```

---

## Self-Review

**Spec coverage：**
- ✅ `normalizers/merger.py` 內容完整移入 `table.py`（Step 2）
- ✅ `normalizers/header_path.py` 內容完整移入 `table.py`（Step 2）
- ✅ `formatters/formatter.py` 內容完整移入 `table.py`（Step 2）
- ✅ `pipeline.py` import 更新（Step 3）
- ✅ 所有 5 個測試檔案 import 更新（Step 4-8）
- ✅ 舊目錄刪除（Step 10）
- ✅ `adapters/` 不動

**Placeholder 掃描：** 無 TBD / TODO。

**Type consistency：**
- `LabelledCell` 和 `LabelledTable` 的定義在 `table.py` 中，`build_header_paths` 回傳 `LabelledTable`，`linearize_kv`/`to_json`/`to_markdown` 接受 `LabelledTable` — 類型一致。
- `expand_spans` 和 `merge_cross_page` 接受並回傳 `IRTable` — 與 `pipeline.py` 的使用方式一致。
- `_merge` 和 `_can_merge` 等 helper 僅在 `table.py` 內部使用 — 正確，不暴露。
