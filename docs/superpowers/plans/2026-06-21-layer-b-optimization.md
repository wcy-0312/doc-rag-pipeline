# Layer B 程式碼優化計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 消除 `layer_b/pipeline.py` 中三處重複計算，並修正兩處小型程式碼問題，讓資料流更明確、每份文件少執行 4 次重複函式呼叫。

**Architecture:** `process_document()` 是唯一的對外入口；`_paragraph_path`、`_figure_path`、`_high_graphics_path` 是它的 private helpers。目前這三個 helpers 各自對同一個 `raw` dict 重複呼叫 `_doc_confidence()` 和 `_build_page_context_map()`。修法是在 `process_document()` 計算一次後以參數傳入，helpers 收到預計算值時直接使用。函式對外簽章（`process_document`）不變。

**Tech Stack:** Python 3.10, dataclasses (stdlib), pytest

## Global Constraints

- 僅修改 `layer_b/pipeline.py`；不改任何其他模組或測試檔
- `process_document(raw: dict) -> list[RetrievalUnit]` 簽章不變
- 所有修改對外行為完全相同（pure refactor，無邏輯變更）
- 測試命令：`conda run -n hospital-rag pytest tests/test_layer_b/ -q --ignore=tests/test_layer_b/test_integration_real_docs.py --ignore=tests/test_layer_b/test_integration_v3_real_docs.py`（已知兩個 integration test 有無關的既有 import 錯誤，不在本 PR 範圍）

---

## 修改檔案總覽

| 動作 | 路徑 | 改動說明 |
|------|------|----------|
| Modify | `layer_b/pipeline.py` | 全部修改（兩個 task） |

**不需新增任何檔案，不需新增任何測試**（這是 pure refactor；既有 104 個測試即為驗收標準）。

---

## Task 1：提升共用狀態計算，消除三處重複呼叫

**目的：** 將 `_doc_confidence(raw)` 和 `_build_page_context_map(data, norm_tool)` 從三個 helper 函式中移除，改由 `process_document()` 計算一次後傳入。同步將 `from dataclasses import replace` 移到檔案頂部（現在它藏在 `process_document()` 的 if 分支內）。

**Files:**
- Modify: `layer_b/pipeline.py:1-10`（頂部 import）
- Modify: `layer_b/pipeline.py:470-586`（`_paragraph_path`）
- Modify: `layer_b/pipeline.py:670-739`（`_figure_path`）
- Modify: `layer_b/pipeline.py:742-812`（`_high_graphics_path`）
- Modify: `layer_b/pipeline.py:847-876`（`process_document`）

**Interfaces:**
- Produces: 以下三個 private 函式新增選用參數，`process_document` 簽章不變
  - `_paragraph_path(raw, source_tool, doc_prefix, doc_metadata, confidence: tuple | None = None)`
  - `_figure_path(raw, source_tool, doc_prefix, doc_metadata, confidence: tuple | None = None, page_context_map: dict | None = None)`
  - `_high_graphics_path(raw, doc_prefix, norm_tool, doc_metadata, confidence: tuple | None = None, page_context_map: dict | None = None, covered_pages: set[int] | None = None)`

- [ ] **Step 1：確認測試基線為綠**

```bash
conda run -n hospital-rag pytest tests/test_layer_b/ -q \
  --ignore=tests/test_layer_b/test_integration_real_docs.py \
  --ignore=tests/test_layer_b/test_integration_v3_real_docs.py
```

期望輸出：`104 passed`

- [ ] **Step 2：在檔案頂部加入 `from dataclasses import replace`**

找到 `layer_b/pipeline.py` 的第 1-4 行：

```python
from __future__ import annotations
import re
from pathlib import Path as _Path
from layer_b.adapters import adapt, get_source_tool
```

修改為：

```python
from __future__ import annotations
import re
from dataclasses import replace
from pathlib import Path as _Path
from layer_b.adapters import adapt, get_source_tool
```

- [ ] **Step 3：修改 `_paragraph_path` 接受預計算的 `confidence`**

找到（pipeline.py:470）：

```python
def _paragraph_path(raw: dict, source_tool: str, doc_prefix: str = "", doc_metadata: dict | None = None) -> list[RetrievalUnit]:
    data = raw["data"]
    styles = data.get("styles", [])
    source_tool = _normalize_source_tool(source_tool)
```

替換為：

```python
def _paragraph_path(raw: dict, source_tool: str, doc_prefix: str = "", doc_metadata: dict | None = None, confidence: tuple | None = None) -> list[RetrievalUnit]:
    data = raw["data"]
    styles = data.get("styles", [])
    source_tool = _normalize_source_tool(source_tool)
```

找到（pipeline.py:513，在 `if not candidates: return []` 之後）：

```python
    confidence_level, quality_flag, retrieval_weight = _doc_confidence(raw)
```

替換為：

```python
    confidence_level, quality_flag, retrieval_weight = confidence if confidence is not None else _doc_confidence(raw)
```

- [ ] **Step 4：修改 `_figure_path` 接受預計算的 `confidence` 和 `page_context_map`**

找到（pipeline.py:670-681）：

```python
def _figure_path(raw: dict, source_tool: str, doc_prefix: str, doc_metadata: dict | None = None) -> list[RetrievalUnit]:
    """Generate RetrievalUnit for each meaningful figure.

    Meaningful = area >= 0.5 page units (filters tiny decorative icons).
    embedding_text uses: caption > linked paragraph elements > page context fallback.
    """
    data = raw.get("data", {})
    figures = data.get("figures", [])
    paragraphs = data.get("paragraphs", [])
    confidence_level, quality_flag, retrieval_weight = _doc_confidence(raw)
    norm_tool = _normalize_source_tool(source_tool)
    page_context_map = _build_page_context_map(data, norm_tool)
```

替換為：

```python
def _figure_path(raw: dict, source_tool: str, doc_prefix: str, doc_metadata: dict | None = None, confidence: tuple | None = None, page_context_map: dict | None = None) -> list[RetrievalUnit]:
    """Generate RetrievalUnit for each meaningful figure.

    Meaningful = area >= 0.5 page units (filters tiny decorative icons).
    embedding_text uses: caption > linked paragraph elements > page context fallback.
    """
    data = raw.get("data", {})
    figures = data.get("figures", [])
    paragraphs = data.get("paragraphs", [])
    confidence_level, quality_flag, retrieval_weight = confidence if confidence is not None else _doc_confidence(raw)
    norm_tool = _normalize_source_tool(source_tool)
    page_context_map = page_context_map if page_context_map is not None else _build_page_context_map(data, norm_tool)
```

- [ ] **Step 5：修改 `_high_graphics_path` 接受預計算的 `confidence` 和 `page_context_map`**

找到（pipeline.py:742-765）：

```python
def _high_graphics_path(
    raw: dict,
    doc_prefix: str,
    norm_tool: str,
    doc_metadata: dict | None = None,
    covered_pages: set[int] | None = None,
) -> list[RetrievalUnit]:
    """為有 page_image 但無任何其他 RetrievalUnit 的頁面生成文件級 RetrievalUnit。

    目的：讓癌症診療指引的流程圖/決策樹頁面可被 RAG 查詢命中。
    covered_pages: 已由 table/paragraph/figure path 生成 unit 的頁碼集合，這些頁面不重複生成。
    """
    data = raw.get("data", {})
    page_images = data.get("page_images", {})

    if not page_images:
        return []

    # 已被前面三個 path 處理過的頁面，直接跳過
    pages_with_figures: set[int] = covered_pages or set()

    confidence_level, quality_flag, retrieval_weight = _doc_confidence(raw)

    page_context_map = _build_page_context_map(data, norm_tool)
```

替換為：

```python
def _high_graphics_path(
    raw: dict,
    doc_prefix: str,
    norm_tool: str,
    doc_metadata: dict | None = None,
    confidence: tuple | None = None,
    page_context_map: dict | None = None,
    covered_pages: set[int] | None = None,
) -> list[RetrievalUnit]:
    """為有 page_image 但無任何其他 RetrievalUnit 的頁面生成文件級 RetrievalUnit。

    目的：讓癌症診療指引的流程圖/決策樹頁面可被 RAG 查詢命中。
    covered_pages: 已由 table/paragraph/figure path 生成 unit 的頁碼集合，這些頁面不重複生成。
    """
    data = raw.get("data", {})
    page_images = data.get("page_images", {})

    if not page_images:
        return []

    # 已被前面三個 path 處理過的頁面，直接跳過
    pages_with_figures: set[int] = covered_pages or set()

    confidence_level, quality_flag, retrieval_weight = confidence if confidence is not None else _doc_confidence(raw)

    page_context_map = page_context_map if page_context_map is not None else _build_page_context_map(data, norm_tool)
```

- [ ] **Step 6：更新 `process_document()` — 計算一次，傳給三個 helpers**

找到（pipeline.py:847-876）：

```python
def process_document(raw: dict) -> list[RetrievalUnit]:
    """Unified entry point: Conversion Layer JSON → list[RetrievalUnit].

    Routes by extractor_metadata.tool:
      - vision_llm → document path (IRDocument → element-per-unit)
      - others     → table path + paragraph path
    """
    source_tool = get_source_tool(raw)
    if source_tool == "vision_llm":
        return _document_path(raw)
    doc_prefix = _doc_prefix(raw)
    doc_metadata = _build_doc_metadata(raw)
    units = []
    units.extend(_table_path(raw, source_tool, doc_prefix, doc_metadata))
    units.extend(_paragraph_path(raw, source_tool, doc_prefix, doc_metadata))
    norm_tool = _normalize_source_tool(source_tool)
    units.extend(_figure_path(raw, source_tool, doc_prefix, doc_metadata))
    _covered = {p for u in units for p in (u.source_pages or [])}
    units.extend(_high_graphics_path(raw, doc_prefix, norm_tool, doc_metadata, covered_pages=_covered))
    vision_desc = raw.get("vision_description", "")
    if vision_desc:
        from dataclasses import replace
        units = [
            replace(u, embedding_text=vision_desc + "\n" + u.embedding_text)
            if u.embedding_text
            else replace(u, embedding_text=vision_desc)
            for u in units
        ]
    return units
```

替換為：

```python
def process_document(raw: dict) -> list[RetrievalUnit]:
    """Unified entry point: Conversion Layer JSON → list[RetrievalUnit].

    Routes by extractor_metadata.tool:
      - vision_llm → document path (IRDocument → element-per-unit)
      - others     → table path + paragraph path
    """
    source_tool = get_source_tool(raw)
    if source_tool == "vision_llm":
        return _document_path(raw)
    doc_prefix = _doc_prefix(raw)
    doc_metadata = _build_doc_metadata(raw)
    norm_tool = _normalize_source_tool(source_tool)
    confidence = _doc_confidence(raw)
    page_context_map = _build_page_context_map(raw.get("data", {}), norm_tool)
    units = []
    units.extend(_table_path(raw, source_tool, doc_prefix, doc_metadata))
    units.extend(_paragraph_path(raw, source_tool, doc_prefix, doc_metadata, confidence))
    units.extend(_figure_path(raw, source_tool, doc_prefix, doc_metadata, confidence, page_context_map))
    _covered = {p for u in units for p in (u.source_pages or [])}
    units.extend(_high_graphics_path(raw, doc_prefix, norm_tool, doc_metadata, confidence, page_context_map, covered_pages=_covered))
    vision_desc = raw.get("vision_description", "")
    if vision_desc:
        units = [
            replace(u, embedding_text=vision_desc + "\n" + u.embedding_text)
            if u.embedding_text
            else replace(u, embedding_text=vision_desc)
            for u in units
        ]
    return units
```

- [ ] **Step 7：執行測試確認仍為綠**

```bash
conda run -n hospital-rag pytest tests/test_layer_b/ -q \
  --ignore=tests/test_layer_b/test_integration_real_docs.py \
  --ignore=tests/test_layer_b/test_integration_v3_real_docs.py
```

期望輸出：`104 passed`

- [ ] **Step 8：Commit**

```bash
git add layer_b/pipeline.py
git commit -m "refactor(layer_b): hoist _doc_confidence and _build_page_context_map to process_document

Eliminates 4 redundant function calls per document:
- _doc_confidence(raw) was called in _paragraph_path, _figure_path, _high_graphics_path
- _build_page_context_map() was called in both _figure_path and _high_graphics_path

Now computed once in process_document() and passed as optional params.
Also moves 'from dataclasses import replace' to top-level import."
```

---

## Task 2：修正 `_row_to_text` 雙重呼叫

**目的：** `_table_path` 中 `[_row_to_text(r) for r in rows if _row_to_text(r)]` 對每個不為空的 row 呼叫 `_row_to_text` 兩次。用 walrus operator 消除重複。

**Files:**
- Modify: `layer_b/pipeline.py:615`

**Interfaces:**
- 無 signature 變更，純行為等價替換

- [ ] **Step 1：修改 `_table_path` 中的 list comprehension**

找到（pipeline.py:615）：

```python
        row_texts = [_row_to_text(r) for r in json_out.get("rows", []) if _row_to_text(r)]
```

替換為：

```python
        row_texts = [t for r in json_out.get("rows", []) if (t := _row_to_text(r))]
```

- [ ] **Step 2：執行測試確認仍為綠**

```bash
conda run -n hospital-rag pytest tests/test_layer_b/ -q \
  --ignore=tests/test_layer_b/test_integration_real_docs.py \
  --ignore=tests/test_layer_b/test_integration_v3_real_docs.py
```

期望輸出：`104 passed`

- [ ] **Step 3：Commit**

```bash
git add layer_b/pipeline.py
git commit -m "refactor(layer_b): fix double _row_to_text call with walrus operator"
```

---

## Self-Review 檢查結果

**Spec coverage（對照 code-optimization-review 決策表）：**
- ✅ #1 `_doc_confidence` 3× → 1×：Task 1 Step 3, 4, 5, 6
- ✅ #2 `_build_page_context_map` 2× → 1×：Task 1 Step 4, 5, 6
- ✅ #3 API 不一致（`_figure_path` vs `_high_graphics_path`）：Task 1 Step 6（`_figure_path` 現在和 `_high_graphics_path` 一樣，都從 caller 接 `page_context_map`）
- ✅ #4 `_row_to_text` 雙重呼叫：Task 2
- ✅ #5 `from dataclasses import replace` 移到頂部：Task 1 Step 2
- ❌ #6 `_quality()` 保留（tests 依賴）：正確不做

**Placeholder 掃描：** 無 TBD / TODO / 模糊描述。

**Type consistency：**
- `confidence` 型別：`tuple | None`，解包為 `(confidence_level, quality_flag, retrieval_weight)`，與 `_doc_confidence()` 回傳的 `tuple[str, str, float]` 相容。
- `page_context_map` 型別：`dict | None`，使用 `is not None` 判斷（避免空 dict `{}` 被誤判為 falsy）。
