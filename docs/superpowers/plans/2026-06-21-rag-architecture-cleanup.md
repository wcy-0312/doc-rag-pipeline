# RAG Architecture Cleanup — Original PDF Always Accessible

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the dead `page_image_refs` pattern across all layers, fix Azure CU figure extraction (currently producing 0 figures due to a broken filter), add a document registry for pdf_path lookup, and add opt-in LLM semantic enrichment for tables and figures at ingestion time.

**Architecture:** `page_image_refs` was designed for pre-rendered page image storage but is always empty for Azure CU (Layer A never saves `page_images`); since the original PDF is always accessible, on-demand rendering via `get_full_page_image()` replaces it. The document registry (`layer_d/document_registry.py`) stores `doc_id → pdf_path` so Layer E can resolve paths without per-chunk duplication. Semantic enrichment (`layer_b/enrichment.py`) is an optional post-processing step that prepends LLM-generated clinical summaries to `embedding_text`, improving retrieval for clinically notated queries (e.g. `cT2N1M0`).

**Tech Stack:** Python 3.11, pytest, dataclasses, json (stdlib), existing `layer_e.llm_client.LLMClient` interface

## Global Constraints

- Run all tests with: `conda run -n hospital-rag pytest tests/ -x -q` — must pass before each commit
- Python 3.11, no new third-party dependencies
- Follow existing dataclass patterns in `layer_b/models.py` and `layer_d/models.py`
- YAGNI: do not add visual (image-based) figure enrichment — text-only for now
- `process_document()` signature in `layer_b/pipeline.py` must remain unchanged
- All new public functions must have a docstring

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Modify | `layer_b/models.py` | Remove `page_image_refs` from `IRTable`, `RetrievalUnit` |
| Modify | `layer_b/adapters/azure_cu_adapter.py` | Remove `_resolve_image_path`, `page_images`, `page_image_refs` from `IRTable()` |
| Modify | `layer_b/pipeline.py` | Remove `_resolve_page_image()` + all `page_image_refs` construction; fix `_figure_path` |
| Modify | `layer_d/models.py` | Remove `page_image_refs` from `RankedResult` |
| Modify | `layer_d/ingestion.py` | Remove `page_image_refs` from Qdrant payload (appears twice) |
| Modify | `layer_d/retrieval.py` | Remove `page_image_refs` from two `RankedResult()` constructors |
| Modify | `layer_e/context_packer.py` | Remove `collect_image_paths()`, remove `page_image_refs` from evidence_map |
| Modify | `layer_e/pipeline.py` | Remove dead multimodal branch (lines 38–43) |
| Create | `layer_d/document_registry.py` | Persistent `doc_id → pdf_path` mapping |
| Modify | `pipeline/runner.py` | Accept `registry_path`, `pdf_path`, `doc_id`, `enrich_llm` in `ingest()` |
| Create | `layer_b/enrichment.py` | LLM-based semantic enrichment for tables and figures |
| Modify | `tests/test_layer_b/test_adapters.py` | Remove 3 `page_image_refs` tests; fix line 175 assertion |
| Modify | `tests/test_layer_b/test_confidence.py` | Remove `test_fallback_available`, `test_no_fallback`; fix `_table()` helper |
| Modify | `tests/test_layer_b/test_pipeline.py` | Fix `_labelled()` helper; add figure extraction test |
| Modify | `tests/test_layer_d/test_retrieval.py` | Remove `TestPageImageRefsInRankedResult` class |
| Modify | `tests/test_layer_e/test_context_packer.py` | Remove 5 `page_image_refs` / `collect_image_paths` tests |
| Modify | `tests/test_layer_e/test_pipeline.py` | Remove multimodal branch tests |
| Create | `tests/test_layer_d/test_document_registry.py` | Tests for `DocumentRegistry` |
| Create | `tests/test_layer_b/test_enrichment.py` | Tests for `enrich_units()` |

---

### Task 1: Remove `page_image_refs` from Layer B

**Files:**
- Modify: `layer_b/models.py`
- Modify: `layer_b/adapters/azure_cu_adapter.py`
- Modify: `layer_b/pipeline.py`
- Test: `tests/test_layer_b/test_adapters.py`
- Test: `tests/test_layer_b/test_confidence.py`
- Test: `tests/test_layer_b/test_pipeline.py`

**Interfaces:**
- Produces: `IRTable` without `page_image_refs`; `RetrievalUnit` without `page_image_refs`
- Consumes: nothing from earlier tasks

- [ ] **Step 1: Write the failing tests**

In `tests/test_layer_b/test_pipeline.py`, the `_labelled()` helper passes `page_image_refs` to `IRTable`. After removing the field, this will fail at construction. Confirm it fails first:

```bash
conda run -n hospital-rag pytest tests/test_layer_b/ -x -q
```

Expected: tests pass (field still exists). Good — we'll confirm failure after removing the field.

- [ ] **Step 2: Remove `page_image_refs` from `layer_b/models.py`**

In `layer_b/models.py`, make these two changes:

**`IRTable`** — remove the last field:
```python
@dataclass
class IRTable:
    table_id: str
    source_tool: str
    source_pages: list[int]
    cells: list[IRCell]
    qc: QC = field(default_factory=QC)
    # REMOVED: page_image_refs: dict[str, str] = field(default_factory=dict)
```

**`RetrievalUnit`** — remove the `page_image_refs` line (the one between `source_pages` and `row_texts`):
```python
    source_pages: list[int] = field(default_factory=list)
    # REMOVED: page_image_refs: dict = field(default_factory=dict)
    row_texts: list[str] = field(default_factory=list)
```

- [ ] **Step 3: Fix `layer_b/adapters/azure_cu_adapter.py`**

Remove the `_resolve_image_path()` function (lines 14–28). Remove `page_images` from `adapt()`. The `IRTable()` constructor no longer takes `page_image_refs`:

```python
def adapt(raw: dict) -> list[IRTable]:
    """Convert Azure CU native output to list of IRTable."""
    metadata = raw.get("metadata", {})
    # REMOVED: page_images: dict[str, str] = raw.get("data", {}).get("page_images", {})
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
            # REMOVED: page_image_refs=...
        ))

    return result
```

- [ ] **Step 4: Fix `layer_b/pipeline.py`**

Four changes:

**4a** — Remove `_resolve_page_image()` function (lines 452–461 in the original). Delete the entire function.

**4b** — In `assess()`, remove `fallback_available` and `page_image_refs` from the returned dict:
```python
    return {
        "level": level,
        "score": score,
        "reasons": reasons,
        # REMOVED: "fallback_available": bool(table.page_image_refs),
        # REMOVED: "page_image_refs": table.page_image_refs,
    }
```

**4c** — In `_table_path()`, remove `page_image_refs` from `RetrievalUnit()`:
```python
        units.append(RetrievalUnit(
            retrieval_unit_id=table_id,
            source_tool=t.source_tool,
            embedding_text=embedding_text,
            structured_json=json_out,
            display_markdown=markdown_out,
            confidence_level=level,
            quality_flag=flag,
            retrieval_weight=weight,
            source_pages=t.source_pages,
            # REMOVED: page_image_refs=conf["page_image_refs"],
            row_texts=row_texts,
            doc_metadata=doc_metadata or {},
        ))
```

**4d** — In `_paragraph_path()`, remove `page_images = data.get("page_images", {})` and all `img_path` assignments and `page_image_refs={...}` from the two `RetrievalUnit()` constructors (short-doc path and per-paragraph path). Also remove from `_figure_path()` and `_high_graphics_path()`.

The short-doc `RetrievalUnit()` at line ~526:
```python
        return [RetrievalUnit(
            retrieval_unit_id=short_doc_id,
            source_tool=source_tool,
            embedding_text=markdown_content,
            structured_json={...},
            display_markdown=markdown_content,
            row_texts=[],
            confidence_level=confidence_level,
            quality_flag=quality_flag,
            retrieval_weight=retrieval_weight,
            source_pages=source_pages,
            # REMOVED: page_image_refs={...},
            doc_metadata=doc_metadata or {},
        )]
```

The per-paragraph `RetrievalUnit()` at line ~565:
```python
        units.append(RetrievalUnit(
            retrieval_unit_id=para_id,
            source_tool=source_tool,
            embedding_text=embedding_text,
            structured_json={...},
            display_markdown=display_markdown,
            row_texts=[],
            confidence_level=confidence_level,
            quality_flag=quality_flag,
            retrieval_weight=retrieval_weight,
            source_pages=[page] if page is not None else [],
            # REMOVED: page_image_refs={str(page): img_path} if img_path else {},
            doc_metadata=doc_metadata or {},
        ))
```

In `_figure_path()`, remove `img_path = _resolve_page_image(page_images, page)`, `page_image_refs = {str(page): img_path} if img_path else {}`, and the `page_image_refs=page_image_refs,` from `RetrievalUnit()`.

In `_high_graphics_path()`, remove `page_image_refs={str(page): img_path} if img_path else {},` from `RetrievalUnit()`. Also remove the `img_path` variables (both the dict and string branches) since they're only used for `page_image_refs`.

- [ ] **Step 5: Update tests in `test_adapters.py`**

Remove the three functions (and the `if __name__ == "__main__"` calls to them):
- `test_azure_cu_page_image_refs_dict_structure` (around line 242)
- `test_docling_page_image_refs_dict_structure` (around line 256)
- `test_azure_di_page_image_refs_dict_structure` (around line 270)

Remove line 175: `assert tables[0].page_image_refs == {}`

- [ ] **Step 6: Update tests in `test_confidence.py`**

In the `_table()` helper, remove `page_images` parameter and `page_image_refs=page_images` from the `IRTable()` call:
```python
def _table(
    cells,
    estimated_info_loss_rate=None,
    word_avg=None,
    low_confidence_rate=None,
    empty_cell_rate=0.0,
    warnings=None,
    # REMOVED: page_images=None,
):
    return IRTable(
        table_id="t_test",
        source_tool="azure_cu",
        source_pages=[1],
        cells=cells,
        qc=QC(
            empty_cell_rate=empty_cell_rate,
            qc_level="ok",
            warnings=warnings or [],
            word_avg=word_avg,
            low_confidence_rate=low_confidence_rate,
            estimated_info_loss_rate=estimated_info_loss_rate,
        ),
        # REMOVED: page_image_refs=page_images,
    )
```

Remove `test_fallback_available` and `test_no_fallback` functions entirely.

- [ ] **Step 7: Update tests in `test_pipeline.py`**

Fix `_labelled()` — remove the `page_image_refs` argument from `IRTable()`:
```python
def _labelled(cells, table_id="t_001"):
    t = IRTable(table_id, "azure_cu", [1], cells, QC())  # removed {"1": "img/p1.png"}
    return build_header_paths(t)
```

Remove any `page_image_refs={}` references in RetrievalUnit construction (line 190).

- [ ] **Step 8: Run tests and verify**

```bash
conda run -n hospital-rag pytest tests/test_layer_b/ -x -q
```

Expected: All pass. No `page_image_refs` references remain in Layer B.

- [ ] **Step 9: Commit**

```bash
git add layer_b/models.py layer_b/adapters/azure_cu_adapter.py layer_b/pipeline.py
git add tests/test_layer_b/test_adapters.py tests/test_layer_b/test_confidence.py tests/test_layer_b/test_pipeline.py
git commit -m "refactor(layer_b): remove page_image_refs — on-demand PDF rendering replaces pre-stored images"
```

---

### Task 2: Fix Azure CU `_figure_path`

**Files:**
- Modify: `layer_b/pipeline.py`
- Test: `tests/test_layer_b/test_pipeline.py`

**Interfaces:**
- Consumes: `IRTable`/`RetrievalUnit` without `page_image_refs` (from Task 1)
- Produces: `_figure_path()` correctly returns `RetrievalUnit` for Azure CU figures; previously returned `[]` because `has_image`/`area_sqin` filter always failed

**Context:** Azure CU figure objects have `source`, `span`, `elements`, `id` — NOT `has_image` or `area_sqin`. The current filter `if not fig.get("has_image") or fig.get("area_sqin", 0.0) < 0.5` always evaluates to `True` (skip), so 0 figures are produced. The fix: parse area from the `D(page, x1,y1,x2,y2,x3,y3,x4,y4)` bounding box in `source`, and extract text from `elements` (linked paragraph indices).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_layer_b/test_pipeline.py`:

```python
def _azure_cu_raw_with_figure() -> dict:
    """Minimal Azure CU payload with one meaningful figure (flowchart-like)."""
    return {
        "extractor_metadata": {"tool": "azure_content_understanding"},
        "metadata": {"qc": {"empty_cell_rate": 0.0, "qc_level": "ok", "warnings": []}},
        "data": {
            "tables": [],
            "figures": [
                {
                    "id": "3.1",
                    # D(page, x1,y1, x2,y2, x3,y3, x4,y4) — page 3, ~3×2 inch area
                    "source": "D(3,1.0,1.0,4.0,1.0,4.0,3.0,1.0,3.0)",
                    "elements": ["/paragraphs/0", "/paragraphs/1"],
                },
                {
                    "id": "3.2",
                    # tiny icon — area < 0.5
                    "source": "D(3,0.1,0.1,0.3,0.1,0.3,0.2,0.1,0.2)",
                    "elements": [],
                },
            ],
            "paragraphs": [
                {"content": "cT2N1M0 治療流程", "source": "D(3,1.0,1.0,4.0,1.5)"},
                {"content": "新輔助化療建議", "source": "D(3,1.0,1.5,4.0,2.0)"},
            ],
            "page_images": {},
        },
    }


def test_figure_path_azure_cu():
    """_figure_path extracts figures for Azure CU using source coords + elements text."""
    raw = _azure_cu_raw_with_figure()
    units = process_document(raw)

    fig_units = [u for u in units if u.structured_json.get("type") == "figure"]
    assert len(fig_units) == 1, f"Expected 1 figure (tiny icon filtered), got {len(fig_units)}"

    f = fig_units[0]
    assert f.source_pages == [3]
    assert "cT2N1M0 治療流程" in f.embedding_text
    assert "新輔助化療建議" in f.embedding_text
    assert f.structured_json["area"] > 0.5
```

Run:
```bash
conda run -n hospital-rag pytest tests/test_layer_b/test_pipeline.py::test_figure_path_azure_cu -v
```

Expected: FAIL — `AssertionError: Expected 1 figure, got 0`

- [ ] **Step 2: Add `_parse_figure_area()` helper and fix `_figure_path()` in `layer_b/pipeline.py`**

Add the helper near the top of `pipeline.py`, after the existing `_SOURCE_PAGE_RE`:

```python
_FIGURE_COORDS_RE = re.compile(r'^D\((\d+),(.+)\)$')


def _parse_figure_area(source_str: str) -> tuple[int | None, float]:
    """Parse Azure CU figure source string D(page, x1,y1,...) → (page, area).

    Returns (None, 0.0) if source_str is missing or malformed.
    Area is width × height in page units (inches for Azure CU).
    """
    m = _FIGURE_COORDS_RE.match(source_str or "")
    if not m:
        return None, 0.0
    page = int(m.group(1))
    try:
        coords = [float(x) for x in m.group(2).split(",")]
    except ValueError:
        return page, 0.0
    if len(coords) >= 8:
        xs = coords[0::2]
        ys = coords[1::2]
        return page, (max(xs) - min(xs)) * (max(ys) - min(ys))
    return page, 0.0
```

Replace the entire `_figure_path()` function body with:

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

    units = []
    seq = 0
    for fig in figures:
        page, area = _parse_figure_area(fig.get("source", ""))
        if page is None or area < 0.5:
            continue

        cap_raw = fig.get("caption") or {}
        caption_text = (cap_raw.get("content", "") if isinstance(cap_raw, dict) else "").strip()

        elem_texts = []
        for elem_ref in fig.get("elements", []):
            try:
                idx = int(elem_ref.split("/")[-1])
                content = paragraphs[idx].get("content", "").strip()
                if len(content) >= 3:
                    elem_texts.append(content)
            except (IndexError, ValueError, KeyError):
                pass
        elem_text = " ".join(elem_texts[:15])[:400]

        if caption_text:
            embedding_text = caption_text
        elif elem_text:
            embedding_text = elem_text
        elif page_context_map.get(page):
            embedding_text = f"[圖表 第{page}頁] {page_context_map[page]}"
        else:
            embedding_text = f"[圖表 第{page}頁]"

        display = f"[圖表 第{page}頁]"
        if caption_text:
            display += f"\n\n{caption_text}"
        elif elem_text:
            display += f"\n\n{elem_text[:200]}"

        unit_id = f"{doc_prefix}_f_{seq:03d}" if doc_prefix else f"f_{seq:03d}"
        units.append(RetrievalUnit(
            retrieval_unit_id=unit_id,
            source_tool=norm_tool,
            embedding_text=embedding_text,
            structured_json={
                "type": "figure",
                "page": page,
                "caption": caption_text,
                "area": round(area, 4),
            },
            display_markdown=display,
            confidence_level=confidence_level,
            quality_flag=quality_flag,
            retrieval_weight=retrieval_weight,
            source_pages=[page],
            row_texts=[],
            doc_metadata=doc_metadata or {},
        ))
        seq += 1
    return units
```

- [ ] **Step 3: Run the test**

```bash
conda run -n hospital-rag pytest tests/test_layer_b/test_pipeline.py::test_figure_path_azure_cu -v
```

Expected: PASS

- [ ] **Step 4: Run full Layer B tests**

```bash
conda run -n hospital-rag pytest tests/test_layer_b/ -x -q
```

Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add layer_b/pipeline.py tests/test_layer_b/test_pipeline.py
git commit -m "fix(layer_b): restore Azure CU figure extraction using source coords + elements text"
```

---

### Task 3: Remove `page_image_refs` from Layer D + E

**Files:**
- Modify: `layer_d/models.py`
- Modify: `layer_d/ingestion.py`
- Modify: `layer_d/retrieval.py`
- Modify: `layer_e/context_packer.py`
- Modify: `layer_e/pipeline.py`
- Test: `tests/test_layer_d/test_retrieval.py`
- Test: `tests/test_layer_e/test_context_packer.py`
- Test: `tests/test_layer_e/test_pipeline.py`

**Interfaces:**
- Consumes: `RetrievalUnit` without `page_image_refs` (Task 1)
- Produces: `RankedResult` without `page_image_refs`; `evidence_map` entries without `page_image_refs`; `generate()` never calls `generate_multimodal()`

**Context:** `context_packer.collect_image_paths()` was always returning `[]` in practice (page_image_refs always empty for Azure CU), so the `generate_multimodal` branch in `pipeline.py` was dead code. Remove it all cleanly.

- [ ] **Step 1: Write the failing tests**

These tests currently pass but will fail after we make changes. Confirm they exist:
```bash
conda run -n hospital-rag pytest tests/test_layer_d/test_retrieval.py::TestPageImageRefsInRankedResult tests/test_layer_e/test_context_packer.py::test_evidence_map_contains_page_image_refs -v
```

Expected: PASS (they test functionality we're about to remove).

- [ ] **Step 2: Update `layer_d/models.py`**

Remove `page_image_refs` from `RankedResult`:

```python
@dataclass
class RankedResult:
    chunk_id: str
    chunk_type: str
    parent_chunk_id: Optional[str]
    retrieval_unit_id: str
    final_score: float
    rrf_score: float
    retrieval_weight: float
    display_markdown: str
    metadata: dict
    source_tool: str
    source_pages: List[int]
    embedding_text: str = ""
    rerank_score: float = 0.0
    # REMOVED: page_image_refs: dict = field(default_factory=dict)
```

- [ ] **Step 3: Update `layer_d/ingestion.py`**

Remove both occurrences of the `"page_image_refs"` line from the payload dict (around line 119). The dict has it listed twice (a duplication bug) — remove both:

```python
                {
                    "retrieval_unit_id": chunk.retrieval_unit_id,
                    "source_tool":       chunk.metadata.get("source_tool"),
                    "confidence_level":  chunk.metadata.get("confidence_level"),
                    "quality_flag":      chunk.metadata.get("quality_flag"),
                    "retrieval_weight":  chunk.metadata.get("retrieval_weight", 1.0),
                    "source_pages":      chunk.metadata.get("source_pages", []),
                    "has_handwriting":   chunk.metadata.get("has_handwriting", False),
                    "embedding_text":    chunk.embedding_text,
                    "display_markdown":  chunk.display_markdown,
                    # REMOVED: "page_image_refs": chunk.metadata.get("page_image_refs", {}),
                    "patient_id":        chunk.metadata.get("patient_id"),
                    "document_type":     chunk.metadata.get("document_type"),
                }
```

- [ ] **Step 4: Update `layer_d/retrieval.py`**

Remove `page_image_refs=payload.get("page_image_refs", {}),` from both `RankedResult()` constructors (around lines 71 and 258).

- [ ] **Step 5: Update `layer_e/context_packer.py`**

Remove the entire `collect_image_paths()` function (lines 27–40).

Remove `"page_image_refs": getattr(parent, "page_image_refs", {}),` from the first evidence_map entry (the parent-chunk branch, around line 101).

Remove `"page_image_refs": getattr(r, "page_image_refs", {}),` from the second evidence_map entry (the standalone chunk branch, around line 150).

- [ ] **Step 6: Update `layer_e/pipeline.py`**

Replace lines 38–43 (the image_paths/multimodal branch) with a direct call:

Before:
```python
    image_paths = context_packer.collect_image_paths(evidence_map)
    if image_paths:
        messages = prompt_builder.build_multimodal_messages(
            prompt["system"], prompt["user"], image_paths
        )
        llm_output = llm_client.generate_multimodal(messages)
    else:
        llm_output = llm_client.generate(system=prompt["system"], user=prompt["user"])
```

After:
```python
    llm_output = llm_client.generate(system=prompt["system"], user=prompt["user"])
```

- [ ] **Step 7: Update `tests/test_layer_d/test_retrieval.py`**

Remove the entire `TestPageImageRefsInRankedResult` class and the `_make_scored_point_with_image_refs()` helper function. Also remove `"page_image_refs": page_image_refs if page_image_refs is not None else {},` from any payload dicts in the remaining tests.

- [ ] **Step 8: Update `tests/test_layer_e/test_context_packer.py`**

Remove the `collect_image_paths` import and these five test functions:
- `test_evidence_map_contains_page_image_refs`
- `test_evidence_map_page_image_refs_absent_when_not_set`
- `test_collect_image_paths_absolute`
- `test_collect_image_paths_relative`
- `test_collect_image_paths_dedup`
- `test_collect_image_paths_empty`

Also remove any `"page_image_refs": {...}` key from evidence_map dicts in the remaining tests.

- [ ] **Step 9: Update `tests/test_layer_e/test_pipeline.py`**

Remove these two test functions (they test the now-removed multimodal branch):
- `test_pipeline_uses_multimodal_when_images_present`
- `test_pipeline_fallback_when_llm_no_vision`

In the stub `_FakeRankedResult` dataclass and related helpers, remove any `page_image_refs` field.

- [ ] **Step 10: Run all tests**

```bash
conda run -n hospital-rag pytest tests/ -x -q
```

Expected: All pass. No `page_image_refs` references remain anywhere.

- [ ] **Step 11: Confirm no stray references**

```bash
grep -rn "page_image_refs" /home/wangcy0312/doc-rag-pipeline/layer_b /home/wangcy0312/doc-rag-pipeline/layer_d /home/wangcy0312/doc-rag-pipeline/layer_e /home/wangcy0312/doc-rag-pipeline/pipeline /home/wangcy0312/doc-rag-pipeline/tests
```

Expected: zero output.

- [ ] **Step 12: Commit**

```bash
git add layer_d/models.py layer_d/ingestion.py layer_d/retrieval.py
git add layer_e/context_packer.py layer_e/pipeline.py
git add tests/test_layer_d/test_retrieval.py tests/test_layer_e/test_context_packer.py tests/test_layer_e/test_pipeline.py
git commit -m "refactor(layer_d,layer_e): remove page_image_refs from D/E and dead multimodal branch"
```

---

### Task 4: Document Registry

**Files:**
- Create: `layer_d/document_registry.py`
- Modify: `pipeline/runner.py`
- Test: `tests/test_layer_d/test_document_registry.py`

**Interfaces:**
- Produces: `DocumentRegistry(registry_path)` with `.register(doc_id, pdf_path, collection_name)` and `.get_pdf_path(doc_id) -> str | None`
- Produces: `RAGPipeline.__init__()` gains `registry_path: str | None = None`
- Produces: `RAGPipeline.ingest()` gains `pdf_path: str | None = None, doc_id: str | None = None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_layer_d/test_document_registry.py`:

```python
import json
import pytest
from layer_d.document_registry import DocumentRegistry


def test_register_and_get(tmp_path):
    reg = DocumentRegistry(tmp_path / "registry.json")
    reg.register("乳癌診療指引-2026年", "/docs/乳癌.pdf", collection_name="乳癌診療指引-2026年")
    assert reg.get_pdf_path("乳癌診療指引-2026年") == "/docs/乳癌.pdf"


def test_get_unknown_returns_none(tmp_path):
    reg = DocumentRegistry(tmp_path / "registry.json")
    assert reg.get_pdf_path("nonexistent") is None


def test_persists_to_disk(tmp_path):
    path = tmp_path / "registry.json"
    reg = DocumentRegistry(path)
    reg.register("doc_a", "/path/to/a.pdf")
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["doc_a"]["pdf_path"] == "/path/to/a.pdf"


def test_loads_existing_data(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps({"doc_b": {"pdf_path": "/b.pdf", "collection_name": "col_b"}}),
        encoding="utf-8",
    )
    reg = DocumentRegistry(path)
    assert reg.get_pdf_path("doc_b") == "/b.pdf"


def test_register_overwrites(tmp_path):
    reg = DocumentRegistry(tmp_path / "registry.json")
    reg.register("doc_c", "/old.pdf")
    reg.register("doc_c", "/new.pdf")
    assert reg.get_pdf_path("doc_c") == "/new.pdf"


def test_get_collection_name(tmp_path):
    reg = DocumentRegistry(tmp_path / "registry.json")
    reg.register("doc_d", "/d.pdf", collection_name="col_d")
    assert reg.get_collection_name("doc_d") == "col_d"


def test_get_collection_name_unknown(tmp_path):
    reg = DocumentRegistry(tmp_path / "registry.json")
    assert reg.get_collection_name("unknown") is None
```

Run:
```bash
conda run -n hospital-rag pytest tests/test_layer_d/test_document_registry.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'layer_d.document_registry'`

- [ ] **Step 2: Create `layer_d/document_registry.py`**

```python
from __future__ import annotations
import json
from pathlib import Path


class DocumentRegistry:
    """Persistent mapping from doc_id to PDF file path and Qdrant collection name.

    Stores data as a JSON file so it survives process restarts.
    doc_id is the PDF filename stem (e.g. "乳癌診療指引-2026年").
    """

    def __init__(self, registry_path: str | Path):
        self._path = Path(registry_path)
        self._data: dict[str, dict] = {}
        if self._path.exists():
            self._data = json.loads(self._path.read_text(encoding="utf-8"))

    def register(self, doc_id: str, pdf_path: str, collection_name: str = "") -> None:
        """Add or update an entry and persist to disk immediately."""
        self._data[doc_id] = {
            "pdf_path": str(pdf_path),
            "collection_name": collection_name,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_pdf_path(self, doc_id: str) -> str | None:
        """Return pdf_path for doc_id, or None if not registered."""
        entry = self._data.get(doc_id)
        return entry["pdf_path"] if entry else None

    def get_collection_name(self, doc_id: str) -> str | None:
        """Return collection_name for doc_id, or None if not registered."""
        entry = self._data.get(doc_id)
        return entry.get("collection_name") if entry else None
```

- [ ] **Step 3: Run the tests**

```bash
conda run -n hospital-rag pytest tests/test_layer_d/test_document_registry.py -v
```

Expected: All 7 tests PASS.

- [ ] **Step 4: Update `pipeline/runner.py`**

In `RAGPipeline.__init__()`, add `registry_path` parameter and import:

```python
from layer_d.document_registry import DocumentRegistry

class RAGPipeline:
    def __init__(
        self,
        embedding_provider,
        qdrant_client,
        collection_name: str,
        llm_client=None,
        abstention_threshold: float = 0.10,
        reranker=None,
        registry_path: str | None = None,   # NEW
    ):
        ...
        # add at end of __init__:
        self._registry = DocumentRegistry(registry_path) if registry_path else None
```

In `RAGPipeline.ingest()`, add optional parameters and register the document:

```python
    def ingest(
        self,
        raw_document: dict,
        pdf_path: str | None = None,     # NEW
        doc_id: str | None = None,       # NEW
    ) -> int:
        units = process_document(raw_document)
        chunks = process_and_embed(units, self._provider)
        self._ingester.create_collection_if_not_exists()
        n = self._ingester.ingest(chunks)
        if self._registry is not None and pdf_path and doc_id:
            self._registry.register(
                doc_id, pdf_path, self._ingester.collection_name
            )
        return n
```

- [ ] **Step 5: Run all tests**

```bash
conda run -n hospital-rag pytest tests/ -x -q
```

Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add layer_d/document_registry.py pipeline/runner.py tests/test_layer_d/test_document_registry.py
git commit -m "feat(layer_d): add DocumentRegistry for doc_id→pdf_path lookup; wire into RAGPipeline.ingest()"
```

---

### Task 5: Semantic Enrichment for Tables and Figures

**Files:**
- Create: `layer_b/enrichment.py`
- Test: `tests/test_layer_b/test_enrichment.py`

**Interfaces:**
- Consumes: `list[RetrievalUnit]` (output of `process_document()`), any object with `.generate(system: str, user: str) -> dict`
- Produces: `enrich_units(units, llm_client) -> list[RetrievalUnit]` — returns new list; enriched units have a semantic prefix prepended to `embedding_text`; paragraph units returned unchanged
- Integration: caller (e.g. `runner.py`) calls `enrich_units()` between `process_document()` and `process_and_embed()`

**Context:** The LLM is asked to judge whether a table/figure contains meaningful clinical data. If yes, it returns `semantic_summary`, `applicability`, `answers_questions`. These are prepended to `embedding_text`. If no, the unit is returned unchanged. Layer 1 structural pre-filter skips trivially empty tables (≤ 2 cells) and all-date tables (version history) before touching the LLM. The `llm_client.generate()` interface already handles JSON parsing via `_parse_json_response`.

**Enrichment prompt design:** `llm_client.generate(system, user)` returns whatever JSON the LLM outputs. The enrichment prompts ask for `{"meaningful": bool, ...}` schema — this is a different schema from the generation pipeline's `{"answer", "claims", "abstain"}` schema, but `generate()` in `GPT41Client` simply calls `_parse_json_response(raw)` which parses any JSON.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_layer_b/test_enrichment.py`:

```python
import pytest
from dataclasses import dataclass, field
from layer_b.models import RetrievalUnit
from layer_b.enrichment import enrich_units, _is_trivial_table


# ── Stubs ──────────────────────────────────────────────────────────────────

class _MeaningfulLLM:
    """Always returns meaningful=True with a fixed enrichment."""
    def generate(self, system: str, user: str) -> dict:
        return {
            "meaningful": True,
            "semantic_summary": "乳癌分期對照表",
            "applicability": "需要確認分期的病人",
            "answers_questions": ["T2N1M0是哪個期?", "Stage IIB的TNM條件?"],
        }


class _TrivialLLM:
    """Always returns meaningful=False."""
    def generate(self, system: str, user: str) -> dict:
        return {"meaningful": False}


class _FailingLLM:
    """Always raises an exception."""
    def generate(self, system: str, user: str) -> dict:
        raise RuntimeError("LLM error")


def _table_unit(embedding_text: str = "T0 N1 M0 Stage I", rows=None) -> RetrievalUnit:
    if rows is None:
        rows = [
            {"row_header_path": [], "cells": [
                {"col_header_path": ["TNM"], "value": "T0 N1 M0"},
                {"col_header_path": ["Stage"], "value": "Stage I"},
                {"col_header_path": ["Grade"], "value": "G2"},
            ]},
        ]
    return RetrievalUnit(
        retrieval_unit_id="doc_t_001",
        source_tool="azure_cu",
        embedding_text=embedding_text,
        structured_json={"type": "table", "rows": rows, "merge_rate": 0.1},
        display_markdown="| TNM | Stage |",
        confidence_level="high",
        quality_flag="ok",
        retrieval_weight=1.0,
        source_pages=[5],
        row_texts=["T0 N1 M0為Stage I"],
    )


def _figure_unit(embedding_text: str = "cT2N1M0 治療流程 新輔助化療建議") -> RetrievalUnit:
    return RetrievalUnit(
        retrieval_unit_id="doc_f_001",
        source_tool="azure_cu",
        embedding_text=embedding_text,
        structured_json={"type": "figure", "page": 3, "caption": "", "area": 6.0},
        display_markdown="[圖表 第3頁]",
        confidence_level="high",
        quality_flag="ok",
        retrieval_weight=1.0,
        source_pages=[3],
        row_texts=[],
    )


def _para_unit() -> RetrievalUnit:
    return RetrievalUnit(
        retrieval_unit_id="doc_p_001",
        source_tool="azure_cu",
        embedding_text="化療方案建議",
        structured_json={"type": "paragraph", "content": "化療方案建議"},
        display_markdown="化療方案建議",
        confidence_level="high",
        quality_flag="ok",
        retrieval_weight=1.0,
        source_pages=[4],
        row_texts=[],
    )


# ── Tests ──────────────────────────────────────────────────────────────────

def test_table_gets_enrichment_prefix():
    units = enrich_units([_table_unit()], _MeaningfulLLM())
    assert len(units) == 1
    u = units[0]
    assert "[摘要] 乳癌分期對照表" in u.embedding_text
    assert "T0 N1 M0 Stage I" in u.embedding_text  # original text preserved


def test_table_unchanged_when_not_meaningful():
    original = _table_unit()
    units = enrich_units([original], _TrivialLLM())
    assert units[0].embedding_text == original.embedding_text


def test_paragraph_always_unchanged():
    original = _para_unit()
    units = enrich_units([original], _MeaningfulLLM())
    assert units[0].embedding_text == original.embedding_text


def test_figure_gets_enrichment_prefix():
    units = enrich_units([_figure_unit()], _MeaningfulLLM())
    assert "[摘要]" in units[0].embedding_text
    assert "cT2N1M0 治療流程" in units[0].embedding_text


def test_llm_failure_leaves_unit_unchanged():
    original = _table_unit()
    units = enrich_units([original], _FailingLLM())
    assert units[0].embedding_text == original.embedding_text


def test_trivial_table_skips_llm():
    """Table with <= 2 cells never calls the LLM."""
    call_count = []

    class _CountingLLM:
        def generate(self, system, user):
            call_count.append(1)
            return {"meaningful": True, "semantic_summary": "x", "applicability": "y", "answers_questions": []}

    tiny_rows = [{"row_header_path": [], "cells": [{"col_header_path": ["A"], "value": "B"}]}]
    units = enrich_units([_table_unit(rows=tiny_rows)], _CountingLLM())
    assert call_count == []


def test_is_trivial_table_date_only():
    rows = [
        {"row_header_path": [], "cells": [{"col_header_path": ["Version"], "value": "2025/11/11 Version 19.0"}]},
        {"row_header_path": [], "cells": [{"col_header_path": ["Version"], "value": "2024/09/24 Version 17.0"}]},
        {"row_header_path": [], "cells": [{"col_header_path": ["Version"], "value": "2023/11/14 Version 16.0"}]},
        {"row_header_path": [], "cells": [{"col_header_path": ["Version"], "value": "2022/12/27 Version 15.0"}]},
    ]
    unit = _table_unit(rows=rows)
    assert _is_trivial_table(unit) is True


def test_mixed_units_only_table_and_figure_enriched():
    units = [_para_unit(), _table_unit(), _figure_unit()]
    result = enrich_units(units, _MeaningfulLLM())
    assert result[0].embedding_text == "化療方案建議"      # para unchanged
    assert "[摘要]" in result[1].embedding_text            # table enriched
    assert "[摘要]" in result[2].embedding_text            # figure enriched
```

Run:
```bash
conda run -n hospital-rag pytest tests/test_layer_b/test_enrichment.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'layer_b.enrichment'`

- [ ] **Step 2: Create `layer_b/enrichment.py`**

```python
from __future__ import annotations
import json
import logging
import re
from typing import Any

from layer_b.models import RetrievalUnit

logger = logging.getLogger(__name__)

_MIN_TABLE_CELLS = 3
_DATE_ONLY_RE = re.compile(r'^\d{4}/\d{2}/\d{2}')

_TABLE_SYSTEM = (
    "你是醫療知識庫整理員。請判斷以下表格是否包含對臨床決策有意義的醫療資訊"
    "（如疾病分期、治療建議、用藥劑量、診斷標準等）。\n"
    "若無意義（例如版本歷史、行政表格），只輸出 {\"meaningful\": false}\n"
    "若有意義，輸出 JSON：\n"
    "{\"meaningful\": true, \"semantic_summary\": \"一句話描述此表格的醫療用途\","
    " \"applicability\": \"適用對象或情境（20字以內）\","
    " \"answers_questions\": [\"可回答的具體臨床問題，最多3條\"]}\n"
    "只輸出 JSON，不要其他說明。"
)

_FIGURE_SYSTEM = (
    "你是醫療知識庫整理員。請根據以下文字描述判斷圖表是否包含對臨床決策有意義的醫療資訊"
    "（如治療流程圖、演算法、決策樹、臨床指引步驟等）。\n"
    "若無意義（例如標誌、裝飾圖、空白頁），只輸出 {\"meaningful\": false}\n"
    "若有意義，輸出 JSON：\n"
    "{\"meaningful\": true, \"semantic_summary\": \"一句話描述此圖的醫療用途\","
    " \"applicability\": \"適用對象或情境（20字以內）\","
    " \"answers_questions\": [\"可回答的具體臨床問題，最多3條\"]}\n"
    "只輸出 JSON，不要其他說明。"
)


def _is_trivial_table(unit: RetrievalUnit) -> bool:
    """Layer 1 pre-filter: True for structurally trivial tables that need no enrichment."""
    rows = unit.structured_json.get("rows", [])
    if not rows:
        return True
    all_cells = [c for r in rows for c in r.get("cells", [])]
    if len(all_cells) <= _MIN_TABLE_CELLS:
        return True
    # All non-empty values start with a date — version history table
    values = [c.get("value", "") for c in all_cells if c.get("value", "").strip()]
    return bool(values) and all(_DATE_ONLY_RE.match(v) for v in values)


def _is_trivial_figure(unit: RetrievalUnit) -> bool:
    """Layer 1 pre-filter: True for figures with no useful text content."""
    text = unit.embedding_text.strip()
    if not text:
        return True
    # Fallback placeholder text — no elements were linked
    if text.startswith("[圖表"):
        return True
    return False


def _build_prefix(result: dict) -> str:
    lines = [f"[摘要] {result['semantic_summary']}"]
    if result.get("applicability"):
        lines.append(f"[適用] {result['applicability']}")
    questions = result.get("answers_questions") or []
    if questions:
        lines.append(f"[可解答] {'；'.join(q for q in questions[:3] if q)}")
    return "\n".join(lines) + "\n"


def _call_llm(llm_client: Any, system: str, user: str) -> dict | None:
    """Call llm_client.generate() and return parsed dict, or None on failure."""
    try:
        response = llm_client.generate(system=system, user=user)
        # generate() returns a dict (already parsed by _parse_json_response).
        # If the LLM followed the enrichment prompt, it contains "meaningful".
        if isinstance(response, dict) and "meaningful" in response:
            return response
        # Fallback: try parsing the "answer" key if present (should not happen with
        # a compliant LLM, but defensive against prompt drift)
        answer_text = response.get("answer", "") if isinstance(response, dict) else ""
        if not answer_text:
            return None
        raw = answer_text.strip()
        if raw.startswith("```"):
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw.strip())
        return json.loads(raw)
    except Exception as exc:
        logger.warning("Enrichment LLM call failed: %s", exc)
        return None


def enrich_units(
    units: list[RetrievalUnit],
    llm_client: Any,
) -> list[RetrievalUnit]:
    """Apply LLM semantic enrichment to table and figure units.

    Table and figure units that pass the Layer 1 structural pre-filter are sent
    to the LLM with a domain-specific prompt. If meaningful=true, a semantic
    prefix (摘要/適用/可解答) is prepended to embedding_text. Paragraph units
    and trivial tables/figures are returned unchanged.

    Args:
        units: Output of process_document().
        llm_client: Any object with .generate(system, user) -> dict.
                    Uses GPT41Client or any LLMClient subclass in practice.
    """
    from dataclasses import replace

    result = []
    for unit in units:
        unit_type = unit.structured_json.get("type")

        if unit_type == "table":
            if not _is_trivial_table(unit):
                enrichment = _call_llm(llm_client, _TABLE_SYSTEM, unit.embedding_text[:600])
                if enrichment and enrichment.get("meaningful"):
                    unit = replace(unit, embedding_text=_build_prefix(enrichment) + unit.embedding_text)

        elif unit_type == "figure":
            if not _is_trivial_figure(unit):
                enrichment = _call_llm(llm_client, _FIGURE_SYSTEM, unit.embedding_text[:600])
                if enrichment and enrichment.get("meaningful"):
                    unit = replace(unit, embedding_text=_build_prefix(enrichment) + unit.embedding_text)

        result.append(unit)
    return result
```

- [ ] **Step 3: Run the tests**

```bash
conda run -n hospital-rag pytest tests/test_layer_b/test_enrichment.py -v
```

Expected: All 9 tests PASS.

- [ ] **Step 4: Run all tests**

```bash
conda run -n hospital-rag pytest tests/ -x -q
```

Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add layer_b/enrichment.py tests/test_layer_b/test_enrichment.py
git commit -m "feat(layer_b): add semantic enrichment module with LLM quality filter for tables and figures"
```

---

## Self-Review

**Spec coverage:**
1. ✅ Remove `page_image_refs` from all layers — Tasks 1 + 3
2. ✅ Fix Azure CU `_figure_path` (was producing 0 figures) — Task 2
3. ✅ Document registry (doc_id → pdf_path) — Task 4
4. ✅ Semantic enrichment for tables and figures — Task 5
5. ✅ `process_document()` signature unchanged — `enrich_units()` is a separate call
6. ✅ No new third-party dependencies — stdlib json + existing dataclasses

**Placeholder scan:** None found. All code blocks are complete.

**Type consistency:**
- `enrich_units(units: list[RetrievalUnit], llm_client: Any) -> list[RetrievalUnit]` — consistent across test stubs and implementation
- `DocumentRegistry.register(doc_id, pdf_path, collection_name)` — consistent between tests and implementation
- `RAGPipeline.ingest(raw_document, pdf_path=None, doc_id=None)` — consistent with existing call sites (positional `raw_document` unchanged)
- `_parse_figure_area(source_str) -> tuple[int | None, float]` — consistent between Task 2 usage and definition

**Edge cases covered:**
- LLM failure in enrichment → unit returned unchanged (test: `test_llm_failure_leaves_unit_unchanged`)
- Trivial table skips LLM entirely (test: `test_trivial_table_skips_llm`)
- Figure with no elements text skips enrichment (test: `_is_trivial_figure`)
- Azure CU tiny icon figure filtered by area < 0.5 (test: `test_figure_path_azure_cu`)
- Registry file doesn't exist yet → empty dict (test: `test_register_and_get`)
