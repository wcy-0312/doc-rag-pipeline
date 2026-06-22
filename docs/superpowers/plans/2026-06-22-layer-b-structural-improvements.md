# Layer B Structural Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the PDF (Azure CU) pipeline in Layer B with four coordinated changes: decouple OCR quality from retrieval scores, eliminate figure-paragraph duplication, build full hierarchical heading breadcrumbs, and extract a compact document index that gives the agentic loop pre-query structural navigation.

**Architecture:** `retrieval_weight` is fixed at 1.0 for all units — quality signals live in `quality_flag`/`confidence_level` for display only. The `sections[]` tree from Azure CU drives two new features: per-chunk full heading paths (replaces the single-level `current_heading` scan), and a compact ToC stored in Qdrant as a special point (retrieval_weight=0.0, invisible to normal searches). The agentic loop (`AgenticPipeline`) loads the ToC at init time and includes it in the system prompt so the LLM can plan `retrieve_more` calls against specific sections.

**Tech Stack:** Python 3.11+, Qdrant (qdrant_client), layer_b/pipeline.py, layer_d/ingestion.py + retrieval.py, layer_e/agentic_pipeline.py.

## Global Constraints

- `process_document()` return type stays `list[RetrievalUnit]` — no signature change
- New helper functions in `layer_b/pipeline.py` are module-level (not nested), prefixed with `_`
- No external API or Qdrant calls in unit tests — use pytest fixtures with synthetic dicts
- Run tests: `pytest tests/test_layer_b/ tests/test_layer_e/test_context_packer.py -v`
- Every function that is new or changed must have at least one test

---

## File Map

| File | Change |
|---|---|
| `layer_b/pipeline.py` | Tasks 1–4: `_doc_confidence`, `_table_path`, `_high_graphics_path`, `_extract_azure_cu_paragraphs`, `_paragraph_path`, `process_document` updated; 3 new functions added |
| `layer_e/context_packer.py` | Task 1: replace `retrieval_weight < 0.5` with `metadata.get("quality_flag") == "low"` (2 places) |
| `layer_d/ingestion.py` | Task 4: add `DocumentIngester.store_document_index()` method |
| `layer_d/retrieval.py` | Task 4: add `HybridRetriever.get_document_index()` method; add `MatchValue` import |
| `layer_e/agentic_pipeline.py` | Task 4: load document_index at init; append outline to system prompt |
| `tests/test_layer_b/test_pipeline.py` | Tasks 1, 3, 4: new test functions |
| `tests/test_layer_b/test_paragraph_path.py` | Task 2: new test for figure dedup |
| `tests/test_layer_e/test_context_packer.py` | Task 1: update low-confidence tests |

---

### Task 1: Decouple quality from retrieval_weight

**Files:**
- Modify: `layer_b/pipeline.py` (lines 437–463, 603–607, 808)
- Modify: `layer_e/context_packer.py` (lines 49, 97)
- Test: `tests/test_layer_b/test_pipeline.py`
- Test: `tests/test_layer_e/test_context_packer.py`

**Interfaces:**
- Consumes: nothing new
- Produces: `_doc_confidence()` still returns `(str, str, float)` — the float (weight) is now always `1.0`

- [ ] **Step 1: Write failing tests**

In `tests/test_layer_b/test_pipeline.py`, add after the existing `test_continuous_weight`:

```python
def test_doc_confidence_weight_always_one_no_qc():
    """PDF path: no qc key → weight must be 1.0."""
    raw = {"metadata": {}}
    level, flag, weight = _doc_confidence(raw)
    assert weight == 1.0
    assert level == "high"
    assert flag == "ok"


def test_doc_confidence_weight_always_one_with_high_loss():
    """Word/DI path: high info_loss → level=low, flag=low, but weight still 1.0."""
    from layer_b.pipeline import _doc_confidence
    raw = {"metadata": {"qc": {"estimated_info_loss_rate": 0.20, "qc_level": "danger"}}}
    level, flag, weight = _doc_confidence(raw)
    assert weight == 1.0
    assert level == "low"
    assert flag == "low"


def test_doc_confidence_fully_scanned_is_low():
    """Fully scanned doc → level=low, flag=low regardless of info_loss."""
    raw = {
        "metadata": {
            "qc": {"estimated_info_loss_rate": 0.01},
            "extractor_metadata": {"is_fully_scanned": True},
        }
    }
    level, flag, weight = _doc_confidence(raw)
    assert level == "low"
    assert flag == "low"
    assert weight == 1.0
```

Add to imports at top of the test file (if not already):
```python
from layer_b.pipeline import _doc_confidence
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_layer_b/test_pipeline.py::test_doc_confidence_weight_always_one_no_qc \
       tests/test_layer_b/test_pipeline.py::test_doc_confidence_weight_always_one_with_high_loss \
       tests/test_layer_b/test_pipeline.py::test_doc_confidence_fully_scanned_is_low -v
```

Expected: FAIL — `_doc_confidence` currently returns non-1.0 weights.

- [ ] **Step 3: Replace `_doc_confidence()` in `layer_b/pipeline.py`**

Replace the entire function (lines 437–463):

```python
def _doc_confidence(raw: dict) -> tuple[str, str, float]:
    """Return (confidence_level, quality_flag, retrieval_weight).

    retrieval_weight 固定 1.0 — 品質不影響 retrieval 排名。
    quality_flag="low" が display 層の [低信心] ラベルを駆動する。
    """
    doc_metadata = raw.get("metadata", {})
    if "qc" not in doc_metadata:
        return "high", "ok", 1.0

    info_loss = None
    try:
        info_loss = doc_metadata["qc"]["estimated_info_loss_rate"]
    except (KeyError, TypeError):
        pass

    is_fully_scanned = doc_metadata.get("extractor_metadata", {}).get("is_fully_scanned", False)

    if is_fully_scanned or (info_loss is not None and info_loss > _INFO_LOSS_HIGH):
        level, flag = "low", "low"
    elif info_loss is not None and info_loss > _INFO_LOSS_LOW:
        level, flag = "medium", "ok"
    else:
        level, flag = "high", "ok"

    return level, flag, 1.0
```

- [ ] **Step 4: Fix `_table_path()` — remove quality-based weight**

In `layer_b/pipeline.py` line ~606, replace:

```python
        weight = _continuous_weight(conf["score"])
```

with:

```python
        weight = 1.0
```

- [ ] **Step 5: Fix `_high_graphics_path()` — remove 0.9 penalty**

In `layer_b/pipeline.py` line ~808, replace:

```python
            retrieval_weight=retrieval_weight * 0.9,  # 輕微降權，因無直接文字
```

with:

```python
            retrieval_weight=retrieval_weight,
```

- [ ] **Step 6: Update `context_packer.py` — use `quality_flag` for [低信心] label**

In `layer_e/context_packer.py`, find **both** occurrences (lines ~49 and ~97) of:

```python
            if r.retrieval_weight < 0.5:
                content = "[低信心] " + content
```

Replace each with:

```python
            if r.metadata.get("quality_flag") == "low":
                content = "[低信心] " + content
```

- [ ] **Step 7: Update context_packer tests**

In `tests/test_layer_e/test_context_packer.py`, the existing `_make_para` helper creates `RankedResult` with `metadata={}`. Find the two tests that check `[低信心]` behaviour (around line 76–88) and update them:

```python
def test_pack_low_confidence_label():
    """quality_flag='low' → [低信心] prefix in packed content."""
    r = _make_para(
        "c1", rerank_score=0.9,
        content="some meaningful text here",
        retrieval_weight=1.0,          # weight no longer drives label
        metadata={"quality_flag": "low"},
    )
    evidence_list, _ = pack([r])
    assert evidence_list[0].content.startswith("[低信心] ")


def test_pack_normal_confidence_no_label():
    """quality_flag absent → no [低信心] prefix."""
    r = _make_para(
        "c1", rerank_score=0.9,
        content="some meaningful text here",
        retrieval_weight=1.0,
        metadata={},
    )
    evidence_list, _ = pack([r])
    assert not evidence_list[0].content.startswith("[低信心]")
```

Also update `_make_para` helper to accept `metadata` kwarg:

```python
def _make_para(chunk_id, rerank_score, content="default test content",
               retrieval_weight=1.0, source_pages=None, metadata=None):
    return RankedResult(
        chunk_id=chunk_id,
        chunk_type="paragraph",
        parent_chunk_id=None,
        retrieval_unit_id=chunk_id,
        final_score=rerank_score,
        rrf_score=rerank_score,
        retrieval_weight=retrieval_weight,
        display_markdown=content,
        metadata=metadata or {},
        source_tool="docling",
        source_pages=source_pages or [1],
        rerank_score=rerank_score,
    )
```

- [ ] **Step 8: Run all affected tests**

```bash
pytest tests/test_layer_b/test_pipeline.py tests/test_layer_e/test_context_packer.py -v
```

Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add layer_b/pipeline.py layer_e/context_packer.py \
        tests/test_layer_b/test_pipeline.py tests/test_layer_e/test_context_packer.py
git commit -m "refactor(layer_b): decouple OCR quality from retrieval_weight

retrieval_weight is now fixed at 1.0 for all units. quality_flag='low'
drives the [低信心] display label in context_packer instead of
retrieval_weight < 0.5. _high_graphics_path drops the 0.9 penalty."
```

---

### Task 2: Deduplicate figures[].elements[] paragraphs

**Files:**
- Modify: `layer_b/pipeline.py`
- Test: `tests/test_layer_b/test_paragraph_path.py`

**Interfaces:**
- Consumes: `raw["data"]["figures"][*]["elements"]` — list of strings like `"/paragraphs/5"`
- Produces: `_build_figure_element_set(figures) -> set[int]` (new); `_extract_azure_cu_paragraphs` gains `skip_indices` parameter; `_paragraph_path` gains `figure_element_set` parameter

- [ ] **Step 1: Write failing test**

In `tests/test_layer_b/test_paragraph_path.py`, add:

```python
from layer_b.pipeline import _build_figure_element_set, _extract_azure_cu_paragraphs


def test_build_figure_element_set_basic():
    figures = [
        {"elements": ["/paragraphs/3", "/paragraphs/7"]},
        {"elements": ["/paragraphs/7", "/paragraphs/12"]},
    ]
    result = _build_figure_element_set(figures)
    assert result == {3, 7, 12}


def test_build_figure_element_set_empty():
    assert _build_figure_element_set([]) == set()
    assert _build_figure_element_set([{"elements": []}]) == set()


def test_extract_azure_cu_paragraphs_skips_figure_elements():
    """Paragraphs referenced in figures[].elements[] must not appear as candidates."""
    data = {
        "paragraphs": [
            {"content": "chapter heading", "role": "sectionHeading",
             "source": "D(1,0,0,1,0,1,1,0,1)", "spans": []},
            {"content": "caption for figure 1", "role": None,
             "source": "D(1,0,0,1,0,1,1,0,1)", "spans": []},
            {"content": "Normal paragraph with enough text to pass filter.",
             "role": None, "source": "D(1,0,0,1,0,1,1,0,1)", "spans": []},
        ],
    }
    # paragraph index 1 is a figure element — should be skipped
    candidates = _extract_azure_cu_paragraphs(data, skip_indices={1})
    contents = [c["content"] for c in candidates]
    assert "caption for figure 1" not in contents
    assert "Normal paragraph with enough text to pass filter." in contents
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_layer_b/test_paragraph_path.py::test_build_figure_element_set_basic \
       tests/test_layer_b/test_paragraph_path.py::test_extract_azure_cu_paragraphs_skips_figure_elements -v
```

Expected: FAIL — `_build_figure_element_set` doesn't exist; `_extract_azure_cu_paragraphs` has no `skip_indices` param.

- [ ] **Step 3: Add `_build_figure_element_set` to `layer_b/pipeline.py`**

Insert after `_parse_figure_area` (around line 35):

```python
def _build_figure_element_set(figures: list[dict]) -> set[int]:
    """Return set of paragraph indices referenced in any figure's elements[].

    These paragraphs are captions/labels belonging to figures and must not
    be emitted again as standalone paragraph RetrievalUnits.
    """
    indices: set[int] = set()
    for fig in figures:
        for elem_ref in fig.get("elements", []):
            try:
                indices.add(int(str(elem_ref).split("/")[-1]))
            except (ValueError, IndexError):
                pass
    return indices
```

- [ ] **Step 4: Add `skip_indices` parameter to `_extract_azure_cu_paragraphs`**

Change function signature (around line 242):

```python
def _extract_azure_cu_paragraphs(
    data: dict,
    skip_indices: set[int] | None = None,
) -> list[dict]:
```

Inside the loop, add skip check as the very first line:

```python
    for i, para in enumerate(data.get("paragraphs", [])):
        if skip_indices and i in skip_indices:
            continue
        role = para.get("role")
        ...
```

- [ ] **Step 5: Thread `figure_element_set` through `_paragraph_path` and `process_document`**

Add parameter to `_paragraph_path` signature:

```python
def _paragraph_path(
    raw: dict,
    source_tool: str,
    doc_prefix: str = "",
    doc_metadata: dict | None = None,
    confidence: tuple | None = None,
    figure_element_set: set[int] | None = None,
) -> list[RetrievalUnit]:
```

Inside, change the `azure_cu` branch:

```python
    if source_tool == "azure_cu":
        candidates = _extract_azure_cu_paragraphs(data, skip_indices=figure_element_set)
```

In `process_document()`, add before `units = []`:

```python
    figure_element_set = _build_figure_element_set(raw.get("data", {}).get("figures", []))
```

And update the `_paragraph_path` call:

```python
    units.extend(_paragraph_path(
        raw, source_tool, doc_prefix, doc_metadata, confidence,
        figure_element_set=figure_element_set,
    ))
```

- [ ] **Step 6: Run tests**

```bash
pytest tests/test_layer_b/test_paragraph_path.py -v
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add layer_b/pipeline.py tests/test_layer_b/test_paragraph_path.py
git commit -m "fix(layer_b): skip figure-referenced paragraphs in paragraph path

Paragraphs listed in figures[].elements[] were being emitted as
standalone RetrievalUnits AND used as figure embedding text, causing
duplicate content. _build_figure_element_set() extracts the index set;
_extract_azure_cu_paragraphs() now accepts skip_indices to exclude them."
```

---

### Task 3: Full hierarchical heading_breadcrumb from sections[] tree

**Files:**
- Modify: `layer_b/pipeline.py`
- Test: `tests/test_layer_b/test_pipeline.py`

**Interfaces:**
- Consumes: `raw["data"]["sections"]` — Azure CU sections[] tree
- Produces: `_build_section_path_map(sections) -> dict[int, list[str]]` (new); `_extract_azure_cu_paragraphs` gains `section_path_map` parameter; `process_document` builds and passes it

- [ ] **Step 1: Write failing tests**

In `tests/test_layer_b/test_pipeline.py`, add:

```python
from layer_b.pipeline import _build_section_path_map


def test_build_section_path_map_flat():
    """Single-level sections → paragraph gets one-element path."""
    sections = [
        {
            "title": "第一章 流行病學",
            "elements": ["/paragraphs/0", "/paragraphs/1"],
        },
        {
            "title": "第二章 治療",
            "elements": ["/paragraphs/2"],
        },
    ]
    result = _build_section_path_map(sections)
    assert result[0] == ["第一章 流行病學"]
    assert result[1] == ["第一章 流行病學"]
    assert result[2] == ["第二章 治療"]


def test_build_section_path_map_nested():
    """Nested sections → paragraph gets full path list."""
    sections = [
        {
            "title": "",                              # root, no title
            "elements": ["/sections/1", "/sections/2"],
        },
        {
            "title": "第三章 治療",
            "elements": ["/paragraphs/10", "/sections/3"],
        },
        {
            "title": "第一章",
            "elements": ["/paragraphs/5"],
        },
        {
            "title": "3.1 一線化療",
            "elements": ["/paragraphs/12"],
        },
    ]
    result = _build_section_path_map(sections)
    assert result[5] == ["第一章"]
    assert result[10] == ["第三章 治療"]
    assert result[12] == ["第三章 治療", "3.1 一線化療"]
    assert 0 not in result   # para 0 not referenced by any section


def test_extract_azure_cu_paragraphs_uses_section_map():
    """Paragraphs use full section path when section_path_map provided."""
    data = {
        "paragraphs": [
            {"content": "劑量每日 5mg，第 1-5 天給藥，每 28 天一個療程。",
             "role": None, "source": "D(1,0,0,1,0,1,1,0,1)", "spans": []},
        ],
    }
    section_path_map = {0: ["第三章 治療", "3.1 一線化療"]}
    candidates = _extract_azure_cu_paragraphs(data, section_path_map=section_path_map)
    assert candidates[0]["heading_breadcrumb"] == "第三章 治療 > 3.1 一線化療"
```

Add to imports at top of test file:
```python
from layer_b.pipeline import _build_section_path_map, _extract_azure_cu_paragraphs
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_layer_b/test_pipeline.py::test_build_section_path_map_flat \
       tests/test_layer_b/test_pipeline.py::test_build_section_path_map_nested \
       tests/test_layer_b/test_pipeline.py::test_extract_azure_cu_paragraphs_uses_section_map -v
```

Expected: FAIL — `_build_section_path_map` doesn't exist.

- [ ] **Step 3: Add `_build_section_path_map` to `layer_b/pipeline.py`**

Insert after `_build_figure_element_set` (Task 2):

```python
def _build_section_path_map(sections: list[dict]) -> dict[int, list[str]]:
    """Build {paragraph_index: [title, subtitle, ...]} from sections[] tree.

    Traverses the tree recursively. A paragraph's breadcrumb is the ordered
    list of non-empty section titles from the nearest root down to the section
    that directly references it. If a paragraph appears in multiple sections
    the first encounter wins.
    """
    section_by_idx: dict[int, dict] = {i: s for i, s in enumerate(sections)}
    result: dict[int, list[str]] = {}

    def _visit(sec_idx: int, path: list[str]) -> None:
        sec = section_by_idx.get(sec_idx)
        if sec is None:
            return
        title = (sec.get("title") or "").strip()
        current_path = path + [title] if title else list(path)
        for elem_ref in sec.get("elements", []):
            try:
                parts = str(elem_ref).strip("/").split("/")
                kind, idx = parts[-2], int(parts[-1])
            except (IndexError, ValueError):
                continue
            if kind == "paragraphs" and idx not in result:
                result[idx] = current_path
            elif kind == "sections":
                _visit(idx, current_path)

    # Identify root sections — those NOT referenced as a child by any other section
    child_indices: set[int] = set()
    for sec in sections:
        for elem_ref in sec.get("elements", []):
            try:
                parts = str(elem_ref).strip("/").split("/")
                if parts[-2] == "sections":
                    child_indices.add(int(parts[-1]))
            except (IndexError, ValueError):
                pass

    for i in range(len(sections)):
        if i not in child_indices:
            _visit(i, [])

    return result
```

- [ ] **Step 4: Add `section_path_map` parameter to `_extract_azure_cu_paragraphs`**

Update function signature to accept both new parameters together (combining with Task 2):

```python
def _extract_azure_cu_paragraphs(
    data: dict,
    skip_indices: set[int] | None = None,
    section_path_map: dict[int, list[str]] | None = None,
) -> list[dict]:
```

Inside the loop, after the skip check, update the `heading_breadcrumb` assignment in the non-sectionHeading branch:

```python
        else:
            if len(content) < MIN_PARA_LEN:
                continue
            if _VERSION_HISTORY_RE.match(content):
                continue
            # Prefer full section path from sections[] tree; fall back to tracked heading
            if section_path_map is not None and i in section_path_map:
                path = section_path_map[i]
                hb = " > ".join(p for p in path if p) or None
            else:
                hb = current_heading
            candidates.append({
                "content": content,
                "page": page,
                "role": None,
                "label": None,
                "heading_breadcrumb": hb,
                "excluded_items": [],
                "spans": spans,
            })
```

- [ ] **Step 5: Thread `section_path_map` through `_paragraph_path` and `process_document`**

Add parameter to `_paragraph_path` signature:

```python
def _paragraph_path(
    raw: dict,
    source_tool: str,
    doc_prefix: str = "",
    doc_metadata: dict | None = None,
    confidence: tuple | None = None,
    figure_element_set: set[int] | None = None,
    section_path_map: dict[int, list[str]] | None = None,
) -> list[RetrievalUnit]:
```

Update the `azure_cu` branch inside `_paragraph_path`:

```python
    if source_tool == "azure_cu":
        candidates = _extract_azure_cu_paragraphs(
            data,
            skip_indices=figure_element_set,
            section_path_map=section_path_map,
        )
```

In `process_document()`, add after `figure_element_set = ...`:

```python
    sections = raw.get("data", {}).get("sections", [])
    section_path_map = _build_section_path_map(sections) if sections else {}
```

Update the `_paragraph_path` call:

```python
    units.extend(_paragraph_path(
        raw, source_tool, doc_prefix, doc_metadata, confidence,
        figure_element_set=figure_element_set,
        section_path_map=section_path_map,
    ))
```

- [ ] **Step 6: Run tests**

```bash
pytest tests/test_layer_b/test_pipeline.py tests/test_layer_b/test_paragraph_path.py -v
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add layer_b/pipeline.py tests/test_layer_b/test_pipeline.py
git commit -m "feat(layer_b): build full heading_breadcrumb from sections[] tree

Replace single-level current_heading tracking with a tree traversal of
sections[]. Each paragraph now gets its full path e.g.
'第三章 治療 > 3.1 一線化療'. Falls back to the heading-scan approach
when sections[] is absent (DI/Docling paths)."
```

---

### Task 4: document_index extraction, storage, and agentic use

**Files:**
- Modify: `layer_b/pipeline.py` (new `extract_document_index`)
- Modify: `layer_d/ingestion.py` (new `store_document_index`)
- Modify: `layer_d/retrieval.py` (new `get_document_index`, add `MatchValue` import)
- Modify: `layer_e/agentic_pipeline.py` (load index at init, inject into system prompt)
- Test: `tests/test_layer_b/test_pipeline.py`
- Test: `tests/test_layer_d/test_ingestion.py`
- Test: `tests/test_layer_e/test_agentic_pipeline.py`

**Interfaces:**
- `extract_document_index(raw: dict) -> dict | None` — produces compact ToC dict
- `DocumentIngester.store_document_index(doc_stem, index)` — consumes ToC dict
- `HybridRetriever.get_document_index(doc_stem) -> dict | None` — returns ToC dict
- `AgenticPipeline.__init__` — calls `retriever.get_document_index(doc_stem)` internally

- [ ] **Step 1: Write failing tests**

In `tests/test_layer_b/test_pipeline.py`, add:

```python
from layer_b.pipeline import extract_document_index


def test_extract_document_index_flat():
    raw = {
        "data": {
            "sections": [
                {"title": "第一章 流行病學", "elements": ["/paragraphs/0"]},
                {"title": "第二章 治療", "elements": ["/paragraphs/1"]},
            ]
        }
    }
    idx = extract_document_index(raw)
    assert idx is not None
    titles = [s["title"] for s in idx["sections"]]
    assert "第一章 流行病學" in titles
    assert "第二章 治療" in titles


def test_extract_document_index_nested():
    raw = {
        "data": {
            "sections": [
                {"title": "", "elements": ["/sections/1", "/sections/2"]},
                {"title": "第三章 治療", "elements": ["/sections/3"]},
                {"title": "第一章", "elements": ["/paragraphs/5"]},
                {"title": "3.1 一線化療", "elements": ["/paragraphs/12"]},
            ]
        }
    }
    idx = extract_document_index(raw)
    assert idx is not None
    # Top-level should be 第三章 and 第一章 (children of the empty root)
    root_titles = [s.get("title", "") for s in idx["sections"]]
    assert "第三章 治療" in root_titles
    # 3.1 一線化療 should be nested under 第三章 治療
    ch3 = next(s for s in idx["sections"] if s.get("title") == "第三章 治療")
    assert "sections" in ch3
    assert ch3["sections"][0]["title"] == "3.1 一線化療"


def test_extract_document_index_no_sections():
    assert extract_document_index({"data": {}}) is None
    assert extract_document_index({"data": {"sections": []}}) is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_layer_b/test_pipeline.py::test_extract_document_index_flat \
       tests/test_layer_b/test_pipeline.py::test_extract_document_index_nested \
       tests/test_layer_b/test_pipeline.py::test_extract_document_index_no_sections -v
```

Expected: FAIL — `extract_document_index` doesn't exist.

- [ ] **Step 3: Add `extract_document_index` to `layer_b/pipeline.py`**

Insert after `_build_section_path_map`:

```python
def extract_document_index(raw: dict) -> dict | None:
    """Build a compact Table-of-Contents dict from sections[] tree.

    Returns None if sections[] is absent or has no titled sections.
    The returned dict is stored in Qdrant by DocumentIngester.store_document_index()
    and retrieved by HybridRetriever.get_document_index() for the agentic loop.

    Output shape:
        {"sections": [{"title": "...", "sections": [...optional children...]}, ...]}
    """
    sections = raw.get("data", {}).get("sections", [])
    if not sections:
        return None

    section_by_idx: dict[int, dict] = {i: s for i, s in enumerate(sections)}

    def _node(sec_idx: int) -> dict | None:
        sec = section_by_idx.get(sec_idx)
        if sec is None:
            return None
        title = (sec.get("title") or "").strip()
        children = []
        for elem_ref in sec.get("elements", []):
            try:
                parts = str(elem_ref).strip("/").split("/")
                if parts[-2] == "sections":
                    child = _node(int(parts[-1]))
                    if child is not None:
                        children.append(child)
            except (IndexError, ValueError):
                pass
        # Skip nodes with no title AND no children (structural noise)
        if not title and not children:
            return None
        node: dict = {}
        if title:
            node["title"] = title
        if children:
            node["sections"] = children
        return node

    child_indices: set[int] = set()
    for sec in sections:
        for elem_ref in sec.get("elements", []):
            try:
                parts = str(elem_ref).strip("/").split("/")
                if parts[-2] == "sections":
                    child_indices.add(int(parts[-1]))
            except (IndexError, ValueError):
                pass

    roots = [
        node
        for i in range(len(sections))
        if i not in child_indices
        for node in [_node(i)]
        if node is not None
    ]

    return {"sections": roots} if roots else None
```

- [ ] **Step 4: Run Layer B tests**

```bash
pytest tests/test_layer_b/test_pipeline.py -v
```

Expected: all pass.

- [ ] **Step 5: Add `store_document_index` to `DocumentIngester` in `layer_d/ingestion.py`**

Add after the `ingest` method:

```python
    def store_document_index(self, doc_stem: str, document_index: dict) -> None:
        """Store a compact document ToC as a special Qdrant point.

        retrieval_weight=0.0 keeps it invisible to all normal searches
        (search filter requires >=0.3). Retrieve it with
        HybridRetriever.get_document_index(doc_stem).
        """
        chunk_id = f"{doc_stem}__document_index"
        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(
                id=_chunk_id_to_point_id(chunk_id),
                vector={
                    "dense": [0.0] * 1024,
                    "sparse": SparseVector(indices=[], values=[]),
                },
                payload={
                    "chunk_id":         chunk_id,
                    "chunk_type":       "document_index",
                    "retrieval_weight": 0.0,
                    "document_index":   document_index,
                },
            )],
        )
```

- [ ] **Step 6: Write ingestion test**

In `tests/test_layer_d/test_ingestion.py`, add:

```python
def test_store_document_index(mock_qdrant_client):
    """store_document_index upserts a point with chunk_type=document_index."""
    ingester = DocumentIngester(client=mock_qdrant_client)
    index = {"sections": [{"title": "第一章"}]}
    ingester.store_document_index("my_doc", index)

    calls = mock_qdrant_client.upsert.call_args_list
    assert len(calls) == 1
    points = calls[0].kwargs["points"]
    assert len(points) == 1
    payload = points[0].payload
    assert payload["chunk_type"] == "document_index"
    assert payload["retrieval_weight"] == 0.0
    assert payload["chunk_id"] == "my_doc__document_index"
    assert payload["document_index"] == index
```

Check existing test file for the `mock_qdrant_client` fixture pattern and follow it.

- [ ] **Step 7: Add `get_document_index` to `HybridRetriever` in `layer_d/retrieval.py`**

Add `MatchValue` to the existing import block:

```python
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
    MatchText,
    MatchValue,       # add this
    Prefetch,
    Range,
    SparseVector,
)
```

Add method after `make_doc_filter`:

```python
    def get_document_index(self, doc_stem: str) -> dict | None:
        """Return the stored document_index for doc_stem, or None if absent."""
        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(must=[
                FieldCondition(key="chunk_type", match=MatchValue(value="document_index")),
                FieldCondition(key="chunk_id", match=MatchText(text=f"{doc_stem}__")),
            ]),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not results:
            return None
        return results[0].payload.get("document_index")
```

- [ ] **Step 8: Add `_format_document_index` helper and update `AgenticPipeline` in `layer_e/agentic_pipeline.py`**

Add module-level helper before the `AgenticPipeline` class:

```python
def _format_document_index(index: dict) -> str:
    """Format document_index dict as indented text for the system prompt."""
    lines: list[str] = []

    def _render(sections: list, depth: int) -> None:
        for sec in sections:
            title = sec.get("title", "")
            if title:
                lines.append("  " * depth + title)
            _render(sec.get("sections", []), depth + 1)

    _render(index.get("sections", []), 0)
    return "\n".join(lines)
```

In `AgenticPipeline.__init__`, add after `self._abstention_threshold = abstention_threshold`:

```python
        # Load document index for pre-query navigation
        document_index = None
        if hasattr(retriever, "get_document_index"):
            try:
                document_index = retriever.get_document_index(doc_stem)
            except Exception:
                pass
        self._document_outline: str | None = (
            _format_document_index(document_index) if document_index else None
        )
```

In `AgenticPipeline.run()`, after `system_content = _SYSTEM_TEMPLATE.format(evidence_block=evidence_block)`:

```python
        if self._document_outline:
            system_content += (
                "\n\n文件結構概覽（可輔助決定 retrieve_more 應搜尋哪個章節）：\n"
                + self._document_outline
            )
```

- [ ] **Step 9: Write agentic_pipeline test**

In `tests/test_layer_e/test_agentic_pipeline.py`, find the existing test pattern and add:

```python
def test_document_outline_appears_in_system_prompt(mock_llm, mock_retriever):
    """When retriever returns a document_index, system prompt includes the outline."""
    mock_retriever.get_document_index.return_value = {
        "sections": [
            {"title": "第一章 流行病學"},
            {"title": "第三章 治療", "sections": [{"title": "3.1 一線化療"}]},
        ]
    }
    pipeline = AgenticPipeline(
        llm_client=mock_llm,
        retriever=mock_retriever,
        pdf_path="/tmp/test.pdf",
        doc_stem="test_doc",
    )
    assert pipeline._document_outline is not None
    assert "第一章 流行病學" in pipeline._document_outline
    assert "3.1 一線化療" in pipeline._document_outline


def test_no_document_outline_when_retriever_returns_none(mock_llm, mock_retriever):
    """When retriever returns None, _document_outline is None and prompt is unaffected."""
    mock_retriever.get_document_index.return_value = None
    pipeline = AgenticPipeline(
        llm_client=mock_llm,
        retriever=mock_retriever,
        pdf_path="/tmp/test.pdf",
        doc_stem="test_doc",
    )
    assert pipeline._document_outline is None
```

- [ ] **Step 10: Run all tests**

```bash
pytest tests/test_layer_b/ tests/test_layer_d/test_ingestion.py \
       tests/test_layer_e/test_agentic_pipeline.py tests/test_layer_e/test_context_packer.py -v
```

Expected: all pass.

- [ ] **Step 11: Commit**

```bash
git add layer_b/pipeline.py layer_d/ingestion.py layer_d/retrieval.py \
        layer_e/agentic_pipeline.py \
        tests/test_layer_b/test_pipeline.py tests/test_layer_d/test_ingestion.py \
        tests/test_layer_e/test_agentic_pipeline.py
git commit -m "feat(layer_b/d/e): add document_index for agentic loop navigation

extract_document_index() builds a compact ToC from sections[] tree.
DocumentIngester.store_document_index() persists it as a special Qdrant
point (retrieval_weight=0.0, invisible to normal searches).
AgenticPipeline loads it at init and appends the section outline to the
system prompt, enabling the LLM to plan retrieve_more calls by section."
```

---

## Self-Review

**Spec coverage check:**

| Agreed change | Covered by |
|---|---|
| retrieval_weight 固定 1.0 | Task 1 `_doc_confidence`, `_table_path`, `_high_graphics_path` |
| `[低信心]` label uses quality_flag | Task 1 `context_packer.py` |
| figures dedup | Task 2 `_build_figure_element_set` + `skip_indices` |
| Full heading_breadcrumb from sections[] | Task 3 `_build_section_path_map` + `section_path_map` param |
| document_index extraction | Task 4 `extract_document_index` |
| document_index storage in Qdrant | Task 4 `store_document_index` |
| document_index served to agentic loop | Task 4 `get_document_index` + `AgenticPipeline` |

**Placeholder scan:** No TBD, no "implement later", no "similar to Task N" patterns found.

**Type consistency:**
- `_build_section_path_map` returns `dict[int, list[str]]` — matches the `section_path_map` parameter type in `_extract_azure_cu_paragraphs` and `_paragraph_path` ✓
- `extract_document_index` returns `dict | None` — matches `store_document_index(doc_stem, document_index: dict)` caller ✓
- `get_document_index` returns `dict | None` — matches `_format_document_index(document_index: dict)` caller (None check guards the call) ✓
- `_format_document_index` returns `str` — matches `self._document_outline: str | None` ✓
