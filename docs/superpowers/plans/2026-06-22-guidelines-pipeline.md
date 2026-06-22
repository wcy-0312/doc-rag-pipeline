# 癌症診療指引 QA Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `doc_type="guidelines"` pathway in Layer B that improves retrieval quality for cancer treatment guidelines by (1) tagging evidence levels and (2) merging drug-condition paragraph blocks so that health insurance prerequisites are never separated from their drug entry.

**Architecture:** A thin dispatcher `layer_b/processor.py` wraps the existing generic `layer_b/pipeline.py` and routes to `layer_b/strategies/guidelines.py` when `doc_type="guidelines"`. The guidelines strategy is a pure post-processor: it receives the generic `list[RetrievalUnit]` and returns an augmented one. `pipeline/runner.py` exposes `doc_type` to callers. Layers C/D/E are untouched.

**Tech Stack:** Python 3.11, dataclasses (`replace`), `re`, pytest

## Global Constraints

- All new public functions return `list[RetrievalUnit]`; no new fields added to `RetrievalUnit` — metadata stored in existing `doc_metadata: dict`
- `layer_b/pipeline.py` must not be modified; the strategies directory wraps it
- `doc_type="generic"` must produce output identical to calling `layer_b.pipeline.process_document` directly
- Run `conda run -n hospital-rag python3 -m pytest` to execute tests
- Existing 31 tests in `tests/test_layer_b/test_pipeline.py` must continue to pass after every task
- Merge threshold constant `_MAX_MERGE_CHARS = 2000` (chars, not tokens)
- Evidence level regex pattern: `\[([IVX]+,[A-D])\]` (e.g. `[I,A]`, `[II,B]`, `[III,C]`)
- Insurance keywords list: `("健保申請條件", "健保給付", "健保規範")`

---

### Task 1: Dispatcher + `doc_type` parameter

**Files:**
- Create: `layer_b/strategies/__init__.py`
- Create: `layer_b/processor.py`
- Modify: `pipeline/runner.py:97-133` — `ingest()` signature + import
- Create: `tests/test_layer_b/test_processor.py`

**Interfaces:**
- Produces: `layer_b.processor.process_document(raw: dict, doc_type: str = "generic") -> list[RetrievalUnit]`
- Produces: `layer_b.processor.extract_document_index(raw: dict) -> dict | None`
- `pipeline/runner.py` imports from `layer_b.processor` (replaces `layer_b.pipeline` import for these two functions)

---

- [ ] **Step 1: Write the failing tests**

Create `tests/test_layer_b/test_processor.py`:

```python
import pytest
from layer_b.processor import process_document, extract_document_index
from layer_b.models import RetrievalUnit


def _minimal_raw() -> dict:
    return {
        "extractor_metadata": {"tool": "azure_content_understanding"},
        "metadata": {},
        "data": {
            "paragraphs": [
                {"content": "乳癌診療指引", "role": "sectionHeading",
                 "boundingRegions": [{"pageNumber": 1}], "spans": []},
                {"content": "建議使用trastuzumab。[I,A]",
                 "boundingRegions": [{"pageNumber": 1}], "spans": []},
            ],
            "tables": [],
            "figures": [],
            "sections": [],
        },
    }


def test_generic_doc_type_returns_retrieval_units():
    units = process_document(_minimal_raw(), doc_type="generic")
    assert isinstance(units, list)
    assert all(isinstance(u, RetrievalUnit) for u in units)


def test_guidelines_doc_type_returns_retrieval_units():
    units = process_document(_minimal_raw(), doc_type="guidelines")
    assert isinstance(units, list)
    assert all(isinstance(u, RetrievalUnit) for u in units)


def test_default_doc_type_matches_generic():
    raw = _minimal_raw()
    default_units = process_document(raw)
    generic_units = process_document(raw, doc_type="generic")
    assert len(default_units) == len(generic_units)
    assert [u.retrieval_unit_id for u in default_units] == [
        u.retrieval_unit_id for u in generic_units
    ]


def test_extract_document_index_returns_none_for_empty_sections():
    raw = _minimal_raw()
    assert extract_document_index(raw) is None


def test_unknown_doc_type_falls_back_to_generic():
    raw = _minimal_raw()
    units = process_document(raw, doc_type="unknown_type")
    generic = process_document(raw, doc_type="generic")
    assert len(units) == len(generic)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n hospital-rag python3 -m pytest tests/test_layer_b/test_processor.py -v
```

Expected: `ModuleNotFoundError: No module named 'layer_b.processor'`

- [ ] **Step 3: Create `layer_b/strategies/__init__.py`**

```python
```

(Empty file — marks `layer_b/strategies/` as a package.)

- [ ] **Step 4: Create `layer_b/processor.py`**

```python
from __future__ import annotations
from layer_b.models import RetrievalUnit


def process_document(raw: dict, doc_type: str = "generic") -> list[RetrievalUnit]:
    """Dispatch to the appropriate Layer B processor based on doc_type.

    Parameters
    ----------
    raw:
        Output from a layer_a extractor (azure_cu, docling, etc.).
    doc_type:
        "generic" uses layer_b.pipeline.process_document unchanged.
        "guidelines" applies domain-specific post-processing for clinical
        treatment guidelines (evidence level tagging + drug-block merging).
        Any unrecognised value falls back to "generic".
    """
    from layer_b.pipeline import process_document as _generic
    units = _generic(raw)
    if doc_type == "guidelines":
        from layer_b.strategies.guidelines import post_process
        return post_process(units)
    return units


def extract_document_index(raw: dict) -> dict | None:
    """Thin re-export — always uses the generic implementation."""
    from layer_b.pipeline import extract_document_index as _extract
    return _extract(raw)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
conda run -n hospital-rag python3 -m pytest tests/test_layer_b/test_processor.py -v
```

Expected: 5 passed — but `test_guidelines_doc_type_returns_retrieval_units` will FAIL because `layer_b/strategies/guidelines.py` doesn't exist yet. That's expected at this step; the guidelines strategy is Task 2.

Actually — to avoid a hard import error, create a stub `layer_b/strategies/guidelines.py` now:

```python
# layer_b/strategies/guidelines.py  — stub (Task 2 will fill this in)
from __future__ import annotations
from layer_b.models import RetrievalUnit


def post_process(units: list[RetrievalUnit]) -> list[RetrievalUnit]:
    return list(units)
```

Re-run:

```bash
conda run -n hospital-rag python3 -m pytest tests/test_layer_b/test_processor.py -v
```

Expected: 5 passed

- [ ] **Step 6: Update `pipeline/runner.py`**

Change the import at line 20:

```python
# Before:
from layer_b.pipeline import process_document, extract_document_index

# After:
from layer_b.processor import process_document, extract_document_index
```

Add `doc_type` parameter to `ingest()` (lines 97-133). Replace the method signature and the `process_document` call:

```python
def ingest(
    self,
    raw_document: dict,
    pdf_path: str | None = None,
    doc_id: str | None = None,
    doc_type: str = "generic",
) -> int:
    """Structure, embed, and index one document.

    Parameters
    ----------
    raw_document:
        Output from a layer_a extractor (azure_cu, azure_di, docling, llm).
    pdf_path:
        Optional path to the original PDF file. Used to populate the registry.
    doc_id:
        Optional document identifier (e.g. PDF filename stem). Used as the
        registry key. Both pdf_path and doc_id must be provided for registration.
    doc_type:
        Layer B processing strategy. "generic" (default) or "guidelines".

    Returns
    -------
    int
        Number of chunks ingested.
    """
    units = process_document(raw_document, doc_type=doc_type)  # B: → list[RetrievalUnit]
    chunks = process_and_embed(units, self._provider)           # C: → list[EmbeddedChunk]
    self._ingester.create_collection_if_not_exists()
    n = self._ingester.ingest(chunks)                           # D: → Qdrant
    doc_index = extract_document_index(raw_document)
    if doc_index is not None:
        _stem = Path(raw_document.get("metadata", {}).get("file_name", "")).stem
        _doc_stem = _re.sub(r'[^\w\-]', '_', _stem) if _stem else "doc"
        self._ingester.store_document_index(_doc_stem, doc_index)
    if self._registry is not None and pdf_path and doc_id:
        self._registry.register(
            doc_id, pdf_path, self._ingester.collection_name
        )
    return n
```

- [ ] **Step 7: Verify existing tests still pass**

```bash
conda run -n hospital-rag python3 -m pytest tests/test_layer_b/ tests/test_pipeline/ -q --tb=short
```

Expected: all existing tests pass (processor tests: 5 passed, pipeline tests: 31 passed, runner tests: 3 passed)

- [ ] **Step 8: Commit**

```bash
git add layer_b/strategies/__init__.py layer_b/strategies/guidelines.py \
        layer_b/processor.py pipeline/runner.py \
        tests/test_layer_b/test_processor.py
git commit -m "feat(layer_b): add doc_type dispatcher + guidelines stub"
```

---

### Task 2: Guidelines strategy — evidence levels + drug-block merging

**Files:**
- Modify: `layer_b/strategies/guidelines.py` — replace stub with full implementation
- Create: `tests/test_layer_b/test_guidelines_strategy.py`

**Interfaces:**
- Consumes: `list[RetrievalUnit]` from Task 1's dispatcher
- `post_process(units)` calls `_tag_evidence_levels(units)` then `_merge_drug_blocks(units)`
- `_tag_evidence_levels`: adds `doc_metadata["evidence_levels"]: list[str]` when `[I,A]`-style markers exist
- `_merge_drug_blocks`: merges consecutive same-breadcrumb paragraph units that contain an insurance keyword and fit within `_MAX_MERGE_CHARS`; sets `doc_metadata["merged_paragraph_count"]: int`

---

- [ ] **Step 1: Write the failing tests**

Create `tests/test_layer_b/test_guidelines_strategy.py`:

```python
import pytest
from dataclasses import replace
from layer_b.models import RetrievalUnit
from layer_b.strategies.guidelines import (
    _tag_evidence_levels,
    _merge_drug_blocks,
    post_process,
    _MAX_MERGE_CHARS,
)


def _para(embedding_text: str, breadcrumb: str = "Section A",
          pages: list[int] | None = None) -> RetrievalUnit:
    return RetrievalUnit(
        retrieval_unit_id=f"p_{embedding_text[:8]}",
        source_tool="azure_cu",
        embedding_text=embedding_text,
        structured_json={"type": "paragraph", "heading_breadcrumb": breadcrumb,
                         "content": embedding_text},
        display_markdown=embedding_text,
        confidence_level="high",
        quality_flag="ok",
        retrieval_weight=1.0,
        source_pages=pages or [1],
        doc_metadata={},
    )


def _table() -> RetrievalUnit:
    return RetrievalUnit(
        retrieval_unit_id="t_001",
        source_tool="azure_cu",
        embedding_text="table content",
        structured_json={"type": "table"},
        display_markdown="| col |",
        confidence_level="high",
        quality_flag="ok",
        retrieval_weight=1.0,
        source_pages=[1],
        doc_metadata={},
    )


# ── _tag_evidence_levels ───────────────────────────────────────────────────────

def test_tag_evidence_levels_single_marker():
    unit = _para("建議使用trastuzumab。[I,A]")
    result = _tag_evidence_levels([unit])
    assert result[0].doc_metadata["evidence_levels"] == ["I,A"]


def test_tag_evidence_levels_multiple_markers():
    unit = _para("方案A [I,A] 或方案B [II,B]")
    result = _tag_evidence_levels([unit])
    assert result[0].doc_metadata["evidence_levels"] == ["I,A", "II,B"]


def test_tag_evidence_levels_deduplicates():
    unit = _para("[I,A] 重複 [I,A]")
    result = _tag_evidence_levels([unit])
    assert result[0].doc_metadata["evidence_levels"] == ["I,A"]


def test_tag_evidence_levels_no_marker_unchanged():
    unit = _para("一般描述文字，無實證等級")
    result = _tag_evidence_levels([unit])
    assert "evidence_levels" not in result[0].doc_metadata


def test_tag_evidence_levels_preserves_existing_metadata():
    unit = replace(_para("text [I,A]"), doc_metadata={"existing_key": "value"})
    result = _tag_evidence_levels([unit])
    assert result[0].doc_metadata["existing_key"] == "value"
    assert result[0].doc_metadata["evidence_levels"] == ["I,A"]


# ── _merge_drug_blocks ────────────────────────────────────────────────────────

def test_merge_drug_blocks_merges_insurance_group():
    units = [
        _para("Olaparib 主要說明", breadcrumb="First-line"),
        _para("健保申請條件：須完成6週期化療", breadcrumb="First-line"),
    ]
    result = _merge_drug_blocks(units)
    assert len(result) == 1
    assert "Olaparib 主要說明" in result[0].embedding_text
    assert "健保申請條件" in result[0].embedding_text
    assert result[0].doc_metadata["merged_paragraph_count"] == 2


def test_merge_drug_blocks_no_insurance_keyword_no_merge():
    units = [
        _para("段落一", breadcrumb="Section A"),
        _para("段落二", breadcrumb="Section A"),
    ]
    result = _merge_drug_blocks(units)
    assert len(result) == 2


def test_merge_drug_blocks_different_breadcrumb_no_merge():
    units = [
        _para("健保申請條件段落", breadcrumb="Section A"),
        _para("另一段", breadcrumb="Section B"),
    ]
    result = _merge_drug_blocks(units)
    assert len(result) == 2


def test_merge_drug_blocks_exceeds_threshold_no_merge():
    long_a = "A" * (_MAX_MERGE_CHARS // 2 + 1)
    long_b = "B" * (_MAX_MERGE_CHARS // 2 + 1) + " 健保申請條件"
    units = [
        _para(long_a, breadcrumb="Sec"),
        _para(long_b, breadcrumb="Sec"),
    ]
    result = _merge_drug_blocks(units)
    assert len(result) == 2


def test_merge_drug_blocks_table_is_separator():
    units = [
        _para("Para A 健保申請條件", breadcrumb="Sec"),
        _table(),
        _para("Para B", breadcrumb="Sec"),
    ]
    result = _merge_drug_blocks(units)
    assert len(result) == 3


def test_merge_drug_blocks_merges_source_pages():
    units = [
        _para("主說明 健保給付", breadcrumb="Drug", pages=[5]),
        _para("條件細節", breadcrumb="Drug", pages=[6]),
    ]
    result = _merge_drug_blocks(units)
    assert result[0].source_pages == [5, 6]


def test_merge_drug_blocks_single_unit_group_unchanged():
    units = [_para("健保申請條件：唯一段落", breadcrumb="Sec")]
    result = _merge_drug_blocks(units)
    assert len(result) == 1
    assert "merged_paragraph_count" not in result[0].doc_metadata


def test_post_process_applies_both_operations():
    units = [
        _para("Olaparib [I,A]", breadcrumb="Drug"),
        _para("健保申請條件：6週期", breadcrumb="Drug"),
    ]
    result = post_process(units)
    assert len(result) == 1
    assert result[0].doc_metadata.get("merged_paragraph_count") == 2
    assert result[0].doc_metadata.get("evidence_levels") == ["I,A"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n hospital-rag python3 -m pytest tests/test_layer_b/test_guidelines_strategy.py -v
```

Expected: most tests fail with `ImportError` (functions don't exist in stub yet)

- [ ] **Step 3: Replace stub with full implementation**

Overwrite `layer_b/strategies/guidelines.py`:

```python
from __future__ import annotations
import re
from dataclasses import replace
from layer_b.models import RetrievalUnit

_EVIDENCE_LEVEL_RE = re.compile(r'\[([IVX]+,[A-D])\]')
_INSURANCE_KEYWORDS = ("健保申請條件", "健保給付", "健保規範")
_MAX_MERGE_CHARS = 2000


def post_process(units: list[RetrievalUnit]) -> list[RetrievalUnit]:
    """Apply guidelines-specific post-processing to generic RetrievalUnits.

    Operations (in order):
    1. _tag_evidence_levels  — extract [I,A]-style markers into doc_metadata
    2. _merge_drug_blocks    — merge consecutive same-section paragraphs that
                               contain a health-insurance keyword
    """
    units = _tag_evidence_levels(units)
    units = _merge_drug_blocks(units)
    return units


def _tag_evidence_levels(units: list[RetrievalUnit]) -> list[RetrievalUnit]:
    """Extract evidence level markers and store in doc_metadata["evidence_levels"].

    Pattern: [I,A], [II,B], [III,C], [IV,D] etc. as defined by ESMO/ASCO grading.
    Only units that contain at least one marker are modified.
    """
    result: list[RetrievalUnit] = []
    for u in units:
        levels = _EVIDENCE_LEVEL_RE.findall(u.embedding_text)
        if levels:
            seen: dict[str, None] = {}
            for lvl in levels:
                seen[lvl] = None
            meta = {**u.doc_metadata, "evidence_levels": list(seen)}
            result.append(replace(u, doc_metadata=meta))
        else:
            result.append(u)
    return result


def _merge_drug_blocks(units: list[RetrievalUnit]) -> list[RetrievalUnit]:
    """Merge consecutive paragraph units that form a drug-condition block.

    A group of paragraphs is merged when ALL three conditions hold:
    - All share the same heading_breadcrumb (same leaf section)
    - At least one paragraph contains a health-insurance keyword
    - Combined embedding_text length ≤ _MAX_MERGE_CHARS

    Tables and figures are never merged and act as separators between groups.
    When a group is not merged, units are passed through one at a time.
    """
    result: list[RetrievalUnit] = []
    i = 0
    while i < len(units):
        u = units[i]
        if u.structured_json.get("type") != "paragraph":
            result.append(u)
            i += 1
            continue

        breadcrumb = u.structured_json.get("heading_breadcrumb", "")

        # Collect consecutive same-breadcrumb paragraph units
        group: list[RetrievalUnit] = [u]
        j = i + 1
        while j < len(units):
            nxt = units[j]
            if (nxt.structured_json.get("type") == "paragraph" and
                    nxt.structured_json.get("heading_breadcrumb", "") == breadcrumb):
                group.append(nxt)
                j += 1
            else:
                break

        combined = "\n".join(g.embedding_text for g in group)
        has_insurance = any(kw in combined for kw in _INSURANCE_KEYWORDS)

        if len(group) > 1 and has_insurance and len(combined) <= _MAX_MERGE_CHARS:
            pages = sorted({p for g in group for p in g.source_pages})
            merged_meta = {
                **group[0].doc_metadata,
                "merged_paragraph_count": len(group),
            }
            result.append(replace(
                group[0],
                embedding_text=combined,
                display_markdown="\n\n".join(g.display_markdown for g in group),
                source_pages=pages,
                doc_metadata=merged_meta,
            ))
            i = j
        else:
            result.append(u)
            i += 1

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n hospital-rag python3 -m pytest tests/test_layer_b/test_guidelines_strategy.py -v
```

Expected: 14 passed

- [ ] **Step 5: Verify no regressions**

```bash
conda run -n hospital-rag python3 -m pytest tests/test_layer_b/ tests/test_pipeline/ -q --tb=short
```

Expected: all tests pass (31 pipeline + 5 processor + 14 guidelines strategy + 3 runner = 53 total)

- [ ] **Step 6: Commit**

```bash
git add layer_b/strategies/guidelines.py tests/test_layer_b/test_guidelines_strategy.py
git commit -m "feat(layer_b): guidelines strategy — evidence level tagging + drug-block merging"
```

---

### Task 3: Validate improvement on Q7

**Files:**
- Modify: `scripts/test_batch_qa.py` — delete enriched cache for Q7 re-run, pass `doc_type="guidelines"`

**Interfaces:**
- Consumes: `RAGPipeline.ingest(raw, doc_type="guidelines")` from Task 1
- No new interfaces

---

- [ ] **Step 1: Update `scripts/test_batch_qa.py`**

The script currently calls `ingester.ingest(chunks)` directly (Layer D only). It bypasses `RAGPipeline`. The `doc_type` must be threaded into the Layer B call.

Locate the Layer B section (lines ~96–115) and update the `process_document` import and call:

```python
# Change import (top of file, after other layer_b imports):
from layer_b.processor import process_document  # was: from layer_b.pipeline import process_document

# Change the Layer B call in main() — find where process_document(raw) is called:
# Before:
units = process_document(raw)
# After:
units = process_document(raw, doc_type="guidelines")
```

Also fix the existing bug on line 169 — wrong key for document_index children:

```python
# Before:
children = doc_index.get("children", [])
# After:
children = doc_index.get("sections", [])
```

- [ ] **Step 2: Delete stale enriched cache to force re-processing**

```bash
rm /home/wangcy0312/doc-rag-pipeline/output/layer_b/retrieval_units_乳癌診療指引-2026年_enriched.json
```

The enriched cache was built with the generic processor. Deleting it forces a fresh Layer B + B.5 run with `doc_type="guidelines"`.

The non-enriched cache (`retrieval_units_乳癌診療指引-2026年.json`) is also now stale since it was built with generic processor:

```bash
rm /home/wangcy0312/doc-rag-pipeline/output/layer_b/retrieval_units_乳癌診療指引-2026年.json
```

- [ ] **Step 3: Run only Q7 to validate drug-block merging**

Add a `--q7-only` shortcut by temporarily commenting out batches A, B, C in `BATCHES` and leaving only:

```python
BATCHES = [
    # ... (comment out batches A, B, C temporarily)
    {
        "name": "壓軸：邊界測試",  # rename temporarily
        "questions": [
            "BRCA1/2 胚系突變的三陰性乳癌（TNBC），指引中有哪些額外治療選項？",
        ],
    },
]
```

```bash
conda run -n hospital-rag python3 scripts/test_batch_qa.py 2>&1 | tail -30
```

Expected output should now show olaparib answer with complete conditions listed explicitly (not "需符合特定條件").

- [ ] **Step 4: Restore full BATCHES list and run all 10 questions**

Restore `BATCHES` to original 4-batch list.

```bash
conda run -n hospital-rag python3 scripts/test_batch_qa.py 2>&1 | grep -E "Q[0-9]|報告"
```

Expected: 10 questions answered, report written to `output/eval/batch_qa_乳癌診療指引-2026年.md`

- [ ] **Step 5: Commit**

```bash
git add scripts/test_batch_qa.py
git commit -m "test(batch_qa): use doc_type=guidelines + fix document_index children key"
```
