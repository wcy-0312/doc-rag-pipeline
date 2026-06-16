"""Integration tests against real Conversion Layer JSON outputs.

These tests use actual medical document JSON files produced by the
Conversion Layer (old format: extractor info in metadata.processing.extractor,
no extractor_metadata field).

File paths are fixed; each test skips gracefully if the file is absent.
"""
from __future__ import annotations

import json

import pytest

# sys.path managed by pytest from repo root


from layer_b.pipeline import process_document, _continuous_weight
from layer_b.adapters import adapt
from layer_b.normalizers.merger import merge_cross_page
from layer_b.normalizers.normalizer import expand_spans
from layer_b.normalizers.header_path import build_header_paths
from layer_b.normalizers.confidence import assess
from layer_b.formatters.formatter import linearize_kv, to_json, to_markdown
from layer_b.models import RetrievalUnit

# ── File paths ────────────────────────────────────────────────────────────────

_OUTPUT_DIR = "/home/wangcy0312/doc-convert-api/output"

THYROID_JSON = os.path.join(
    _OUTPUT_DIR,
    "2026-06-11_06-58-09_甲狀腺癌指引v9.0.pdf.json",
)
PAIN_JSON = os.path.join(
    _OUTPUT_DIR,
    "2026-06-09_07-47-17_護理部_A31000-Q02-W-A14_慢性疼痛照護作業標準.pdf.json",
)
CHECKLIST_JSON = os.path.join(
    _OUTPUT_DIR,
    "2026-06-11_06-58-09_護理部_A31000-Q16-F-B12_急救護理用品基數效期查核表-2.1版.docx.json",
)


def _load(path: str) -> dict:
    """Load JSON or skip the test if the file is missing."""
    if not os.path.exists(path):
        pytest.skip(f"Real-doc fixture not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Test 1: PDF azure_cu — normal document (甲狀腺癌指引) ─────────────────────

def test_1_pdf_azure_cu_normal():
    """甲狀腺癌指引 (26 tables, info_loss=0.0102): table units have expected
    source_tool, weight, and non-empty output fields.

    Note: process_document() now returns table + paragraph units; only table
    units (id starting with 't_') are validated here.
    """
    raw = _load(THYROID_JSON)

    units = process_document(raw)
    table_units = [u for u in units if "rows" in u.structured_json]

    assert isinstance(units, list), "process_document must return a list"
    assert len(table_units) == 23, f"Expected 23 table units after cross-page merge, got {len(table_units)}"

    for u in table_units:
        assert isinstance(u, RetrievalUnit), f"Expected RetrievalUnit, got {type(u)}"
        assert isinstance(u.retrieval_weight, float), "retrieval_weight must be a float"
        # Weight = 1.0 - round(0.0102, 3) = 1.0 - 0.01 = 0.99
        assert 0.9 < u.retrieval_weight <= 1.0, (
            f"Expected weight in (0.9, 1.0] for info_loss=0.0102, got {u.retrieval_weight}"
        )
        assert u.display_markdown, "display_markdown must not be empty"
        assert u.embedding_text, "embedding_text must not be empty"
        assert u.confidence_level in ("high", "medium"), (
            f"Expected high or medium for info_loss=0.0102, got {u.confidence_level!r}"
        )

    # At least one unit must have body rows
    units_with_rows = [u for u in table_units if u.row_texts]
    assert units_with_rows, "At least one table unit should have non-empty row_texts"


# ── Test 2: PDF azure_cu — high info_loss (慢性疼痛照護作業標準) ───────────────

def test_2_pdf_azure_cu_high_info_loss():
    """慢性疼痛照護作業標準 (13 tables, info_loss=0.1158): table units have
    low confidence, quality_flag='low', and retrieval_weight < 0.9.

    Only table units (id starting with 't_') are validated.
    """
    raw = _load(PAIN_JSON)

    units = process_document(raw)
    table_units = [u for u in units if "rows" in u.structured_json]

    assert isinstance(units, list)
    assert len(table_units) == 5, f"Expected 5 table units after cross-page merge, got {len(table_units)}"

    # All tables share the same document-level QC → same weight/level
    # weight = 1.0 - round(0.1158, 3) = 1.0 - 0.116 = 0.884
    expected_weight = pytest.approx(1.0 - round(0.1158, 3), abs=0.001)

    assert any(u.confidence_level == "low" for u in table_units), (
        "At least one table unit should have confidence_level='low' for info_loss=0.1158"
    )
    assert any(u.quality_flag == "low" for u in table_units), (
        "At least one table unit should have quality_flag='low'"
    )
    assert any(u.retrieval_weight < 0.9 for u in table_units), (
        "At least one table unit should have retrieval_weight < 0.9 for info_loss > 0.10"
    )

    for u in table_units:
        assert u.retrieval_weight == expected_weight, (
            f"All table units should share the same weight ({expected_weight}), got {u.retrieval_weight}"
        )


# ── Test 3: DOCX docling (急救護理用品基數效期查核表) ──────────────────────────

def test_3_docx_docling():
    """急救護理用品查核表 (2 tables, DOCX/docling): adapt + normalize pipeline
    returns 2 labelled tables with non-empty formatter output."""
    raw = _load(CHECKLIST_JSON)

    # Old format: use adapt() directly with explicit source_tool to bypass
    # get_source_tool() fallback that would route to azure_content_understanding.
    tables = adapt(raw, "docling")
    tables = merge_cross_page(tables)
    tables = [expand_spans(t) for t in tables]
    labelled = [build_header_paths(t) for t in tables]

    assert len(tables) == 2, f"Expected 2 tables, got {len(tables)}"

    for t in tables:
        assert t.source_tool == "docling", (
            f"source_tool should be 'docling', got {t.source_tool!r}"
        )

    for lt in labelled:
        assert lt.cells, "Labelled table must have cells"
        kv = linearize_kv(lt)
        assert kv, "linearize_kv must return a non-empty string"
        md = to_markdown(lt)
        assert "|" in md, "to_markdown must return a string containing '|'"
        j = to_json(lt)
        assert j["rows"], "to_json must return non-empty rows"


# ── Test 4: row_texts content validation (甲狀腺癌指引) ──────────────────────

def test_4_row_texts_content():
    """row_texts must contain natural language with '為', and count must match
    non-empty rows in structured_json."""
    raw = _load(THYROID_JSON)
    units = process_document(raw)

    # Find the first unit that has body rows in row_texts
    unit = next((u for u in units if u.row_texts), None)
    assert unit is not None, "At least one unit must have non-empty row_texts"

    assert len(unit.row_texts) > 0, "row_texts must be non-empty"

    # Natural language format: key為value — at least one row must use this pattern
    assert any("為" in rt for rt in unit.row_texts), (
        "At least one row_text should contain '為' (key-value NL format)"
    )

    # row_texts count should match rows with cells in structured_json
    rows_with_cells = [r for r in unit.structured_json["rows"] if r.get("cells")]
    assert len(unit.row_texts) == len(rows_with_cells), (
        f"row_texts length ({len(unit.row_texts)}) should equal "
        f"rows-with-cells count ({len(rows_with_cells)})"
    )


# ── Test 5: retrieval_weight continuous value (甲狀腺癌指引) ─────────────────

def test_5_retrieval_weight_continuous():
    """Table units from 甲狀腺癌指引 must have continuous weight from
    document-level info_loss=0.0102 (rounded to 3dp → 0.01 → weight=0.99).

    Only table units (id starting with 't_') are validated; paragraph units
    use a separate weight derivation.
    """
    raw = _load(THYROID_JSON)
    units = process_document(raw)
    table_units = [u for u in units if "rows" in u.structured_json]

    # weight = _continuous_weight(round(0.0102, 3)) = _continuous_weight(0.01) = 0.99
    expected_weight = _continuous_weight(round(0.0102, 3))

    for u in table_units:
        assert isinstance(u.retrieval_weight, float), (
            f"retrieval_weight must be a float, got {type(u.retrieval_weight)}"
        )
        assert u.retrieval_weight == pytest.approx(expected_weight, abs=0.001), (
            f"Expected weight ≈ {expected_weight}, got {u.retrieval_weight}"
        )
