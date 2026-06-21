
import pytest
from layer_b.models import IRCell, IRTable, QC, RetrievalUnit
from layer_b.normalizers.header_path import build_header_paths
from layer_b.pipeline import assess
from layer_b.formatters.formatter import to_markdown
from layer_b.pipeline import process_document, _quality, _continuous_weight


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cell(row, col, content, is_col_header=False, is_row_header=False):
    return IRCell(
        row_index=row, col_index=col, row_span=1, col_span=1,
        content=content, is_col_header=is_col_header, is_row_header=is_row_header,
    )


def _labelled(cells, table_id="t_001"):
    t = IRTable(table_id, "azure_cu", [1], cells, QC())
    return build_header_paths(t)


def _azure_cu_raw(cells: list[dict]) -> dict:
    """Build a minimal Conversion Layer azure_content_understanding payload."""
    return {
        "extractor_metadata": {"tool": "azure_content_understanding"},
        "metadata": {"qc": {"empty_cell_rate": 0.0, "qc_level": "ok", "warnings": []}},
        "data": {
            "tables": [{"cells": cells}],
            "page_images": {"1": "img/p1.png"},
        },
    }


def _vision_llm_raw(sections: list[dict]) -> dict:
    """Build a minimal Conversion Layer vision_llm payload."""
    return {
        "extractor_metadata": {"tool": "vision_llm"},
        "metadata": {"doc_id": "doc_test_001", "qc": {"qc_level": "ok", "warnings": []}},
        "data": {"sections": sections},
    }


# ── Test 1: table path via azure_cu ──────────────────────────────────────────

def test_process_document_table_path():
    """azure_cu payload → 1 RetrievalUnit with expected fields."""
    cells = [
        {"rowIndex": 0, "columnIndex": 0, "rowSpan": 1, "columnSpan": 1,
         "content": "護理問題", "kind": "columnHeader", "confidence": 0.99,
         "boundingRegions": [{"pageNumber": 1, "polygon": [0, 0, 1, 0, 1, 1, 0, 1]}]},
        {"rowIndex": 0, "columnIndex": 1, "rowSpan": 1, "columnSpan": 1,
         "content": "措施", "kind": "columnHeader", "confidence": 0.99,
         "boundingRegions": [{"pageNumber": 1, "polygon": [1, 0, 2, 0, 2, 1, 1, 1]}]},
        {"rowIndex": 1, "columnIndex": 0, "rowSpan": 1, "columnSpan": 1,
         "content": "疼痛管理", "kind": "content", "confidence": 0.95,
         "boundingRegions": [{"pageNumber": 1, "polygon": [0, 1, 1, 1, 1, 2, 0, 2]}]},
        {"rowIndex": 1, "columnIndex": 1, "rowSpan": 1, "columnSpan": 1,
         "content": "給藥PRN", "kind": "content", "confidence": 0.93,
         "boundingRegions": [{"pageNumber": 1, "polygon": [1, 1, 2, 1, 2, 2, 1, 2]}]},
    ]
    raw = _azure_cu_raw(cells)
    units = process_document(raw)

    assert isinstance(units, list)
    assert len(units) == 1

    u = units[0]
    assert isinstance(u, RetrievalUnit)
    assert u.source_tool == "azure_cu"
    assert isinstance(u.embedding_text, str) and len(u.embedding_text) > 0
    assert "|" in u.display_markdown
    assert u.confidence_level in ("high", "medium", "low")
    assert u.quality_flag in ("ok", "low")
    assert isinstance(u.retrieval_weight, float)
    assert 0.0 <= u.retrieval_weight <= 1.0
    assert isinstance(u.row_texts, list)


# ── Test 2: document path via vision_llm ─────────────────────────────────────

def test_process_document_vision_llm():
    """vision_llm payload with 1 section + 2 elements → 2 RetrievalUnits."""
    sections = [
        {
            "section_id": "sec_001",
            "title": "護理評估",
            "level": 1,
            "page_start": 1,
            "page_end": 1,
            "semantic_type": "assessment",
            "elements": [
                {
                    "element_id": "elem_001",
                    "type": "text",
                    "content": "患者主訴疼痛",
                    "page_no": 1,
                    "reading_order": 1,
                    "entities": {"symptom": ["疼痛"]},
                    "document_signals": ["pain_related"],
                },
                {
                    "element_id": "elem_002",
                    "type": "text",
                    "content": "血壓 120/80",
                    "page_no": 1,
                    "reading_order": 2,
                    "entities": {"vital_sign": ["血壓"]},
                    "document_signals": ["vital_signs"],
                },
            ],
        }
    ]
    raw = _vision_llm_raw(sections)
    units = process_document(raw)

    assert len(units) == 2

    for u in units:
        assert isinstance(u, RetrievalUnit)
        assert u.source_tool == "vision_llm"
        assert u.confidence_level == "medium"
        assert u.quality_flag == "ok"
        assert u.retrieval_weight == 0.7
        assert u.entities is not None
        assert u.document_signals is not None
        assert "護理評估" in u.display_markdown
        assert u.row_texts == []


# ── Test 3: to_markdown basic ─────────────────────────────────────────────────

def test_to_markdown_basic():
    """2-column table with header + 1 body row → markdown contains | and ---."""
    cells = [
        _cell(0, 0, "護理問題", is_col_header=True),
        _cell(0, 1, "措施", is_col_header=True),
        _cell(1, 0, "疼痛管理"),
        _cell(1, 1, "給藥PRN"),
    ]
    lt = _labelled(cells)
    md = to_markdown(lt)
    assert "|" in md
    assert "---" in md
    assert "護理問題" in md
    assert "疼痛管理" in md


# ── Test 4: to_markdown empty table ──────────────────────────────────────────

def test_to_markdown_empty():
    """Empty table returns empty string."""
    lt = _labelled([])
    md = to_markdown(lt)
    assert md == ""


# ── Test 5: confidence levels → retrieval_weight mapping ─────────────────────

def test_retrieval_unit_confidence_levels():
    """_quality() correctly maps confidence levels to (flag, weight)."""
    # high confidence
    flag, weight = _quality("high")
    assert weight == 1.0
    assert flag == "ok"

    # medium confidence
    flag, weight = _quality("medium")
    assert weight == 0.7
    assert flag == "ok"

    # low confidence
    flag, weight = _quality("low")
    assert weight == 0.4
    assert flag == "low"


def test_retrieval_unit_confidence_levels_via_assess():
    """assess() with known inputs returns expected levels used by pipeline."""
    from layer_b.models import IRTable, QC, IRCell

    def _make_table(info_loss, word_avg=None):
        c = IRCell(row_index=0, col_index=0, row_span=1, col_span=1,
                   content="x", is_col_header=False, is_row_header=False)
        return IRTable(
            table_id="t_x", source_tool="azure_cu", source_pages=[1],
            cells=[c],
            qc=QC(estimated_info_loss_rate=info_loss, word_avg=word_avg),
        )

    # high: info_loss=0.01, word_avg=0.95
    result = assess(_make_table(0.01, 0.95))
    assert result["level"] == "high"
    flag, weight = _quality(result["level"])
    assert weight == 1.0 and flag == "ok"

    # medium: info_loss=0.05
    result = assess(_make_table(0.05))
    assert result["level"] == "medium"
    flag, weight = _quality(result["level"])
    assert weight == 0.7 and flag == "ok"

    # low: info_loss=0.15
    result = assess(_make_table(0.15))
    assert result["level"] == "low"
    flag, weight = _quality(result["level"])
    assert weight == 0.4 and flag == "low"


# ── Test 6: empty tables list ────────────────────────────────────────────────

def test_process_document_empty_tables():
    """Payload with empty tables list returns empty list without crash."""
    raw = {
        "extractor_metadata": {"tool": "azure_content_understanding"},
        "metadata": {"qc": {"empty_cell_rate": 0.0, "qc_level": "ok", "warnings": []}},
        "data": {
            "tables": [],
            "page_images": {},
        },
    }
    units = process_document(raw)
    assert units == []


# ── Test 7: row_texts generated for table with header + body rows ─────────────

def test_row_texts_generated():
    """Table with 1 header row + 2 body rows → row_texts has 2 non-empty entries."""
    cells = [
        {"rowIndex": 0, "columnIndex": 0, "rowSpan": 1, "columnSpan": 1,
         "content": "護理問題", "kind": "columnHeader", "confidence": 0.99,
         "boundingRegions": [{"pageNumber": 1, "polygon": [0, 0, 1, 0, 1, 1, 0, 1]}]},
        {"rowIndex": 0, "columnIndex": 1, "rowSpan": 1, "columnSpan": 1,
         "content": "措施", "kind": "columnHeader", "confidence": 0.99,
         "boundingRegions": [{"pageNumber": 1, "polygon": [1, 0, 2, 0, 2, 1, 1, 1]}]},
        {"rowIndex": 1, "columnIndex": 0, "rowSpan": 1, "columnSpan": 1,
         "content": "疼痛管理", "kind": "content", "confidence": 0.95,
         "boundingRegions": [{"pageNumber": 1, "polygon": [0, 1, 1, 1, 1, 2, 0, 2]}]},
        {"rowIndex": 1, "columnIndex": 1, "rowSpan": 1, "columnSpan": 1,
         "content": "給藥PRN", "kind": "content", "confidence": 0.93,
         "boundingRegions": [{"pageNumber": 1, "polygon": [1, 1, 2, 1, 2, 2, 1, 2]}]},
        {"rowIndex": 2, "columnIndex": 0, "rowSpan": 1, "columnSpan": 1,
         "content": "感染風險", "kind": "content", "confidence": 0.92,
         "boundingRegions": [{"pageNumber": 1, "polygon": [0, 2, 1, 2, 1, 3, 0, 3]}]},
        {"rowIndex": 2, "columnIndex": 1, "rowSpan": 1, "columnSpan": 1,
         "content": "傷口換藥", "kind": "content", "confidence": 0.91,
         "boundingRegions": [{"pageNumber": 1, "polygon": [1, 2, 2, 2, 2, 3, 1, 3]}]},
    ]
    raw = _azure_cu_raw(cells)
    units = process_document(raw)

    assert len(units) == 1
    u = units[0]
    assert len(u.row_texts) == 2
    for rt in u.row_texts:
        assert isinstance(rt, str) and len(rt) > 0


# ── Test 8: _continuous_weight ────────────────────────────────────────────────

def test_continuous_weight():
    """_continuous_weight() maps info_loss to [0.0, 1.0] correctly."""
    assert _continuous_weight(0.02) == pytest.approx(0.98, abs=0.001)
    assert _continuous_weight(0.15) == pytest.approx(0.85, abs=0.001)
    assert _continuous_weight(None) == 0.7
    assert _continuous_weight(0.0) == 1.0
    assert _continuous_weight(1.5) == 0.0  # clip to 0


# ── Test 9: EmbeddingProvider protocol ───────────────────────────────────────

def test_embedding_provider_protocol():
    """NullEmbeddingProvider satisfies EmbeddingProvider protocol."""
    from layer_b.embedding import EmbeddingProvider, NullEmbeddingProvider
    provider = NullEmbeddingProvider()
    result = provider.embed_texts(["test text"])
    assert result == [[]]
    assert isinstance(provider, EmbeddingProvider)


# ── Test ③b: vision_description prepended to embedding_text ──────────────────

def test_process_document_vision_description_prepended():
    """若 raw 含 vision_description，所有 units 的 embedding_text 應以其為前綴。"""
    cells = [
        {"rowIndex": 0, "columnIndex": 0, "rowSpan": 1, "columnSpan": 1,
         "content": "護理問題", "kind": "columnHeader", "confidence": 0.99,
         "boundingRegions": [{"pageNumber": 1, "polygon": [0, 0, 1, 0, 1, 1, 0, 1]}]},
        {"rowIndex": 1, "columnIndex": 0, "rowSpan": 1, "columnSpan": 1,
         "content": "疼痛管理", "kind": "content", "confidence": 0.95,
         "boundingRegions": [{"pageNumber": 1, "polygon": [0, 1, 1, 1, 1, 2, 0, 2]}]},
    ]
    raw = _azure_cu_raw(cells)
    raw["vision_description"] = "此頁為護理紀錄表格，包含護理問題與對應措施。"

    units = process_document(raw)

    assert len(units) >= 1, "應至少產生 1 個 RetrievalUnit"
    vision_desc = raw["vision_description"]
    for u in units:
        assert vision_desc in u.embedding_text, (
            f"embedding_text 應包含 vision_description，實際為: {u.embedding_text!r}"
        )
        assert u.embedding_text.startswith(vision_desc), (
            f"embedding_text 應以 vision_description 為前綴，實際為: {u.embedding_text!r}"
        )


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


if __name__ == "__main__":
    test_process_document_table_path()
    test_process_document_vision_llm()
    test_to_markdown_basic()
    test_to_markdown_empty()
    test_retrieval_unit_confidence_levels()
    test_retrieval_unit_confidence_levels_via_assess()
    test_process_document_empty_tables()
    test_row_texts_generated()
    test_continuous_weight()
    test_embedding_provider_protocol()
    test_process_document_vision_description_prepended()
    print("All pipeline tests passed.")
