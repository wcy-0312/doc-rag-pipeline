
import pytest
from layer_b.models import IRCell, IRTable, QC, RetrievalUnit
from layer_b.table import build_header_paths, to_markdown
from layer_b.pipeline import assess
from layer_b.pipeline import process_document, _continuous_weight, _doc_confidence
from layer_b.pipeline import _build_section_path_map, _extract_azure_cu_paragraphs, extract_document_index, _build_figure_element_set


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


# ── Test 5: assess() confidence levels ───────────────────────────────────────

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

    assert assess(_make_table(0.01, 0.95))["level"] == "high"
    assert assess(_make_table(0.05))["level"] == "medium"
    assert assess(_make_table(0.15))["level"] == "low"


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


def test_doc_confidence_weight_always_one_no_qc():
    """PDF path: no qc key → weight must be 1.0."""
    raw = {"metadata": {}}
    level, flag, weight = _doc_confidence(raw)
    assert weight == 1.0
    assert level == "high"
    assert flag == "ok"


def test_doc_confidence_weight_always_one_with_high_loss():
    """Word/DI path: high info_loss → level=low, flag=low, but weight still 1.0."""
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


def test_doc_confidence_fully_scanned_no_qc_key():
    """Fully scanned doc with no qc key must still return level=low, not high."""
    raw = {
        "metadata": {
            "extractor_metadata": {"is_fully_scanned": True},
        }
    }
    level, flag, weight = _doc_confidence(raw)
    assert level == "low"
    assert flag == "low"
    assert weight == 1.0


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


# ── Test: _build_section_path_map (Task 3) ────────────────────────────────────

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



# ── Test: extract_document_index (Task 4) ────────────────────────────────────

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


# ── Fix 1: _doc_confidence handles string estimated_info_loss_rate ────────────

def test_doc_confidence_string_info_loss_no_exception():
    """_doc_confidence must not raise TypeError when info_loss is stored as a string."""
    raw = {"metadata": {"qc": {"estimated_info_loss_rate": "0.15"}}}
    level, flag, weight = _doc_confidence(raw)
    assert level in ("high", "medium", "low")
    assert flag in ("ok", "low")
    assert weight == 1.0
    # "0.15" > _INFO_LOSS_HIGH (0.10) → should be low
    assert level == "low"
    assert flag == "low"


def test_doc_confidence_string_info_loss_medium():
    """String info_loss within medium band returns medium level."""
    raw = {"metadata": {"qc": {"estimated_info_loss_rate": "0.05"}}}
    level, flag, weight = _doc_confidence(raw)
    assert level == "medium"
    assert flag == "ok"


# ── Fix 2: _build_section_path_map cycle detection ────────────────────────────

def test_build_section_path_map_self_referencing_no_recursion():
    """A self-referencing section must not raise RecursionError."""
    # Section 0 references itself as a child → infinite recursion without fix
    sections = [
        {
            "title": "循環節",
            "elements": ["/sections/0", "/paragraphs/0"],
        }
    ]
    result = _build_section_path_map(sections)
    # Must return without crashing; paragraph 0 may or may not be mapped
    assert isinstance(result, dict)


def test_build_section_path_map_mutual_cycle_no_recursion():
    """Sections that reference each other in a cycle must not raise RecursionError."""
    sections = [
        {"title": "A", "elements": ["/sections/1"]},
        {"title": "B", "elements": ["/sections/0", "/paragraphs/5"]},
    ]
    result = _build_section_path_map(sections)
    assert isinstance(result, dict)


# ── Fix 3: extract_document_index cycle detection ────────────────────────────

def test_extract_document_index_self_referencing_no_recursion():
    """A self-referencing section must not raise RecursionError."""
    raw = {
        "data": {
            "sections": [
                {"title": "循環", "elements": ["/sections/0"]},
            ]
        }
    }
    result = extract_document_index(raw)
    # Must return without crashing
    assert result is None or isinstance(result, dict)


def test_extract_document_index_mutual_cycle_no_recursion():
    """Mutually referencing sections must not raise RecursionError."""
    raw = {
        "data": {
            "sections": [
                {"title": "A", "elements": ["/sections/1"]},
                {"title": "B", "elements": ["/sections/0"]},
            ]
        }
    }
    result = extract_document_index(raw)
    assert result is None or isinstance(result, dict)


# ── Fix 4: _build_figure_element_set ignores non-paragraph refs ───────────────

def test_build_figure_element_set_ignores_table_refs():
    """Only /paragraphs/N refs should be included; /tables/N must be ignored."""
    figures = [
        {
            "elements": [
                "/paragraphs/3",
                "/tables/3",
                "/figures/2",
                "/paragraphs/7",
            ]
        }
    ]
    result = _build_figure_element_set(figures)
    assert result == {3, 7}


def test_build_figure_element_set_no_paragraph_refs():
    figures = [{"elements": ["/tables/0", "/figures/1"]}]
    result = _build_figure_element_set(figures)
    assert result == set()


# ── Fix 5: _table_path propagates is_fully_scanned to table quality_flag ──────

def _azure_cu_raw_with_table_and_scanned_flag(cells: list[dict]) -> dict:
    """Build a minimal azure_cu payload with is_fully_scanned=True."""
    return {
        "extractor_metadata": {"tool": "azure_content_understanding"},
        "metadata": {
            "qc": {"empty_cell_rate": 0.0, "qc_level": "ok", "warnings": []},
            "extractor_metadata": {"is_fully_scanned": True},
        },
        "data": {
            "tables": [{"cells": cells}],
            "page_images": {},
        },
    }


def test_table_unit_quality_flag_low_for_fully_scanned():
    """Table units from a fully-scanned document must have quality_flag='low'."""
    cells = [
        {"rowIndex": 0, "columnIndex": 0, "rowSpan": 1, "columnSpan": 1,
         "content": "護理問題", "kind": "columnHeader", "confidence": 0.99,
         "boundingRegions": [{"pageNumber": 1, "polygon": [0, 0, 1, 0, 1, 1, 0, 1]}]},
        {"rowIndex": 1, "columnIndex": 0, "rowSpan": 1, "columnSpan": 1,
         "content": "疼痛管理", "kind": "content", "confidence": 0.95,
         "boundingRegions": [{"pageNumber": 1, "polygon": [0, 1, 1, 1, 1, 2, 0, 2]}]},
    ]
    raw = _azure_cu_raw_with_table_and_scanned_flag(cells)
    units = process_document(raw)

    table_units = [u for u in units if u.structured_json.get("type") == "table"]
    assert len(table_units) >= 1, "Expected at least 1 table unit"
    for u in table_units:
        assert u.quality_flag == "low", (
            f"Expected quality_flag='low' for fully-scanned doc, got {u.quality_flag!r}"
        )
        assert u.confidence_level == "low"


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
