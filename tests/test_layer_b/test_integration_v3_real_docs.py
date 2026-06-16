"""
整合測試：對三份 schema-v3.0 真實文件 JSON 驗證 Structure-aware Layer 端到端行為。

Fixtures（需存在才執行，否則 pytest.skip）：
  - MRI報告_2024.json        → azure_content_understanding，28 tables
  - 中醫護理衛教指導.json    → azure_document_intelligence，0 tables
  - 護理品質監測.json        → docling，2 tables
"""

import json

import pytest

# sys.path managed by pytest from repo root

from layer_b.adapters import get_source_tool
from layer_b.models import RetrievalUnit
from layer_b.pipeline import _continuous_weight, process_document

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

_BASE = "/home/wangcy0312/doc-convert-api/output/for_lead_b"
MRI_JSON = os.path.join(_BASE, "MRI報告_2024.json")
ADI_JSON = os.path.join(_BASE, "中醫護理衛教指導.json")
DOC_JSON = os.path.join(_BASE, "護理品質監測.json")


def _load(path: str) -> dict:
    if not os.path.exists(path):
        pytest.skip(f"fixture not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# test_v3_1_routing
# ---------------------------------------------------------------------------


def test_v3_1_routing():
    """三個檔案的 extractor_metadata.tool 路由到正確的 source tool 字串。"""
    mri = _load(MRI_JSON)
    adi = _load(ADI_JSON)
    doc = _load(DOC_JSON)

    assert get_source_tool(mri) == "azure_content_understanding"
    assert get_source_tool(adi) == "azure_document_intelligence"
    assert get_source_tool(doc) == "docling"


# ---------------------------------------------------------------------------
# test_v3_2_pdf_azure_cu_full
# ---------------------------------------------------------------------------


def test_v3_2_pdf_azure_cu_full():
    """MRI_JSON → process_document() → 28 table units，每個 table unit 基本欄位正確。

    paragraph units 加入後 total unit count 增加；只驗 table units（id 以 't_' 開頭）。
    """
    raw = _load(MRI_JSON)
    units = process_document(raw)
    table_units = [u for u in units if "rows" in u.structured_json]

    assert len(table_units) == 23, f"expected 23 table units after cross-page merge, got {len(table_units)}"

    for u in table_units:
        assert isinstance(u, RetrievalUnit)
        assert u.source_tool == "azure_cu"
        assert isinstance(u.retrieval_weight, float)
        assert 0.9 < u.retrieval_weight <= 1.0, (
            f"retrieval_weight {u.retrieval_weight} not in (0.9, 1.0]"
        )
        assert u.confidence_level in ("high", "medium"), (
            f"unexpected confidence_level: {u.confidence_level}"
        )
        assert u.quality_flag in ("ok", "low"), (
            f"unexpected quality_flag: {u.quality_flag}"
        )
        assert u.display_markdown, "display_markdown should not be empty"
        has_rows = bool(u.structured_json.get("rows"))
        has_embed = bool(u.embedding_text)
        assert has_rows == has_embed, (
            f"unit {u.retrieval_unit_id}: embedding_text and rows should both be "
            f"empty or both non-empty (rows={has_rows}, embedding_text={has_embed})"
        )


# ---------------------------------------------------------------------------
# test_v3_3_pdf_row_texts
# ---------------------------------------------------------------------------


def test_v3_3_pdf_row_texts():
    """MRI_JSON units：至少一個 row_texts 非空；有 row_texts 的 unit，
    len(row_texts) == 非空文字的 rows-with-cells 數量（pipeline 過濾空字串）。"""
    raw = _load(MRI_JSON)
    units = process_document(raw)

    assert any(u.row_texts for u in units), "at least one unit should have row_texts"

    for u in units:
        if u.row_texts:
            rows_with_cells = [
                r for r in u.structured_json.get("rows", []) if r.get("cells")
            ]
            # pipeline 以 _row_to_text() 結果非空為條件過濾，計算非空文字行數
            from layer_b.pipeline import _row_to_text

            non_empty_count = sum(
                1 for r in rows_with_cells if _row_to_text(r)
            )
            assert len(u.row_texts) == non_empty_count, (
                f"unit {u.retrieval_unit_id}: "
                f"row_texts={len(u.row_texts)}, non_empty_rows={non_empty_count}"
            )


# ---------------------------------------------------------------------------
# test_v3_4_pdf_retrieval_weight_continuous
# ---------------------------------------------------------------------------


def test_v3_4_pdf_retrieval_weight_continuous():
    """MRI 所有 table units 的 retrieval_weight 應相同，等於 _continuous_weight(0.013)。

    info_loss=0.0126 → round(0.0126, 3)=0.013 → weight=1.0-0.013=0.987
    paragraph units 使用 0.7 fixed weight，不在此驗證範圍。
    """
    raw = _load(MRI_JSON)
    units = process_document(raw)
    table_units = [u for u in units if "rows" in u.structured_json]

    assert table_units, "should have at least one table unit"
    expected = pytest.approx(_continuous_weight(round(0.0126, 3)), abs=0.001)

    for u in table_units:
        assert u.retrieval_weight == expected, (
            f"unit {u.retrieval_unit_id}: weight={u.retrieval_weight}, expected≈0.987"
        )


# ---------------------------------------------------------------------------
# test_v3_5_azure_di_no_tables
# ---------------------------------------------------------------------------


def test_v3_5_azure_di_no_tables():
    """中醫護理衛教指導.json（0 tables）→ process_document() 不拋例外，table units 為空。

    段落路徑加入後，整份文件觸發短文件策略，產生 1 個 type="document" unit。
    這裡只驗 table units 為空，且至少有 1 個 paragraph/document unit。
    """
    raw = _load(ADI_JSON)

    units = process_document(raw)
    table_units = [u for u in units if "rows" in u.structured_json]
    para_units = [u for u in units if u.structured_json.get("type") in ("paragraph", "document")]

    assert isinstance(units, list), "process_document should return a list"
    assert table_units == [], (
        f"expected no table units for zero-table document, got {len(table_units)}"
    )
    assert para_units, (
        f"expected at least 1 paragraph/document unit from short-doc strategy, got {len(para_units)}"
    )
    doc_unit = para_units[0]
    assert doc_unit.structured_json["type"] == "document", (
        f"expected type='document' for short-doc unit, got {doc_unit.structured_json['type']}"
    )
    assert doc_unit.structured_json["excluded_items"], (
        "excluded_items should list the :unselected: items from 中醫護理衛教指導"
    )


# ---------------------------------------------------------------------------
# test_v3_6_docling_full
# ---------------------------------------------------------------------------


def test_v3_6_docling_full():
    """護理品質監測.json → process_document() → table units 各欄位符合規格。

    段落路徑加入後 total unit count 會增加，這裡只驗 table unit（id 以 't_' 開頭）。
    """
    raw = _load(DOC_JSON)
    units = process_document(raw)

    table_units = [u for u in units if "rows" in u.structured_json]
    assert len(table_units) == 2, f"expected 2 table units, got {len(table_units)}"

    for u in table_units:
        assert u.source_tool == "docling"
        assert isinstance(u.retrieval_weight, float)
        assert 0.0 < u.retrieval_weight <= 1.0, (
            f"retrieval_weight {u.retrieval_weight} out of (0, 1]"
        )
        assert u.confidence_level in ("high", "medium", "low"), (
            f"unexpected confidence_level: {u.confidence_level}"
        )
        assert "|" in u.display_markdown, (
            "display_markdown should contain '|' (Markdown table)"
        )
        assert u.embedding_text, "embedding_text should not be empty"

    assert any(u.row_texts for u in table_units), (
        "at least one docling table unit should have row_texts"
    )


# ---------------------------------------------------------------------------
# test_v3_7_docling_weight
# ---------------------------------------------------------------------------


def test_v3_7_docling_weight():
    """護理品質監測 info_loss=0.0706 → table unit weight = _continuous_weight(0.071) ≈ 0.929。

    只驗 table units（id 以 't_' 開頭）；paragraph units 使用不同的 weight 計算。
    """
    raw = _load(DOC_JSON)
    units = process_document(raw)

    table_units = [u for u in units if "rows" in u.structured_json]
    assert table_units, "should have at least one table unit"

    expected = pytest.approx(_continuous_weight(round(0.0706, 3)), abs=0.001)

    for u in table_units:
        assert u.retrieval_weight == expected, (
            f"unit {u.retrieval_unit_id}: weight={u.retrieval_weight}, expected≈0.929"
        )


# ---------------------------------------------------------------------------
# Slim fixture paths
# ---------------------------------------------------------------------------

SLIM_MRI = os.path.join(_BASE, "slim_MRI報告_2024.json")
SLIM_CSF = os.path.join(_BASE, "slim_CSF外送檢體報告.json")
SLIM_DOC = os.path.join(_BASE, "slim_護理品質監測.json")


# ---------------------------------------------------------------------------
# test_v3_slim_mri
# ---------------------------------------------------------------------------


def test_v3_slim_mri():
    """slim_MRI報告_2024.json（azure_cu，1 table）→ 1 unit，基本欄位正確。"""
    raw = _load(SLIM_MRI)
    units = process_document(raw)

    assert len(units) == 1, f"expected 1 unit, got {len(units)}"

    u = units[0]
    assert u.source_tool == "azure_cu"
    assert u.retrieval_weight == pytest.approx(0.987, abs=0.001), (
        f"retrieval_weight={u.retrieval_weight}, expected≈0.987"
    )
    assert u.confidence_level == "high", (
        f"expected confidence_level='high', got '{u.confidence_level}'"
    )
    assert len(u.row_texts) == 14, (
        f"expected 14 row_texts, got {len(u.row_texts)}"
    )
    assert "|" in u.display_markdown, (
        "display_markdown should contain '|' (Markdown table)"
    )


# ---------------------------------------------------------------------------
# test_v3_slim_csf_azure_di_with_tables
# ---------------------------------------------------------------------------


def test_v3_slim_csf_azure_di_with_tables():
    """slim_CSF外送檢體報告.json（azure_di，1 table）— Azure DI 有表格的第一個整合測試。

    重點驗證 azure_di 表格路徑正確走通：cells[].confidence 為 null 不影響輸出，
    header heuristic（rowIndex == 0）正常運作，boundingRegions 不拋例外。
    """
    raw = _load(SLIM_CSF)
    units = process_document(raw)

    assert len(units) == 1, f"expected 1 unit, got {len(units)}"

    u = units[0]
    assert u.source_tool == "azure_di", (
        f"expected source_tool='azure_di', got '{u.source_tool}'"
    )

    assert isinstance(u.retrieval_weight, float), "retrieval_weight must be float"
    assert 0.0 < u.retrieval_weight <= 1.0, (
        f"retrieval_weight {u.retrieval_weight} out of (0, 1]"
    )

    assert u.confidence_level in ("high", "medium", "low"), (
        f"unexpected confidence_level: '{u.confidence_level}'"
    )

    assert u.display_markdown, "display_markdown should not be empty"
    assert "|" in u.display_markdown, (
        "display_markdown should contain '|' (Markdown table)"
    )

    assert u.embedding_text, "embedding_text should not be empty"

    assert len(u.row_texts) >= 1, (
        f"row_texts should be non-empty, got {len(u.row_texts)}"
    )

    assert u.structured_json.get("rows"), (
        "structured_json['rows'] should be non-empty"
    )


# ---------------------------------------------------------------------------
# test_v3_slim_docling
# ---------------------------------------------------------------------------


def test_v3_slim_docling():
    """slim_護理品質監測.json（docling，1 table）→ 1 unit，基本欄位正確。"""
    raw = _load(SLIM_DOC)
    units = process_document(raw)

    assert len(units) == 1, f"expected 1 unit, got {len(units)}"

    u = units[0]
    assert u.source_tool == "docling", (
        f"expected source_tool='docling', got '{u.source_tool}'"
    )
    assert u.retrieval_weight == pytest.approx(0.929, abs=0.001), (
        f"retrieval_weight={u.retrieval_weight}, expected≈0.929"
    )
    assert "|" in u.display_markdown, (
        "display_markdown should contain '|' (Markdown table)"
    )
    assert len(u.row_texts) == 3, (
        f"expected 3 row_texts, got {len(u.row_texts)}"
    )
