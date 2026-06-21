
from layer_b.adapters import llm_adapter
from layer_b.adapters import adapt as unified_adapt
from layer_b.models import IRDocument, IRSection


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _raw_with_sections():
    return {
        "schema_version": "v3.0",
        "metadata": {
            "doc_id": "test_doc_001",
            "qc": {"qc_level": "ok", "warnings": []},
        },
        "data": {
            "sections": [
                {
                    "section_id": "s001",
                    "title": "疼痛管理",
                    "level": 1,
                    "page_start": 1,
                    "page_end": 2,
                    "semantic_type": "procedure",
                    "elements": [
                        {
                            "element_id": "s001_e001",
                            "type": "text",
                            "page_no": 1,
                            "reading_order": 1,
                            "content": "給藥 PRN Q4H",
                            "entities": {
                                "medications": [{"text": "PRN", "certainty": "high"}]
                            },
                            "document_signals": [
                                {
                                    "signal_type": "dosage",
                                    "basis": "explicit_phrase",
                                    "markers": ["Q4H"],
                                }
                            ],
                        }
                    ],
                },
                {
                    "section_id": "s002",
                    "title": "跌倒預防",
                    "level": 1,
                    "page_start": 3,
                    "page_end": 3,
                    "semantic_type": "assessment",
                    "elements": [],
                },
            ],
        },
        "extractor_metadata": {
            "tool": "vision_llm",
            "model": "claude-sonnet-4-6",
        },
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_llm_adapt_basic():
    """基本輸入，確認回傳 list[IRDocument]，長度為 1，sections 數正確。"""
    raw = _raw_with_sections()
    result = llm_adapter.adapt(raw)
    assert isinstance(result, list)
    assert len(result) == 1
    doc = result[0]
    assert isinstance(doc, IRDocument)
    assert len(doc.sections) == 2


def test_llm_adapt_sections():
    """確認 IRSection 欄位（title, semantic_type, elements）正確對應。"""
    raw = _raw_with_sections()
    doc = llm_adapter.adapt(raw)[0]
    s = doc.sections[0]
    assert isinstance(s, IRSection)
    assert s.section_id == "s001"
    assert s.title == "疼痛管理"
    assert s.level == 1
    assert s.page_start == 1
    assert s.page_end == 2
    assert s.semantic_type == "procedure"
    assert len(s.elements) == 1
    assert s.elements[0]["element_id"] == "s001_e001"


def test_llm_adapt_empty_sections():
    """`data.sections` 為空時，回傳一個 IRDocument，sections 為 []。"""
    raw = {
        "schema_version": "v3.0",
        "data": {"sections": []},
        "extractor_metadata": {"tool": "vision_llm"},
    }
    result = llm_adapter.adapt(raw)
    assert len(result) == 1
    doc = result[0]
    assert isinstance(doc, IRDocument)
    assert doc.sections == []
    assert doc.doc_id == "doc_001"


def test_llm_adapt_qc():
    """qc.qc_level 和 qc.warnings 正確從 metadata.qc 讀取。"""
    raw = {
        "schema_version": "v3.0",
        "metadata": {
            "doc_id": "d001",
            "qc": {"qc_level": "warning", "warnings": ["low_confidence"]},
        },
        "data": {"sections": []},
        "extractor_metadata": {"tool": "vision_llm"},
    }
    doc = llm_adapter.adapt(raw)[0]
    assert doc.qc.qc_level == "warning"
    assert "low_confidence" in doc.qc.warnings


def test_llm_adapt_via_unified():
    """使用統一入口 adapt，source_tool='vision_llm' 時正確路由，回傳 list[IRDocument]。"""
    raw = _raw_with_sections()
    result = unified_adapt(raw, "vision_llm")
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], IRDocument)
    assert result[0].source_tool == "vision_llm"


