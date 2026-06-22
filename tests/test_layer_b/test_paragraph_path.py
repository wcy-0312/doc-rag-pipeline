
import pytest
from layer_b.pipeline import (
    _para_has_handwriting,
    _process_checkbox_content,
    _is_formatting_artifact,
    _build_figure_element_set,
    _extract_azure_cu_paragraphs,
    _extract_azure_di_paragraphs,
    _extract_docling_paragraphs,
    _paragraph_path,
    SHORT_DOC_THRESHOLD,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_raw(source_tool: str, data: dict, info_loss: float = 0.05) -> dict:
    return {
        "extractor_metadata": {
            "tool": source_tool,
            "estimated_info_loss_rate": info_loss,
        },
        "data": data,
    }


# ── 1. _para_has_handwriting() ────────────────────────────────────────────────

def test_para_has_handwriting_overlap():
    """span 重疊 → True"""
    para_spans = [{"offset": 10, "length": 20}]  # p: [10, 30)
    styles = [{"is_handwritten": True, "confidence": 0.8, "spans": [{"offset": 15, "length": 5}]}]  # s: [15, 20)
    assert _para_has_handwriting(para_spans, styles) is True


def test_para_has_handwriting_no_overlap():
    """span 不重疊 → False"""
    para_spans = [{"offset": 10, "length": 5}]   # p: [10, 15)
    styles = [{"is_handwritten": True, "confidence": 0.8, "spans": [{"offset": 20, "length": 5}]}]  # s: [20, 25)
    assert _para_has_handwriting(para_spans, styles) is False


def test_para_has_handwriting_low_confidence():
    """confidence < 0.5 → False（即使重疊）"""
    para_spans = [{"offset": 10, "length": 20}]
    styles = [{"is_handwritten": True, "confidence": 0.3, "spans": [{"offset": 15, "length": 5}]}]
    assert _para_has_handwriting(para_spans, styles) is False


def test_para_has_handwriting_empty_styles():
    """styles=[] → False"""
    assert _para_has_handwriting([{"offset": 0, "length": 10}], []) is False


# ── 2. _process_checkbox_content() ───────────────────────────────────────────

def test_process_checkbox_selected_only():
    """:selected: 慢性腎病的中醫認識 → ("慢性腎病的中醫認識", [])"""
    cleaned, excluded = _process_checkbox_content(":selected: 慢性腎病的中醫認識")
    assert cleaned == "慢性腎病的中醫認識"
    assert excluded == []


def test_process_checkbox_unselected_multiple():
    """:unselected: 均衡飲食 :unselected: 其他 → ("", ["均衡飲食", "其他"])"""
    cleaned, excluded = _process_checkbox_content(":unselected: 均衡飲食 :unselected: 其他")
    assert cleaned == ""
    assert excluded == ["均衡飲食", "其他"]


def test_process_checkbox_unselected_two_items():
    """:unselected: 不亂吃藥 :unselected: 自我監測 → ("", ["不亂吃藥", "自我監測"])"""
    cleaned, excluded = _process_checkbox_content(":unselected: 不亂吃藥 :unselected: 自我監測")
    assert cleaned == ""
    assert excluded == ["不亂吃藥", "自我監測"]


def test_process_checkbox_mixed():
    """:selected: 慢性腎病的中醫認識 :unselected: 健康生活型態 → ("慢性腎病的中醫認識", ["健康生活型態"])"""
    cleaned, excluded = _process_checkbox_content(":selected: 慢性腎病的中醫認識 :unselected: 健康生活型態")
    assert cleaned == "慢性腎病的中醫認識"
    assert excluded == ["健康生活型態"]


def test_process_checkbox_no_marker():
    """一般文字段落（無 marker）→ ("一般文字段落", [])"""
    cleaned, excluded = _process_checkbox_content("一般文字段落")
    assert cleaned == "一般文字段落"
    assert excluded == []


# ── 3. _is_formatting_artifact() ─────────────────────────────────────────────

def test_is_formatting_artifact_empty():
    """'' → True"""
    assert _is_formatting_artifact("") is True


def test_is_formatting_artifact_short_non_chinese():
    """'╴╴' → True（len=2，非中文）"""
    assert _is_formatting_artifact("╴╴") is True


def test_is_formatting_artifact_short_with_chinese():
    """'年度：' → False（len=3，含中文）"""
    assert _is_formatting_artifact("年度：") is False


def test_is_formatting_artifact_long_text():
    """'Hello World, this is a long sentence' → False"""
    assert _is_formatting_artifact("Hello World, this is a long sentence") is False


# ── 4. _extract_azure_cu_paragraphs() ────────────────────────────────────────

def test_extract_azure_cu_paragraphs_role_filter():
    """pageHeader 被過濾掉，sectionHeading 和 None role 保留。"""
    data = {
        "paragraphs": [
            {"content": "Page Header Text", "role": "pageHeader",
             "boundingRegions": [{"pageNumber": 1}], "spans": []},
            {"content": "Section One", "role": "sectionHeading",
             "boundingRegions": [{"pageNumber": 1}], "spans": []},
            {"content": "This is a normal paragraph body text here.", "role": None,
             "boundingRegions": [{"pageNumber": 1}], "spans": []},
        ]
    }
    candidates = _extract_azure_cu_paragraphs(data)
    contents = [c["content"] for c in candidates]
    assert "Page Header Text" not in contents
    assert "Section One" in contents
    assert "This is a normal paragraph body text here." in contents


def test_extract_azure_cu_paragraphs_length_filter():
    """role=None 且 content 長度 < MIN_PARA_LEN (12) 的段落被過濾掉；sectionHeading 豁免。"""
    data = {
        "paragraphs": [
            {"content": "短文", "role": None,
             "boundingRegions": [{"pageNumber": 1}], "spans": []},
            # Under threshold (10 chars): filtered out
            {"content": "Stage I-IV", "role": None,
             "boundingRegions": [{"pageNumber": 1}], "spans": []},
            # Meets threshold (14 chars): kept — clinical stage labels must not be dropped
            {"content": "Stage III - IV", "role": None,
             "boundingRegions": [{"pageNumber": 1}], "spans": []},
            # sectionHeading always kept regardless of length
            {"content": "Leads", "role": "sectionHeading",
             "boundingRegions": [{"pageNumber": 1}], "spans": []},
        ]
    }
    candidates = _extract_azure_cu_paragraphs(data)
    contents = [c["content"] for c in candidates]
    assert "短文" not in contents
    assert "Stage I-IV" not in contents
    assert "Stage III - IV" in contents
    assert "Leads" in contents


def test_extract_azure_cu_paragraphs_heading_breadcrumb():
    """sectionHeading 後的 None role 段落，heading_breadcrumb 等於 sectionHeading 的 content。"""
    data = {
        "paragraphs": [
            {"content": "Leads", "role": "sectionHeading",
             "boundingRegions": [{"pageNumber": 5}], "spans": []},
            {"content": "RV Pace/Sense: Medtronic 5076-58 cm", "role": None,
             "boundingRegions": [{"pageNumber": 5}], "spans": []},
        ]
    }
    candidates = _extract_azure_cu_paragraphs(data)
    assert len(candidates) == 2
    assert candidates[0]["heading_breadcrumb"] is None  # sectionHeading 自己不加 breadcrumb
    assert candidates[1]["heading_breadcrumb"] == "Leads"


# ── 4b. _build_figure_element_set() + skip_indices ───────────────────────────

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


# ── 5. _extract_azure_di_paragraphs() ────────────────────────────────────────

def test_extract_azure_di_paragraphs_checkbox():
    """含 :selected: 和 :unselected: 的段落，:unselected: 存入 excluded_items，cleaned content 正確。"""
    data = {
        "paragraphs": [
            {"content": ":selected: 慢性腎病的中醫認識 :unselected: 健康生活型態",
             "boundingRegions": [{"pageNumber": 1}], "spans": []},
        ]
    }
    candidates = _extract_azure_di_paragraphs(data)
    assert len(candidates) == 1
    assert candidates[0]["content"] == "慢性腎病的中醫認識"
    assert candidates[0]["excluded_items"] == ["健康生活型態"]


def test_extract_azure_di_paragraphs_returns_all_including_empty():
    """全 :unselected: 段落的 content="" 仍保留在 candidates，excluded_items 有值。"""
    data = {
        "paragraphs": [
            {"content": ":unselected: 均衡飲食 :unselected: 其他",
             "boundingRegions": [{"pageNumber": 1}], "spans": []},
        ]
    }
    candidates = _extract_azure_di_paragraphs(data)
    assert len(candidates) == 1
    assert candidates[0]["content"] == ""
    assert "均衡飲食" in candidates[0]["excluded_items"]
    assert "其他" in candidates[0]["excluded_items"]


# ── 6. _extract_docling_paragraphs() ─────────────────────────────────────────

def test_extract_docling_paragraphs_artifact_filter():
    """╴╴（artifact）被過濾，正常文字保留。"""
    data = {
        "texts": [
            {"label": "text", "text": "╴╴", "prov": [{"page_no": 1}]},
            {"label": "text", "text": "正常內容文字段落", "prov": [{"page_no": 1}]},
        ]
    }
    candidates = _extract_docling_paragraphs(data)
    contents = [c["content"] for c in candidates]
    assert "╴╴" not in contents
    assert "正常內容文字段落" in contents


def test_extract_docling_paragraphs_section_header_tracking():
    """section_header label 後的 text label，heading_breadcrumb = section_header 的 text。"""
    data = {
        "texts": [
            {"label": "section_header", "text": "第一節：病史", "prov": [{"page_no": 1}]},
            {"label": "text", "text": "患者有高血壓病史多年。", "prov": [{"page_no": 1}]},
        ]
    }
    candidates = _extract_docling_paragraphs(data)
    assert len(candidates) == 2
    assert candidates[0]["label"] == "section_header"
    assert candidates[0]["heading_breadcrumb"] is None
    assert candidates[1]["heading_breadcrumb"] == "第一節：病史"


def test_extract_docling_paragraphs_empty_prov():
    """prov=[] 時（DOCX xml_native 模式的已知行為）page 應為 None，而非 1。

    Docling 2.x 在 Word 格式下不輸出 prov，所以不應 hardcode fallback 到 page=1。
    source_pages=[] 比錯誤的 [1] 更誠實，不會誤導 RAG 定位到不正確頁碼。
    """
    data = {
        "texts": [
            {"label": "text", "text": "護理品質監測指標共三項，針對未達閾值項目進行原因分析。", "prov": []},
            {"label": "section_header", "text": "報告摘要", "prov": []},
        ]
    }
    candidates = _extract_docling_paragraphs(data)
    assert len(candidates) == 2
    assert candidates[0]["page"] is None, "prov=[] 時 page 應為 None，不應 hardcode 為 1"
    assert candidates[1]["page"] is None


# ── 7. _paragraph_path() ─────────────────────────────────────────────────────

def test_paragraph_path_empty_candidates_returns_empty():
    """docling data.texts=[] → 回傳 []（不產生空 document unit）。"""
    raw = _make_raw("docling", {"texts": []})
    units = _paragraph_path(raw, "docling")
    assert units == []


def test_paragraph_path_short_doc_strategy():
    """azure_di，所有過濾後 total_len < 500 → 回傳 1 個 type=document unit。"""
    data = {
        "paragraphs": [
            {"content": ":selected: 慢性腎病 :unselected: 均衡飲食",
             "boundingRegions": [{"pageNumber": 1}], "spans": []},
            {"content": "短文本內容",
             "boundingRegions": [{"pageNumber": 1}], "spans": []},
        ],
        "styles": [],
    }
    raw = _make_raw("azure_di", data)
    units = _paragraph_path(raw, "azure_di")

    assert len(units) == 1
    u = units[0]
    assert u.retrieval_unit_id == "p_azure_di_doc_001"
    assert u.structured_json["type"] == "document"
    assert "均衡飲食" in u.structured_json["excluded_items"]


def test_paragraph_path_normal_paragraphs():
    """azure_cu，total_len >= 500 → 回傳多個 type=paragraph unit，id 格式正確。"""
    long_content = "這是一段很長的段落內容，用來確保超過閾值。" * 10  # ~200 chars each

    data = {
        "paragraphs": [
            {"content": "臨床摘要", "role": "sectionHeading",
             "boundingRegions": [{"pageNumber": 1}], "spans": []},
            {"content": long_content, "role": None,
             "boundingRegions": [{"pageNumber": 1}], "spans": []},
            {"content": long_content, "role": None,
             "boundingRegions": [{"pageNumber": 2}], "spans": []},
            {"content": long_content, "role": None,
             "boundingRegions": [{"pageNumber": 2}], "spans": []},
        ],
        "styles": [],
    }
    raw = _make_raw("azure_cu", data)
    units = _paragraph_path(raw, "azure_cu")

    assert len(units) > 1
    for u in units:
        assert u.structured_json["type"] == "paragraph"
        assert u.retrieval_unit_id.startswith("p_azure_cu_")

    # sectionHeading 後的段落 embedding_text 應包含 heading_breadcrumb
    body_units = [u for u in units if u.structured_json["role"] is None
                  and u.structured_json["heading_breadcrumb"] == "臨床摘要"]
    assert len(body_units) > 0
    for u in body_units:
        assert "臨床摘要" in u.embedding_text


def test_paragraph_path_has_handwriting_flag():
    """azure_di，styles 含重疊的 is_handwritten span → has_handwriting=True。"""
    para_offset = 0
    para_length = 30
    data = {
        "paragraphs": [
            {
                "content": "手寫段落內容，包含手寫文字記錄。",
                "boundingRegions": [{"pageNumber": 1}],
                "spans": [{"offset": para_offset, "length": para_length}],
            },
        ],
        "styles": [
            {
                "is_handwritten": True,
                "confidence": 0.8,
                "spans": [{"offset": 5, "length": 10}],  # overlaps with [0, 30)
            }
        ],
    }
    raw = _make_raw("azure_di", data)
    units = _paragraph_path(raw, "azure_di")

    assert len(units) >= 1
    # Find the unit whose content corresponds to the handwritten paragraph
    hw_units = [u for u in units if u.structured_json.get("has_handwriting") is True]
    assert len(hw_units) >= 1


if __name__ == "__main__":
    test_para_has_handwriting_overlap()
    test_para_has_handwriting_no_overlap()
    test_para_has_handwriting_low_confidence()
    test_para_has_handwriting_empty_styles()
    test_process_checkbox_selected_only()
    test_process_checkbox_unselected_multiple()
    test_process_checkbox_unselected_two_items()
    test_process_checkbox_mixed()
    test_process_checkbox_no_marker()
    test_is_formatting_artifact_empty()
    test_is_formatting_artifact_short_non_chinese()
    test_is_formatting_artifact_short_with_chinese()
    test_is_formatting_artifact_long_text()
    test_extract_azure_cu_paragraphs_role_filter()
    test_extract_azure_cu_paragraphs_length_filter()
    test_extract_azure_cu_paragraphs_heading_breadcrumb()
    test_extract_azure_di_paragraphs_checkbox()
    test_extract_azure_di_paragraphs_returns_all_including_empty()
    test_extract_docling_paragraphs_artifact_filter()
    test_extract_docling_paragraphs_section_header_tracking()
    test_paragraph_path_empty_candidates_returns_empty()
    test_paragraph_path_short_doc_strategy()
    test_paragraph_path_normal_paragraphs()
    test_paragraph_path_has_handwriting_flag()
    print("All paragraph path tests passed.")
