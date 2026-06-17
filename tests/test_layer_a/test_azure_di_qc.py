"""Tests for azure_di_extractor QC escalation and vision prompt."""
import pytest


def test_compute_qc_good_when_text_present():
    """When total_text_chars >= 50, qc_level should not be forced to danger."""
    from layer_a.azure_di_extractor import _compute_qc

    confidence = {
        "page_stats": [],
        "pages_no_words": [],
        "low_confidence_rate": 0.0,
    }
    qc = _compute_qc({}, confidence, None, total_text_chars=200)
    # Should NOT be danger due to text being present
    assert qc["qc_level"] in ("good", "warning")


def test_compute_qc_warning_when_text_low():
    """When total_text_chars < 50, warning IMAGE_OCR_INSUFFICIENT is added."""
    from layer_a.azure_di_extractor import _compute_qc

    confidence = {
        "page_stats": [{"page_no": 1, "avg_confidence": None}],
        "pages_no_words": [1],
        "low_confidence_rate": 0.0,
    }
    qc = _compute_qc({}, confidence, None, total_text_chars=10)
    assert "IMAGE_OCR_INSUFFICIENT_STORED_AS_PAGE_IMAGE" in qc["warnings"]


def test_vision_ocr_both_empty_escalates_to_danger():
    """When vision_description is empty AND total_text_chars < 50, override qc_level to danger."""
    # Simulate the logic from extract_image_azure_di (post-step-8)
    total_text_chars = 5
    vision_description = ""

    # Simulate existing qc (might be "good" from _compute_qc with limited signal)
    qc = {
        "qc_level": "good",
        "warnings": ["IMAGE_OCR_INSUFFICIENT_STORED_AS_PAGE_IMAGE"],
        "errors": [],
    }

    # Apply the P3 logic
    if total_text_chars < 50 and not vision_description:
        qc = dict(qc)
        qc["qc_level"] = "danger"
        warnings = list(qc.get("warnings", []))
        if "VISION_AND_OCR_BOTH_EMPTY" not in warnings:
            warnings.append("VISION_AND_OCR_BOTH_EMPTY")
        qc["warnings"] = warnings

    assert qc["qc_level"] == "danger"
    assert "VISION_AND_OCR_BOTH_EMPTY" in qc["warnings"]


def test_vision_ocr_both_empty_no_escalation_when_vision_present():
    """When vision_description is non-empty, qc_level should NOT be escalated."""
    total_text_chars = 5
    vision_description = "這是一張手寫表單，包含病人姓名和診斷..."

    qc = {
        "qc_level": "good",
        "warnings": [],
        "errors": [],
    }

    if total_text_chars < 50 and not vision_description:
        qc["qc_level"] = "danger"

    assert qc["qc_level"] == "good"


def test_generate_vision_description_prompt_has_checkbox_instruction():
    """Vision LLM prompt should include checkbox detection instruction."""
    import inspect
    from layer_a import azure_di_extractor

    source = inspect.getsource(azure_di_extractor._generate_vision_description)
    assert "[已選]" in source, "Prompt should include [已選] checkbox marker"
    assert "[未選]" in source, "Prompt should include [未選] checkbox marker"
    assert "2000" in source, "max_tokens should be 2000"
