"""Tests for checkbox detection and embedded image trigger in azure_cu_extractor."""
import pytest


def test_detect_checkbox_states_empty_doc():
    """_detect_checkbox_states returns [] for page with no checkboxes."""
    from unittest.mock import MagicMock, patch

    mock_page = MagicMock()
    mock_page.widgets.return_value = []
    mock_page.get_text.return_value = {"blocks": []}
    mock_page.get_drawings.return_value = []

    mock_doc = MagicMock()
    mock_doc.__getitem__ = lambda self, i: mock_page

    from layer_a.azure_cu_extractor import _detect_checkbox_states
    result = _detect_checkbox_states(mock_doc, 1)
    assert result == []


def test_detect_checkbox_states_visual_checked():
    """Visual detection: □ overlapping with black-filled drawing → checked=True."""
    import fitz
    from layer_a.azure_cu_extractor import _detect_checkbox_states

    # Create a real in-memory PDF with a □ and a black filled rect
    doc = fitz.open()
    page = doc.new_page(width=200, height=100)

    # Insert □ character using htmlbox (supports full Unicode, unlike insert_text)
    page.insert_htmlbox(fitz.Rect(40, 40, 100, 60), "□")

    # Locate the □ char bbox from the rendered page to draw the overlap correctly
    rawdict = page.get_text("rawdict")
    sq_bbox = None
    for block in rawdict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                for char in span.get("chars", []):
                    if char.get("c") == "□":
                        sq_bbox = char["bbox"]
                        break

    assert sq_bbox is not None, "□ character was not found in rendered PDF"

    # Draw a black filled rect that overlaps the □
    r = fitz.Rect(sq_bbox)
    page.draw_rect(r, color=(0, 0, 0), fill=(0, 0, 0))

    results = _detect_checkbox_states(doc, 1)
    doc.close()

    # At least one checkbox should be detected
    assert len(results) >= 1
    # The one overlapping with the black rect should be checked
    checked_results = [r for r in results if r["checked"]]
    assert len(checked_results) >= 1


def test_detect_checkbox_states_visual_unchecked():
    """Visual detection: □ with no overlapping drawing → checked=False."""
    import fitz
    from layer_a.azure_cu_extractor import _detect_checkbox_states

    doc = fitz.open()
    page = doc.new_page(width=200, height=100)

    # Insert □ character using htmlbox (supports full Unicode)
    page.insert_htmlbox(fitz.Rect(40, 40, 100, 60), "□")

    # Draw a black filled rect FAR AWAY (no overlap with □ at ~40-51 x range)
    rect = fitz.Rect(150, 80, 170, 100)
    page.draw_rect(rect, color=(0, 0, 0), fill=(0, 0, 0))

    results = _detect_checkbox_states(doc, 1)
    doc.close()

    assert len(results) >= 1
    # None should be checked (no overlap)
    checked = [r for r in results if r["checked"]]
    assert len(checked) == 0


def test_page_image_refs_embedded_image_trigger(tmp_path):
    """_page_image_refs detects pages with large embedded images and returns pdf_path refs."""
    pytest.importorskip("PIL")
    import fitz
    from PIL import Image
    import io
    from layer_a.azure_cu_extractor import _page_image_refs

    # Create a PDF with a large embedded image (> 2.0 sqin)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)  # 8.5×11 inches @ 72dpi

    # Create a 400×300 JPEG image (400/72 × 300/72 ≈ 5.55 × 4.16 = 23.1 sqin)
    img = Image.new("RGB", (400, 300), color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    page.insert_image(fitz.Rect(50, 50, 450, 350), stream=buf.read())

    pdf_path = tmp_path / "test_embedded.pdf"
    doc.save(str(pdf_path))
    doc.close()

    # confidence with no pages_no_words (so trigger 2 doesn't fire)
    confidence = {"pages_no_words": [], "page_stats": []}
    figures = []  # no figures (so trigger 1 doesn't fire)

    result = _page_image_refs(pdf_path, figures, confidence, category="", page_count=1)

    # Page 1 should be detected due to embedded image trigger
    assert 1 in result
    assert result[1]["has_image"] is True
    assert result[1]["source_path"] == str(pdf_path)
    assert result[1]["source_type"] == "pdf"
    assert result[1]["page_no"] == 1
