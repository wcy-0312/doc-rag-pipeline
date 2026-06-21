import base64
import pytest
import fitz
from layer_e.pdf_tools import get_full_page_image


@pytest.fixture
def sample_pdf(tmp_path):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Test page 1")
    pdf_path = str(tmp_path / "sample.pdf")
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_returns_valid_base64_png(sample_pdf):
    result = get_full_page_image(sample_pdf, page_no=1)
    assert isinstance(result, str)
    decoded = base64.b64decode(result)
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


def test_invalid_page_raises(sample_pdf):
    with pytest.raises(ValueError, match="out of range"):
        get_full_page_image(sample_pdf, page_no=999)


def test_zero_page_raises(sample_pdf):
    with pytest.raises(ValueError):
        get_full_page_image(sample_pdf, page_no=0)


def test_missing_file_raises():
    with pytest.raises(Exception):
        get_full_page_image("/nonexistent/file.pdf", page_no=1)
