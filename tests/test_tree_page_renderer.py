import io
import pytest
import fitz
from layer_f.page_renderer import render_pages


def _make_pdf(n_pages: int = 3) -> bytes:
    """Create a minimal in-memory PDF with n pages."""
    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 100), f"Page {i + 1} content")
    return doc.tobytes()


def test_render_returns_jpeg_bytes(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(_make_pdf(3))
    images = render_pages(str(pdf_path), [1, 2])
    assert len(images) == 2
    for img in images:
        assert isinstance(img, bytes)
        assert img[:3] == b'\xff\xd8\xff'  # JPEG magic bytes


def test_render_skips_out_of_range(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(_make_pdf(2))
    images = render_pages(str(pdf_path), [1, 99])  # page 99 out of range
    assert len(images) == 1


def test_render_caps_at_10_pages(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(_make_pdf(15))
    images = render_pages(str(pdf_path), list(range(1, 16)))  # 15 pages
    assert len(images) == 10


def test_render_deduplicates_pages(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(_make_pdf(3))
    images = render_pages(str(pdf_path), [1, 1, 2])  # page 1 duplicated
    assert len(images) == 2


def test_render_empty_list(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(_make_pdf(2))
    assert render_pages(str(pdf_path), []) == []
