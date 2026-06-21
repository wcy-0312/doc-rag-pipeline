import base64
import fitz


def get_full_page_image(pdf_path: str, page_no: int, dpi: int = 150) -> str:
    """Return base64-encoded PNG of a single PDF page (1-based page_no)."""
    doc = fitz.open(pdf_path)
    try:
        if page_no < 1 or page_no > len(doc):
            raise ValueError(f"page_no {page_no} out of range 1..{len(doc)}")
        pixmap = doc[page_no - 1].get_pixmap(dpi=dpi)
        png_bytes = pixmap.tobytes("png")
    finally:
        doc.close()
    return base64.b64encode(png_bytes).decode()
