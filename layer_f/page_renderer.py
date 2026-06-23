from __future__ import annotations

_MAX_PAGES = 10


def render_pages(pdf_path: str, pages: list[int], dpi: int = 150) -> list[bytes]:
    """Render PDF pages to JPEG bytes using pymupdf.

    Args:
        pdf_path: Absolute or relative path to the PDF file.
        pages: 1-based page numbers to render. Duplicates are removed.
               Pages outside the document range are silently skipped.
        dpi: Rendering resolution. 150 is readable for medical flowcharts.

    Returns:
        List of JPEG bytes, one per valid page, capped at _MAX_PAGES.
    """
    import fitz

    unique_pages = sorted(set(p for p in pages if p > 0))[:_MAX_PAGES]
    if not unique_pages:
        return []

    result: list[bytes] = []
    doc = fitz.open(pdf_path)
    try:
        n = doc.page_count
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        for page_num in unique_pages:
            if page_num > n:
                continue
            pix = doc[page_num - 1].get_pixmap(matrix=mat)
            result.append(pix.tobytes("jpeg"))
    finally:
        doc.close()
    return result
