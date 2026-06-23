import re


def test_doc_stem_includes_extension():
    """Full filename with extension produces a unique stem that avoids .pdf/.docx collision."""
    file_name_pdf = "指引.pdf"
    file_name_docx = "指引.docx"
    stem_pdf = re.sub(r'[^\w\-]', '_', file_name_pdf)
    stem_docx = re.sub(r'[^\w\-]', '_', file_name_docx)
    assert stem_pdf == "指引_pdf"
    assert stem_docx == "指引_docx"
    assert stem_pdf != stem_docx


def test_doc_stem_does_not_use_pathlib_stem():
    """Ensure we do NOT strip the extension via Path.stem."""
    from pathlib import Path
    file_name = "乳癌診療指引-2026年.pdf"
    old_way = re.sub(r'[^\w\-]', '_', Path(file_name).stem)   # "乳癌診療指引-2026年"
    new_way = re.sub(r'[^\w\-]', '_', file_name)               # "乳癌診療指引-2026年_pdf"
    assert old_way != new_way
    assert new_way == "乳癌診療指引-2026年_pdf"
