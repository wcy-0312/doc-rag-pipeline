from __future__ import annotations

import pytest
from layer_c.chunker import chunk_units


def make_table_unit(retrieval_unit_id="t_000", row_texts=None, embedding_text="Col1: A | Col2: B"):
    """建立 table unit（structured_json 無 type 欄位，模擬真實 MRI 資料）"""
    if row_texts is None:
        row_texts = ["Row 0", "Row 1", "Row 2"]
    return {
        "retrieval_unit_id": retrieval_unit_id,
        "source_tool": "azure_di",
        "embedding_text": embedding_text,
        "structured_json": {
            "rows": [{"cell": f"row{i}"} for i in range(len(row_texts))],
        },
        "display_markdown": "| Col1 | Col2 |\n| A | B |",
        "confidence_level": "high",
        "quality_flag": "ok",
        "retrieval_weight": 1.0,
        "source_pages": [1],
        "page_image_refs": {},
        "row_texts": row_texts,
    }


def make_paragraph_unit(retrieval_unit_id="p_001", embedding_text="Leads\nLead model"):
    """建立 paragraph unit"""
    return {
        "retrieval_unit_id": retrieval_unit_id,
        "source_tool": "docling",
        "embedding_text": embedding_text,
        "structured_json": {
            "type": "paragraph",
            "has_handwriting": False,
            "excluded_items": [],
        },
        "display_markdown": embedding_text,
        "confidence_level": "high",
        "quality_flag": "ok",
        "retrieval_weight": 1.0,
        "source_pages": [1],
        "page_image_refs": {},
        "row_texts": [],
    }


def make_document_unit(retrieval_unit_id="doc_001", embedding_text="病患姓名: 王小明"):
    """建立 document unit，含 excluded_items"""
    return {
        "retrieval_unit_id": retrieval_unit_id,
        "source_tool": "azure_cu",
        "embedding_text": embedding_text,
        "structured_json": {
            "type": "document",
            "has_handwriting": False,
            "excluded_items": ["A", "B"],
        },
        "display_markdown": embedding_text,
        "confidence_level": "medium",
        "quality_flag": "ok",
        "retrieval_weight": 0.8,
        "source_pages": [1, 2],
        "page_image_refs": {},
        "row_texts": [],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_table_unit_produces_table_and_row_chunks():
    unit = make_table_unit(row_texts=["Row 0", "Row 1", "Row 2"])
    chunks = chunk_units([unit])

    assert len(chunks) == 4

    chunk_types = [c.chunk_type for c in chunks]
    assert chunk_types.count("table") == 1
    assert chunk_types.count("row") == 3

    table_chunk = chunks[0]
    assert table_chunk.parent_chunk_id is None

    row_chunks = [c for c in chunks if c.chunk_type == "row"]
    for rc in row_chunks:
        assert rc.parent_chunk_id == table_chunk.chunk_id

    row_ids = [c.chunk_id for c in row_chunks]
    assert row_ids == ["t_000_row0", "t_000_row1", "t_000_row2"]


def test_paragraph_unit_produces_one_chunk():
    unit = make_paragraph_unit()
    chunks = chunk_units([unit])

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_type == "paragraph"
    assert chunk.parent_chunk_id is None
    assert "has_handwriting" in chunk.metadata
    assert "excluded_items" in chunk.metadata
    assert "row_index" not in chunk.metadata


def test_document_unit_produces_one_chunk():
    unit = make_document_unit()
    chunks = chunk_units([unit])

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_type == "document"
    assert chunk.parent_chunk_id is None
    assert chunk.metadata["excluded_items"] == ["A", "B"]
    assert chunk.metadata["has_handwriting"] == False


def test_empty_embedding_text_skipped():
    unit = make_table_unit(embedding_text="")
    chunks = chunk_units([unit])
    assert len(chunks) == 0


def test_empty_row_text_skipped():
    unit = make_table_unit(row_texts=["有內容", "", "也有內容"])
    chunks = chunk_units([unit])

    assert len(chunks) == 3  # 1 table + 2 row

    row_chunks = [c for c in chunks if c.chunk_type == "row"]
    assert len(row_chunks) == 2

    row_ids = [c.chunk_id for c in row_chunks]
    assert row_ids == ["t_000_row0", "t_000_row2"]


def test_table_metadata_no_handwriting_excluded_items():
    unit = make_table_unit()
    chunks = chunk_units([unit])

    table_chunk = chunks[0]
    assert "has_handwriting" not in table_chunk.metadata
    assert "excluded_items" not in table_chunk.metadata


def test_retrieval_unit_id_preserved():
    unit = make_table_unit(retrieval_unit_id="t_xyz", row_texts=["Row A", "Row B"])
    chunks = chunk_units([unit])

    for chunk in chunks:
        assert chunk.retrieval_unit_id == "t_xyz"


def test_multiple_units_mixed():
    table_unit = make_table_unit(row_texts=["Row 0", "Row 1"])
    para_unit = make_paragraph_unit()
    doc_unit = make_document_unit()

    chunks = chunk_units([table_unit, para_unit, doc_unit])

    assert len(chunks) == 5  # 3 (table) + 1 (paragraph) + 1 (document)
