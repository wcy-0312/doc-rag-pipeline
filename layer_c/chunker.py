from __future__ import annotations

import dataclasses
from typing import List

from layer_c.models import EmbeddedChunk


def _build_para_doc_metadata(unit: dict) -> dict:
    doc_meta = unit.get("doc_metadata", {})
    return {
        "source_tool": unit["source_tool"],
        "confidence_level": unit["confidence_level"],
        "quality_flag": unit["quality_flag"],
        "retrieval_weight": unit["retrieval_weight"],
        "source_pages": unit["source_pages"],
        "has_handwriting": unit["structured_json"].get("has_handwriting", False),
        "excluded_items": unit["structured_json"].get("excluded_items", []),
        "patient_id": doc_meta.get("patient_id"),
        "document_type": doc_meta.get("document_type"),
        "file_name": doc_meta.get("file_name", ""),
    }


def _build_table_metadata(unit: dict) -> dict:
    doc_meta = unit.get("doc_metadata", {})
    return {
        "source_tool": unit["source_tool"],
        "confidence_level": unit["confidence_level"],
        "quality_flag": unit["quality_flag"],
        "retrieval_weight": unit["retrieval_weight"],
        "source_pages": unit["source_pages"],
        "patient_id": doc_meta.get("patient_id"),
        "document_type": doc_meta.get("document_type"),
        "file_name": doc_meta.get("file_name", ""),
    }


def retrieval_unit_to_chunks(unit: dict) -> List[EmbeddedChunk]:
    # 通用規則：空 embedding_text 直接跳過
    if unit["embedding_text"] == "":
        return []

    unit_type = unit["structured_json"].get("type") if unit["structured_json"] else None

    if unit_type == "paragraph":
        chunk = EmbeddedChunk(
            chunk_id=unit["retrieval_unit_id"],
            chunk_type="paragraph",
            parent_chunk_id=None,
            embedding_text=unit["embedding_text"],
            structured_json=unit["structured_json"],
            display_markdown=unit["display_markdown"],
            metadata=_build_para_doc_metadata(unit),
            retrieval_unit_id=unit["retrieval_unit_id"],
        )
        return [chunk]

    elif unit_type == "document":
        chunk = EmbeddedChunk(
            chunk_id=unit["retrieval_unit_id"],
            chunk_type="document",
            parent_chunk_id=None,
            embedding_text=unit["embedding_text"],
            structured_json=unit["structured_json"],
            display_markdown=unit["display_markdown"],
            metadata=_build_para_doc_metadata(unit),
            retrieval_unit_id=unit["retrieval_unit_id"],
        )
        return [chunk]

    elif unit_type == "figure":
        chunk = EmbeddedChunk(
            chunk_id=unit["retrieval_unit_id"],
            chunk_type="figure",
            parent_chunk_id=None,
            embedding_text=unit["embedding_text"],
            structured_json=unit["structured_json"],
            display_markdown=unit["display_markdown"],
            metadata=_build_table_metadata(unit),
            retrieval_unit_id=unit["retrieval_unit_id"],
        )
        return [chunk]

    else:
        # table 路由（type 為 None、"table" 或任何其他值）
        table_chunk_type = "element" if unit["source_tool"] == "vision_llm" else "table"

        table_chunk = EmbeddedChunk(
            chunk_id=unit["retrieval_unit_id"],
            chunk_type=table_chunk_type,
            parent_chunk_id=None,
            embedding_text=unit["embedding_text"],
            structured_json=unit["structured_json"],
            display_markdown=unit["display_markdown"],
            metadata=_build_table_metadata(unit),
            retrieval_unit_id=unit["retrieval_unit_id"],
        )

        result: List[EmbeddedChunk] = [table_chunk]

        # Skip row chunks for heavily merged tables — KV linearization breaks when
        # multiple rows share the same header path due to merged cells.
        merge_rate = (unit["structured_json"].get("merge_rate", 0.0)
                      if unit["structured_json"] else 0.0)
        if merge_rate > 0.4:
            return result

        row_texts = unit.get("row_texts", [])
        structured_rows = unit["structured_json"].get("rows") if unit["structured_json"] else None

        for i, row_text in enumerate(row_texts):
            if row_text == "":
                continue

            if structured_rows is not None and i < len(structured_rows):
                row_structured_json = structured_rows[i]
            else:
                row_structured_json = {}

            row_chunk = EmbeddedChunk(
                chunk_id=f"{unit['retrieval_unit_id']}_row{i}",
                chunk_type="row",
                parent_chunk_id=unit["retrieval_unit_id"],
                embedding_text=row_text,
                structured_json=row_structured_json,
                display_markdown=row_text,
                metadata={**_build_table_metadata(unit), "row_index": i},
                retrieval_unit_id=unit["retrieval_unit_id"],
            )
            result.append(row_chunk)

        return result


def chunk_units(units: list) -> list[EmbeddedChunk]:
    """將 RetrievalUnit 物件或 dict 的列表轉換為 EmbeddedChunk 列表。

    layer_b.process_document() 返回 list[RetrievalUnit] dataclass 物件，
    但 retrieval_unit_to_chunks() 使用 dict 語法存取欄位，
    因此在這裡統一轉換為 dict（使用 dataclasses.asdict）。
    """
    result = []
    for unit in units:
        if dataclasses.is_dataclass(unit) and not isinstance(unit, type):
            unit = dataclasses.asdict(unit)
        result.extend(retrieval_unit_to_chunks(unit))
    return result
