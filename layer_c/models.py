from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EmbeddedChunk:
    chunk_id: str
    chunk_type: str              # "table" | "row" | "paragraph" | "document" | "element"
    parent_chunk_id: Optional[str]   # row-level chunk 指向 table-level；其餘為 None
    embedding_text: str
    structured_json: dict
    display_markdown: str
    metadata: dict
    retrieval_unit_id: str
    vector: List[float] = field(default_factory=list)  # filled by pipeline.py after embedding
