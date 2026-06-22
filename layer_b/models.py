from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class IRCell:
    row_index: int
    col_index: int
    row_span: int
    col_span: int
    content: str
    is_row_header: bool
    is_col_header: bool
    # "flag" = from tool, "heuristic" = inferred
    header_source: str = "flag"
    confidence: Optional[float] = None


@dataclass
class QC:
    empty_cell_rate: float = 0.0
    qc_level: str = "ok"
    warnings: list[str] = field(default_factory=list)
    word_avg: Optional[float] = None
    low_confidence_rate: Optional[float] = None
    estimated_info_loss_rate: Optional[float] = None


@dataclass
class IRTable:
    table_id: str
    source_tool: str  # "azure_cu" | "docling" | "azure_di"
    source_pages: list[int]
    cells: list[IRCell]
    qc: QC = field(default_factory=QC)


@dataclass
class IRSection:
    section_id: str
    title: str
    level: int
    page_start: int
    page_end: int
    semantic_type: str
    elements: list[dict]  # 保留原始 element dict（含 entities/document_signals）


@dataclass
class IRDocument:
    doc_id: str          # 從 metadata.doc_id 或 extractor_metadata 生成，fallback: "doc_001"
    source_tool: str     # "vision_llm"
    sections: list[IRSection]
    qc: QC = field(default_factory=QC)


@dataclass
class RetrievalUnit:
    retrieval_unit_id: str
    source_tool: str         # "azure_cu" | "docling" | "azure_di" | "vision_llm"

    # 三層表示（表格路徑三者都有；語意路徑只有 embedding_text 和 structured_json）
    embedding_text: str      # Linearized KV（表格）或 element content（語意）
    structured_json: dict    # to_json() 輸出（表格）或 element dict（語意）
    display_markdown: str    # Markdown 表格（表格）或 section 標題 + content（語意）

    # Quality 欄位
    confidence_level: str    # "high" | "medium" | "low"
    quality_flag: str        # "ok" | "degraded" | "low"
    retrieval_weight: float  # always 1.0 for structured paths; 0.7 for vision_llm (legacy)

    # Metadata
    source_pages: list[int] = field(default_factory=list)

    # Row-level text representations（表格路徑有值；語意路徑為 []）
    row_texts: list[str] = field(default_factory=list)

    # Document-level metadata (applies to all units)
    doc_metadata: dict = field(default_factory=dict)

    # 語意路徑專屬（表格路徑為 None）
    doc_id: Optional[str] = None
    section_id: Optional[str] = None
    section_title: Optional[str] = None
    semantic_type: Optional[str] = None
    page_no: Optional[int] = None
    reading_order: Optional[int] = None
    element_type: Optional[str] = None
    entities: Optional[dict] = None
    document_signals: Optional[list] = None
