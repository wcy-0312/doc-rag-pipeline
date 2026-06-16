from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EmbeddedChunk:
    chunk_id: str
    chunk_type: str
    parent_chunk_id: Optional[str]
    embedding_text: str
    structured_json: dict
    display_markdown: str
    metadata: dict
    retrieval_unit_id: str
    vector: List[float]


@dataclass
class RankedResult:
    chunk_id: str
    chunk_type: str
    parent_chunk_id: Optional[str]
    retrieval_unit_id: str
    final_score: float
    rrf_score: float
    retrieval_weight: float
    display_markdown: str
    metadata: dict
    source_tool: str
    source_pages: List[int]
    embedding_text: str = ""
    rerank_score: float = 0.0
    page_image_refs: dict = field(default_factory=dict)


@dataclass
class SyntheticQuery:
    query_id: str
    query_text: str
    source_chunk_id: str
    source_doc_id: str


@dataclass
class RelevanceJudgment:
    query_id: str
    chunk_id: str
    score: int  # 0-3


@dataclass
class EvaluationResult:
    ndcg_at_10: float
    per_query_scores: dict          # {query_id: float}
    low_score_queries: list         # query_id list（ndcg < 0.5）
    total_queries: int
