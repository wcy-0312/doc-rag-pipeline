"""
Public retrieve() interface for Eval layer.

Usage:
    from layer_d.retrieve import retrieve, build_retriever

    # One-shot: auto-ingest all embedded_chunks and search
    results = retrieve("洗手步驟有哪些？", top_k=10)

    # Or: build once, call multiple times (faster for batch evaluation)
    retriever = build_retriever()
    results = retriever.search_text("洗手步驟有哪些？", top_k=10)

Each result is a RankedResult with:
    - chunk_id          : str   — unique chunk identifier
    - retrieval_unit_id : str   — maps to benchmark primary_evidence_unit_ids
    - rerank_score      : float — cross-encoder score (BGE-reranker-v2-m3, normalize=True)
    - final_score       : float — rrf_score × retrieval_weight (pre-rerank ordering score)
    - display_markdown  : str   — LLM-readable content
    - chunk_type        : str   — table / row / paragraph / document
    - source_pages      : List[int]
    - source_tool       : str
"""
from __future__ import annotations

import glob
import json
import logging
import os
from typing import List, Optional

from layer_d.models import EmbeddedChunk, RankedResult
from layer_d.ingestion import DocumentIngester
from layer_d.retrieval import HybridRetriever
from layer_d.reranker import BGEReranker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known document stems and their query-trigger aliases
# ---------------------------------------------------------------------------

KNOWN_DOC_STEMS = [
    "MRI報告_2024",
    "T-Adult-070_鼻胃管灌食",
    "T-Adult-094_鼻噴劑使用方法",
    "中醫護理衛教指導",
    "外院胃鏡報告1",
    "大腸癌診療指引V19.1",
    "護理品質監測",
    "護理部_A31000-Q03-W-A01_腦中風",
    "護理部_A31000-Q04-W-A02_腹部超音波",
    "護理部_A31000-Q05-W-A05_洗手技術",
    "護理部_A31000-Q05-W-A12_肌肉注射技術",
    "護理部_A31000-Q05-W-B08_鼻套管氧氣吸入法",
    "護理部_A31000-Q07-F-006_跌倒的防範及處置評核表",
    "輸血治療同意書-1",
]

# Alias → doc_stem: short substrings that identify a specific document.
# Listed from most-specific to least-specific so that longer aliases match first.
_ALIAS_MAP: List[tuple[str, str]] = [
    ("MRI", "MRI報告_2024"),
    ("鼻胃管", "T-Adult-070_鼻胃管灌食"),
    ("鼻噴劑", "T-Adult-094_鼻噴劑使用方法"),
    ("大腸癌", "大腸癌診療指引V19.1"),
    ("腦中風", "護理部_A31000-Q03-W-A01_腦中風"),
    ("腹部超音波", "護理部_A31000-Q04-W-A02_腹部超音波"),
    ("洗手技術", "護理部_A31000-Q05-W-A05_洗手技術"),
    ("肌肉注射", "護理部_A31000-Q05-W-A12_肌肉注射技術"),
    ("鼻套管", "護理部_A31000-Q05-W-B08_鼻套管氧氣吸入法"),
    ("跌倒", "護理部_A31000-Q07-F-006_跌倒的防範及處置評核表"),
    ("輸血", "輸血治療同意書-1"),
    ("外院胃鏡", "外院胃鏡報告1"),
    ("中醫護理", "中醫護理衛教指導"),
    ("護理品質", "護理品質監測"),
]


def detect_docs(query_text: str) -> List[str]:
    """Return list of doc_stems whose content is referenced in query_text.

    Checks full doc stems first, then alias shortcuts.  Deduplicates results
    while preserving first-seen order.
    """
    seen: set[str] = set()
    detected: List[str] = []

    def _add(stem: str) -> None:
        if stem not in seen:
            seen.add(stem)
            detected.append(stem)

    for stem in KNOWN_DOC_STEMS:
        if stem in query_text:
            _add(stem)

    for alias, stem in _ALIAS_MAP:
        if alias in query_text:
            _add(stem)

    return detected


_EMBEDDED_CHUNKS_DIR = os.getenv(
    "EMBEDDED_CHUNKS_DIR",
    "/home/wangcy0312/doc-chunk-embed-layer/output",
)
_COLLECTION = os.getenv("QDRANT_COLLECTION", "medical_docs")

# Module-level singleton: lazily initialized on first call to retrieve()
_retriever: Optional["RerankedRetriever"] = None


class RerankedRetriever:
    """
    Wraps HybridRetriever + BGEReranker.

    search_text() pipeline:
        1. BGE-M3 hybrid (dense + BM25/sparse) with RRF → top prefetch_k
        2. BGEReranker cross-encoder → fills rerank_score on each result
        3. Sort by rerank_score descending → return top_k
    """

    def __init__(
        self,
        hybrid: HybridRetriever,
        reranker: BGEReranker,
        prefetch_k: int = 50,
    ) -> None:
        self._hybrid = hybrid
        self._reranker = reranker
        self._prefetch_k = prefetch_k

    def search_text(
        self,
        query_text: str,
        top_k: int = 10,
        min_retrieval_weight: float = 0.3,
        filter_chunk_types: Optional[List[str]] = None,
        include_parent_context: bool = False,
    ) -> List[RankedResult]:
        detected_docs = detect_docs(query_text)

        # Baseline: broader prefetch when cross-doc query detected
        prefetch_k = 100 if len(detected_docs) >= 2 else self._prefetch_k
        candidates = self._hybrid.search_text(
            query_text=query_text,
            top_k=prefetch_k,
            prefetch_k=prefetch_k,
            min_retrieval_weight=min_retrieval_weight,
            filter_chunk_types=filter_chunk_types,
            include_parent_context=include_parent_context,
        )

        if len(detected_docs) >= 2:
            for doc_stem in detected_docs:
                doc_filter = self._hybrid.make_doc_filter(doc_stem)
                sub_candidates = self._hybrid.search_text(
                    query_text=query_text,
                    top_k=10,
                    prefetch_k=10,
                    min_retrieval_weight=min_retrieval_weight,
                    filter_chunk_types=filter_chunk_types,
                    include_parent_context=include_parent_context,
                    doc_filter=doc_filter,
                )
                candidates.extend(sub_candidates)

            # Deduplicate: keep highest final_score per chunk_id
            seen: dict[str, RankedResult] = {}
            for c in candidates:
                if c.chunk_id not in seen or c.final_score > seen[c.chunk_id].final_score:
                    seen[c.chunk_id] = c
            candidates = list(seen.values())
            logger.debug(
                "search_text: cross-doc query detected %s docs, %d candidates after dedup",
                detected_docs,
                len(candidates),
            )

        if not candidates:
            return []
        return self._reranker.rerank(query_text, candidates, top_k=top_k)


def build_retriever(
    embedded_chunks_dir: str = _EMBEDDED_CHUNKS_DIR,
    collection_name: str = _COLLECTION,
    reranker_model: Optional[str] = None,
    prefetch_k: int = 50,
) -> RerankedRetriever:
    """
    Build an in-memory Qdrant retriever (with BGEReranker) by ingesting all
    embedded_chunks_*.json.

    Call once before batch evaluation; the returned RerankedRetriever is
    safe to call search_text() on many times.
    """
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")
    ingester = DocumentIngester(client=client, collection_name=collection_name)
    ingester.create_collection_if_not_exists()

    chunk_files = sorted(glob.glob(os.path.join(embedded_chunks_dir, "embedded_chunks_*.json")))
    if not chunk_files:
        raise FileNotFoundError(
            f"No embedded_chunks_*.json found in {embedded_chunks_dir}"
        )

    total_ingested = 0
    for fpath in chunk_files:
        doc_label = os.path.basename(fpath).replace("embedded_chunks_", "").replace(".json", "")
        raw = json.load(open(fpath, encoding="utf-8"))
        chunks = [
            EmbeddedChunk(
                chunk_id=c["chunk_id"],
                chunk_type=c["chunk_type"],
                parent_chunk_id=c.get("parent_chunk_id"),
                embedding_text=c.get("embedding_text", ""),
                structured_json=c.get("structured_json", {}),
                display_markdown=c.get("display_markdown", ""),
                metadata=c.get("metadata", {}),
                retrieval_unit_id=c.get("retrieval_unit_id", c["chunk_id"]),
                vector=c.get("vector", []),
            )
            for c in raw
        ]
        n = ingester.ingest(chunks)
        total_ingested += n
        logger.info("build_retriever: %s → %d chunks ingested", doc_label, n)

    if total_ingested == 0:
        raise RuntimeError(
            "No chunks were ingested — check that embedded_chunks files have non-empty vectors."
        )

    logger.info("build_retriever: total %d points in collection '%s'", total_ingested, collection_name)

    hybrid = HybridRetriever(client=client, collection_name=collection_name)
    reranker = BGEReranker(model_name=reranker_model)
    return RerankedRetriever(hybrid=hybrid, reranker=reranker, prefetch_k=prefetch_k)


def retrieve(
    query: str,
    top_k: int = 10,
    embedded_chunks_dir: str = _EMBEDDED_CHUNKS_DIR,
    collection_name: str = _COLLECTION,
) -> List[RankedResult]:
    """
    Convenience function: auto-initialize retriever on first call (lazy singleton),
    then search. Not safe for concurrent use; use build_retriever() for parallelism.

    Returns top_k RankedResult objects sorted by rerank_score descending.
    Key fields for Eval:
        result.chunk_id          — unique chunk identifier
        result.retrieval_unit_id — maps to benchmark primary_evidence_unit_ids
        result.rerank_score      — cross-encoder score (normalize=True, range 0~1)
        result.final_score       — rrf_score × retrieval_weight
    """
    global _retriever
    if _retriever is None:
        _retriever = build_retriever(
            embedded_chunks_dir=embedded_chunks_dir,
            collection_name=collection_name,
        )
    return _retriever.search_text(query, top_k=top_k)
