"""
RAGPipeline — unified A→E five-layer pipeline runner.

Layer interfaces:
  A (layer_a):  file path → raw dict (call extractor directly, pass result here)
  B (layer_b):  raw dict → list[RetrievalUnit]
  C (layer_c):  list[RetrievalUnit] → list[EmbeddedChunk]
  D (layer_d):  list[EmbeddedChunk] → Qdrant index; query → list[RankedResult]
  E (layer_e):  list[RankedResult] → GenerationResult

Each layer can also be used standalone by importing directly from its package,
e.g.: `from layer_b.pipeline import process_document`
"""
from __future__ import annotations

import re as _re
from pathlib import Path
from typing import List, Optional

from layer_b.pipeline import process_document, extract_document_index
from layer_c.pipeline import process_and_embed
from layer_d.document_registry import DocumentRegistry
from layer_d.ingestion import DocumentIngester
from layer_d.reranker import BGEReranker
from layer_d.retrieval import HybridRetriever
from layer_e.agentic_pipeline import AgenticPipeline
from layer_e.llm_client import GPT41Client
from layer_e.pipeline import generate, GenerationPipeline


class RAGPipeline:
    """Five-layer pipeline: raw document dict → grounded answer.

    Usage
    -----
    1. Prepare a raw document dict from layer_a extractor:
       >>> from layer_a import get_extractor
       >>> raw = get_extractor("azure_cu")(pdf_path, ...)

    2. Instantiate and ingest:
       >>> pipeline = RAGPipeline(provider, qdrant_client, "my_collection")
       >>> pipeline.ingest(raw)

    3. Query:
       >>> result = pipeline.query("請問第一期的治療方式？")
       >>> print(result.answer)
    """

    def __init__(
        self,
        embedding_provider,
        qdrant_client,
        collection_name: str,
        llm_client=None,
        abstention_threshold: float = 0.10,
        reranker=None,
        registry_path: str | None = None,
    ):
        """
        Parameters
        ----------
        embedding_provider:
            Any object implementing EmbeddingProvider (layer_c.providers).
            Recommended: BGEm3Provider from layer_c.providers.bge_m3.
        qdrant_client:
            QdrantClient instance (in-memory or remote).
        collection_name:
            Qdrant collection to use for this document set.
        llm_client:
            LLM client implementing generate().
            Defaults to Gemma3Client (layer_e.llm_client) if None.
        abstention_threshold:
            Minimum rerank_score to answer; below this the pipeline abstains.
            Set to 0.0 when no reranker is deployed.
        reranker:
            BGEReranker instance (or compatible). Defaults to BGEReranker() if None.
        """
        _reranker = reranker if reranker is not None else BGEReranker()
        self._provider = embedding_provider
        self._ingester = DocumentIngester(
            client=qdrant_client,
            collection_name=collection_name,
        )
        self._retriever = HybridRetriever(
            client=qdrant_client,
            collection_name=collection_name,
            reranker=_reranker,
        )
        self._gen = GenerationPipeline(
            llm_client=llm_client,
            abstention_threshold=abstention_threshold,
        )
        self._registry = DocumentRegistry(registry_path) if registry_path else None

    # ── Ingestion (A + B + C + D) ─────────────────────────────────────────

    def ingest(
        self,
        raw_document: dict,
        pdf_path: str | None = None,
        doc_id: str | None = None,
    ) -> int:
        """Structure, embed, and index one document.

        Parameters
        ----------
        raw_document:
            Output from a layer_a extractor (azure_cu, azure_di, docling, llm).
        pdf_path:
            Optional path to the original PDF file. Used to populate the registry.
        doc_id:
            Optional document identifier (e.g. PDF filename stem). Used as the
            registry key. Both pdf_path and doc_id must be provided for registration.

        Returns
        -------
        int
            Number of chunks ingested.
        """
        units = process_document(raw_document)           # B: → list[RetrievalUnit]
        chunks = process_and_embed(units, self._provider) # C: → list[EmbeddedChunk]
        self._ingester.create_collection_if_not_exists()
        n = self._ingester.ingest(chunks)                 # D: → Qdrant
        doc_index = extract_document_index(raw_document)
        if doc_index is not None:
            _stem = Path(raw_document.get("metadata", {}).get("file_name", "")).stem
            _doc_stem = _re.sub(r'[^\w\-]', '_', _stem) if _stem else "doc"
            self._ingester.store_document_index(_doc_stem, doc_index)
        if self._registry is not None and pdf_path and doc_id:
            self._registry.register(
                doc_id, pdf_path, self._ingester.collection_name
            )
        return n

    # ── Query (D + E) ─────────────────────────────────────────────────────

    def query(self, query_text: str, top_k: int = 5, prefetch_k: int = 20, rerank: bool = True):
        """Retrieve evidence and generate a grounded answer.

        Parameters
        ----------
        query_text:
            Natural language question in any language.
        top_k:
            Number of ranked results to pass to generation.
        prefetch_k:
            Number of candidates fetched before RRF fusion.
        rerank:
            Whether to apply the cross-encoder reranker. Default True.

        Returns
        -------
        GenerationResult
            .answer        — grounded answer string
            .claims        — list[ClaimCitation] with evidence IDs
            .evidence_map  — dict mapping evidence ID → chunk metadata
            .abstain       — True if pipeline refused to answer
            .safety_verdict — "safe" | "needs_review" | "abstained"
        """
        ranked = self._retriever.search_text(
            query_text, top_k=top_k, prefetch_k=prefetch_k, rerank=rerank
        )                                                 # D: hybrid retrieval
        return self._gen.run(query_text, ranked)          # E: generation

    def query_agentic(
        self,
        query_text: str,
        pdf_path: str,
        top_k: int = 5,
        prefetch_k: int = 20,
        rerank: bool = True,
        llm_client=None,
    ):
        """Retrieve evidence then run the agentic loop with GPT-4.1 tool calling.

        Parameters
        ----------
        query_text:
            Natural language question.
        pdf_path:
            Absolute path to the original PDF file (for on-demand screenshots).
        top_k, prefetch_k, rerank:
            Same as query().
        llm_client:
            Optional LLM client override. Defaults to GPT41Client() if None.
            Useful for testing with a stub client.

        Returns
        -------
        GenerationResult
            Same structure as query(), plus .steps_log with the agentic trace.
        """
        ranked = self._retriever.search_text(
            query_text, top_k=top_k, prefetch_k=prefetch_k, rerank=rerank
        )
        _raw_stem = Path(pdf_path).stem
        doc_stem = _re.sub(r'[^\w\-]', '_', _raw_stem) if _raw_stem else "doc"
        agentic = AgenticPipeline(
            llm_client=llm_client or GPT41Client(),
            retriever=self._retriever,
            pdf_path=pdf_path,
            doc_stem=doc_stem,
        )
        return agentic.run(query_text, ranked)
