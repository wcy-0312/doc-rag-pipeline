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

from typing import List, Optional

from layer_b.pipeline import process_document
from layer_c.pipeline import process_and_embed
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
            LLM client implementing generate()/generate_multimodal().
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

    # ── Ingestion (A + B + C + D) ─────────────────────────────────────────

    def ingest(self, raw_document: dict) -> int:
        """Structure, embed, and index one document.

        Parameters
        ----------
        raw_document:
            Output from a layer_a extractor (azure_cu, azure_di, docling, llm).

        Returns
        -------
        int
            Number of chunks ingested.
        """
        units = process_document(raw_document)           # B: → list[RetrievalUnit]
        chunks = process_and_embed(units, self._provider) # C: → list[EmbeddedChunk]
        self._ingester.create_collection_if_not_exists()
        return self._ingester.ingest(chunks)              # D: → Qdrant

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

        Returns
        -------
        GenerationResult
            Same structure as query(), plus .steps_log with the agentic trace.
        """
        ranked = self._retriever.search_text(
            query_text, top_k=top_k, prefetch_k=prefetch_k, rerank=rerank
        )
        doc_stem = self._ingester.collection_name  # collection name doubles as doc stem
        agentic = AgenticPipeline(
            llm_client=GPT41Client(),
            retriever=self._retriever,
            pdf_path=pdf_path,
            doc_stem=doc_stem,
        )
        return agentic.run(query_text, ranked)
