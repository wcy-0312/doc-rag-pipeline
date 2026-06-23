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
from layer_b.tree_builder import build_tree as _build_tree_from_raw
from layer_c.pipeline import process_and_embed
from layer_d.document_registry import DocumentRegistry
from layer_d.ingestion import DocumentIngester
from layer_d.models import RankedResult
from layer_d.reranker import BGEReranker
from layer_d.retrieval import HybridRetriever
from layer_e.agentic_pipeline import AgenticPipeline
from layer_e.llm_client import GPT41Client
from layer_e.pipeline import generate, GenerationPipeline
from layer_f.tree_models import TreeNode
from layer_f.tree_store import TreeStore
from layer_f.tree_search import TreeSearcher

_AGENTIC_SYNTHESIS_SYSTEM = "你是醫療文件分析助理，協助根據診療指引和病人資料回答臨床問題。"

_AGENTIC_SYNTHESIS_TEMPLATE = """\
查詢：{query}

【病人資料摘要】
{patient_context}

【相關診療指引內容】
{guideline_content}

請根據以上資訊，用繁體中文直接回答查詢。說明診療指引的建議是否適用於此病人，並給出具體說明。"""


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
        self._tree_store = TreeStore()
        self._qdrant_client = qdrant_client  # keep reference for tree store
        self._collection_name_ref = collection_name

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

    # ── PageIndex Tree path (F) ───────────────────────────────────────────

    def build_tree(
        self,
        raw_document: dict,
        doc_id: str,
        static: bool = True,
        session_id: str | None = None,
        llm_client=None,
    ) -> TreeNode | None:
        """Build a PageIndexTree from a raw document dict and store it.

        Parameters
        ----------
        raw_document:
            Output from a layer_a extractor (same format as ingest()).
        doc_id:
            Document identifier used as the storage key (e.g. PDF filename stem).
        static:
            True → store in Qdrant (persists across sessions).
            False → store in session memory (session_id required).
        session_id:
            Required when static=False. Identifies the user session.
        llm_client:
            LLM client for generating node summaries. If None, summaries are empty.
        """
        import re as _re_local
        from pathlib import Path as _Path
        file_name = raw_document.get("metadata", {}).get("file_name") or doc_id
        doc_stem = _re_local.sub(r'[^\w\-]', '_', file_name)

        tree = _build_tree_from_raw(raw_document, llm_client=llm_client)
        if tree is None:
            return None

        if static:
            self._ingester.create_collection_if_not_exists()
            self._tree_store.store_static(
                doc_stem, tree, self._qdrant_client, self._collection_name_ref
            )
        else:
            if not session_id:
                raise ValueError("session_id is required for dynamic trees (static=False)")
            self._tree_store.store_dynamic(session_id, doc_stem, tree)

        return tree

    def preload_trees(self, doc_ids: list[str]) -> list[str]:
        """Session startup: eagerly load specified static trees into memory.

        Parameters
        ----------
        doc_ids:
            Full filenames (with extension) passed to build_tree(), e.g.
            ["乳癌診療指引-2026年.pdf", "肺癌診療指引v20.1.pdf"].
            Internally sanitised the same way as build_tree().

        Returns
        -------
        list[str]
            doc_stems that were successfully loaded (missing trees are silently skipped).
        """
        import re as _re_local
        stems = [_re_local.sub(r'[^\w\-]', '_', doc_id) for doc_id in doc_ids]
        return self._tree_store.preload_static(
            stems, self._qdrant_client, self._collection_name_ref
        )

    @staticmethod
    def _tree_nodes_to_ranked_results(nodes: list[TreeNode]) -> list[RankedResult]:
        """Convert TreeNode leaf content into RankedResult for GenerationPipeline."""
        results = []
        for node in nodes:
            pages = (
                list(range(node.start_page, node.end_page + 1))
                if node.start_page is not None and node.end_page is not None
                else []
            )
            display = f"**{node.title}**\n\n{node.content}" if node.title else node.content
            results.append(RankedResult(
                chunk_id=node.node_id,
                chunk_type="paragraph",
                parent_chunk_id=None,
                retrieval_unit_id=node.node_id,
                final_score=1.0,
                rrf_score=1.0,
                retrieval_weight=1.0,
                display_markdown=display,
                metadata={
                    "source_pages": pages,
                    "file_name": "",
                    "confidence_level": "high",
                    "quality_flag": "ok",
                    "source_tool": "page_index_tree",
                    "has_handwriting": False,
                    "excluded_items": [],
                    "patient_id": None,
                    "document_type": None,
                },
                source_tool="page_index_tree",
                source_pages=pages,
                embedding_text=node.content,
            ))
        return results

    def query_tree(
        self,
        query_text: str,
        doc_ids: list[str],
        llm_client=None,
    ):
        """Example 1: Top-down tree traversal on one or more static trees.

        Parameters
        ----------
        query_text:
            Natural language question.
        doc_ids:
            List of doc_stem keys for static trees to search.
        llm_client:
            LLM client for tree traversal. Defaults to self._gen's llm_client.

        Returns
        -------
        GenerationResult (same structure as query())
        """
        import re as _re_local
        _llm = llm_client or self._gen._llm_client

        trees = []
        for doc_id in doc_ids:
            raw_stem = _re_local.sub(r'[^\w\-]', '_', doc_id) if doc_id else doc_id
            tree = self._tree_store.load_static(
                raw_stem, self._qdrant_client, self._collection_name_ref
            )
            if tree is not None:
                trees.append(tree)

        if not trees:
            from layer_e.models import GenerationResult
            return GenerationResult(
                answer="", claims=[], evidence_map={}, unsupported_claims=[],
                abstain=True, abstain_reason="找不到對應的文件樹",
                safety_verdict="abstained", steps_log=[],
            )

        searcher = TreeSearcher(_llm)
        all_nodes = []
        for tree in trees:
            result = searcher.search(query_text, tree)
            all_nodes.extend(result.matched_nodes)

        ranked = self._tree_nodes_to_ranked_results(all_nodes)
        return self._gen.run(query_text, ranked)

    def query_tree_cross(
        self,
        query_text: str,
        guideline_doc_id: str,
        session_id: str,
        patient_doc_stem: str,
        llm_client=None,
    ):
        """Example 2: Cross-tree query — static guideline + dynamic patient record.

        Parameters
        ----------
        query_text:
            Natural language question (e.g. "病人是否符合免疫治療給付？").
        guideline_doc_id:
            doc_stem of the static guideline tree.
        session_id:
            Session identifier for the dynamic patient tree.
        patient_doc_stem:
            doc_stem of the dynamic patient tree.
        llm_client:
            LLM client. Defaults to self._gen's llm_client.

        Returns
        -------
        GenerationResult where .answer contains the cross-tree synthesis.
        """
        import re as _re_local
        from layer_e.models import GenerationResult, ClaimCitation

        _llm = llm_client or self._gen._llm_client
        g_stem = _re_local.sub(r'[^\w\-]', '_', guideline_doc_id) if guideline_doc_id else guideline_doc_id
        p_stem = _re_local.sub(r'[^\w\-]', '_', patient_doc_stem) if patient_doc_stem else patient_doc_stem

        guideline_tree = self._tree_store.load_static(
            g_stem, self._qdrant_client, self._collection_name_ref
        )
        patient_tree = self._tree_store.load_dynamic(session_id, p_stem)

        if guideline_tree is None or patient_tree is None:
            return GenerationResult(
                answer="", claims=[], evidence_map={}, unsupported_claims=[],
                abstain=True,
                abstain_reason=f"缺少必要的文件樹（指引：{guideline_tree is None}，病歷：{patient_tree is None}）",
                safety_verdict="abstained", steps_log=[],
            )

        searcher = TreeSearcher(_llm)
        cross_result = searcher.search_cross(query_text, guideline_tree, patient_tree)

        # Package synthesis as a GenerationResult so the caller has a uniform interface
        return GenerationResult(
            answer=cross_result.synthesis,
            claims=[ClaimCitation(text=cross_result.synthesis, citations=[])],
            evidence_map={},
            unsupported_claims=[],
            abstain=False,
            abstain_reason=None,
            safety_verdict="safe",
            steps_log=[],
        )

    def query_tree_agentic(
        self,
        query: str,
        session_id: str | None = None,
        llm_client=None,
    ):
        """LLM-routed multi-tree query.

        The LLM decides whether patient (dynamic) trees are needed, which static
        guideline trees to search, then synthesises a final answer.

        Parameters
        ----------
        query:
            Natural language question.
        session_id:
            If provided, dynamic trees for this session are available for
            patient-context extraction.
        llm_client:
            LLM client for routing, traversal, and synthesis.
            Defaults to self._gen's llm_client.

        Returns
        -------
        GenerationResult
            Same structure as query(). .abstain is True when no preloaded trees
            exist or the router finds no relevant guideline.
        """
        import re as _re_local
        from layer_e.models import GenerationResult, ClaimCitation
        from layer_f.tree_router import TreeRouter
        from layer_f.tree_search import TreeSearcher

        _llm = llm_client or self._gen._llm_client

        # ── Step 1: Route ─────────────────────────────────────────────────────
        static_summaries = self._tree_store.get_static_summaries()
        if not static_summaries:
            return GenerationResult(
                answer="", claims=[], evidence_map={}, unsupported_claims=[],
                abstain=True, abstain_reason="沒有預載的診療指引，請先呼叫 preload_trees()",
                safety_verdict="abstained", steps_log=[],
            )

        dynamic_stems = list(self._tree_store._dynamic.get(session_id or "", {}).keys())
        has_dynamic = bool(session_id and dynamic_stems)

        decision = TreeRouter(_llm).route(query, static_summaries, has_dynamic)

        # ── Step 2: Dynamic search (optional) ─────────────────────────────────
        patient_context = ""
        if decision.need_patient_context and has_dynamic:
            searcher = TreeSearcher(_llm)
            parts = []
            for stem in dynamic_stems:
                tree = self._tree_store.load_dynamic(session_id, stem)
                if tree:
                    result = searcher.search(query, tree)
                    for node in result.matched_nodes:
                        if node.content:
                            parts.append(f"【{node.title}】\n{node.content[:500]}")
            patient_context = "\n\n".join(parts)

        # ── Step 3: Static tree search ─────────────────────────────────────────
        if not decision.selected_stems:
            return GenerationResult(
                answer="", claims=[], evidence_map={}, unsupported_claims=[],
                abstain=True, abstain_reason="LLM 判斷無相關診療指引可回答此問題",
                safety_verdict="abstained", steps_log=[],
            )

        searcher = TreeSearcher(_llm)
        effective_query = (
            f"{query}\n\n【病人資料摘要】\n{patient_context}" if patient_context else query
        )
        all_static_nodes = []
        for stem in decision.selected_stems:
            tree = self._tree_store._static_cache.get(stem)
            if tree:
                result = searcher.search(effective_query, tree)
                all_static_nodes.extend(result.matched_nodes)

        if not all_static_nodes:
            return GenerationResult(
                answer="", claims=[], evidence_map={}, unsupported_claims=[],
                abstain=True, abstain_reason="靜態樹 Top-down 搜尋無結果",
                safety_verdict="abstained", steps_log=[],
            )

        # ── Step 4: Synthesise ─────────────────────────────────────────────────
        if patient_context:
            guideline_text = "\n\n".join(
                f"【{n.title}】\n{n.content[:1000]}"
                for n in all_static_nodes if n.content
            ) or "（無相關章節）"
            synthesis_prompt = _AGENTIC_SYNTHESIS_TEMPLATE.format(
                query=query,
                patient_context=patient_context[:2000],
                guideline_content=guideline_text,
            )
            answer = _llm.generate_text(synthesis_prompt, system=_AGENTIC_SYNTHESIS_SYSTEM)
            return GenerationResult(
                answer=answer,
                claims=[ClaimCitation(text=answer, citations=[])],
                evidence_map={},
                unsupported_claims=[],
                abstain=False,
                abstain_reason=None,
                safety_verdict="safe",
                steps_log=[],
            )

        ranked = self._tree_nodes_to_ranked_results(all_static_nodes)
        return self._gen.run(query, ranked)
