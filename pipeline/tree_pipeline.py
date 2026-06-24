"""
TreeRAGPipeline — Tree RAG 路徑的獨立入口。

流程：
  build_tree()    raw document dict → TreeNode 階層 → 存入 Qdrant
  preload_trees() Session 啟動時將靜態樹載入記憶體
  query()         Route → Search → Synthesise → 答案字串
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import fitz

from layer_b.tree_builder import build_tree as _build_tree_from_raw
from layer_d.ingestion import DocumentIngester
from layer_f.tree_store import TreeStore

_VISION_SYSTEM = (
    "你是醫療文件分析助理。根據提供的頁面圖片（包含流程圖、表格）與章節文字，"
    "用繁體中文回答問題。請仔細讀取圖片中的決策流程與條件分支。"
)

_VISION_USER_TEMPLATE = """\
查詢：{query}

【章節文字摘要】
{node_text}

請根據上方章節文字以及以下頁面圖片中的流程圖、表格，完整回答查詢。\
回答中請保留原文的實證等級標注，例如 [I,A]、[II,B]、[V,A]。\
"""


def _sanitise_stem(name: str) -> str:
    return re.sub(r'[^\w\-]', '_', unicodedata.normalize('NFKC', name))


def _node_pages(node) -> list[int]:
    if node.start_page is None:
        return []
    end = node.end_page if node.end_page is not None else node.start_page
    return list(range(node.start_page, end + 1))


class TreeRAGPipeline:
    """Tree RAG 路徑：建樹、載入、查詢。

    Parameters
    ----------
    qdrant_client:
        QdrantClient 實例。
    collection_name:
        Qdrant collection 名稱。
    llm_client:
        實作 generate_text() / generate_text_multimodal() 的 LLM client。
        預設使用 Gemma3Client。
    """

    def __init__(self, qdrant_client, collection_name: str, llm_client=None):
        if llm_client is None:
            from layer_e.llm_client import Gemma3Client
            llm_client = Gemma3Client()
        self._llm = llm_client
        self._qdrant_client = qdrant_client
        self._collection_name = collection_name
        self._ingester = DocumentIngester(
            client=qdrant_client,
            collection_name=collection_name,
        )
        self._tree_store = TreeStore()
        self._pdf_paths: dict[str, str] = {}

    # ── Build ─────────────────────────────────────────────────────────────

    def build_tree(self, raw_document: dict, doc_id: str, pdf_path: str = "") -> None:
        """解析 raw document，建 TreeNode 階層，存入 Qdrant。

        Parameters
        ----------
        raw_document:
            layer_a extractor 輸出的 dict。
        doc_id:
            文件識別符（例如 PDF 檔名）。metadata["file_name"] 存在時以它為準。
        pdf_path:
            PDF 原始檔的絕對路徑，Synthesise 階段渲染頁面圖片時使用。
            Word 文件可省略，自動從 raw_document["data"]["pdf_path_for_render"] 讀取。
        """
        file_name = raw_document.get("metadata", {}).get("file_name") or doc_id
        doc_stem = _sanitise_stem(file_name)

        # Use raw_document's render PDF if no explicit pdf_path given (Word documents)
        if not pdf_path:
            pdf_path = raw_document.get("data", {}).get("pdf_path_for_render") or ""

        self._pdf_paths[doc_stem] = pdf_path

        tree = _build_tree_from_raw(raw_document)
        if tree is None:
            return

        self._ingester.create_collection_if_not_exists()
        self._tree_store.store_static(
            doc_stem, tree, self._qdrant_client, self._collection_name
        )

    # ── Preload ───────────────────────────────────────────────────────────

    def preload_trees(self, doc_ids: list[str]) -> list[str]:
        """Session 啟動：將指定靜態樹從 Qdrant 載入記憶體。

        Parameters
        ----------
        doc_ids:
            build_tree() 時使用的 doc_id 清單。

        Returns
        -------
        list[str]
            成功載入的 doc_stems。
        """
        stems = [_sanitise_stem(d) for d in doc_ids]
        return self._tree_store.preload_static(
            stems, self._qdrant_client, self._collection_name
        )

    # ── Query ─────────────────────────────────────────────────────────────

    def query(
        self,
        query: str,
        session_id: str | None = None,
        vision_dpi: int = 100,
    ) -> str:
        """Route → Search → Synthesise，回傳答案字串。

        Parameters
        ----------
        query:
            自然語言問題。
        session_id:
            若有動態病患樹（build_tree static=False），提供 session_id 可納入病患資料。
        vision_dpi:
            PDF 頁面渲染解析度。

        Returns
        -------
        str
            答案字串。無相關指引或搜尋無結果時回傳空字串。
        """
        from layer_f.tree_router import TreeRouter
        from layer_f.tree_search import TreeSearcher

        # Step 1: Route
        static_summaries = self._tree_store.get_static_summaries()
        if not static_summaries:
            return ""

        dynamic_stems = list(self._tree_store._dynamic.get(session_id or "", {}).keys())
        has_dynamic = bool(session_id and dynamic_stems)
        decision = TreeRouter(self._llm).route(query, static_summaries, has_dynamic)

        # Step 2: Dynamic search (optional)
        patient_context = ""
        if decision.need_patient_context and has_dynamic:
            searcher = TreeSearcher(self._llm)
            parts = []
            for stem in dynamic_stems:
                tree = self._tree_store.load_dynamic(session_id, stem)
                if tree:
                    result = searcher.search(query, tree)
                    for node in result.matched_nodes:
                        if node.content:
                            parts.append(f"【{node.title}】\n{node.content[:1000]}")
            patient_context = "\n\n".join(parts)

        # Step 3: Static tree search
        if not decision.selected_stems:
            return ""

        searcher = TreeSearcher(self._llm)
        effective_query = (
            f"{query}\n\n【病人資料摘要】\n{patient_context}" if patient_context else query
        )
        all_nodes = []
        for stem in decision.selected_stems:
            tree = self._tree_store._static_cache.get(stem)
            if tree:
                result = searcher.search(effective_query, tree)
                all_nodes.extend(result.matched_nodes)

        if not all_nodes:
            return ""

        # Step 4: Synthesise
        return self._synthesise(query, all_nodes, decision.selected_stems, vision_dpi, patient_context)

    # ── Internal ──────────────────────────────────────────────────────────

    def _synthesise(
        self,
        query: str,
        all_nodes: list,
        stems: list[str],
        vision_dpi: int,
        patient_context: str,
    ) -> str:
        from layer_f.page_renderer import render_pages

        pages = sorted({p for n in all_nodes for p in _node_pages(n)})
        all_images: list[bytes] = []
        for stem in stems:
            pdf_path = self._pdf_paths.get(stem)
            if not pdf_path:
                continue
            if pages:
                all_images.extend(render_pages(pdf_path, pages, dpi=vision_dpi))
            else:
                # Limitation: if this query mixes PDF stems (pages known) with Word stems
                # (start_page=None), the Word stem falls into the `if pages:` branch above
                # and attempts to render specific page numbers from the Word render PDF,
                # which may produce incorrect results. This edge case is not yet handled.
                # Word nodes have no page numbers — render all pages (up to 10)
                try:
                    doc = fitz.open(pdf_path)
                    all_pages = list(range(1, min(doc.page_count + 1, 11)))
                    doc.close()
                    all_images.extend(render_pages(pdf_path, all_pages, dpi=vision_dpi))
                except Exception:
                    pass

        if not all_images:
            return ""

        node_text = "\n\n".join(
            f"【{n.title}】\n{n.content[:2000]}"
            for n in all_nodes if n.content
        ) or "（無文字摘要）"

        vision_query = (
            f"{query}\n\n【病人資料摘要】\n{patient_context}"
            if patient_context else query
        )
        prompt = _VISION_USER_TEMPLATE.format(query=vision_query, node_text=node_text)
        return self._llm.generate_text_multimodal(prompt, all_images, system=_VISION_SYSTEM)
