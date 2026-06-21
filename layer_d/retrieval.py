from __future__ import annotations

import os
from typing import List, Optional
from uuid import uuid5, NAMESPACE_DNS

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
    MatchText,
    Prefetch,
    Range,
    SparseVector,
)

from layer_d.models import RankedResult

_COLLECTION = os.getenv("QDRANT_COLLECTION", "medical_docs")
_HOST = os.getenv("QDRANT_HOST", "localhost")
_PORT = int(os.getenv("QDRANT_PORT", "6333"))

_sparse_model = None


def _get_sparse_model():
    global _sparse_model
    if _sparse_model is None:
        from FlagEmbedding import BGEM3FlagModel
        _sparse_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    return _sparse_model


def _encode_query(query_text: str) -> tuple[List[float], dict]:
    """Return (dense_vector, sparse_lexical_weights) for a query string."""
    model = _get_sparse_model()
    out = model.encode(
        [query_text],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense = out["dense_vecs"][0].tolist()
    sparse = out["lexical_weights"][0]
    return dense, sparse


def _chunk_id_to_point_id(chunk_id: str) -> str:
    return str(uuid5(NAMESPACE_DNS, chunk_id))


def _point_to_ranked_result(p, rrf_score: float) -> RankedResult:
    payload = p.payload or {}
    retrieval_weight = float(payload.get("retrieval_weight", 1.0))
    return RankedResult(
        chunk_id=payload.get("chunk_id", str(p.id)),
        chunk_type=payload.get("chunk_type", ""),
        parent_chunk_id=payload.get("parent_chunk_id"),
        retrieval_unit_id=payload.get("retrieval_unit_id", ""),
        final_score=rrf_score * retrieval_weight,
        rrf_score=rrf_score,
        retrieval_weight=retrieval_weight,
        display_markdown=payload.get("display_markdown", ""),
        metadata=payload,
        source_tool=payload.get("source_tool", ""),
        source_pages=payload.get("source_pages", []),
        embedding_text=payload.get("embedding_text", ""),
    )


class HybridRetriever:
    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        collection_name: str = _COLLECTION,
        reranker=None,
    ) -> None:
        self.client = client or QdrantClient(host=_HOST, port=_PORT)
        self.collection_name = collection_name
        self._reranker = reranker

    def _make_doc_filter(self, doc_ids: List[str]) -> Filter:
        """Return a Qdrant Filter that restricts results to chunks from any of the given doc_ids."""
        conditions = [
            FieldCondition(key="chunk_id", match=MatchText(text=f"{stem}__"))
            for stem in doc_ids
        ]
        if len(conditions) == 1:
            return Filter(must=conditions)
        return Filter(should=conditions)  # multiple docs: OR condition

    def make_doc_filter(self, doc_stem: str) -> Filter:
        """Return a Qdrant Filter that restricts results to chunks from doc_stem."""
        return Filter(
            must=[
                FieldCondition(
                    key="chunk_id",
                    match=MatchText(text=f"{doc_stem}__"),
                )
            ]
        )

    def search(
        self,
        query_dense: List[float],
        query_sparse: dict,
        top_k: int = 10,
        prefetch_k: int = 50,
        min_retrieval_weight: float = 0.3,
        filter_chunk_types: Optional[List[str]] = None,
        include_parent_context: bool = False,
        doc_ids: Optional[List[str]] = None,
    ) -> List[RankedResult]:
        """Hybrid dense+sparse search with RRF fusion and retrieval_weight weighting."""
        must_conditions = [
            FieldCondition(
                key="retrieval_weight",
                range=Range(gte=min_retrieval_weight),
            )
        ]
        if filter_chunk_types:
            must_conditions.append(
                FieldCondition(
                    key="chunk_type",
                    match=MatchAny(any=filter_chunk_types),
                )
            )
        base_filter = Filter(must=must_conditions)
        if doc_ids is not None:
            doc_filter = self._make_doc_filter(doc_ids)
            if len(doc_filter.must or []) > 0:
                query_filter = Filter(must=[*base_filter.must, *doc_filter.must])
            elif len(doc_filter.should or []) > 0:
                query_filter = Filter(must=base_filter.must, should=doc_filter.should)
            else:
                query_filter = base_filter
        else:
            query_filter = base_filter

        sparse_indices = [int(k) for k in query_sparse.keys()]
        sparse_values = [float(v) for v in query_sparse.values()]

        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=query_dense,
                    using="dense",
                    limit=prefetch_k,
                    filter=query_filter,
                ),
                Prefetch(
                    query=SparseVector(indices=sparse_indices, values=sparse_values),
                    using="sparse",
                    limit=prefetch_k,
                    filter=query_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        results = [_point_to_ranked_result(p, p.score) for p in response.points]
        results.sort(key=lambda r: r.final_score, reverse=True)

        if include_parent_context:
            results = self._aggregate_parent_context(results)

        return results

    def search_text(
        self,
        query_text: str,
        top_k: int = 10,
        prefetch_k: int = 50,
        min_retrieval_weight: float = 0.3,
        filter_chunk_types: Optional[List[str]] = None,
        include_parent_context: bool = False,
        doc_ids: Optional[List[str]] = None,
        rerank: bool = True,
    ) -> List[RankedResult]:
        """Convenience method: encode query text then call search()."""
        dense, sparse = _encode_query(query_text)
        fetch_k = prefetch_k if (self._reranker and rerank) else top_k
        if prefetch_k > top_k:
            try:
                total = self.client.count(
                    collection_name=self.collection_name, exact=True
                ).count
                prefetch_k = max(top_k, min(prefetch_k, total))
                fetch_k = prefetch_k if (self._reranker and rerank) else top_k
            except Exception:
                pass
        results = self.search(
            query_dense=dense,
            query_sparse=sparse,
            top_k=fetch_k,
            prefetch_k=prefetch_k,
            min_retrieval_weight=min_retrieval_weight,
            filter_chunk_types=filter_chunk_types,
            include_parent_context=include_parent_context,
            doc_ids=doc_ids,
        )
        if self._reranker and rerank:
            results = self._reranker.rerank(query_text, results, top_k=top_k)
        else:
            results = results[:top_k]
        return results

    def _aggregate_parent_context(
        self, results: List[RankedResult]
    ) -> List[RankedResult]:
        """Fetch parent table chunks for any row-level hits not already in results."""
        existing_chunk_ids = {r.chunk_id for r in results}
        parent_chunk_ids_needed = {
            r.parent_chunk_id
            for r in results
            if r.chunk_type == "row" and r.parent_chunk_id
            and r.parent_chunk_id not in existing_chunk_ids
        }
        if not parent_chunk_ids_needed:
            return results

        parent_point_ids = [
            _chunk_id_to_point_id(cid) for cid in parent_chunk_ids_needed
        ]
        parent_points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=parent_point_ids,
            with_payload=True,
        )

        already_chunk_ids = existing_chunk_ids.copy()
        for pp in parent_points:
            payload = pp.payload or {}
            cid = payload.get("chunk_id", str(pp.id))
            if cid in already_chunk_ids:
                continue
            already_chunk_ids.add(cid)
            results.append(
                RankedResult(
                    chunk_id=cid,
                    chunk_type=payload.get("chunk_type", "table"),
                    parent_chunk_id=payload.get("parent_chunk_id"),
                    retrieval_unit_id=payload.get("retrieval_unit_id", ""),
                    final_score=0.0,
                    rrf_score=0.0,
                    retrieval_weight=float(payload.get("retrieval_weight", 1.0)),
                    display_markdown=payload.get("display_markdown", ""),
                    metadata=payload,
                    source_tool=payload.get("source_tool", ""),
                    source_pages=payload.get("source_pages", []),
                )
            )

        return results
