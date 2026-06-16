from __future__ import annotations

import os
from typing import List, Optional


class BGEReranker:
    """Cross-encoder reranker using BAAI/bge-reranker-v2-m3.

    Lazy-loads the FlagReranker model on first call to rerank(), so that
    importing this module does not trigger GPU allocation.

    Reference:
        BGE-reranker-v2-m3 — BAAI (2024).
        https://huggingface.co/BAAI/bge-reranker-v2-m3
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        use_fp16: bool = True,
    ) -> None:
        self.model_name: str = (
            model_name
            or os.environ.get("BGE_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
        )
        self._use_fp16 = use_fp16
        self._model = None  # Lazy loading — not loaded at construction time

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query_text: str,
        candidates: list,
        top_k: int = 10,
    ) -> list:
        """Rerank candidates with the cross-encoder and return top_k results.

        Parameters
        ----------
        query_text:
            The user's search query string.
        candidates:
            List of RankedResult objects produced by HybridRetriever.
        top_k:
            Number of top results to return after reranking.

        Returns
        -------
        List of RankedResult objects with ``rerank_score`` populated,
        sorted by ``rerank_score`` descending, limited to top_k.
        """
        if not candidates:
            return []

        model = self._load_model()

        pairs = [(query_text, c.embedding_text) for c in candidates]
        scores: List[float] = model.compute_score(pairs, normalize=True)

        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = float(score)

        candidates.sort(key=lambda c: c.rerank_score, reverse=True)
        return candidates[:top_k]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        """Return the FlagReranker model, loading it on first call."""
        if self._model is not None:
            return self._model

        from FlagEmbedding import FlagReranker  # noqa: PLC0415 — intentional lazy import

        self._model = FlagReranker(self.model_name, use_fp16=self._use_fp16)
        return self._model
