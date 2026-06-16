from __future__ import annotations

import os
import re
from math import log2
from typing import List, Optional
from uuid import uuid4

from layer_d.models import EvaluationResult, SyntheticQuery

# ---------------------------------------------------------------------------
# LLM client helpers
# ---------------------------------------------------------------------------

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-sonnet-4-6")


def _get_llm_client():
    if JUDGE_MODEL.startswith("claude"):
        import anthropic
        return anthropic.Anthropic()
    else:
        import openai
        return openai.OpenAI(
            base_url=os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1"),
            api_key="local",
        )


def _call_llm(client, prompt: str) -> str:
    if hasattr(client, "messages"):   # anthropic
        resp = client.messages.create(
            model=JUDGE_MODEL, max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.content[0].text.strip()
    else:                             # openai-compatible
        resp = client.chat.completions.create(
            model=JUDGE_MODEL, max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# SyntheticQueryGenerator
# ---------------------------------------------------------------------------

class SyntheticQueryGenerator:
    """Generate synthetic clinical queries from EmbeddedChunk objects.

    Uses LLM-as-generator (Reverse RAG) to produce query-document pairs
    for offline retrieval evaluation. [1]

    [1] Ma et al., "Zero-Shot Listwise Document Reranking with a Large
        Language Model", arXiv 2023.
    """

    def __init__(self, llm_client=None) -> None:
        self._client = llm_client or _get_llm_client()

    def generate(
        self,
        chunks: list,
        target_count: int = 35,
        seed_ids: Optional[List[str]] = None,
    ) -> List[SyntheticQuery]:
        """Generate up to *target_count* synthetic queries from *chunks*.

        Parameters
        ----------
        chunks:
            List of EmbeddedChunk objects to draw queries from.
        target_count:
            Stop after collecting this many valid queries.
        seed_ids:
            If provided, only attempt generation from chunks whose
            ``chunk_id`` is in this list.
        """
        results: List[SyntheticQuery] = []

        for chunk in chunks:
            if len(results) >= target_count:
                break
            if seed_ids is not None and chunk.chunk_id not in seed_ids:
                continue
            sq = self._generate_for_chunk(chunk)
            if sq is not None:
                results.append(sq)

        return results

    def _generate_for_chunk(self, chunk) -> Optional[SyntheticQuery]:
        """Attempt to generate one SyntheticQuery from a single chunk.

        Returns None if the chunk text is too short or the LLM output is
        not usable.
        """
        if len(chunk.embedding_text) < 20:
            return None

        prompt = (
            "你是台灣醫療臨床專家。以下是從醫療文件提取的段落：\n\n"
            "---\n"
            f"{chunk.embedding_text[:800]}\n"
            "---\n\n"
            "請根據此段落，生成 1 個臨床問題，要求：\n"
            "1. 問題可以直接從此段落找到答案\n"
            "2. 使用臨床工作者的自然問法\n"
            "3. 不要在問題中直接引用段落原文\n"
            "4. 問題長度至少 10 個字\n\n"
            "只輸出問題本身，不需要任何解釋。"
        )

        raw = _call_llm(self._client, prompt)
        question = raw.strip().strip('"').strip("'").strip("「」『』")

        if len(question) < 10:
            return None

        return SyntheticQuery(
            query_id=str(uuid4()),
            query_text=question,
            source_chunk_id=chunk.chunk_id,
            source_doc_id=chunk.metadata.get("source_doc_id", ""),
        )


# ---------------------------------------------------------------------------
# RelevanceJudge
# ---------------------------------------------------------------------------

class RelevanceJudge:
    """LLM-as-judge that scores (query, chunk) relevance on a 0-3 scale.

    Scoring rubric (TREC-style graded relevance):
        3 = Perfectly answers the clinical question
        2 = Partially answers
        1 = Topically related but does not directly answer
        0 = Irrelevant

    Reference: Voorhees, E. (2001). Evaluation by highly relevant documents.
    SIGIR '01.
    """

    def __init__(self, llm_client=None) -> None:
        self._client = llm_client or _get_llm_client()

    def judge(self, query_text: str, chunk_text: str) -> int:
        """Return a relevance score in {0, 1, 2, 3}."""
        prompt = (
            "你是醫療資訊檢索評審專家。\n\n"
            "【臨床問題】\n"
            f"{query_text}\n\n"
            "【檢索結果】\n"
            f"{chunk_text[:600]}\n\n"
            "【評分】3=完全回答，2=部分回答，1=相關但不能直接回答，0=無關\n"
            "只輸出一個整數（0-3）。"
        )
        result = _call_llm(self._client, prompt)
        match = re.search(r"[0-3]", result)
        return int(match.group()) if match else 0


# ---------------------------------------------------------------------------
# NDCGEvaluator
# ---------------------------------------------------------------------------

class NDCGEvaluator:
    """Evaluate retrieval quality using NDCG@10 with LLM-judged relevance.

    NDCG (Normalized Discounted Cumulative Gain) is the standard metric for
    graded-relevance retrieval evaluation. [2]

    [2] Järvelin, K. & Kekäläinen, J. (2002). Cumulated gain-based
        evaluation of IR techniques. ACM TOIS 20(4).
    """

    def __init__(self, judge: RelevanceJudge) -> None:
        self.judge = judge

    def evaluate(
        self,
        queries: List[SyntheticQuery],
        retriever,
        reranker=None,
        top_k: int = 10,
    ) -> EvaluationResult:
        """Run end-to-end evaluation over *queries*.

        For each query:
          1. Retrieve top_k*3 candidates via hybrid search.
          2. Optionally rerank to top_k.
          3. Judge each result with LLM-as-judge.
          4. Compute NDCG@k.

        Returns an EvaluationResult with aggregate and per-query statistics.
        """
        per_query_scores: dict = {}

        for query in queries:
            results = retriever.search_text(query.query_text, top_k=top_k * 3)

            if reranker is not None:
                results = reranker.rerank(query.query_text, results, top_k=top_k)
            else:
                results = results[:top_k]

            scores = [
                self.judge.judge(query.query_text, result.display_markdown)
                for result in results
            ]
            per_query_scores[query.query_id] = self.ndcg_at_k(scores, k=top_k)

        if per_query_scores:
            ndcg_mean = sum(per_query_scores.values()) / len(per_query_scores)
        else:
            ndcg_mean = 0.0

        low_score_queries = [
            qid for qid, score in per_query_scores.items() if score < 0.5
        ]

        return EvaluationResult(
            ndcg_at_10=ndcg_mean,
            per_query_scores=per_query_scores,
            low_score_queries=low_score_queries,
            total_queries=len(queries),
        )

    @staticmethod
    def ndcg_at_k(scores: List[int], k: int = 10) -> float:
        """Compute NDCG@k for a single ranked list of graded relevance scores."""
        s = scores[:k]
        if not s or max(s) == 0:
            return 0.0

        dcg = s[0] + sum(v / log2(i + 2) for i, v in enumerate(s[1:], 1))
        ideal = sorted(s, reverse=True)
        idcg = ideal[0] + sum(v / log2(i + 2) for i, v in enumerate(ideal[1:], 1))
        return dcg / idcg if idcg > 0 else 0.0
