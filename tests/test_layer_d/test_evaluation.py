from __future__ import annotations

from math import log2
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from layer_d.evaluation import NDCGEvaluator, RelevanceJudge, SyntheticQueryGenerator
from layer_d.models import EvaluationResult, RankedResult, SyntheticQuery


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(
    chunk_id: str = "chunk-a",
    embedding_text: str = "這是一段足夠長的醫療文件內容，用於測試合成問題生成功能。",
    source_doc_id: str = "doc-1",
) -> SimpleNamespace:
    return SimpleNamespace(
        chunk_id=chunk_id,
        embedding_text=embedding_text,
        display_markdown=f"# {chunk_id}",
        metadata={"source_doc_id": source_doc_id},
    )


def _make_ranked_result(chunk_id: str = "r0", display_markdown: str = "content") -> RankedResult:
    return RankedResult(
        chunk_id=chunk_id,
        chunk_type="paragraph",
        parent_chunk_id=None,
        retrieval_unit_id=chunk_id,
        final_score=0.5,
        rrf_score=0.5,
        retrieval_weight=1.0,
        display_markdown=display_markdown,
        metadata={},
        source_tool="azure_cu",
        source_pages=[1],
    )


def _make_synthetic_query(query_id: str = "q1", query_text: str = "病患應如何服用此藥物？") -> SyntheticQuery:
    return SyntheticQuery(
        query_id=query_id,
        query_text=query_text,
        source_chunk_id="chunk-a",
        source_doc_id="doc-1",
    )


# ---------------------------------------------------------------------------
# NDCG@k unit tests
# ---------------------------------------------------------------------------

class TestNDCGAtK:
    def test_ndcg_perfect(self):
        """Perfect ranking: DCG == IDCG → 1.0."""
        scores = [3, 2, 1, 0, 0, 0, 0, 0, 0, 0]
        result = NDCGEvaluator.ndcg_at_k(scores, k=10)
        assert result == pytest.approx(1.0)

    def test_ndcg_zero(self):
        """All zeros → 0.0."""
        scores = [0] * 10
        result = NDCGEvaluator.ndcg_at_k(scores, k=10)
        assert result == pytest.approx(0.0)

    def test_ndcg_empty(self):
        """Empty list → 0.0."""
        result = NDCGEvaluator.ndcg_at_k([], k=10)
        assert result == pytest.approx(0.0)

    def test_ndcg_single(self):
        """Single perfect score [3] → 1.0 (DCG == IDCG)."""
        result = NDCGEvaluator.ndcg_at_k([3], k=10)
        assert result == pytest.approx(1.0)

    def test_ndcg_truncates_at_k(self):
        """Scores beyond k must not affect the result."""
        scores_short = [3, 2, 1]
        scores_long  = [3, 2, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        result_short = NDCGEvaluator.ndcg_at_k(scores_short, k=3)
        result_long  = NDCGEvaluator.ndcg_at_k(scores_long,  k=3)
        assert result_short == pytest.approx(result_long)

    def test_ndcg_partial(self):
        """Reversed order [0,3] is worse than ideal [3,0] → score < 1.0."""
        worst  = NDCGEvaluator.ndcg_at_k([0, 3], k=2)
        best   = NDCGEvaluator.ndcg_at_k([3, 0], k=2)
        assert worst < best
        assert worst < 1.0
        assert best  == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# SyntheticQueryGenerator tests
# ---------------------------------------------------------------------------

class TestSyntheticQueryGenerator:
    def _make_generator(self, llm_response: str) -> SyntheticQueryGenerator:
        mock_client = MagicMock()
        with patch("layer_d.evaluation._call_llm", return_value=llm_response):
            gen = SyntheticQueryGenerator(llm_client=mock_client)
        gen._client = mock_client
        return gen

    def test_generator_skips_short_text(self):
        """embedding_text < 20 chars must return None from _generate_for_chunk."""
        mock_client = MagicMock()
        gen = SyntheticQueryGenerator(llm_client=mock_client)
        chunk = _make_chunk(embedding_text="短")
        with patch("layer_d.evaluation._call_llm") as mock_call:
            result = gen._generate_for_chunk(chunk)
        assert result is None
        mock_call.assert_not_called()

    def test_generator_skips_short_question(self):
        """LLM returning < 10 chars must cause _generate_for_chunk to return None."""
        mock_client = MagicMock()
        gen = SyntheticQueryGenerator(llm_client=mock_client)
        chunk = _make_chunk()
        with patch("layer_d.evaluation._call_llm", return_value="短問題"):
            result = gen._generate_for_chunk(chunk)
        assert result is None

    def test_generator_returns_synthetic_query(self):
        """Valid LLM response (≥10 chars) must produce a SyntheticQuery."""
        mock_client = MagicMock()
        gen = SyntheticQueryGenerator(llm_client=mock_client)
        chunk = _make_chunk(chunk_id="chunk-x", source_doc_id="doc-42")
        question = "此藥物在臨床使用時有哪些需要注意的禁忌症？"
        with patch("layer_d.evaluation._call_llm", return_value=question):
            result = gen._generate_for_chunk(chunk)
        assert isinstance(result, SyntheticQuery)
        assert result.query_text == question
        assert result.source_chunk_id == "chunk-x"
        assert result.source_doc_id == "doc-42"
        assert len(result.query_id) == 36  # UUID4 canonical form

    def test_generator_generate_respects_target_count(self):
        """generate() must stop at target_count even when more chunks exist."""
        mock_client = MagicMock()
        gen = SyntheticQueryGenerator(llm_client=mock_client)
        chunks = [_make_chunk(chunk_id=f"c{i}") for i in range(20)]
        question = "病患術後應如何進行傷口護理以避免感染？"
        with patch("layer_d.evaluation._call_llm", return_value=question):
            results = gen.generate(chunks, target_count=5)
        assert len(results) == 5


# ---------------------------------------------------------------------------
# RelevanceJudge tests
# ---------------------------------------------------------------------------

class TestRelevanceJudge:
    def test_judge_parse_valid(self):
        """LLM returning '2' → judge score is 2."""
        mock_client = MagicMock()
        judge = RelevanceJudge(llm_client=mock_client)
        with patch("layer_d.evaluation._call_llm", return_value="2"):
            score = judge.judge("問題", "文件內容")
        assert score == 2

    def test_judge_parse_fallback(self):
        """LLM returning 'N/A' (no digit 0-3) → judge score falls back to 0."""
        mock_client = MagicMock()
        judge = RelevanceJudge(llm_client=mock_client)
        with patch("layer_d.evaluation._call_llm", return_value="N/A"):
            score = judge.judge("問題", "文件內容")
        assert score == 0

    def test_judge_parse_all_valid_scores(self):
        """Each digit 0-3 must be parsed correctly."""
        mock_client = MagicMock()
        judge = RelevanceJudge(llm_client=mock_client)
        for expected in [0, 1, 2, 3]:
            with patch("layer_d.evaluation._call_llm", return_value=str(expected)):
                assert judge.judge("q", "doc") == expected

    def test_judge_uses_first_digit_in_multi_digit_response(self):
        """When LLM returns text like '評分：2 分', the first 0-3 digit is used."""
        mock_client = MagicMock()
        judge = RelevanceJudge(llm_client=mock_client)
        with patch("layer_d.evaluation._call_llm", return_value="評分：2 分"):
            score = judge.judge("q", "doc")
        assert score == 2


# ---------------------------------------------------------------------------
# NDCGEvaluator integration tests
# ---------------------------------------------------------------------------

class TestNDCGEvaluatorIntegration:
    def _make_evaluator(self, judge_scores: list[int]) -> tuple[NDCGEvaluator, MagicMock]:
        mock_judge = MagicMock(spec=RelevanceJudge)
        mock_judge.judge.side_effect = judge_scores
        evaluator = NDCGEvaluator(judge=mock_judge)
        return evaluator, mock_judge

    def _make_retriever(self, results: list[RankedResult]) -> MagicMock:
        mock_retriever = MagicMock()
        mock_retriever.search_text.return_value = results
        return mock_retriever

    def test_evaluate_result_structure(self):
        """EvaluationResult must have all required fields with correct types."""
        results = [_make_ranked_result(f"r{i}") for i in range(3)]
        retriever = self._make_retriever(results)
        evaluator, _ = self._make_evaluator([3, 2, 1])

        queries = [_make_synthetic_query("q1")]
        outcome = evaluator.evaluate(queries, retriever, reranker=None, top_k=3)

        assert isinstance(outcome, EvaluationResult)
        assert isinstance(outcome.ndcg_at_10, float)
        assert isinstance(outcome.per_query_scores, dict)
        assert isinstance(outcome.low_score_queries, list)
        assert outcome.total_queries == 1
        assert "q1" in outcome.per_query_scores

    def test_evaluate_with_reranker(self):
        """When reranker is provided, rerank() must be called once per query."""
        results = [_make_ranked_result(f"r{i}") for i in range(3)]
        retriever = self._make_retriever(results)

        mock_reranker = MagicMock()
        reranked = [_make_ranked_result(f"rr{i}") for i in range(3)]
        mock_reranker.rerank.return_value = reranked

        evaluator, _ = self._make_evaluator([3, 2, 1])
        queries = [_make_synthetic_query("q1")]
        evaluator.evaluate(queries, retriever, reranker=mock_reranker, top_k=3)

        mock_reranker.rerank.assert_called_once()

    def test_evaluate_low_score_queries_identified(self):
        """Queries with NDCG < 0.5 must appear in low_score_queries."""
        # [0, 0, 0] → NDCG = 0.0, which is < 0.5
        results = [_make_ranked_result(f"r{i}") for i in range(3)]
        retriever = self._make_retriever(results)
        evaluator, _ = self._make_evaluator([0, 0, 0])

        queries = [_make_synthetic_query("q-low")]
        outcome = evaluator.evaluate(queries, retriever, reranker=None, top_k=3)

        assert "q-low" in outcome.low_score_queries

    def test_evaluate_perfect_score_not_in_low(self):
        """A query with NDCG = 1.0 must NOT be in low_score_queries."""
        results = [_make_ranked_result(f"r{i}") for i in range(3)]
        retriever = self._make_retriever(results)
        evaluator, _ = self._make_evaluator([3, 2, 1])

        queries = [_make_synthetic_query("q-good")]
        outcome = evaluator.evaluate(queries, retriever, reranker=None, top_k=3)

        assert "q-good" not in outcome.low_score_queries

    def test_evaluate_empty_queries(self):
        """Empty query list must return EvaluationResult with ndcg_at_10=0.0."""
        retriever = self._make_retriever([])
        evaluator, _ = self._make_evaluator([])

        outcome = evaluator.evaluate([], retriever, reranker=None, top_k=10)

        assert outcome.ndcg_at_10 == pytest.approx(0.0)
        assert outcome.total_queries == 0
        assert outcome.per_query_scores == {}
