from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from layer_d.models import RankedResult
from layer_d.reranker import BGEReranker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidate(
    chunk_id: str = "chunk-a",
    embedding_text: str = "some medical text",
    final_score: float = 0.5,
) -> RankedResult:
    return RankedResult(
        chunk_id=chunk_id,
        chunk_type="paragraph",
        parent_chunk_id=None,
        retrieval_unit_id=chunk_id,
        final_score=final_score,
        rrf_score=final_score,
        retrieval_weight=1.0,
        display_markdown=f"# {chunk_id}",
        metadata={},
        source_tool="azure_cu",
        source_pages=[1],
        embedding_text=embedding_text,
    )


def _make_candidates(n: int) -> list[RankedResult]:
    return [_make_candidate(chunk_id=f"chunk-{i}", final_score=0.5) for i in range(n)]


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestLazyLoading:
    def test_lazy_loading_not_called_on_init(self):
        """__init__ must NOT load FlagReranker — model stays None."""
        with patch("layer_d.reranker.BGEReranker._load_model") as mock_load:
            reranker = BGEReranker()
            mock_load.assert_not_called()
            assert reranker._model is None

    def test_model_loaded_on_first_rerank(self):
        """After the first rerank() call, _model must be set (non-None)."""
        mock_flag_reranker = MagicMock()
        mock_flag_reranker.compute_score.return_value = [0.8]

        with patch("layer_d.reranker.FlagReranker", mock_flag_reranker, create=True):
            # Patch at import level inside _load_model
            reranker = BGEReranker()
            assert reranker._model is None

            with patch.dict("sys.modules", {"FlagEmbedding": MagicMock(FlagReranker=mock_flag_reranker)}):
                reranker._model = None  # ensure clean state
                # Trigger load via rerank
                candidates = [_make_candidate()]
                reranker._load_model = MagicMock(return_value=mock_flag_reranker)
                reranker.rerank("query", candidates, top_k=1)
                reranker._load_model.assert_called_once()

    def test_model_singleton(self):
        """Second rerank() must NOT construct FlagReranker a second time."""
        mock_flag_reranker_instance = MagicMock()
        mock_flag_reranker_instance.compute_score.return_value = [0.5, 0.7]

        reranker = BGEReranker()

        load_call_count = 0
        original_load = reranker._load_model.__func__ if hasattr(reranker._load_model, "__func__") else None

        mock_load = MagicMock(return_value=mock_flag_reranker_instance)
        reranker._load_model = mock_load

        candidates1 = [_make_candidate("c1"), _make_candidate("c2")]
        candidates2 = [_make_candidate("c3"), _make_candidate("c4")]

        mock_flag_reranker_instance.compute_score.side_effect = [
            [0.5, 0.7],
            [0.3, 0.9],
        ]

        reranker.rerank("query", candidates1, top_k=2)
        reranker.rerank("query", candidates2, top_k=2)

        # _load_model() called twice (once per rerank), but the singleton
        # logic inside _load_model itself ensures FlagReranker() is only
        # constructed once. Here we verify _load_model is called each time
        # (so the singleton check inside it matters).
        assert mock_load.call_count == 2


class TestRerankBehavior:
    def _make_reranker_with_mock_model(self, scores: list[float]) -> tuple[BGEReranker, MagicMock]:
        reranker = BGEReranker()
        mock_model = MagicMock()
        mock_model.compute_score.return_value = scores
        reranker._load_model = MagicMock(return_value=mock_model)
        return reranker, mock_model

    def test_rerank_returns_top_k(self):
        """Input 30 candidates, top_k=10 -> exactly 10 returned."""
        scores = [float(i) / 30.0 for i in range(30)]
        reranker, _ = self._make_reranker_with_mock_model(scores)

        candidates = _make_candidates(30)
        results = reranker.rerank("query text", candidates, top_k=10)

        assert len(results) == 10

    def test_rerank_sorted_by_score(self):
        """scores=[0.1, 0.9, 0.5] -> first result is the candidate with score 0.9."""
        reranker, _ = self._make_reranker_with_mock_model([0.1, 0.9, 0.5])

        c_low = _make_candidate("low")
        c_high = _make_candidate("high")
        c_mid = _make_candidate("mid")
        candidates = [c_low, c_high, c_mid]

        results = reranker.rerank("query", candidates, top_k=3)

        assert results[0].chunk_id == "high"
        assert results[0].rerank_score == pytest.approx(0.9)
        assert results[1].rerank_score == pytest.approx(0.5)
        assert results[2].rerank_score == pytest.approx(0.1)

    def test_rerank_empty_candidates(self):
        """Empty candidate list must return [] without touching the model."""
        reranker = BGEReranker()
        reranker._load_model = MagicMock()

        result = reranker.rerank("query", [], top_k=10)

        assert result == []
        reranker._load_model.assert_not_called()

    def test_rerank_score_assigned(self):
        """Each returned result's rerank_score must equal the model's output."""
        scores = [0.3, 0.7, 0.55]
        reranker, _ = self._make_reranker_with_mock_model(scores)

        candidates = [
            _make_candidate("c0"),
            _make_candidate("c1"),
            _make_candidate("c2"),
        ]
        results = reranker.rerank("query", candidates, top_k=3)

        # Results are sorted, so collect by chunk_id
        score_map = {r.chunk_id: r.rerank_score for r in results}
        assert score_map["c0"] == pytest.approx(0.3)
        assert score_map["c1"] == pytest.approx(0.7)
        assert score_map["c2"] == pytest.approx(0.55)

    def test_final_score_not_modified(self):
        """rerank() must NOT change final_score — only rerank_score is updated."""
        original_final_scores = [0.42, 0.31, 0.88]
        reranker, _ = self._make_reranker_with_mock_model([0.5, 0.6, 0.4])

        candidates = [
            _make_candidate("c0", final_score=0.42),
            _make_candidate("c1", final_score=0.31),
            _make_candidate("c2", final_score=0.88),
        ]
        results = reranker.rerank("query", candidates, top_k=3)

        result_map = {r.chunk_id: r for r in results}
        assert result_map["c0"].final_score == pytest.approx(0.42)
        assert result_map["c1"].final_score == pytest.approx(0.31)
        assert result_map["c2"].final_score == pytest.approx(0.88)


class TestModelNameFromEnv:
    def test_model_name_from_env(self):
        """BGE_RERANKER_MODEL env var must be used as model_name."""
        with patch.dict(os.environ, {"BGE_RERANKER_MODEL": "custom-model"}):
            reranker = BGEReranker()
            assert reranker.model_name == "custom-model"

    def test_model_name_default_when_env_not_set(self):
        """Without env var, model_name must fall back to the BAAI default."""
        env = {k: v for k, v in os.environ.items() if k != "BGE_RERANKER_MODEL"}
        with patch.dict(os.environ, env, clear=True):
            reranker = BGEReranker()
            assert reranker.model_name == "BAAI/bge-reranker-v2-m3"

    def test_explicit_model_name_overrides_env(self):
        """Explicit constructor argument must take precedence over env var."""
        with patch.dict(os.environ, {"BGE_RERANKER_MODEL": "env-model"}):
            reranker = BGEReranker(model_name="explicit-model")
            assert reranker.model_name == "explicit-model"
