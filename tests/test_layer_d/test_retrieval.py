from __future__ import annotations

import pytest
from types import SimpleNamespace
from typing import List
from unittest.mock import MagicMock, patch

from layer_d.retrieval import HybridRetriever, _chunk_id_to_point_id
from layer_d.retrieve import detect_docs
from layer_d.models import RankedResult


# ---------------------------------------------------------------------------
# Fake data helpers
# ---------------------------------------------------------------------------

DENSE_DIM = 1024
FAKE_DENSE = [0.1] * DENSE_DIM
FAKE_SPARSE = {101: 0.5, 2023: 0.3, 5432: 0.8}


def _make_scored_point(
    chunk_id: str = "chunk-a",
    chunk_type: str = "paragraph",
    parent_chunk_id=None,
    retrieval_unit_id: str = "unit-a",
    retrieval_weight: float = 1.0,
    score: float = 0.5,
    source_tool: str = "azure_cu",
):
    payload = {
        "chunk_id": chunk_id,
        "chunk_type": chunk_type,
        "parent_chunk_id": parent_chunk_id,
        "retrieval_unit_id": retrieval_unit_id,
        "retrieval_weight": retrieval_weight,
        "source_tool": source_tool,
        "source_pages": [1],
        "display_markdown": f"content of {chunk_id}",
        "confidence_level": "high",
        "quality_flag": "ok",
        "has_handwriting": False,
        "embedding_text": "some text",
    }
    pt = MagicMock()
    pt.id = _chunk_id_to_point_id(chunk_id)
    pt.score = score
    pt.payload = payload
    return pt


def _make_retriever(scored_points: list) -> tuple[HybridRetriever, MagicMock]:
    client = MagicMock()
    client.query_points.return_value = SimpleNamespace(points=scored_points)
    client.retrieve.return_value = []
    retriever = HybridRetriever(client=client, collection_name="medical_docs")
    return retriever, client


# ---------------------------------------------------------------------------
# Prefetch structure
# ---------------------------------------------------------------------------

class TestPrefetchStructure:
    def test_two_prefetch_entries(self):
        retriever, client = _make_retriever([])
        retriever.search(FAKE_DENSE, FAKE_SPARSE)

        kwargs = client.query_points.call_args.kwargs
        prefetch = kwargs["prefetch"]
        assert len(prefetch) == 2

    def test_dense_prefetch_uses_dense_vector_name(self):
        retriever, client = _make_retriever([])
        retriever.search(FAKE_DENSE, FAKE_SPARSE, prefetch_k=30)

        prefetch = client.query_points.call_args.kwargs["prefetch"]
        dense_pf = next(pf for pf in prefetch if pf.using == "dense")
        assert dense_pf.limit == 30
        assert dense_pf.query == FAKE_DENSE

    def test_sparse_prefetch_uses_sparse_vector_name(self):
        retriever, client = _make_retriever([])
        retriever.search(FAKE_DENSE, FAKE_SPARSE)

        prefetch = client.query_points.call_args.kwargs["prefetch"]
        sparse_pf = next(pf for pf in prefetch if pf.using == "sparse")
        assert sparse_pf.query.indices == [101, 2023, 5432]
        assert sparse_pf.query.values == pytest.approx([0.5, 0.3, 0.8])

    def test_rrf_fusion_query(self):
        retriever, client = _make_retriever([])
        retriever.search(FAKE_DENSE, FAKE_SPARSE)

        from qdrant_client.models import Fusion
        kwargs = client.query_points.call_args.kwargs
        assert kwargs["query"].fusion == Fusion.RRF

    def test_top_k_passed_as_limit(self):
        retriever, client = _make_retriever([])
        retriever.search(FAKE_DENSE, FAKE_SPARSE, top_k=7)

        assert client.query_points.call_args.kwargs["limit"] == 7

    def test_filter_applied_in_prefetch(self):
        retriever, client = _make_retriever([])
        retriever.search(FAKE_DENSE, FAKE_SPARSE, min_retrieval_weight=0.5)

        prefetch = client.query_points.call_args.kwargs["prefetch"]
        for pf in prefetch:
            must = pf.filter.must
            weight_cond = next(c for c in must if c.key == "retrieval_weight")
            assert weight_cond.range.gte == pytest.approx(0.5)

    def test_chunk_type_filter_in_prefetch(self):
        retriever, client = _make_retriever([])
        retriever.search(FAKE_DENSE, FAKE_SPARSE, filter_chunk_types=["table", "row"])

        prefetch = client.query_points.call_args.kwargs["prefetch"]
        for pf in prefetch:
            must = pf.filter.must
            type_cond = next(c for c in must if c.key == "chunk_type")
            assert set(type_cond.match.any) == {"table", "row"}

    def test_with_payload_true(self):
        retriever, client = _make_retriever([])
        retriever.search(FAKE_DENSE, FAKE_SPARSE)
        assert client.query_points.call_args.kwargs["with_payload"] is True


# ---------------------------------------------------------------------------
# Score weighting
# ---------------------------------------------------------------------------

class TestScoreWeighting:
    def test_final_score_equals_rrf_times_weight(self):
        pt = _make_scored_point(score=0.4, retrieval_weight=0.8)
        retriever, _ = _make_retriever([pt])

        results = retriever.search(FAKE_DENSE, FAKE_SPARSE)

        assert len(results) == 1
        r = results[0]
        assert r.rrf_score == pytest.approx(0.4)
        assert r.retrieval_weight == pytest.approx(0.8)
        assert r.final_score == pytest.approx(0.4 * 0.8)

    def test_full_weight_chunk_unchanged(self):
        pt = _make_scored_point(score=0.6, retrieval_weight=1.0)
        retriever, _ = _make_retriever([pt])
        results = retriever.search(FAKE_DENSE, FAKE_SPARSE)
        assert results[0].final_score == pytest.approx(0.6)

    def test_zero_weight_chunk_scores_zero(self):
        pt = _make_scored_point(score=0.9, retrieval_weight=0.0)
        retriever, _ = _make_retriever([pt])
        results = retriever.search(FAKE_DENSE, FAKE_SPARSE)
        assert results[0].final_score == pytest.approx(0.0)

    def test_results_sorted_by_final_score_descending(self):
        pts = [
            _make_scored_point("c1", score=0.8, retrieval_weight=0.5),   # final=0.40
            _make_scored_point("c2", score=0.5, retrieval_weight=1.0),   # final=0.50
            _make_scored_point("c3", score=0.6, retrieval_weight=0.9),   # final=0.54
        ]
        retriever, _ = _make_retriever(pts)
        results = retriever.search(FAKE_DENSE, FAKE_SPARSE)

        scores = [r.final_score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert results[0].chunk_id == "c3"
        assert results[1].chunk_id == "c2"
        assert results[2].chunk_id == "c1"

    def test_ranked_result_fields_populated(self):
        pt = _make_scored_point(
            chunk_id="chunk-x",
            chunk_type="table",
            retrieval_unit_id="unit-x",
            source_tool="docling",
            score=0.7,
            retrieval_weight=0.9,
        )
        retriever, _ = _make_retriever([pt])
        results = retriever.search(FAKE_DENSE, FAKE_SPARSE)

        r = results[0]
        assert r.chunk_id == "chunk-x"
        assert r.chunk_type == "table"
        assert r.retrieval_unit_id == "unit-x"
        assert r.source_tool == "docling"
        assert r.display_markdown == "content of chunk-x"

    def test_empty_results(self):
        retriever, _ = _make_retriever([])
        assert retriever.search(FAKE_DENSE, FAKE_SPARSE) == []


# ---------------------------------------------------------------------------
# Parent aggregation
# ---------------------------------------------------------------------------

class TestParentAggregation:
    def _make_parent_point(self, chunk_id: str) -> MagicMock:
        payload = {
            "chunk_id": chunk_id,
            "chunk_type": "table",
            "parent_chunk_id": None,
            "retrieval_unit_id": chunk_id,
            "retrieval_weight": 0.95,
            "source_tool": "azure_cu",
            "source_pages": [2],
            "display_markdown": f"table: {chunk_id}",
            "confidence_level": "high",
            "quality_flag": "ok",
        }
        pt = MagicMock()
        pt.id = _chunk_id_to_point_id(chunk_id)
        pt.payload = payload
        return pt

    def test_no_aggregation_when_flag_false(self):
        row_pt = _make_scored_point(
            "unit-t_row0", chunk_type="row", parent_chunk_id="unit-t", score=0.5
        )
        retriever, client = _make_retriever([row_pt])
        retriever.search(FAKE_DENSE, FAKE_SPARSE, include_parent_context=False)
        client.retrieve.assert_not_called()

    def test_parent_fetched_for_row_hits(self):
        row_pt = _make_scored_point(
            "unit-t_row0", chunk_type="row", parent_chunk_id="unit-t", score=0.5
        )
        parent_pt = self._make_parent_point("unit-t")
        retriever, client = _make_retriever([row_pt])
        client.retrieve.return_value = [parent_pt]

        results = retriever.search(FAKE_DENSE, FAKE_SPARSE, include_parent_context=True)

        client.retrieve.assert_called_once()
        chunk_ids = {r.chunk_id for r in results}
        assert "unit-t" in chunk_ids
        assert "unit-t_row0" in chunk_ids

    def test_parent_not_duplicated_if_already_in_results(self):
        # parent table chunk appears in search results directly
        parent_pt = _make_scored_point(
            "unit-t", chunk_type="table", score=0.7
        )
        row_pt = _make_scored_point(
            "unit-t_row0", chunk_type="row", parent_chunk_id="unit-t", score=0.5
        )
        retriever, client = _make_retriever([parent_pt, row_pt])

        results = retriever.search(FAKE_DENSE, FAKE_SPARSE, include_parent_context=True)

        # retrieve should not be called because parent already in results
        client.retrieve.assert_not_called()
        assert sum(1 for r in results if r.chunk_id == "unit-t") == 1

    def test_appended_parent_has_zero_score(self):
        row_pt = _make_scored_point(
            "unit-t_row1", chunk_type="row", parent_chunk_id="unit-t", score=0.6
        )
        parent_pt = self._make_parent_point("unit-t")
        retriever, client = _make_retriever([row_pt])
        client.retrieve.return_value = [parent_pt]

        results = retriever.search(FAKE_DENSE, FAKE_SPARSE, include_parent_context=True)

        parent_result = next(r for r in results if r.chunk_id == "unit-t")
        assert parent_result.final_score == pytest.approx(0.0)
        assert parent_result.rrf_score == pytest.approx(0.0)

    def test_non_row_chunks_do_not_trigger_parent_fetch(self):
        para_pt = _make_scored_point("para-a", chunk_type="paragraph", score=0.6)
        retriever, client = _make_retriever([para_pt])

        retriever.search(FAKE_DENSE, FAKE_SPARSE, include_parent_context=True)

        client.retrieve.assert_not_called()

    def test_multiple_row_hits_same_parent_fetched_once(self):
        row0 = _make_scored_point(
            "unit-t_row0", chunk_type="row", parent_chunk_id="unit-t", score=0.6
        )
        row1 = _make_scored_point(
            "unit-t_row1", chunk_type="row", parent_chunk_id="unit-t", score=0.5
        )
        parent_pt = self._make_parent_point("unit-t")
        retriever, client = _make_retriever([row0, row1])
        client.retrieve.return_value = [parent_pt]

        results = retriever.search(FAKE_DENSE, FAKE_SPARSE, include_parent_context=True)

        client.retrieve.assert_called_once()
        assert sum(1 for r in results if r.chunk_id == "unit-t") == 1


# ---------------------------------------------------------------------------
# detect_docs
# ---------------------------------------------------------------------------

class TestDetectDocs:
    def test_full_stem_match(self):
        result = detect_docs("根據MRI報告_2024的結果")
        assert "MRI報告_2024" in result

    def test_alias_match_mri(self):
        result = detect_docs("MRI顯示有異常")
        assert "MRI報告_2024" in result

    def test_alias_match_muscular_injection(self):
        result = detect_docs("請說明肌肉注射的SOP")
        assert "護理部_A31000-Q05-W-A12_肌肉注射技術" in result

    def test_alias_match_stroke(self):
        result = detect_docs("腦中風護理措施")
        assert "護理部_A31000-Q03-W-A01_腦中風" in result

    def test_alias_match_handwash(self):
        result = detect_docs("洗手技術標準流程")
        assert "護理部_A31000-Q05-W-A05_洗手技術" in result

    def test_alias_match_ng_tube(self):
        result = detect_docs("鼻胃管灌食護理")
        assert "T-Adult-070_鼻胃管灌食" in result

    def test_two_docs_detected(self):
        result = detect_docs("肌肉注射SOP及腦中風SOP的修訂日期")
        assert "護理部_A31000-Q05-W-A12_肌肉注射技術" in result
        assert "護理部_A31000-Q03-W-A01_腦中風" in result
        assert len(result) == 2

    def test_no_doc_detected(self):
        result = detect_docs("請問今天天氣如何")
        assert result == []

    def test_deduplication(self):
        # Full stem and alias both match the same doc — should appear once
        result = detect_docs("MRI報告_2024及MRI異常")
        assert result.count("MRI報告_2024") == 1

    def test_make_doc_filter_returns_filter(self):
        from qdrant_client.models import Filter
        retriever, _ = _make_retriever([])
        f = retriever.make_doc_filter("護理部_A31000-Q05-W-A12_肌肉注射技術")
        assert isinstance(f, Filter)
        assert len(f.must) == 1
        assert f.must[0].key == "chunk_id"


