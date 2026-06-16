from dataclasses import dataclass, field
from typing import List, Optional
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from layer_e.context_packer import pack, collect_image_paths


@dataclass
class RankedResult:
    chunk_id: str
    chunk_type: str
    parent_chunk_id: Optional[str]
    retrieval_unit_id: str
    final_score: float
    rrf_score: float
    retrieval_weight: float
    display_markdown: str
    metadata: dict
    source_tool: str
    source_pages: List[int]
    embedding_text: str = ""
    rerank_score: float = 0.0


def _make_para(chunk_id, rerank_score, content="default test content", retrieval_weight=0.8, source_pages=None):
    return RankedResult(
        chunk_id=chunk_id,
        chunk_type="paragraph",
        parent_chunk_id=None,
        retrieval_unit_id=chunk_id,
        final_score=rerank_score,
        rrf_score=rerank_score,
        retrieval_weight=retrieval_weight,
        display_markdown=content,
        metadata={},
        source_tool="docling",
        source_pages=source_pages or [1],
        rerank_score=rerank_score,
    )


def test_basic_packing_three_paragraphs():
    r1 = _make_para("c1", rerank_score=0.9, content="alpha content paragraph")
    r2 = _make_para("c2", rerank_score=0.7, content="beta content paragraph")
    r3 = _make_para("c3", rerank_score=0.5, content="gamma content paragraph")

    evidence_list, evidence_map = pack([r3, r1, r2], max_tokens=12000)

    assert len(evidence_list) == 3
    assert evidence_list[0].id == "E1"
    assert evidence_list[0].chunk_id == "c1"
    assert evidence_list[1].id == "E2"
    assert evidence_list[1].chunk_id == "c2"
    assert evidence_list[2].id == "E3"
    assert evidence_list[2].chunk_id == "c3"


def test_token_budget_drops_low_score_chunks():
    long_text = "x" * 4000
    r1 = _make_para("c1", rerank_score=0.9, content=long_text)
    r2 = _make_para("c2", rerank_score=0.8, content=long_text)
    r3 = _make_para("c3", rerank_score=0.7, content=long_text)

    evidence_list, evidence_map = pack([r1, r2, r3], max_tokens=2500)

    chunk_ids = [e.chunk_id for e in evidence_list]
    assert "c1" in chunk_ids
    assert "c2" in chunk_ids
    assert "c3" not in chunk_ids


def test_low_confidence_prefix():
    r = _make_para("c1", rerank_score=0.9, content="some meaningful text here", retrieval_weight=0.4)

    evidence_list, _ = pack([r])

    assert evidence_list[0].content.startswith("[低信心] ")


def test_high_confidence_no_prefix():
    r = _make_para("c1", rerank_score=0.9, content="some meaningful text here", retrieval_weight=0.6)

    evidence_list, _ = pack([r])

    assert not evidence_list[0].content.startswith("[低信心]")


def test_parent_child_aggregation_single_slot():
    parent = RankedResult(
        chunk_id="table1",
        chunk_type="table",
        parent_chunk_id=None,
        retrieval_unit_id="table1",
        final_score=0.85,
        rrf_score=0.85,
        retrieval_weight=0.9,
        display_markdown="| col1 | col2 |\n|------|------|\n| a    | b    |",
        metadata={},
        source_tool="azure_di",
        source_pages=[2],
        rerank_score=0.85,
    )
    row = RankedResult(
        chunk_id="row1",
        chunk_type="row",
        parent_chunk_id="table1",
        retrieval_unit_id="row1",
        final_score=0.95,
        rrf_score=0.95,
        retrieval_weight=0.9,
        display_markdown="| a | b |",
        metadata={},
        source_tool="azure_di",
        source_pages=[2],
        rerank_score=0.95,
    )

    evidence_list, evidence_map = pack([parent, row])

    assert len(evidence_list) == 1
    assert evidence_list[0].chunk_id == "table1"
    assert evidence_list[0].content == parent.display_markdown
    assert "E1" in evidence_map


def test_row_chunk_without_parent_in_results():
    row = RankedResult(
        chunk_id="row1",
        chunk_type="row",
        parent_chunk_id="table_missing",
        retrieval_unit_id="row1",
        final_score=0.8,
        rrf_score=0.8,
        retrieval_weight=0.75,
        display_markdown="| column_a | column_b |",
        metadata={},
        source_tool="azure_di",
        source_pages=[3],
        rerank_score=0.8,
    )

    evidence_list, evidence_map = pack([row])

    assert len(evidence_list) == 1
    assert evidence_list[0].chunk_id == "row1"
    assert evidence_list[0].content == "| column_a | column_b |"


def test_evidence_map_structure():
    r1 = _make_para("c1", rerank_score=0.9, content="hello world content text", source_pages=[1, 2])
    r2 = _make_para("c2", rerank_score=0.6, content="world content text here", source_pages=[3])

    evidence_list, evidence_map = pack([r1, r2])

    assert set(evidence_map.keys()) == {"E1", "E2"}
    for key in ["E1", "E2"]:
        entry = evidence_map[key]
        assert "chunk_id" in entry
        assert "source_pages" in entry
        assert "source_tool" in entry
        assert "retrieval_weight" in entry

    assert evidence_map["E1"]["chunk_id"] == "c1"
    assert evidence_map["E1"]["source_pages"] == [1, 2]
    assert evidence_map["E1"]["source_tool"] == "docling"
    assert evidence_map["E1"]["retrieval_weight"] == 0.8

    assert evidence_map["E2"]["chunk_id"] == "c2"
    assert evidence_map["E2"]["source_pages"] == [3]


def test_evidence_map_contains_page_image_refs():
    r = _make_para("c1", rerank_score=0.9, content="meaningful test content")
    r.page_image_refs = {"1": "figures/p1.png"}
    _, evidence_map = pack([r])
    assert "page_image_refs" in evidence_map["E1"]
    assert evidence_map["E1"]["page_image_refs"] == {"1": "figures/p1.png"}


def test_evidence_map_page_image_refs_absent_when_not_set():
    r = _make_para("c1", rerank_score=0.9, content="meaningful test content")
    _, evidence_map = pack([r])
    assert evidence_map["E1"]["page_image_refs"] == {}


def test_collect_image_paths_absolute(tmp_path):
    img = tmp_path / "p1.png"
    img.write_bytes(b"\x89PNG")
    evidence_map = {
        "E1": {"page_image_refs": {"1": str(img)}},
    }
    paths = collect_image_paths(evidence_map, base_dir="")
    assert str(img) in paths


def test_collect_image_paths_relative(tmp_path):
    img = tmp_path / "p1.png"
    img.write_bytes(b"\x89PNG")
    evidence_map = {
        "E1": {"page_image_refs": {"1": "p1.png"}},
    }
    paths = collect_image_paths(evidence_map, base_dir=str(tmp_path))
    assert str(tmp_path / "p1.png") in paths


def test_collect_image_paths_dedup(tmp_path):
    img = tmp_path / "p1.png"
    img.write_bytes(b"\x89PNG")
    evidence_map = {
        "E1": {"page_image_refs": {"1": str(img)}},
        "E2": {"page_image_refs": {"1": str(img)}},
    }
    paths = collect_image_paths(evidence_map, base_dir="")
    assert len(paths) == 1


def test_collect_image_paths_empty():
    assert collect_image_paths({}, base_dir="") == []


def test_parent_chunk_higher_score_than_row():
    parent = RankedResult(
        chunk_id="table2",
        chunk_type="table",
        parent_chunk_id=None,
        retrieval_unit_id="table2",
        final_score=0.95,
        rrf_score=0.95,
        retrieval_weight=0.9,
        display_markdown="| col1 | col2 |\n|------|------|\n| x    | y    |",
        metadata={},
        source_tool="azure_di",
        source_pages=[5],
        rerank_score=0.95,
    )
    row = RankedResult(
        chunk_id="row2",
        chunk_type="row",
        parent_chunk_id="table2",
        retrieval_unit_id="row2",
        final_score=0.80,
        rrf_score=0.80,
        retrieval_weight=0.9,
        display_markdown="| x | y |",
        metadata={},
        source_tool="azure_di",
        source_pages=[5],
        rerank_score=0.80,
    )

    evidence_list, evidence_map = pack([parent, row])

    assert len(evidence_list) == 1, "table+row 應只產生一個 evidence slot"
    assert evidence_list[0].chunk_id == "table2"
    assert evidence_list[0].content == parent.display_markdown
    assert "E1" in evidence_map
