import pytest
from layer_e.models import EvidenceItem, ClaimCitation, GenerationResult


def test_evidence_item_instantiation():
    item = EvidenceItem(
        id="E1",
        chunk_id="chunk-001",
        content="Patient has hypertension.",
        retrieval_weight=0.87,
        source_pages=[3, 4],
        source_tool="azure_di",
    )
    assert item.id == "E1"
    assert item.chunk_id == "chunk-001"
    assert item.content == "Patient has hypertension."
    assert item.retrieval_weight == 0.87
    assert item.source_pages == [3, 4]
    assert item.source_tool == "azure_di"


def test_claim_citation_multiple_citations():
    claim = ClaimCitation(
        text="Blood pressure was elevated.",
        citations=["E1", "E3"],
    )
    assert claim.text == "Blood pressure was elevated."
    assert claim.citations == ["E1", "E3"]
    assert len(claim.citations) == 2


def test_generation_result_safe():
    result = GenerationResult(
        answer="The patient was diagnosed with hypertension.",
        claims=[ClaimCitation(text="Diagnosed with hypertension.", citations=["E1"])],
        evidence_map={
            "E1": {
                "chunk_id": "chunk-001",
                "source_pages": [3],
                "source_tool": "azure_di",
                "retrieval_weight": 0.87,
            }
        },
        unsupported_claims=[],
        abstain=False,
        abstain_reason=None,
        safety_verdict="safe",
    )
    assert result.safety_verdict == "safe"
    assert result.abstain is False
    assert result.abstain_reason is None
    assert result.unsupported_claims == []


def test_generation_result_needs_review():
    result = GenerationResult(
        answer="The dosage may be 500mg.",
        claims=[ClaimCitation(text="Dosage may be 500mg.", citations=["E2"])],
        evidence_map={
            "E2": {
                "chunk_id": "chunk-042",
                "source_pages": [7],
                "source_tool": "docling",
                "retrieval_weight": 0.61,
            }
        },
        unsupported_claims=["Dosage may be 500mg."],
        abstain=False,
        abstain_reason=None,
        safety_verdict="needs_review",
    )
    assert result.safety_verdict == "needs_review"
    assert result.abstain is False
    assert "Dosage may be 500mg." in result.unsupported_claims


def test_generation_result_abstained():
    result = GenerationResult(
        answer="",
        claims=[],
        evidence_map={},
        unsupported_claims=[],
        abstain=True,
        abstain_reason="Insufficient evidence to answer safely.",
        safety_verdict="abstained",
    )
    assert result.abstain is True
    assert result.abstain_reason == "Insufficient evidence to answer safely."
    assert result.safety_verdict == "abstained"
    assert result.answer == ""
    assert result.claims == []


def test_generation_result_abstain_reason_set():
    result = GenerationResult(
        answer="",
        claims=[],
        evidence_map={},
        unsupported_claims=[],
        abstain=True,
        abstain_reason="Query out of scope.",
        safety_verdict="abstained",
    )
    assert result.abstain is True
    assert isinstance(result.abstain_reason, str)
    assert result.abstain_reason == "Query out of scope."


def test_evidence_map_schema():
    evidence_map = {
        "E1": {
            "chunk_id": "chunk-001",
            "source_pages": [1, 2],
            "source_tool": "azure_di",
            "retrieval_weight": 0.95,
        },
        "E2": {
            "chunk_id": "chunk-007",
            "source_pages": [5],
            "source_tool": "llm",
            "retrieval_weight": 0.72,
        },
    }
    result = GenerationResult(
        answer="Some answer.",
        claims=[],
        evidence_map=evidence_map,
        unsupported_claims=[],
        abstain=False,
        abstain_reason=None,
        safety_verdict="safe",
    )
    for key, val in result.evidence_map.items():
        assert "chunk_id" in val
        assert "source_pages" in val
        assert "source_tool" in val
        assert "retrieval_weight" in val
        assert isinstance(val["source_pages"], list)
        assert isinstance(val["retrieval_weight"], float)
