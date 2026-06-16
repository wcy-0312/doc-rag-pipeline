import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from layer_e.models import ClaimCitation, EvidenceItem
from layer_e.llm_client import _StubLLMClient
from layer_e.attribution import validate_citations, detect_unsupported_claims


def make_evidence_map():
    return {"E1": {"content": "content 1"}, "E2": {"content": "content 2"}}


def make_evidence_list():
    return [
        EvidenceItem(id="E1", chunk_id="c1", content="content 1", retrieval_weight=1.0, source_pages=[1], source_tool="tool"),
        EvidenceItem(id="E2", chunk_id="c2", content="content 2", retrieval_weight=0.9, source_pages=[2], source_tool="tool"),
    ]


def test_validate_citations_no_out_of_range():
    claims = [ClaimCitation(text="claim1", citations=["E1", "E2"])]
    result = validate_citations(claims, make_evidence_map())
    assert result == []


def test_validate_citations_one_invalid():
    claims = [ClaimCitation(text="claim1", citations=["E1", "E99"])]
    result = validate_citations(claims, make_evidence_map())
    assert result == ["E99"]


def test_validate_citations_multiple_claims_mixed():
    claims = [
        ClaimCitation(text="claim1", citations=["E1"]),
        ClaimCitation(text="claim2", citations=["E2", "E99"]),
    ]
    result = validate_citations(claims, make_evidence_map())
    assert result == ["E99"]


def test_detect_unsupported_empty_citations():
    claims = [ClaimCitation(text="unsupported claim", citations=[])]
    result = detect_unsupported_claims(claims, make_evidence_list(), _StubLLMClient())
    assert "unsupported claim" in result


def test_detect_unsupported_stub_returns_stub_not_unsupported():
    claims = [ClaimCitation(text="supported claim", citations=["E1"])]
    result = detect_unsupported_claims(claims, make_evidence_list(), _StubLLMClient())
    assert "supported claim" not in result


def test_detect_unsupported_no_stub_returns_unsupported():
    class _NoStub:
        def generate(self, system, user):
            return {"answer": "no, not supported"}

    claims = [ClaimCitation(text="bad claim", citations=["E1"])]
    result = detect_unsupported_claims(claims, make_evidence_list(), _NoStub())
    assert "bad claim" in result


def test_detect_unsupported_yes_answer_not_unsupported():
    class _YesStub:
        def generate(self, system, user):
            return {"answer": "yes, supported"}

    claims = [ClaimCitation(text="good claim", citations=["E1"])]
    result = detect_unsupported_claims(claims, make_evidence_list(), _YesStub())
    assert "good claim" not in result
