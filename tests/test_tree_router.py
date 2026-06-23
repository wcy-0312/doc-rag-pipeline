import pytest
from layer_f.tree_router import TreeRouter, RouteDecision


class _StubRouterLLM:
    def __init__(self, response: str):
        self._response = response

    def generate_text(self, user: str, system: str = "") -> str:
        return self._response

    def generate(self, system: str, user: str) -> dict:
        return {}

    def generate_with_tools(self, messages, tools):
        return ([], "")


def test_route_selects_correct_guideline():
    llm = _StubRouterLLM('{"need_patient_context": false, "relevant_guidelines": [1]}')
    router = TreeRouter(llm)
    summaries = {"乳癌指引_pdf": "乳癌", "肺癌指引_pdf": "肺癌", "大腸癌指引_pdf": "大腸癌"}
    decision = router.route("肺腺癌的治療建議", summaries, has_dynamic=False)
    assert decision.need_patient_context is False
    assert decision.selected_stems == ["肺癌指引_pdf"]


def test_route_requests_patient_context_when_needed():
    llm = _StubRouterLLM('{"need_patient_context": true, "relevant_guidelines": [0]}')
    router = TreeRouter(llm)
    summaries = {"乳癌指引_pdf": "乳癌診療指引"}
    decision = router.route("我現在應該接受什麼治療？", summaries, has_dynamic=True)
    assert decision.need_patient_context is True
    assert decision.selected_stems == ["乳癌指引_pdf"]


def test_route_falls_back_to_all_stems_on_invalid_json():
    llm = _StubRouterLLM("抱歉我無法判斷")   # no JSON
    router = TreeRouter(llm)
    summaries = {"doc_a": "A", "doc_b": "B"}
    decision = router.route("任何問題", summaries, has_dynamic=False)
    assert set(decision.selected_stems) == {"doc_a", "doc_b"}


def test_route_returns_empty_for_no_summaries():
    llm = _StubRouterLLM('{"need_patient_context": false, "relevant_guidelines": [0]}')
    router = TreeRouter(llm)
    decision = router.route("任何問題", {}, has_dynamic=False)
    assert decision.selected_stems == []
    assert decision.need_patient_context is False


def test_route_ignores_out_of_range_indices():
    llm = _StubRouterLLM('{"need_patient_context": false, "relevant_guidelines": [0, 99]}')
    router = TreeRouter(llm)
    summaries = {"only_one_pdf": "只有一份"}
    decision = router.route("問題", summaries, has_dynamic=False)
    assert decision.selected_stems == ["only_one_pdf"]   # index 99 ignored


def test_route_returns_empty_when_llm_explicitly_returns_no_guidelines():
    llm = _StubRouterLLM('{"need_patient_context": false, "relevant_guidelines": []}')
    router = TreeRouter(llm)
    summaries = {"doc_a": "A", "doc_b": "B"}
    decision = router.route("不相關的問題", summaries, has_dynamic=False)
    assert decision.selected_stems == []   # LLM says no match, must NOT fall back to all
