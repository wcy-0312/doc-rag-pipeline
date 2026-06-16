

from layer_e.guardrail import check_abstention, compute_safety_verdict


class _FakeResult:
    def __init__(self, rerank_score):
        self.rerank_score = rerank_score


def test_check_abstention_empty():
    should_abstain, reason = check_abstention([])
    assert should_abstain is True
    assert "no results" in reason


def test_check_abstention_below_threshold():
    results = [_FakeResult(0.05)]
    should_abstain, reason = check_abstention(results, threshold=0.10)
    assert should_abstain is True
    assert "0.050" in reason


def test_check_abstention_above_threshold():
    results = [_FakeResult(0.15)]
    should_abstain, reason = check_abstention(results, threshold=0.10)
    assert should_abstain is False
    assert reason is None


def test_check_abstention_multiple_results():
    results = [_FakeResult(0.05), _FakeResult(0.20), _FakeResult(0.10)]
    should_abstain, reason = check_abstention(results, threshold=0.10)
    assert should_abstain is False
    assert reason is None


def test_compute_safety_verdict_abstain_true():
    verdict = compute_safety_verdict(abstain=True, unsupported_claims=[], llm_abstain=False)
    assert verdict == "abstained"


def test_compute_safety_verdict_llm_abstain_true():
    verdict = compute_safety_verdict(abstain=False, unsupported_claims=[], llm_abstain=True)
    assert verdict == "abstained"


def test_compute_safety_verdict_needs_review():
    verdict = compute_safety_verdict(abstain=False, unsupported_claims=["claim1"], llm_abstain=False)
    assert verdict == "needs_review"


def test_compute_safety_verdict_safe():
    verdict = compute_safety_verdict(abstain=False, unsupported_claims=[], llm_abstain=False)
    assert verdict == "safe"
