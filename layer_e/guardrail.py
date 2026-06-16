from typing import Optional


def check_abstention(ranked_results: list, threshold: float = 0.10) -> tuple:
    if not ranked_results:
        return (True, "insufficient evidence: no results")
    # When no reranker is deployed, all rerank_scores are exactly 0.0.
    # Skip score-based abstention in this case — retrieval quality is judged by RRF scores instead.
    if all(r.rerank_score == 0.0 for r in ranked_results):
        return (False, None)
    top_score = max(r.rerank_score for r in ranked_results)
    if top_score < threshold:
        return (True, f"insufficient evidence: top rerank_score {top_score:.3f} below threshold {threshold}")
    return (False, None)


def compute_safety_verdict(
    abstain: bool,
    unsupported_claims: list,
    llm_abstain: bool,
) -> str:
    if abstain or llm_abstain:
        return "abstained"
    elif unsupported_claims:
        return "needs_review"
    else:
        return "safe"
