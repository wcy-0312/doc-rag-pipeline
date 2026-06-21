from typing import List
from .models import ClaimCitation, EvidenceItem

_JUDGE_SYSTEM = (
    "你是一個事實查核助理。你的任務是判斷一個 claim 是否可以從提供的 Evidence 中找到支持。\n"
    "只輸出 JSON，格式為 {\"answer\": \"yes\"} 或 {\"answer\": \"no\"}，不要加入任何額外文字。"
)


def validate_citations(claims: list, evidence_map: dict) -> list:
    out_of_range = []
    for claim in claims:
        for eid in claim.citations:
            if eid not in evidence_map and eid not in out_of_range:
                out_of_range.append(eid)
    return out_of_range


def detect_unsupported_claims(
    claims: list,
    evidence_list: list,
    llm_client,
) -> list:
    evidence_by_id = {e.id: e for e in evidence_list}
    unsupported = []

    for claim in claims:
        if not claim.citations:
            unsupported.append(claim.text)
            continue

        claim_unsupported = False
        for eid in claim.citations:
            evidence = evidence_by_id.get(eid)
            content = evidence.content if evidence else ""
            judge_user = (
                f"Evidence {eid}：{content}\n\n"
                f"Claim：{claim.text}\n\n"
                f"問題：這個 claim 可以從上述 Evidence 中找到支持嗎？\n"
                f"回答 yes 或 no。"
            )
            response = llm_client.generate(system=_JUDGE_SYSTEM, user=judge_user)
            answer = response.get("answer", "")
            if "no" in answer.lower():
                claim_unsupported = True
                break

        if claim_unsupported:
            unsupported.append(claim.text)

    return unsupported
