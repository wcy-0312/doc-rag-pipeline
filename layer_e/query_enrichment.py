from __future__ import annotations
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from layer_e.models import EvidenceItem

logger = logging.getLogger(__name__)

_FILTER_SYSTEM = """\
你是醫療文件審核員。給定一個臨床查詢和多段文件證據，判斷每段證據是否適用於查詢描述的病人或情境。

判斷規則：
- 若查詢描述浸潤性乳癌（有 T/N 分期，如 cT2N1M0），DCIS / 原位癌 / Tis 的建議不適用
- 若查詢指定特定亞型（如 HER2+），其他亞型專屬建議不適用
- 若查詢指定特定分期，明顯屬於其他分期的建議不適用
- 若不確定是否適用，預設 applicable=true（寬鬆過濾，避免漏掉重要資訊）

輸出 JSON（key 為 Evidence ID，如 E1、E2）：
{"E1": {"applicable": true, "note": "適用/不適用原因（10字以內）"}, "E2": {"applicable": false, "note": "..."}}
只輸出 JSON，不要其他說明。\
"""


def filter_evidence(
    query: str,
    evidence_list: list,
    evidence_map: dict,
    llm_client,
) -> tuple[list, dict]:
    """Query-time relevance filter: one LLM call to remove off-target evidence chunks.

    Judges whether each retrieved chunk applies to the patient profile described
    in the query (e.g., filters DCIS guidance when the query is about invasive
    breast cancer with T/N staging).

    Fail-open: any LLM error or unexpected response returns the original lists
    unchanged so that generation is never blocked by this step.
    """
    if not evidence_list:
        return evidence_list, evidence_map

    chunks_text = "\n\n".join(
        f"[{e.id}]（第{e.source_pages}頁）：{e.content[:350]}"
        for e in evidence_list
    )
    user_msg = f"查詢：{query}\n\n---\n{chunks_text}"

    try:
        result = llm_client.generate(system=_FILTER_SYSTEM, user=user_msg)
    except Exception as exc:
        logger.warning("Query enrichment LLM call failed: %s", exc)
        return evidence_list, evidence_map

    if not isinstance(result, dict):
        logger.warning("Query enrichment: unexpected response type %s", type(result))
        return evidence_list, evidence_map

    # result may be the QA-format dict (e.g. from StubLLMClient) rather than
    # the filter-format dict — detect by checking for known QA keys.
    if "answer" in result or "claims" in result:
        logger.warning("Query enrichment: LLM returned QA format instead of filter format; skipping")
        return evidence_list, evidence_map

    filtered_list: list = []
    filtered_map: dict = {}
    for item in evidence_list:
        verdict = result.get(item.id)
        if verdict is None:
            # LLM did not mention this ID — keep it (fail-open)
            filtered_list.append(item)
            filtered_map[item.id] = evidence_map[item.id]
            continue

        applicable = verdict.get("applicable", True)
        note = verdict.get("note", "")

        if applicable:
            filtered_list.append(item)
            filtered_map[item.id] = evidence_map[item.id]
        else:
            logger.info("Query enrichment filtered out %s — %s", item.id, note)

    removed = len(evidence_list) - len(filtered_list)
    if removed:
        logger.info(
            "Query enrichment: kept %d/%d evidence chunks, removed %d",
            len(filtered_list), len(evidence_list), removed,
        )

    return filtered_list, filtered_map
