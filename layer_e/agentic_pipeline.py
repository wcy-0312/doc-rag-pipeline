import json
import logging
import re
from typing import Optional

from layer_e import context_packer, guardrail, query_enrichment
from layer_e.agentic_tools import TOOL_DEFINITIONS, execute_tool
from layer_e.models import ClaimCitation, GenerationResult
from layer_e.prompt_builder import format_evidence_block

logger = logging.getLogger(__name__)

_SYSTEM_TEMPLATE = """\
你是醫療知識庫助理，協助醫護人員查詢文件。

已為你提供以下從文件中檢索到的 Evidence，每筆格式為 [EN]《文件名》第X頁：
{evidence_block}

你可以呼叫工具取得更多資訊：
- get_page_image: 取得 PDF 頁面整頁截圖，適合閱讀流程圖、治療路徑圖
- retrieve_more: 在文件中搜尋更多相關段落或表格

停止規則：
1. 每一個 medical claim 必須在 claims[] 陣列中列出對應的 Evidence 編號；answer 欄位為純文字散文，不需內嵌 [En] 標注
2. 無法找到支持的 claim 標記 citations: ["unsupported"]
3. 當所有 claim 都有 citation（或標記 unsupported）時，輸出最終 JSON
4. 若問題涉及特定 TNM 分期或分類（如 cT3N2M0），只能引用 Evidence 中明確對應該分期的敘述；若文件僅描述一般適用條件，請如實說明文件未針對此分期給出明確建議，並引用文件的一般條件供參考，不得自行推論符合性
5. 當答案涉及藥物適應症或健保給付條件時，必須從 Evidence 中完整列出所有條件（包括前置療程要求、時間限制、排除條件等），不得以「需符合特定條件」等模糊措辭概括
6. 當問題詢問特定分期或病人的「治療建議」時，若 Evidence 未涵蓋系統性治療（化療／靶向／內分泌）面向，建議先呼叫 retrieve_more 補充一次；若補充後仍無相關 Evidence，依現有資訊如實回答，說明指引在該面向未提供明確內容，不必反覆搜尋或拒絕回答

最終答案必須是以下 JSON 格式，不加 markdown fence：
{{"answer": "完整散文答案（不含任何 [En] 標注）", "claims": [{{"text": "claim 原文", "citations": ["E1", "E2"]}}], "evidence_summaries": [{{"id": "E1", "summary": "一到兩句中文摘要，說明此段內容對回答問題的貢獻，避免直接複製原文"}}], "abstain": false, "abstain_reason": null}}
"""

_SOFT_LIMIT_NOTICE = (
    "\n\n[系統提示] 你已使用多次工具。請根據目前已取得的所有資訊給出最佳答案。"
    "對無法找到 citation 的 claim 標記 [unsupported]。請立即輸出最終 JSON。"
)

# P2: coverage hint injected when treatment-recommendation query lacks systemic treatment evidence
_COVERAGE_HINT_TEMPLATE = (
    "\n\n[系統提示] 偵測到「治療建議」類問題，但目前 Evidence 未涵蓋系統性治療（化療／靶向／內分泌）面向。"
    "請在作答前先呼叫 retrieve_more 補充，例如：\n"
    "- retrieve_more(\"{stage}系統性治療 化療\")\n"
    "- retrieve_more(\"{stage}手術 新輔助治療\")\n"
    "確認多面向 Evidence 後再給出最終答案。"
)

_TREATMENT_QUERY_RE = re.compile(
    r'治療建議|如何治療|治療方案|治療原則|應.*治療|能.*治療|[cCpP][tT]\d[nN]\d[mM]\d',
)

_SYSTEMIC_KEYWORDS = [
    '化療', '化學治療', '靶向', '標靶', '內分泌', '荷爾蒙',
    'CDK', 'trastuzumab', 'pertuzumab', 'anthracycline', 'taxane',
    '系統性治療', '輔助治療', 'neoadjuvant', '新輔助',
]

# If the query already names specific systemic treatment modalities, the user is
# asking a comparison/selection question — P2 coverage hint is not needed.
_SYSTEMIC_IN_QUERY_RE = re.compile(
    r'化療|化學治療|靶向|標靶|內分泌|荷爾蒙治療|trastuzumab|pertuzumab|新輔助|neoadjuvant',
    re.IGNORECASE,
)

_TNM_RE = re.compile(r'[cCpP]?[tT](\d)[nN](\d)[mM](\d)')

_UNSUPPORTED_MARKER = "unsupported"


def _is_treatment_recommendation_query(query: str) -> bool:
    if not _TREATMENT_QUERY_RE.search(query):
        return False
    # Query already specifies systemic modalities → user is asking a selection
    # question, not an open "give me a plan" question; skip P2 hint.
    if _SYSTEMIC_IN_QUERY_RE.search(query):
        return False
    return True


def _lacks_systemic_coverage(evidence_list: list) -> bool:
    """Return True when no evidence chunk mentions systemic treatment."""
    combined = " ".join(item.content for item in evidence_list).lower()
    return not any(kw.lower() in combined for kw in _SYSTEMIC_KEYWORDS)


def _extract_stage_hint(query: str) -> str:
    m = _TNM_RE.search(query)
    return m.group(0) if m else "早期乳癌"


def _renumber_evidence(evidence_list: list, evidence_map: dict) -> tuple[list, dict]:
    """Renumber evidence items E1, E2, ... by position after filtering.

    format_evidence_block always assigns labels by position in the list, so
    evidence_map keys must stay in sync — otherwise LLM citations like "E1"
    won't resolve to the correct entry in evidence_map.
    """
    from dataclasses import replace as dc_replace
    new_list = []
    new_map = {}
    for i, item in enumerate(evidence_list, start=1):
        new_id = f"E{i}"
        new_list.append(dc_replace(item, id=new_id))
        new_map[new_id] = evidence_map[item.id]
    return new_list, new_map


def _format_document_index(index: dict) -> str:
    """Format document_index dict as indented text for the system prompt."""
    lines: list[str] = []

    def _render(sections: list, depth: int) -> None:
        for sec in sections:
            title = sec.get("title", "")
            if title:
                lines.append("  " * depth + title)
            _render(sec.get("sections", []), depth + 1)

    _render(index.get("sections", []), 0)
    return "\n".join(lines)


class AgenticPipeline:
    def __init__(
        self,
        llm_client,
        retriever,
        pdf_path: str,
        doc_stem: str,
        soft_limit: int = 8,
        hard_limit: int = 12,
        abstention_threshold: float = 0.0,
    ):
        self._llm = llm_client
        self._retriever = retriever
        self._pdf_path = pdf_path
        self._doc_stem = doc_stem
        self._soft_limit = soft_limit
        self._hard_limit = hard_limit
        self._abstention_threshold = abstention_threshold
        self._enable_query_enrichment = True

        # Load document index for pre-query navigation
        document_index = None
        if hasattr(retriever, "get_document_index"):
            try:
                document_index = retriever.get_document_index(doc_stem)
            except Exception:
                pass
        self._document_outline: str | None = (
            _format_document_index(document_index) if document_index else None
        )

    def run(self, query: str, ranked_results: list) -> GenerationResult:
        # Guardrail: abstain if evidence is insufficient
        should_abstain, reason = guardrail.check_abstention(
            ranked_results, threshold=self._abstention_threshold
        )
        if should_abstain:
            return GenerationResult(
                answer="",
                claims=[],
                evidence_map={},
                unsupported_claims=[],
                abstain=True,
                abstain_reason=reason,
                safety_verdict="abstained",
                steps_log=[],
            )

        evidence_list, evidence_map = context_packer.pack(ranked_results)
        if self._enable_query_enrichment:
            evidence_list, evidence_map = query_enrichment.filter_evidence(
                query, evidence_list, evidence_map, self._llm
            )
            evidence_list, evidence_map = _renumber_evidence(evidence_list, evidence_map)
        evidence_block = format_evidence_block(evidence_list)
        system_content = _SYSTEM_TEMPLATE.format(evidence_block=evidence_block)
        if self._document_outline:
            system_content += (
                "\n\n文件結構概覽（可輔助決定 retrieve_more 應搜尋哪個章節）：\n"
                + self._document_outline
            )
        # P2: inject coverage hint when treatment query lacks systemic evidence
        if _is_treatment_recommendation_query(query) and _lacks_systemic_coverage(evidence_list):
            stage = _extract_stage_hint(query)
            system_content += _COVERAGE_HINT_TEMPLATE.format(stage=stage)
            logger.info("Coverage hint injected for treatment query (stage=%s)", stage)

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]

        steps_log = []
        tool_call_count = 0

        for _ in range(self._hard_limit):
            # Always work on a copy so the original messages list is never mutated
            msgs = list(messages)
            # Inject soft-limit notice into system message when approaching limit
            if tool_call_count >= self._soft_limit:
                msgs[0] = {
                    "role": "system",
                    "content": system_content + _SOFT_LIMIT_NOTICE,
                }

            tool_calls, final_content = self._llm.generate_with_tools(msgs, TOOL_DEFINITIONS)

            if not tool_calls:
                # LLM gave final answer
                return self._parse_final(final_content, evidence_map, steps_log)

            # Execute each tool call
            # Add assistant message with tool_calls to history
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"]),
                        },
                    }
                    for tc in tool_calls
                ],
            })

            pending_images = []
            for tc in tool_calls:
                tool_call_count += 1
                step = {
                    "step_no": len(steps_log) + 1,
                    "tool": tc["name"],
                    "arguments": tc["arguments"],
                    "reason": tc["arguments"].get("reason", ""),
                }
                steps_log.append(step)
                logger.info("Agentic tool call %d: %s(%s)", tool_call_count, tc["name"], tc["arguments"])

                try:
                    text_result, b64_image = execute_tool(
                        tc, self._pdf_path, self._retriever, self._doc_stem
                    )
                except Exception as exc:
                    text_result = f"工具執行失敗：{exc}"
                    b64_image = None

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": text_result,
                })

                if b64_image:
                    pending_images.append(b64_image)

            # Append images as a user message so GPT-4.1 can see them
            if pending_images:
                content = [{"type": "text", "text": f"以上工具附帶 {len(pending_images)} 張截圖："}]
                for b64 in pending_images:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    })
                messages.append({"role": "user", "content": content})

        # Hard limit reached — force conclusion via one final call without tools
        logger.warning("Agentic loop hit hard_limit=%d, forcing conclusion", self._hard_limit)
        messages[0] = {"role": "system", "content": system_content + _SOFT_LIMIT_NOTICE}
        _, final_content = self._llm.generate_with_tools(messages, [])
        return self._parse_final(final_content, evidence_map, steps_log)

    def _parse_final(self, content: str, evidence_map: dict, steps_log: list) -> GenerationResult:
        if content is None:
            logger.warning(
                "generate_with_tools returned ([], None) — LLM gave no content; treating as empty answer"
            )
        try:
            cleaned = re.sub(r',\s*([\]\}])', r'\1', content or "{}")
            data = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            data = {"answer": content or "", "claims": [], "abstain": False, "abstain_reason": None}

        if data.get("abstain"):
            return GenerationResult(
                answer="",
                claims=[],
                evidence_map=evidence_map,
                unsupported_claims=[],
                abstain=True,
                abstain_reason=data.get("abstain_reason"),
                safety_verdict="abstained",
                steps_log=steps_log,
            )

        claims = []
        unsupported = []
        for c in data.get("claims", []):
            raw_citations = c.get("citations", [])
            clean = [eid for eid in raw_citations if eid != _UNSUPPORTED_MARKER]
            if _UNSUPPORTED_MARKER in raw_citations or not raw_citations:
                unsupported.append(c.get("text", ""))
            claims.append(ClaimCitation(text=c.get("text", ""), citations=clean))

        evidence_summaries = {
            item["id"]: item["summary"]
            for item in data.get("evidence_summaries", [])
            if "id" in item and "summary" in item
        }

        safety_verdict = "needs_review" if unsupported else "safe"
        return GenerationResult(
            answer=data.get("answer", ""),
            claims=claims,
            evidence_map=evidence_map,
            unsupported_claims=unsupported,
            abstain=False,
            abstain_reason=None,
            safety_verdict=safety_verdict,
            steps_log=steps_log,
            evidence_summaries=evidence_summaries,
        )
