import json
import logging
from typing import Optional

from layer_e import context_packer, guardrail
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
1. 答案中每一個 medical claim 必須引用 Evidence 編號（如 [E1]）
2. 無法找到支持的 claim 標記 [unsupported]
3. 當所有 claim 都有 citation（或標記 [unsupported]）時，輸出最終 JSON

最終答案必須是以下 JSON 格式，不加 markdown fence：
{{"answer": "完整答案", "claims": [{{"text": "claim", "citations": ["E1"]}}], "abstain": false, "abstain_reason": null}}
"""

_SOFT_LIMIT_NOTICE = (
    "\n\n[系統提示] 你已使用多次工具。請根據目前已取得的所有資訊給出最佳答案。"
    "對無法找到 citation 的 claim 標記 [unsupported]。請立即輸出最終 JSON。"
)

_UNSUPPORTED_MARKER = "unsupported"


class AgenticPipeline:
    def __init__(
        self,
        llm_client,
        retriever,
        pdf_path: str,
        doc_stem: str,
        soft_limit: int = 8,
        hard_limit: int = 12,
        abstention_threshold: float = 0.10,
    ):
        self._llm = llm_client
        self._retriever = retriever
        self._pdf_path = pdf_path
        self._doc_stem = doc_stem
        self._soft_limit = soft_limit
        self._hard_limit = hard_limit
        self._abstention_threshold = abstention_threshold

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
        evidence_block = format_evidence_block(evidence_list)
        system_content = _SYSTEM_TEMPLATE.format(evidence_block=evidence_block)

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
            data = json.loads(content or "{}")
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
        )
