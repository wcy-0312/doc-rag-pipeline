from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from layer_e.llm_client import LLMClient

_JSON_RE = re.compile(r'\{(?:[^{}]|\{[^{}]*\})*"relevant_guidelines"(?:[^{}]|\{[^{}]*\})*\}')

_ROUTE_SYSTEM = "你是醫療查詢路由助理，協助判斷查詢所需的資料來源。"

_ROUTE_USER_TEMPLATE = """\
查詢：{query}

可用的治療指引：
{guidelines_text}

是否有病人個人資料可查詢：{has_patient_data}

請判斷：
1. 此查詢是否需要先查詢病人個人資料才能回答（若查詢本身已包含分期、診斷等足夠資訊則不需要）
2. 哪幾份指引與此查詢相關（可多選，僅選相關的）

只回傳 JSON，格式：{{"need_patient_context": false, "relevant_guidelines": [0, 2]}}
若無相關指引，回傳：{{"need_patient_context": false, "relevant_guidelines": []}}"""


@dataclass
class RouteDecision:
    need_patient_context: bool
    selected_stems: list[str] = field(default_factory=list)


class TreeRouter:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    def route(
        self,
        query: str,
        static_summaries: dict[str, str],
        has_dynamic: bool,
    ) -> RouteDecision:
        if not static_summaries:
            return RouteDecision(need_patient_context=False, selected_stems=[])

        stems = list(static_summaries.keys())
        guidelines_text = "\n".join(
            f"[{i}] {stem} — {summary}"
            for i, (stem, summary) in enumerate(static_summaries.items())
        )
        has_patient_str = "是（病人資料可供查詢）" if has_dynamic else "否"
        prompt = _ROUTE_USER_TEMPLATE.format(
            query=query,
            guidelines_text=guidelines_text,
            has_patient_data=has_patient_str,
        )
        response = self._llm.generate_text(prompt, system=_ROUTE_SYSTEM)
        return self._parse_decision(response, stems)

    def _parse_decision(self, text: str, stems: list[str]) -> RouteDecision:
        m = _JSON_RE.search(text or "")
        if not m:
            return RouteDecision(need_patient_context=False, selected_stems=list(stems))
        try:
            data = json.loads(m.group())
            need_ctx = bool(data.get("need_patient_context", False))
            raw_indices = data.get("relevant_guidelines", [])
            if raw_indices == []:
                # LLM explicitly says no relevant guidelines → honour that
                return RouteDecision(need_patient_context=need_ctx, selected_stems=[])
            indices = [i for i in raw_indices if isinstance(i, int) and 0 <= i < len(stems)]
            selected = [stems[i] for i in indices] if indices else list(stems)
            return RouteDecision(need_patient_context=need_ctx, selected_stems=selected)
        except (json.JSONDecodeError, TypeError):
            return RouteDecision(need_patient_context=False, selected_stems=list(stems))
