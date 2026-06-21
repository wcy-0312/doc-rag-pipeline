from __future__ import annotations
import json
import logging
import re
from typing import Any

from layer_b.models import RetrievalUnit

logger = logging.getLogger(__name__)

_MIN_TABLE_CELLS = 3
_DATE_ONLY_RE = re.compile(r'^\d{4}/\d{2}/\d{2}')

_TABLE_SYSTEM = (
    "你是醫療知識庫整理員。請判斷以下表格是否包含對臨床決策有意義的醫療資訊"
    "（如疾病分期、治療建議、用藥劑量、診斷標準等）。\n"
    "若無意義（例如版本歷史、行政表格），只輸出 {\"meaningful\": false}\n"
    "若有意義，輸出 JSON：\n"
    "{\"meaningful\": true, \"semantic_summary\": \"一句話描述此表格的醫療用途\","
    " \"applicability\": \"適用對象或情境（20字以內）\","
    " \"answers_questions\": [\"可回答的具體臨床問題，最多3條\"]}\n"
    "只輸出 JSON，不要其他說明。"
)

_FIGURE_SYSTEM = (
    "你是醫療知識庫整理員。請根據以下文字描述判斷圖表是否包含對臨床決策有意義的醫療資訊"
    "（如治療流程圖、演算法、決策樹、臨床指引步驟等）。\n"
    "若無意義（例如標誌、裝飾圖、空白頁），只輸出 {\"meaningful\": false}\n"
    "若有意義，輸出 JSON：\n"
    "{\"meaningful\": true, \"semantic_summary\": \"一句話描述此圖的醫療用途\","
    " \"applicability\": \"適用對象或情境（20字以內）\","
    " \"answers_questions\": [\"可回答的具體臨床問題，最多3條\"]}\n"
    "只輸出 JSON，不要其他說明。"
)


def _is_trivial_table(unit: RetrievalUnit) -> bool:
    """Layer 1 pre-filter: True for structurally trivial tables that need no enrichment."""
    rows = unit.structured_json.get("rows", [])
    if not rows:
        return True
    all_cells = [c for r in rows for c in r.get("cells", [])]
    # Skip tables with fewer than _MIN_TABLE_CELLS cells (≤ 2 cells)
    if len(all_cells) < _MIN_TABLE_CELLS:
        return True
    # All non-empty values start with a date — version history table
    values = [c.get("value", "") for c in all_cells if c.get("value", "").strip()]
    return bool(values) and all(_DATE_ONLY_RE.match(v) for v in values)


_FIGURE_PLACEHOLDER_RE = re.compile(r"^\[圖表[^\]]*\]\s*$")


def _is_trivial_figure(unit: RetrievalUnit) -> bool:
    """Layer 1 pre-filter: True for figures with no useful text content."""
    text = unit.embedding_text.strip()
    if not text:
        return True
    # Fallback placeholder text — no elements were linked
    return bool(_FIGURE_PLACEHOLDER_RE.match(text))


def _build_prefix(result: dict) -> str:
    lines = [f"[摘要] {result['semantic_summary']}"]
    if result.get("applicability"):
        lines.append(f"[適用] {result['applicability']}")
    questions = result.get("answers_questions") or []
    if questions:
        lines.append(f"[可解答] {'；'.join(q for q in questions[:3] if q)}")
    return "\n".join(lines) + "\n"


def _call_llm(llm_client: Any, system: str, user: str) -> dict | None:
    """Call llm_client.generate() and return parsed dict, or None on failure."""
    try:
        response = llm_client.generate(system=system, user=user)
        # generate() returns a dict (already parsed by _parse_json_response).
        # If the LLM followed the enrichment prompt, it contains "meaningful".
        if isinstance(response, dict) and "meaningful" in response:
            return response
        # Fallback: try parsing the "answer" key if present (should not happen with
        # a compliant LLM, but defensive against prompt drift)
        answer_text = response.get("answer", "") if isinstance(response, dict) else ""
        if not answer_text:
            return None
        raw = answer_text.strip()
        if raw.startswith("```"):
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw.strip())
        return json.loads(raw)
    except Exception as exc:
        logger.warning("Enrichment LLM call failed: %s", exc)
        return None


def enrich_units(
    units: list[RetrievalUnit],
    llm_client: Any,
) -> list[RetrievalUnit]:
    """Apply LLM semantic enrichment to table and figure units.

    Table and figure units that pass the Layer 1 structural pre-filter are sent
    to the LLM with a domain-specific prompt. If meaningful=true, a semantic
    prefix (摘要/適用/可解答) is prepended to embedding_text. Paragraph units
    and trivial tables/figures are returned unchanged.

    Args:
        units: Output of process_document().
        llm_client: Any object with .generate(system, user) -> dict.
                    Uses GPT41Client or any LLMClient subclass in practice.
    """
    from dataclasses import replace

    result = []
    for unit in units:
        unit_type = unit.structured_json.get("type")

        if unit_type == "table":
            if not _is_trivial_table(unit):
                enrichment = _call_llm(llm_client, _TABLE_SYSTEM, unit.embedding_text[:600])
                if enrichment and enrichment.get("meaningful"):
                    unit = replace(unit, embedding_text=_build_prefix(enrichment) + unit.embedding_text)

        elif unit_type == "figure":
            if not _is_trivial_figure(unit):
                enrichment = _call_llm(llm_client, _FIGURE_SYSTEM, unit.embedding_text[:600])
                if enrichment and enrichment.get("meaningful"):
                    unit = replace(unit, embedding_text=_build_prefix(enrichment) + unit.embedding_text)

        result.append(unit)
    return result
