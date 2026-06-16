"""
keyword_extractor.py — 兩段式文件關鍵字萃取

架構：
  Pass 1：逐 section 送 LLM 萃取候選關鍵字（每次 context 小，適用任意頁數）
  Pass 2：所有候選詞合併後送 LLM dedup + 排名 → 文件級關鍵字

設計原則：
  - 輸入 markdown（所有 extractor 都有此欄位）
  - Pass 1 每個 section 截斷至 800 字，保持速度
  - Pass 2 輸入只有短詞清單，不會爆 context window
"""

from __future__ import annotations
import re, json


def _split_sections(markdown: str) -> list[str]:
    """按 markdown 標題分割為 sections，過濾太短的片段。"""
    parts = re.split(r'\n(?=#{1,3}\s)', markdown)
    return [p.strip() for p in parts if len(p.strip()) > 80]


def _call_llm_json(prompt: str, llm) -> list[str]:
    from langchain_core.messages import HumanMessage
    try:
        raw = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        if raw.startswith("```"):
            raw = re.sub(r'^```(?:json)?\n?', '', raw)
            raw = re.sub(r'\n?```$', '', raw)
        result = json.loads(raw)
        return result if isinstance(result, list) else []
    except Exception:
        return []


def _pass1_section(text: str, llm) -> list[str]:
    excerpt = text[:800]
    prompt = (
        "從以下醫療文件段落中萃取專業關鍵詞（疾病、治療、藥物、程序等）。\n"
        "只輸出 JSON 字串陣列，不要說明。範例：[\"肺癌\", \"化學治療\", \"EGFR\"]\n\n"
        f"段落：\n{excerpt}"
    )
    return _call_llm_json(prompt, llm)


def _pass2_rank(candidates: list[str], llm, max_keywords: int) -> list[str]:
    unique = list(dict.fromkeys(candidates))
    if len(unique) <= max_keywords:
        return unique
    cand_str = "、".join(unique[:100])
    prompt = (
        f"以下是從醫療文件各章節萃取的關鍵詞候選清單：\n{cand_str}\n\n"
        f"請整合並輸出最重要的 {max_keywords} 個文件級關鍵詞。\n"
        "規則：刪除重複、過於通用（「患者」「醫師」「治療」）的詞，保留具體的醫療術語。\n"
        "只輸出 JSON 字串陣列，不要說明。"
    )
    result = _call_llm_json(prompt, llm)
    return result[:max_keywords] if result else unique[:max_keywords]


def extract_keywords(
    markdown: str,
    llm,
    max_keywords: int = 10,
) -> list[str]:
    """兩段式關鍵字萃取。

    Args:
        markdown:     文件的 markdown 全文（schema-v3.0 data.markdown）
        llm:          Vision LLM 實例（支援 gemma3 / gemma4 / gpt41）
        max_keywords: 最終輸出的關鍵字數量上限

    Returns:
        關鍵字清單，萃取失敗時回傳 []
    """
    sections = _split_sections(markdown)
    if not sections:
        return []

    # Pass 1：逐 section 萃取候選
    all_candidates: list[str] = []
    for sec in sections:
        candidates = _pass1_section(sec, llm)
        all_candidates.extend(candidates)

    if not all_candidates:
        return []

    # Pass 2：dedup + 排名
    return _pass2_rank(all_candidates, llm, max_keywords)
