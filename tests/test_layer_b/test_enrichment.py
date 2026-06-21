import pytest
from dataclasses import dataclass, field
from layer_b.models import RetrievalUnit
from layer_b.enrichment import enrich_units, _is_trivial_table


# ── Stubs ──────────────────────────────────────────────────────────────────

class _MeaningfulLLM:
    """Always returns meaningful=True with a fixed enrichment."""
    def generate(self, system: str, user: str) -> dict:
        return {
            "meaningful": True,
            "semantic_summary": "乳癌分期對照表",
            "applicability": "需要確認分期的病人",
            "answers_questions": ["T2N1M0是哪個期?", "Stage IIB的TNM條件?"],
        }


class _TrivialLLM:
    """Always returns meaningful=False."""
    def generate(self, system: str, user: str) -> dict:
        return {"meaningful": False}


class _FailingLLM:
    """Always raises an exception."""
    def generate(self, system: str, user: str) -> dict:
        raise RuntimeError("LLM error")


def _table_unit(embedding_text: str = "T0 N1 M0 Stage I", rows=None) -> RetrievalUnit:
    if rows is None:
        rows = [
            {"row_header_path": [], "cells": [
                {"col_header_path": ["TNM"], "value": "T0 N1 M0"},
                {"col_header_path": ["Stage"], "value": "Stage I"},
                {"col_header_path": ["Grade"], "value": "G2"},
            ]},
        ]
    return RetrievalUnit(
        retrieval_unit_id="doc_t_001",
        source_tool="azure_cu",
        embedding_text=embedding_text,
        structured_json={"type": "table", "rows": rows, "merge_rate": 0.1},
        display_markdown="| TNM | Stage |",
        confidence_level="high",
        quality_flag="ok",
        retrieval_weight=1.0,
        source_pages=[5],
        row_texts=["T0 N1 M0為Stage I"],
    )


def _figure_unit(embedding_text: str = "cT2N1M0 治療流程 新輔助化療建議") -> RetrievalUnit:
    return RetrievalUnit(
        retrieval_unit_id="doc_f_001",
        source_tool="azure_cu",
        embedding_text=embedding_text,
        structured_json={"type": "figure", "page": 3, "caption": "", "area": 6.0},
        display_markdown="[圖表 第3頁]",
        confidence_level="high",
        quality_flag="ok",
        retrieval_weight=1.0,
        source_pages=[3],
        row_texts=[],
    )


def _para_unit() -> RetrievalUnit:
    return RetrievalUnit(
        retrieval_unit_id="doc_p_001",
        source_tool="azure_cu",
        embedding_text="化療方案建議",
        structured_json={"type": "paragraph", "content": "化療方案建議"},
        display_markdown="化療方案建議",
        confidence_level="high",
        quality_flag="ok",
        retrieval_weight=1.0,
        source_pages=[4],
        row_texts=[],
    )


# ── Tests ──────────────────────────────────────────────────────────────────

def test_table_gets_enrichment_prefix():
    units = enrich_units([_table_unit()], _MeaningfulLLM())
    assert len(units) == 1
    u = units[0]
    assert "[摘要] 乳癌分期對照表" in u.embedding_text
    assert "T0 N1 M0 Stage I" in u.embedding_text  # original text preserved


def test_table_unchanged_when_not_meaningful():
    original = _table_unit()
    units = enrich_units([original], _TrivialLLM())
    assert units[0].embedding_text == original.embedding_text


def test_paragraph_always_unchanged():
    original = _para_unit()
    units = enrich_units([original], _MeaningfulLLM())
    assert units[0].embedding_text == original.embedding_text


def test_figure_gets_enrichment_prefix():
    units = enrich_units([_figure_unit()], _MeaningfulLLM())
    assert "[摘要]" in units[0].embedding_text
    assert "cT2N1M0 治療流程" in units[0].embedding_text


def test_llm_failure_leaves_unit_unchanged():
    original = _table_unit()
    units = enrich_units([original], _FailingLLM())
    assert units[0].embedding_text == original.embedding_text


def test_trivial_table_skips_llm():
    """Table with <= 2 cells never calls the LLM."""
    call_count = []

    class _CountingLLM:
        def generate(self, system, user):
            call_count.append(1)
            return {"meaningful": True, "semantic_summary": "x", "applicability": "y", "answers_questions": []}

    tiny_rows = [{"row_header_path": [], "cells": [{"col_header_path": ["A"], "value": "B"}]}]
    units = enrich_units([_table_unit(rows=tiny_rows)], _CountingLLM())
    assert call_count == []


def test_is_trivial_table_date_only():
    rows = [
        {"row_header_path": [], "cells": [{"col_header_path": ["Version"], "value": "2025/11/11 Version 19.0"}]},
        {"row_header_path": [], "cells": [{"col_header_path": ["Version"], "value": "2024/09/24 Version 17.0"}]},
        {"row_header_path": [], "cells": [{"col_header_path": ["Version"], "value": "2023/11/14 Version 16.0"}]},
        {"row_header_path": [], "cells": [{"col_header_path": ["Version"], "value": "2022/12/27 Version 15.0"}]},
    ]
    unit = _table_unit(rows=rows)
    assert _is_trivial_table(unit) is True


def test_mixed_units_only_table_and_figure_enriched():
    units = [_para_unit(), _table_unit(), _figure_unit()]
    result = enrich_units(units, _MeaningfulLLM())
    assert result[0].embedding_text == "化療方案建議"      # para unchanged
    assert "[摘要]" in result[1].embedding_text            # table enriched
    assert "[摘要]" in result[2].embedding_text            # figure enriched
