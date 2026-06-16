from __future__ import annotations
from layer_b.models import IRTable

_INFO_LOSS_LOW = 0.03    # qc=good 門檻
_INFO_LOSS_HIGH = 0.10   # qc=danger 門檻
_WORD_AVG_HIGH = 0.90    # word-level confidence 高門檻
_EMPTY_CELL_RATE_THRESHOLD = 0.30


def assess(table: IRTable) -> dict:
    """Assess table extraction confidence.

    Returns:
        {
            "level": "high" | "medium" | "low",
            "score": float | None,   # estimated_info_loss_rate (lower is better)
            "reasons": list[str],
            "fallback_available": bool,
            "page_image_refs": dict,
        }
    """
    reasons: list[str] = []
    deductions: list[str] = []

    # 1. estimated_info_loss_rate (primary signal, all three extractors)
    info_loss = table.qc.estimated_info_loss_rate
    if info_loss is not None:
        reasons.append(f"estimated_info_loss_rate={info_loss:.3f}")
        if info_loss > _INFO_LOSS_HIGH:
            deductions.append("high_info_loss")
        elif info_loss > _INFO_LOSS_LOW:
            deductions.append("medium_info_loss")
    else:
        reasons.append("estimated_info_loss_rate=null")

    # 2. word_avg (azure_cu and azure_di only; None for docling)
    word_avg = table.qc.word_avg
    if word_avg is not None:
        reasons.append(f"word_avg={word_avg:.3f}")
        if word_avg < 0.70:
            deductions.append("low_word_avg")

    # 3. empty cell rate
    ecr = table.qc.empty_cell_rate
    reasons.append(f"empty_cell_rate={ecr:.2f}")
    if ecr >= _EMPTY_CELL_RATE_THRESHOLD:
        deductions.append(f"high_empty_cell_rate={ecr:.2f}")

    # 4. QC warnings
    for w in table.qc.warnings:
        reasons.append(f"qc_warning={w}")
        if "scan" in w.lower():
            deductions.append("scan_detected")
        if "FALLBACK" in w:
            deductions.append("extractor_fallback")

    # 5. header source
    heuristic_headers = [c for c in table.cells
                         if c.header_source == "heuristic"
                         and (c.is_col_header or c.is_row_header)]
    if heuristic_headers:
        reasons.append("header_source=heuristic")
        deductions.append("heuristic_header")

    # Determine level
    if "high_info_loss" in deductions or "scan_detected" in deductions or "low_word_avg" in deductions:
        level = "low"
    elif not deductions:
        # No deductions: high if info_loss low and word_avg good, else medium
        if info_loss is not None and info_loss <= _INFO_LOSS_LOW:
            if word_avg is None or word_avg >= _WORD_AVG_HIGH:
                level = "high"
            else:
                level = "medium"
        else:
            level = "medium"
    else:
        # Some deductions but not critical
        level = "medium"

    # score: estimated_info_loss_rate if available, else None
    score = round(info_loss, 3) if info_loss is not None else None

    return {
        "level": level,
        "score": score,
        "reasons": reasons,
        "fallback_available": bool(table.page_image_refs),
        "page_image_refs": table.page_image_refs,
    }
