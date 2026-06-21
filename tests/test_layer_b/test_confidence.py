import sys, os

from layer_b.models import IRCell, IRTable, QC
from layer_b.pipeline import assess


def _cell(conf=None, header_source="flag", is_col_header=False, is_row_header=False):
    return IRCell(row_index=0, col_index=0, row_span=1, col_span=1,
                  content="x", is_col_header=is_col_header, is_row_header=is_row_header,
                  header_source=header_source, confidence=conf)


def _table(cells, empty_cell_rate=0.0, warnings=None,
           estimated_info_loss_rate=None, word_avg=None, low_confidence_rate=None):
    return IRTable(
        table_id="t_000", source_tool="azure_cu", source_pages=[1],
        cells=cells,
        qc=QC(
            empty_cell_rate=empty_cell_rate,
            qc_level="ok",
            warnings=warnings or [],
            word_avg=word_avg,
            low_confidence_rate=low_confidence_rate,
            estimated_info_loss_rate=estimated_info_loss_rate,
        ),
    )


# ── 1. test_high_confidence ──────────────────────────────────────────────────

def test_high_confidence():
    """info_loss=0.01, word_avg=0.95 → level=high, score=0.01"""
    result = assess(_table([_cell()], estimated_info_loss_rate=0.01, word_avg=0.95))
    assert result["level"] == "high", f"expected high, got {result['level']}"
    assert result["score"] == 0.01
    assert any("estimated_info_loss_rate" in r for r in result["reasons"])


# ── 2. test_medium_info_loss ──────────────────────────────────────────────────

def test_medium_info_loss():
    """info_loss=0.05 → level=medium"""
    result = assess(_table([_cell()], estimated_info_loss_rate=0.05))
    assert result["level"] == "medium", f"expected medium, got {result['level']}"
    assert result["score"] == 0.05


# ── 3. test_high_info_loss ────────────────────────────────────────────────────

def test_high_info_loss():
    """info_loss=0.15 → level=low"""
    result = assess(_table([_cell()], estimated_info_loss_rate=0.15))
    assert result["level"] == "low", f"expected low, got {result['level']}"
    assert result["score"] == 0.15


# ── 4. test_null_info_loss_no_deductions ──────────────────────────────────────

def test_null_info_loss_no_deductions():
    """info_loss=None, word_avg=None → level=medium"""
    result = assess(_table([_cell()], estimated_info_loss_rate=None, word_avg=None))
    assert result["level"] == "medium", f"expected medium, got {result['level']}"
    assert result["score"] is None
    assert any("null" in r for r in result["reasons"])


# ── 5. test_scan_warning_low ──────────────────────────────────────────────────

def test_scan_warning_low():
    """scan_detected warning → level=low"""
    result = assess(_table([_cell()], warnings=["scan_detected"]))
    assert result["level"] == "low", f"expected low, got {result['level']}"
    assert any("scan_detected" in r for r in result["reasons"])


# ── 6. test_high_empty_cell_rate ─────────────────────────────────────────────

def test_high_empty_cell_rate():
    """empty_cell_rate=0.35, no other deductions → level=medium (single deduction, not critical)"""
    result = assess(_table([_cell()], empty_cell_rate=0.35))
    assert result["level"] == "medium", f"expected medium, got {result['level']}"
    assert any("empty_cell_rate" in r for r in result["reasons"])


# ── 7. test_low_word_avg ──────────────────────────────────────────────────────

def test_low_word_avg():
    """word_avg=0.60 → level=low"""
    result = assess(_table([_cell()], word_avg=0.60))
    assert result["level"] == "low", f"expected low, got {result['level']}"
    assert any("word_avg" in r for r in result["reasons"])


# ── 8. test_heuristic_header_medium ──────────────────────────────────────────

def test_heuristic_header_medium():
    """heuristic header + info_loss=0.01 → level=medium (heuristic alone not enough for low)"""
    cells = [_cell(header_source="heuristic", is_col_header=True)]
    result = assess(_table(cells, estimated_info_loss_rate=0.01))
    assert result["level"] == "medium", f"expected medium, got {result['level']}"
    assert any("heuristic" in r for r in result["reasons"])


if __name__ == "__main__":
    test_high_confidence()
    test_medium_info_loss()
    test_high_info_loss()
    test_null_info_loss_no_deductions()
    test_scan_warning_low()
    test_high_empty_cell_rate()
    test_low_word_avg()
    test_heuristic_header_medium()
    print("All confidence tests passed.")
