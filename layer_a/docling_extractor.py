"""
docling_extractor.py — Docling Extractor（schema-v3.0）

轉換層職責：Word（.doc / .docx）→ Docling 原始結構，最小轉換。

Docling export_to_dict() 原始輸出直接保留：
  texts[]        — 所有文字項目（含 bbox / charspan / label / formatting）
  tables[]       — 所有表格（含 cell-level data：bbox / row_span / col_span / header）
  pictures[]     — 所有圖片（has_image=False；嵌入圖片暫不支援渲染）
  pages{}        — 頁面資訊（含 size）
  groups[]       — 分組結構
  key_value_items[] — 鍵值對
  body           — 文件樹根節點
  furniture      — 頁首/頁尾等非正文元素

另外計算：
  markdown  — Docling export_to_markdown()
  qc         — 資訊損失量化
  metadata   — 標準四層

支援格式：.docx（直接）、.doc（LibreOffice 轉換後遞迴處理）
不支援：PDF、PPTX、Excel — 請分別使用 azure_cu 或其他路徑
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── QC 計算 ──────────────────────────────────────────────────────────────────

import re as _re
# 私用區（U+E000-F8FF）+ 替換字元（U+FFFD），不含 CJK 相容字（U+F900-FAFF）
_PUA_RE = _re.compile(r"[-�]")


def _compute_qc(raw: dict, page_count: int, pictures: list[dict]) -> dict:
    texts  = raw.get("texts",  [])
    tables = raw.get("tables", [])

    # ── 1. 空白頁 & 低密度頁 ──────────────────────────────────────────────
    items_per_page_map: dict[int, int] = {}
    for key in ("texts", "tables", "pictures"):
        for item in raw.get(key, []):
            for prov in item.get("prov", []):
                pn = prov.get("page_no", 0)
                items_per_page_map[pn] = items_per_page_map.get(pn, 0) + 1

    all_page_nos = set(range(1, page_count + 1))
    total_items  = sum(items_per_page_map.values())
    avg_density  = total_items / max(page_count, 1)
    low_threshold = max(1, avg_density * 0.3)
    low_density_pages = sorted(
        pn for pn, cnt in items_per_page_map.items() if cnt < low_threshold
    )

    # ── 2. 亂碼字元率（PUA / 亂碼字元 / 總字元）─────────────────────────
    chars_total   = sum(len(t.get("text") or "") for t in texts)
    chars_garbled = sum(
        len(_PUA_RE.findall(t.get("text") or "")) for t in texts
    )
    garbled_char_rate = round(chars_garbled / max(chars_total, 1), 4)

    # ── 3. 表格空白 cell 率 ───────────────────────────────────────────────
    cells_total = 0
    cells_empty = 0
    for tbl in tables:
        for cell in tbl.get("data", {}).get("table_cells", []):
            cells_total += 1
            if not (cell.get("text") or "").strip():
                cells_empty += 1
    empty_cell_rate = round(cells_empty / max(cells_total, 1), 4)

    # ── 4. 圖片覆蓋：DOCX 嵌入圖片目前不支援渲染（has_image=False），損失率固定 0
    _ = pictures  # 保留參數以備未來支援渲染

    # ── 5. 綜合損失率 ─────────────────────────────────────────────────────
    text_loss  = garbled_char_rate
    page_loss  = round(len(low_density_pages) / max(page_count, 1), 4)
    table_loss = empty_cell_rate

    estimated = round(
        0.45 * text_loss +
        0.25 * page_loss +
        0.20 * table_loss,
        4
    )
    qc_level = (
        "danger"  if estimated > 0.10 else
        "warning" if estimated > 0.03 else
        "good"
    )

    return {
        "estimated_info_loss_rate": estimated,
        "qc_level":                 qc_level,
        "warnings":                 [],
    }


# ── 主入口 ───────────────────────────────────────────────────────────────────

def convert_word_docling(
    word_path: Path,
    llm=None,
    category: str = "",
) -> dict:
    """Word path：.doc / .docx → Docling 原始結構（schema-v3.0）。

    .doc 檔案先由 LibreOffice 轉為 .docx，再遞迴處理。

    Args:
        word_path : .doc 或 .docx 路徑
        llm       : LLM 實例（有值時對 markdown 進行關鍵字萃取）
        category  : 文件類別
    """
    from docling.document_converter import DocumentConverter
    from metadata_builder import build_metadata

    suffix = word_path.suffix.lower()

    if suffix not in {".doc", ".docx"}:
        raise ValueError(
            f"Word 路徑僅接受 .doc / .docx，收到：{suffix}。"
            "PDF 請用 azure_cu；照片請用 azure_di。"
        )

    # .doc → .docx via LibreOffice，再遞迴
    if suffix == ".doc":
        import subprocess, tempfile
        with tempfile.TemporaryDirectory() as _tmpdir:
            result = subprocess.run(
                ["libreoffice", "--headless", "--convert-to", "docx",
                 "--outdir", _tmpdir, str(word_path)],
                capture_output=True, timeout=60,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"libreoffice conversion failed: {result.stderr.decode()}"
                )
            _converted = Path(_tmpdir) / (word_path.stem + ".docx")
            if not _converted.exists():
                raise FileNotFoundError(f"Converted file not found: {_converted}")
            return convert_word_docling(
                _converted,
                llm=llm,
                category=category,
            )

    # 1. Docling 轉換
    try:
        conv           = DocumentConverter()
        docling_result = conv.convert(word_path)
        doc            = docling_result.document
        raw            = doc.export_to_dict()
        markdown       = doc.export_to_markdown()

    except Exception as exc:
        metadata = build_metadata(
            pdf_path=word_path, category=category,
            page_count=0,
            markdown="", llm=llm,
        )
        metadata["qc"] = {
            "qc_level":                 "danger",
            "estimated_info_loss_rate": 1.0,
        }
        metadata["extractor_metadata"] = {
            "tool":             "docling",
            "api_version":      None,
            "is_fully_scanned": False,
            "warnings":         [f"DOCLING_CONVERSION_FAILED: {type(exc).__name__}: {exc}"],
        }
        return {
            "schema_version": "v3.0",
            "metadata":       metadata,
            "data":           {},
            "page_count":     0,
        }

    # 2. 匯出完整 Docling 原始結構
    page_count   = len(raw.get("pages", {})) or 1
    pictures_raw = raw.get("pictures", [])

    # 3. 圖片：DOCX 嵌入圖片暫不支援渲染，標記 has_image=False
    pictures = [dict(p, has_image=False) for p in pictures_raw]
    raw["pictures"] = pictures

    # page_images：記錄有圖片的頁碼（has_image=False，Layer E 尚未支援渲染）
    page_images: dict[int, dict] = {}
    for pic in pictures_raw:
        for prov in pic.get("prov", []):
            pn = prov.get("page_no")
            if pn:
                page_images[pn] = {
                    "source_path": str(word_path),
                    "source_type": "docx",
                    "page_no":     pn,
                    "has_image":   False,
                }

    # 4. metadata
    metadata = build_metadata(
        pdf_path=word_path, category=category,
        page_count=page_count,
        markdown=markdown, llm=llm,
    )

    # 5. QC
    qc = _compute_qc(raw, page_count, pictures)

    metadata["qc"] = {
        "qc_level":                 qc["qc_level"],
        "estimated_info_loss_rate": qc["estimated_info_loss_rate"],
    }
    metadata["extractor_metadata"] = {
        "tool":             "docling",
        "api_version":      None,
        "is_fully_scanned": False,
        "warnings":         qc.get("warnings", []),
    }

    return {
        "schema_version": "v3.0",
        "metadata": metadata,
        "data": {
            "markdown":        markdown,
            "texts":           raw.get("texts",           []),
            "tables":          raw.get("tables",          []),
            "pictures":        pictures,
            "pages":           raw.get("pages",           {}),
            "groups":          raw.get("groups",          []),
            "key_value_items": raw.get("key_value_items", []),
            "body":            raw.get("body",            {}),
            "page_images":     page_images,
            "furniture":       raw.get("furniture",       {}),
        },
        "page_count": page_count,
    }
