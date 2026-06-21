"""
docling_extractor.py — Docling Extractor（schema-v3.0）

轉換層職責：PDF → Docling 原始結構，最小轉換。
和 azure_cu_extractor 一樣：直接保留工具的原始輸出。

Docling export_to_dict() 原始輸出直接保留：
  texts[]        — 所有文字項目（含 bbox / charspan / label / formatting）
  tables[]       — 所有表格（含 cell-level data：bbox / row_span / col_span / header）
  pictures[]     — 所有圖片（含 bbox，裁切比 CU API 更精確）
  pages{}        — 頁面資訊（含 size）
  groups[]       — 分組結構
  key_value_items[] — 鍵值對
  body           — 文件樹根節點
  furniture      — 頁首/頁尾等非正文元素

另外計算：
  markdown  — Docling export_to_markdown()
  confidence — source=pdf_text_layer，available=False
  qc         — 資訊損失量化
  metadata   — 標準四層
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── 圖片裁切（使用 Docling 提供的精確 bbox）────────────────────────────────

def _materialize_pictures(
    pdf_path: Path,
    pictures: list[dict],
    page_sizes: dict,
) -> list[dict]:
    """為每張 Docling figure 記錄 PDF 參考座標（不渲染存圖）。

    回傳結構：pic dict + {pdf_path, page_no, bbox, has_image}。
    Layer E 按需使用 pdf_path + bbox 裁切。
    """
    import fitz

    doc = fitz.open(str(pdf_path))
    result = []

    for pic in pictures:
        pic_out = dict(pic)
        provs = pic.get("prov", [])
        if not provs:
            pic_out.update({"has_image": False})
            result.append(pic_out)
            continue

        prov    = provs[0]
        page_no = prov.get("page_no", 1)
        bbox    = prov.get("bbox", {})
        if page_no - 1 >= len(doc) or not bbox:
            pic_out.update({"has_image": False})
            result.append(pic_out)
            continue

        page = doc[page_no - 1]
        l, t, r, b = bbox["l"], bbox["t"], bbox["r"], bbox["b"]

        # Docling BOTTOMLEFT → fitz TOPLEFT 座標系轉換
        try:
            if bbox.get("coord_origin", "BOTTOMLEFT") == "BOTTOMLEFT":
                ph = page_sizes.get(str(page_no), {}).get("height", page.rect.height)
                x0, x1 = min(l, r), max(l, r)
                y0, y1 = min(ph - t, ph - b), max(ph - t, ph - b)
            else:
                x0, x1 = min(l, r), max(l, r)
                y0, y1 = min(t, b), max(t, b)
            rect = fitz.Rect(x0, y0, x1, y1) & page.rect
        except Exception:
            pic_out.update({"has_image": False})
            result.append(pic_out)
            continue

        if rect.is_empty:
            pic_out.update({"has_image": False})
            result.append(pic_out)
            continue

        pic_out.update({
            "pdf_path": str(pdf_path),
            "page_no":  page_no,
            "bbox":     {"x0": rect.x0, "y0": rect.y0, "x1": rect.x1, "y1": rect.y1},
            "has_image": True,
        })
        result.append(pic_out)

    doc.close()
    return result


# ── QC 計算 ──────────────────────────────────────────────────────────────────

import re as _re
# 私用區（U+E000-F8FF）+ 替換字元（U+FFFD），不含 CJK 相容字（U+F900-FAFF）
_PUA_RE = _re.compile(r"[-�]")


def _compute_qc(raw: dict, page_count: int,
                pictures: list[dict], output_dir_set: bool) -> dict:
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
    # "blank" 在此指「Docling 未萃取出任何 item 的頁面」，不代表頁面真正空白
    pages_no_extracted_items = sorted(all_page_nos - set(items_per_page_map))
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

    # ── 4. 圖片覆蓋 ───────────────────────────────────────────────────────
    figures_total        = len(pictures)
    figures_materialized = sum(1 for p in pictures if p.get("has_image"))
    figure_loss = (
        round(1 - figures_materialized / figures_total, 4)
        if output_dir_set and figures_total > 0 else 0.0
    )

    # ── 5. 綜合損失率 ─────────────────────────────────────────────────────
    text_loss  = garbled_char_rate
    page_loss  = round(len(low_density_pages) / max(page_count, 1), 4)
    table_loss = empty_cell_rate

    # 啟發式加權公式（非精確損失率）：係數依文件內容對 RAG 的重要性估計，無實驗驗證
    estimated = round(
        0.45 * text_loss +
        0.25 * page_loss +
        0.20 * table_loss +
        0.10 * figure_loss,
        4
    )
    qc_level = (
        "danger"  if estimated > 0.10 else
        "warning" if estimated > 0.03 else
        "good"
    )

    return {
        # 文字品質
        "garbled_char_rate":         garbled_char_rate,
        # 頁面密度
        "items_per_page":            round(avg_density, 1),
        "low_density_pages":         low_density_pages,
        "pages_no_extracted_items":  pages_no_extracted_items,
        # 表格品質
        "empty_cell_rate":           empty_cell_rate,
        # 圖片
        "figures_total":             figures_total,
        "figures_materialized":      figures_materialized,
        # 綜合
        "estimated_info_loss_rate":  estimated,
        "qc_level":                  qc_level,
        "warnings":                  [],
        "errors":                    [],
    }


# ── 主入口 ───────────────────────────────────────────────────────────────────

def convert_pdf_docling(
    pdf_path: Path,
    llm=None,
    category: str = "",
    output_dir: Path | None = None,
    describe_visuals: bool = False,
) -> dict:
    """Docling path：PDF → Docling 原始結構（schema-v3.0）。

    export_to_dict() 完整保留 Docling 輸出：
    - 每個 item 的精確 bbox
    - 表格的 cell-level 結構（row/col/span/header/bbox per cell）
    - formatting、hyperlink、charspan
    - 圖片用 bbox 精確裁切（比整頁更準確）

    Args:
        pdf_path       : PDF 路徑
        llm            : LLM 實例（有值時對 markdown 進行關鍵字萃取）
        category       : 文件類別
        output_dir     : 圖片輸出目錄（有值時用 bbox 精確裁切）
        describe_visuals: 保留供 extractor 介面一致性，不使用
    """
    from docling.document_converter import DocumentConverter
    from metadata_builder import build_metadata

    suffix = pdf_path.suffix.lower()

    # .doc → .docx conversion via LibreOffice
    if suffix == ".doc":
        import subprocess, tempfile
        with tempfile.TemporaryDirectory() as _tmpdir:
            result = subprocess.run(
                ["libreoffice", "--headless", "--convert-to", "docx",
                 "--outdir", _tmpdir, str(pdf_path)],
                capture_output=True, timeout=60,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"libreoffice conversion failed: {result.stderr.decode()}"
                )
            _converted = Path(_tmpdir) / (Path(pdf_path).stem + ".docx")
            if not _converted.exists():
                raise FileNotFoundError(f"Converted file not found: {_converted}")
            # Recurse with the converted .docx (won't enter this branch again)
            return convert_pdf_docling(
                _converted,
                llm=llm,
                category=category,
                output_dir=output_dir,
                describe_visuals=describe_visuals,
            )

    # 1. Docling 轉換
    try:
        conv     = DocumentConverter()
        docling_result = conv.convert(pdf_path)
        doc      = docling_result.document
        raw      = doc.export_to_dict()
        markdown = doc.export_to_markdown()

    except Exception as exc:
        # 其他格式失敗：回傳錯誤結構
        page_count = 0
        if suffix == ".pdf":
            try:
                import fitz as _fitz
                _doc = _fitz.open(str(pdf_path))
                page_count = _doc.page_count
                _doc.close()
            except Exception:
                pass
        metadata = build_metadata(
            pdf_path=pdf_path, category=category,
            extractor="docling", page_count=page_count,
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
            "metadata":   metadata,
            "data":       {},
            "page_count": page_count,
        }

    # 2. 匯出完整 Docling 原始結構

    page_count = len(raw.get("pages", {})) or 1
    page_sizes = {k: v.get("size", {}) for k, v in raw.get("pages", {}).items()}

    # 3. 圖片裁切（使用 Docling bbox）
    # PDF：記錄 bbox 參考；DOCX：無可用 bbox，標記 has_image=False
    pictures_raw = raw.get("pictures", [])
    is_pdf = pdf_path.suffix.lower() == ".pdf"
    is_office = suffix in {".docx", ".pptx"}
    if is_pdf and pictures_raw:
        pictures = _materialize_pictures(pdf_path, pictures_raw, page_sizes)
    else:
        pictures = [dict(p, has_image=False) for p in pictures_raw]
    raw["pictures"] = pictures

    # 3b. 視覺頁面參考（不渲染，供 Layer E 按需使用）
    page_images: dict[int, dict] = {}
    if is_pdf:
        # 有圖片的頁
        pic_pages = {
            prov.get("page_no")
            for pic in pictures if pic.get("has_image")
            for prov in pic.get("prov", [])
            if prov.get("page_no")
        }
        # 無文字的頁（掃描頁）— qc 在此計算一次，line 336 直接重用
        qc = _compute_qc(raw, page_count, pictures, output_dir is not None)
        no_item_pages = set(qc.get("pages_no_extracted_items", []))
        for pn in sorted(pic_pages | no_item_pages):
            page_images[pn] = {"source_path": str(pdf_path), "source_type": "pdf", "page_no": pn, "has_image": True}
    elif is_office:
        # DOCX/PPTX：記錄有圖片的頁碼；has_image=False 因 Layer E 尚未支援渲染
        qc = None  # 由下面 _compute_qc 計算
        for pic in pictures_raw:
            for prov in pic.get("prov", []):
                pn = prov.get("page_no")
                if pn:
                    page_images[pn] = {"source_path": str(pdf_path), "source_type": "docx", "page_no": pn, "has_image": False}

    # 4. metadata
    metadata = build_metadata(
        pdf_path=pdf_path, category=category,
        extractor="docling", page_count=page_count,
        markdown=markdown,
        llm=llm,
    )

    # 5. 組合 schema-v3.0 output（metadata + data 兩層）
    # PDF path: qc already computed above when building page_images
    if qc is None:
        qc = _compute_qc(raw, page_count, pictures, output_dir is not None)

    # 掃描型 PDF 偵測：所有頁面都沒有萃取出文字，且是 PDF
    if is_pdf and qc.get("pages_no_extracted_items"):
        no_item_count = len(qc["pages_no_extracted_items"])
        if no_item_count == page_count:
            qc["warnings"].append(
                "SCANNED_PDF_ALL_PAGES_EMPTY: docling 無法從掃描型 PDF 萃取文字。"
                "建議改用 azure_cu（支援 OCR）或 llm extractor。"
            )
        elif no_item_count > page_count // 2:
            qc["warnings"].append(
                f"SCANNED_PDF_PARTIAL: {no_item_count}/{page_count} 頁無文字萃取。"
                "部分頁面可能是掃描影像，建議改用 azure_cu。"
            )

    # 小字體偵測（< 8pt）：臨床關鍵數值可能精準度下降
    if is_pdf:
        try:
            import fitz as _fitz_sf
            _doc_sf = _fitz_sf.open(str(pdf_path))
            small_font_pages = []
            for _pi in range(_doc_sf.page_count):
                _rawdict = _doc_sf[_pi].get_text("rawdict")
                for _blk in _rawdict.get("blocks", []):
                    for _ln in _blk.get("lines", []):
                        for _sp in _ln.get("spans", []):
                            if _sp.get("size", 99) < 8 and _sp.get("text", "").strip():
                                small_font_pages.append(_pi + 1)
                                break
                        else:
                            continue
                        break
            _doc_sf.close()
            if small_font_pages:
                qc["warnings"].append(
                    f"SMALL_FONT_PAGES {sorted(set(small_font_pages))}: "
                    "部分頁面含 <8pt 文字（臨床分期表、劑量值等），OCR 精準度可能下降。"
                )
        except Exception:
            pass

    metadata["qc"] = {
        "qc_level":                 qc["qc_level"],
        "estimated_info_loss_rate": qc["estimated_info_loss_rate"],
    }
    metadata["extractor_metadata"] = {
        "tool":             "docling",
        "api_version":      None,
        "is_fully_scanned": bool(qc.get("is_fully_scanned", False)),
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
