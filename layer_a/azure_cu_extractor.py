"""
azure_cu_extractor.py — CU API Adapter（schema-v3.0）

轉換層職責：PDF → CU API 原始結構 + 圖片裁切 + QC 量化。

CU API 原始輸出直接保留：
  paragraphs[]  — 所有段落（含 role/content/source/span）
  tables[]      — 所有表格（含 cells/rowSpan/columnSpan）
  figures[]     — 所有圖像（含 bbox/caption，完全原樣，不修改）
  sections[]    — 文件樹（element 引用有效）
  pages[]       — 每頁（含 words/lines + 信心分數）
  hyperlinks[]  — 超連結

另外計算：
  confidence    — 從 pages[].words[] 彙總信心統計
  qc            — 資訊損失量化（text/page/figure 各維度 + 綜合評分）
  metadata      — 標準四層

Layer A 獨有（需要 PDF 原檔）：
  page_images       — 視覺頁面的參考結構（pdf_path + page_no，不存圖）
  checkbox_states   — 各頁 checkbox 勾選狀態

不輸出：blocks / chunks（由後續層負責）

CU API endpoint：settings.azure_cu_endpoint（config.env）
analyzer_id    ：prebuilt-layout
"""

from __future__ import annotations
import base64, re, sys
from pathlib import Path

# config.py 在 lib/ 的上層目錄
sys.path.insert(0, str(Path(__file__).parent.parent))

# 這些類別的 PDF 全頁儲存截圖，確保含算法圖/流程表的頁面也有 page_image_refs
_FULL_PAGE_SAVE_CATEGORIES: frozenset[str] = frozenset({
    "癌症診療指引", "臨床指引", "diagnostic_guideline",
})

_NUM = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
_SOURCE_RE = re.compile(
    rf"D\((\d+),({_NUM}),({_NUM}),({_NUM}),({_NUM}),({_NUM}),({_NUM}),({_NUM}),({_NUM})\)"
)


# ── confidence 計算 ───────────────────────────────────────────────────────

def _compute_confidence(raw: dict) -> dict:
    """從 pages[].words[] 計算每頁及整體信心統計。

    無 words 的頁面（純圖像頁或空白頁）avg_confidence 設為 None，
    不假設為 1.0，避免掩蓋真實問題。
    """
    all_confs: list[float] = []
    page_stats: list[dict] = []
    pages_no_words: list[int] = []

    for p in raw.get("pages", []):
        page_no = p.get("pageNumber", len(page_stats) + 1)
        words   = p.get("words", [])
        confs   = [w["confidence"] for w in words if "confidence" in w]
        if confs:
            avg      = round(sum(confs) / len(confs), 3)
            low_cnt  = sum(1 for c in confs if c < 0.5)
            low_rate = round(low_cnt / len(confs), 3)
        else:
            avg, low_cnt, low_rate = None, 0, None
            pages_no_words.append(page_no)
        page_stats.append({
            "page_no":              page_no,
            "word_count":           len(words),
            "avg_confidence":       avg,
            "low_confidence_count": low_cnt,
            "low_confidence_rate":  low_rate,
        })
        all_confs.extend(confs)

    if all_confs:
        word_avg  = round(sum(all_confs) / len(all_confs), 3)
        low_rate  = round(sum(1 for c in all_confs if c < 0.5) / len(all_confs), 3)
    else:
        word_avg, low_rate = None, None

    return {
        "source":              "ocr_azure_cu",
        "available":           True,
        "word_avg":            word_avg,
        "low_confidence_rate": low_rate,
        "pages_no_words":      pages_no_words,
        "page_stats":          page_stats,
    }


# ── 圖片裁切 ─────────────────────────────────────────────────────────────

def _figure_bbox(source_str: str) -> tuple[int, float, float, float, float] | None:
    """D(page,...) → (page_no, x_min, y_min, x_max, y_max) in inches。"""
    first = str(source_str).split(";")[0]
    m = _SOURCE_RE.match(first)
    if not m:
        return None
    page_no = int(m.group(1))
    coords  = [float(m.group(i)) for i in range(2, 10)]
    xs = [coords[0], coords[2], coords[4], coords[6]]
    ys = [coords[1], coords[3], coords[5], coords[7]]
    return page_no, min(xs), min(ys), max(xs), max(ys)


def _page_image_refs(
    pdf_path: Path,
    raw_figures: list[dict],
    confidence: dict,
    category: str = "",
    page_count: int = 0,
    checkbox_pages: set[int] | None = None,
) -> dict[int, dict]:
    """回傳視覺頁面的參考結構（不存圖，按需由 Layer E 渲染）。

    觸發條件（任一）：
    1. 頁面有任何 CU figure（從 source 解析頁碼）
    2. 頁面無文字（pages_no_words → 掃描頁）
    3. category 屬於 _FULL_PAGE_SAVE_CATEGORIES（臨床指引等含算法流程表的文件）
    4. 頁面有嵌入圖 >= 2.0 sqin（fitz get_image_info()）
    5. 頁面在 checkbox_pages 中（有偵測到 checkbox）

    Returns: {page_no: {"source_path": str, "source_type": "pdf", "page_no": int, "has_image": bool}}
    """
    import fitz

    visual_pages: set[int] = set()

    # Trigger 1: 任何 CU figure（解析 source 取頁碼）
    for fig in raw_figures:
        source = str(fig.get("source", "") or "")
        parsed = _figure_bbox(source)
        if parsed:
            visual_pages.add(parsed[0])

    # Trigger 2: 掃描頁
    for page_no in confidence.get("pages_no_words", []):
        visual_pages.add(page_no)

    # Trigger 3: 特定類別全頁存圖
    if category in _FULL_PAGE_SAVE_CATEGORIES and page_count > 0:
        for pn in range(1, page_count + 1):
            visual_pages.add(pn)

    # Trigger 5: checkbox 頁
    if checkbox_pages:
        visual_pages.update(checkbox_pages)

    def _ref(pn: int) -> dict:
        return {"source_path": str(pdf_path), "source_type": "pdf", "page_no": pn, "has_image": True}

    # Trigger 3 fills all pages — skip fitz scan entirely
    if page_count > 0 and len(visual_pages) >= page_count:
        return {pn: _ref(pn) for pn in range(1, page_count + 1)}

    doc = fitz.open(str(pdf_path))
    try:
        # Trigger 4: pages with large embedded images not captured by CU API figures[]
        # (e.g., BIOSTD EKG waveforms, 放射治療計畫 Report Snapshots)
        for page_idx in range(len(doc)):
            page_no_check = page_idx + 1
            if page_no_check in visual_pages:
                continue
            try:
                p = doc[page_idx]
                for img_info in p.get_image_info():
                    bbox = img_info.get("bbox")  # (x0, y0, x1, y1) in points
                    if bbox:
                        w_in = (bbox[2] - bbox[0]) / 72.0
                        h_in = (bbox[3] - bbox[1]) / 72.0
                        if w_in * h_in >= 2.0:
                            visual_pages.add(page_no_check)
                            break
            except Exception:
                pass
    finally:
        doc.close()

    return {pn: _ref(pn) for pn in sorted(visual_pages)}


def _detect_checkbox_states(doc, page_no: int) -> list[dict]:
    """偵測 PDF 頁面中 checkbox 的勾選狀態。

    策略 1：PDF AcroForm widget（page.widgets()）— 直接讀取邏輯值
    策略 2：空間比對 □（U+25A1）字符 × black_filled vector drawings
    """
    try:
        import fitz
        page = doc[page_no - 1]
        results: list[dict] = []

        # Strategy 1: AcroForm widgets
        try:
            for w in page.widgets():
                if w.field_type == fitz.PDF_WIDGET_TYPE_CHECKBOX:
                    checked = (w.field_value == w.on_state())
                    results.append({
                        "label_text": w.field_label or "",
                        "bbox": list(w.rect),
                        "checked": checked,
                        "method": "acroform",
                    })
        except Exception:
            pass

        # Strategy 2: Visual detection (fallback when AcroForm not available)
        if not results:
            # Collect □ character positions
            square_chars: list[tuple] = []  # (bbox, nearby_text)
            try:
                rawdict = page.get_text("rawdict")
                for block in rawdict.get("blocks", []):
                    for line in block.get("lines", []):
                        line_text = "".join(s.get("text", "") for s in line.get("spans", []))
                        for span in line.get("spans", []):
                            for char in span.get("chars", []):
                                if char.get("c") == "□":  # □
                                    square_chars.append((char["bbox"], line_text.strip()))
            except Exception:
                pass

            # Collect black-filled drawings
            black_drawings: list = []
            try:
                for drawing in page.get_drawings():
                    fill = drawing.get("fill")
                    if fill is not None and all(c <= 0.15 for c in fill):
                        black_drawings.append(drawing["rect"])
            except Exception:
                pass

            # Spatial overlap: IoT (Intersection over Target) >= 0.3
            for sq_bbox, nearby_text in square_chars:
                sq_rect = fitz.Rect(sq_bbox)
                sq_area = abs(sq_rect)
                checked = False
                if sq_area > 0:
                    for dr_rect in black_drawings:
                        inter = sq_rect & dr_rect
                        if abs(inter) / sq_area >= 0.3:
                            checked = True
                            break
                results.append({
                    "label_text": nearby_text,
                    "bbox": list(sq_rect),
                    "checked": checked,
                    "method": "visual",
                })

        return results
    except Exception:
        return []


# ── 主入口 ───────────────────────────────────────────────────────────────────

def convert_pdf_azure_cu(
    pdf_path: Path,
    category: str = "",
    llm=None,
) -> dict:
    """CU API path：PDF → structured JSON（schema-v3.0）。

    Args:
        pdf_path : PDF 檔案路徑
        category : 文件類別
        llm      : LLM 實例（有值時對 markdown 進行關鍵字萃取）

    Returns:
        schema-v3.0 dict
    """
    # ── 1. 呼叫 CU API ────────────────────────────────────────────────
    pdf_bytes = pdf_path.read_bytes()
    b64 = base64.b64encode(pdf_bytes).decode()

    raw: dict = {}
    _cu_api_error: str | None = None

    try:
        from azure.ai.contentunderstanding import ContentUnderstandingClient
        from azure.core.credentials import AzureKeyCredential
        from azure.identity import DefaultAzureCredential
        from config import settings

        endpoint   = settings.azure_cu_endpoint
        api_key    = settings.azure_cu_api_key
        credential = (AzureKeyCredential(api_key)
                      if api_key and api_key != "unused"
                      else DefaultAzureCredential())

        client = ContentUnderstandingClient(endpoint=endpoint, credential=credential)
        result = client.begin_analyze(
            analyzer_id="prebuilt-layout",
            body={"inputs": [{"data": b64, "mimeType": "application/pdf"}]},
        ).result()

        if hasattr(result, "contents") and result.contents:
            raw = result.contents[0].as_dict()
        else:
            _cu_api_error = "empty contents"

    except ImportError as e:
        _cu_api_error = f"SDK not installed: {e}"
    except Exception as e:
        _cu_api_error = str(e)

    # ── 2. confidence ────────────────────────────────────────────────
    confidence = _compute_confidence(raw) if raw and not _cu_api_error else {
        "source": "ocr_azure_cu", "available": False,
        "note": _cu_api_error or "CU API unavailable",
    }

    # ── 3. metadata ──────────────────────────────────────────────────
    # CU API 失敗時用 fitz 取得真實頁數，避免 fallback 成誤導性的 1
    cu_page_count = len(raw.get("pages", []))
    if cu_page_count:
        page_count = cu_page_count
    else:
        try:
            import fitz as _fitz
            _d = _fitz.open(str(pdf_path))
            page_count = _d.page_count
            _d.close()
        except Exception:
            page_count = 0

    from metadata_builder import build_metadata, _infer_document_type
    resolved_category = category or _infer_document_type(pdf_path.stem) or ""
    metadata = build_metadata(
        pdf_path=pdf_path,
        category=resolved_category,
        page_count=page_count,
        markdown=raw.get("markdown", "") if not _cu_api_error else "",
        llm=llm,
    )

    # ── 6b. Checkbox 狀態偵測 + 小字體警告 ───────────────────────────────
    checkbox_states_by_page: dict[int, list[dict]] = {}
    small_font_pages: list[int] = []
    try:
        import fitz as _fitz_cb
        _cb_doc = _fitz_cb.open(str(pdf_path))
        for _pno in range(1, page_count + 1):
            _states = _detect_checkbox_states(_cb_doc, _pno)
            if _states:
                checkbox_states_by_page[_pno] = _states
            try:
                _rawdict = _cb_doc[_pno - 1].get_text("rawdict")
                for _blk in _rawdict.get("blocks", []):
                    for _ln in _blk.get("lines", []):
                        for _sp in _ln.get("spans", []):
                            if _sp.get("size", 99) < 8 and _sp.get("text", "").strip():
                                small_font_pages.append(_pno)
                                break
                        else:
                            continue
                        break
            except Exception:
                pass
        _cb_doc.close()
    except Exception:
        pass

    warnings_list: list[str] = ["AZURE_CU_ERROR"] if _cu_api_error else []
    if small_font_pages:
        warnings_list.append(
            f"SMALL_FONT_PAGES {sorted(set(small_font_pages))}: "
            "<8pt 文字（臨床分期/劑量值），OCR 精準度可能下降。"
        )

    # ── 6a. 視覺頁面參考（供 Layer E 按需渲染）──────────────────────────
    page_images = _page_image_refs(
        pdf_path,
        raw_figures=raw.get("figures", []),
        confidence=confidence,
        category=resolved_category,
        page_count=page_count,
        checkbox_pages=set(checkbox_states_by_page.keys()) if checkbox_states_by_page else None,
    ) if not _cu_api_error else {}

    # ── 7. 組合 schema-v3.0 output（metadata + data 兩層）───────────────
    pages_no_words = confidence.get("pages_no_words", []) if isinstance(confidence, dict) else []

    metadata["extractor_metadata"] = {
        "tool":             "azure_content_understanding",
        "api_version":      "2024-12-01-preview",
        "is_fully_scanned": len(pages_no_words) == page_count and page_count > 0,
        "warnings":         warnings_list,
    }

    return {
        "schema_version": "v3.0",
        "metadata": metadata,
        "data": {
            **raw,
            "markdown": (raw.get("markdown", "") if not _cu_api_error
                         else f"[CU API error: {_cu_api_error}]"),
            "page_images":      page_images,
            "checkbox_states":  checkbox_states_by_page,
        },
        "page_count": page_count,
    }
