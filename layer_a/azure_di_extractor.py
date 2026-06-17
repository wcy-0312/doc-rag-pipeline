"""
azure_di_extractor.py — Azure Document Intelligence v4.0 Extractor（照片 path）

轉換層職責：照片（.jpg/.jpeg/.png/.tiff/.heif）→ Azure DI 原始結構 + QC 量化。

照片 path 設計說明（計畫書 §4）：
  - Primary extractor：Azure Document Intelligence v4.0 GA（2024-11-30）
  - 支援格式：JPEG / PNG / TIFF / HEIF
  - 輸出保留 Azure DI 原生欄位：paragraphs / tables / pages / figures / hyperlinks
  - confidence：Azure DI image mode 提供 word-level confidence（pages[].words[].confidence）
    但不提供 per-cell confidence（tables[].cells[].confidence = null）
  - header 旗標：Azure DI 不輸出 column_header / row_header 布林值；
    由 Structure-aware Layer 以 heuristic（rowIndex==0）推斷
  - Fallback：Azure DI API 失敗時降級至 Docling OCR（docling_extractor.py）

已知限制（見 extractor_metadata.known_limitation）：
  - image mode 無 per-cell confidence
  - 無 column_header / row_header 旗標，header 識別需 heuristic
  - 真實醫療文件 TEDS 約 0.699（JBI 2024）
  - HEIF 格式需環境安裝 pillow-heif

與 Structure-aware Layer（組長 B）的介面：
  - 識別：extractor_metadata.tool == "azure_document_intelligence"
  - 表格 cell：rowIndex / columnIndex / rowSpan / columnSpan / content（無 kind，confidence=null）
  - Header：需 heuristic，rowIndex==0 → 推斷為 column header
  - Bounding box：boundingRegions[].polygon（inches，八點多邊形）

Azure DI SDK：azure-ai-documentintelligence（v1.x，對應 API 2024-11-30）
API endpoint：settings.azure_di_endpoint
"""

from __future__ import annotations
import hashlib, os, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# MIME type 對應表
_MIME_MAP: dict[str, str] = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".tiff": "image/tiff",
    ".tif":  "image/tiff",
    ".heif": "image/heif",
    ".heic": "image/heif",
}

_SUPPORTED_SUFFIXES = set(_MIME_MAP.keys())

VISION_DESCRIPTION_ENABLED = os.getenv("VISION_DESCRIPTION_ENABLED", "1") == "1"


def _generate_vision_description(img_path: Path) -> str:
    """用 Vision LLM 描述照片中的主體文件內容。

    失敗時回傳空字串，不中斷主流程。
    """
    try:
        import base64
        from openai import OpenAI
        from config import settings

        endpoint = getattr(settings, "vision_llm_endpoint", None) or os.getenv(
            "VISION_LLM_ENDPOINT", "http://172.31.6.3:8080/gemma3/v1"
        )
        model = getattr(settings, "vision_llm_model", None) or os.getenv(
            "VISION_LLM_MODEL", "/model"
        )

        img_bytes = img_path.read_bytes()
        b64 = base64.b64encode(img_bytes).decode()
        suffix = img_path.suffix.lower()
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                "tiff": "image/tiff", "tif": "image/tiff"}.get(suffix.lstrip("."), "image/jpeg")

        client = OpenAI(api_key="not-needed", base_url=endpoint)
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text",
                     "text": (
                         "這張照片的主體文件是什麼？"
                         "請描述主體文件的完整內容，包含所有可見文字、數值、表格資料與結構。"
                         "若文件含有核取方塊選項（□ 或 ☑ 或手寫打勾標記），請依序列出每個選項並標示勾選狀態："
                         "[已選] 選項文字 或 [未選] 選項文字。"
                         "勾選判斷規則：方框內有任何 ✓ √ V X 或手寫筆觸 = 已選；方框內完全空白 = 未選。"
                         "若照片中有多份文件或背景雜訊，只描述最主要的那份，忽略背景。"
                         "請用繁體中文回答。"
                     )},
                ],
            }],
            temperature=0.0,
            max_tokens=2000,
        )
        return (resp.choices[0].message.content or "").strip()

    except Exception:
        return ""  # 失敗不中斷主流程


# ── confidence 計算（與 azure_cu_extractor 邏輯一致）────────────────────────

def _compute_confidence(raw: dict) -> dict:
    """從 pages[].words[].confidence 計算每頁及整體統計。

    Azure DI image mode 提供 word-level confidence，但無 per-cell confidence。
    無 words 的頁面（純白頁）avg_confidence 設為 None。
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
        word_avg = round(sum(all_confs) / len(all_confs), 3)
        low_rate = round(sum(1 for c in all_confs if c < 0.5) / len(all_confs), 3)
    else:
        word_avg, low_rate = None, None

    return {
        "source":              "ocr_azure_di",
        "available":           True,
        "word_avg":            word_avg,
        "low_confidence_rate": low_rate,
        "pages_no_words":      pages_no_words,
        "page_stats":          page_stats,
    }


# ── 修補 styles 欄位（camelCase → snake_case）───────────────────────────────

def _patch_styles(styles: list[dict]) -> list[dict]:
    """Azure DI as_dict() 回傳 isHandwritten（camelCase），統一正規化為 is_handwritten。

    SDK 的 as_dict() 對 DocumentStyle 保留 JSON wire 格式（isHandwritten），
    但 Python 慣例是 snake_case。修補後讓下游 adapter 可直接用 is_handwritten 讀取。
    """
    result = []
    for s in styles:
        s_out = dict(s)
        if "isHandwritten" in s_out:
            s_out["is_handwritten"] = s_out.pop("isHandwritten")
        result.append(s_out)
    return result


# ── 修補表格欄位（per-cell confidence 明確設為 null）────────────────────────

def _patch_tables(tables: list[dict]) -> list[dict]:
    """將 tables[].cells[].confidence 明確設為 null。

    Azure DI image mode 不提供 per-cell confidence。
    計畫書 §4.4 Step 2：此欄位明確標記為 null，讓下游 Structure-aware Layer
    知道這是「無資料」而非「信心=0」。
    """
    patched = []
    for tbl in tables:
        tbl_out = dict(tbl)
        cells_out = []
        for cell in tbl.get("cells", []):
            c = dict(cell)
            c["confidence"] = None  # image mode 無 per-cell confidence
            cells_out.append(c)
        tbl_out["cells"] = cells_out
        patched.append(tbl_out)
    return patched


# ── 儲存原始圖片（page_images fallback）────────────────────────────────────

def _save_original_image(
    img_path: Path,
    output_dir: Path,
) -> dict:
    """將原始圖片複製至 output_dir/figures/，回傳 page_images[1] 結構。

    Azure DI 照片 path 每張圖片都存一份完整頁面圖，
    確保 OCR 不足時 Structure-aware Layer 仍有視覺參考。
    """
    img_dir = output_dir / "figures"
    img_dir.mkdir(parents=True, exist_ok=True)

    suffix = img_path.suffix.lower()
    dest   = img_dir / f"{img_path.stem}_p1_full{suffix}"
    shutil.copy2(str(img_path), str(dest))

    sha256 = hashlib.sha256(img_path.read_bytes()).hexdigest()
    return {
        "path":      str(dest.relative_to(output_dir)),
        "sha256":    sha256,
        "has_image": True,
    }


def _crop_figures(
    figures: list[dict],
    pages: list[dict],
    img_path: Path,
    output_dir: Path,
) -> list[dict]:
    """從原始圖片裁切 Azure DI 偵測到的 figure 區域，儲存為獨立 PNG。

    Azure DI figures[].boundingRegions[].polygon 座標單位為英吋，
    換算方式：pixel = inch * (image_px / page_inch)。
    裁切失敗時靜默繼續，原始 figure dict 不受影響。
    """
    if not figures or not pages:
        return figures

    try:
        from PIL import Image as _PILImage
    except ImportError:
        return figures

    page_info = pages[0] if pages else {}
    page_w_in = page_info.get("width") or 0.0
    page_h_in = page_info.get("height") or 0.0

    if not page_w_in or not page_h_in:
        return figures

    try:
        img = _PILImage.open(img_path)
        img_w_px, img_h_px = img.size
    except Exception:
        return figures

    px_per_in_x = img_w_px / page_w_in
    px_per_in_y = img_h_px / page_h_in

    img_dir = output_dir / "figures"
    img_dir.mkdir(parents=True, exist_ok=True)

    result = []
    for i, fig in enumerate(figures):
        fig_out = dict(fig)
        bounding = fig.get("boundingRegions", [])
        if not bounding:
            result.append(fig_out)
            continue

        polygon = bounding[0].get("polygon", [])
        if len(polygon) < 8:
            result.append(fig_out)
            continue

        try:
            xs = [polygon[j] * px_per_in_x for j in range(0, len(polygon), 2)]
            ys = [polygon[j] * px_per_in_y for j in range(1, len(polygon), 2)]
            x0, y0 = max(0, int(min(xs))), max(0, int(min(ys)))
            x1, y1 = min(img_w_px, int(max(xs))), min(img_h_px, int(max(ys)))

            if x1 > x0 and y1 > y0:
                cropped = img.crop((x0, y0, x1, y1))
                out_path = img_dir / f"{img_path.stem}_fig{i:03d}.png"
                cropped.save(str(out_path))
                fig_out["path"] = str(out_path.relative_to(output_dir))
                fig_out["has_image"] = True
        except Exception:
            pass

        result.append(fig_out)

    return result


# ── QC 量化 ──────────────────────────────────────────────────────────────────

def _compute_qc(
    raw: dict,
    confidence: dict,
    _di_api_error: str | None,
    total_text_chars: int,
) -> dict:
    """量化照片 path 的資訊損失。

    照片 path 無 figures（照片本身即整頁），QC 重點在文字提取品質。
    """
    page_stats   = confidence.get("page_stats", [])
    pages_total  = len(page_stats) or len(raw.get("pages", [])) or 1

    # unreadable_pages：avg_confidence < 0.5 的頁（OCR 嚴重失敗）
    unreadable_pages = [
        p["page_no"] for p in page_stats
        if p.get("avg_confidence") is not None and p["avg_confidence"] < 0.5
    ]
    no_word_pages  = confidence.get("pages_no_words", [])
    all_bad_pages  = sorted(set(unreadable_pages) | set(no_word_pages))
    pages_unreadable = len(all_bad_pages)

    # 損失率計算（照片 path 無 figures，figure_loss=0）
    text_loss = confidence.get("low_confidence_rate") or 0.0
    page_loss = round(pages_unreadable / pages_total, 4)
    estimated = round(0.6 * text_loss + 0.3 * page_loss, 4)

    qc_level = (
        "danger"  if estimated > 0.10 else
        "warning" if estimated > 0.03 else
        "good"
    )

    warnings_list: list[str] = []
    if _di_api_error:
        warnings_list.append("AZURE_DI_ERROR")
    if total_text_chars < 50:
        warnings_list.append("IMAGE_OCR_INSUFFICIENT_STORED_AS_PAGE_IMAGE")

    return {
        "figures_total":            0,
        "figures_materialized":     0,
        "figures_meaningful":       0,
        "unreadable_pages":         all_bad_pages,
        "pages_unreadable":         pages_unreadable,
        "estimated_info_loss_rate": estimated,
        "qc_level":                 qc_level,
        "warnings":                 warnings_list,
        "errors": (
            [{"stage": "di_api_call", "message": _di_api_error}]
            if _di_api_error else []
        ),
    }


# ── Markdown 生成（從 DI 原生結構）──────────────────────────────────────────

def _build_markdown(raw: dict) -> str:
    """從 Azure DI 原生結構組合 Markdown 文字表示。

    Azure DI 不直接提供 markdown 欄位（與 Azure CU 不同），
    故從 paragraphs[] 組合簡單 Markdown 供下游使用。
    tables[] 以 Markdown 表格格式組合。
    """
    parts: list[str] = []

    # 段落：依 role 決定 Markdown 層級
    para_role_prefix = {
        "title":          "# ",
        "sectionHeading": "## ",
        "footnote":       "> ",
        "pageHeader":     "",
        "pageFooter":     "",
    }
    for para in raw.get("paragraphs", []):
        role    = para.get("role") or "body"
        content = (para.get("content") or "").strip()
        if not content:
            continue
        prefix = para_role_prefix.get(role, "")
        parts.append(f"{prefix}{content}")

    # 表格：簡單 Markdown 表格格式
    for tbl in raw.get("tables", []):
        row_count = tbl.get("rowCount", 0)
        col_count = tbl.get("columnCount", 0)
        if not row_count or not col_count:
            continue

        # 建立二維格陣
        grid: list[list[str]] = [[""] * col_count for _ in range(row_count)]
        for cell in tbl.get("cells", []):
            r = cell.get("rowIndex", 0)
            c = cell.get("columnIndex", 0)
            if r < row_count and c < col_count:
                grid[r][c] = (cell.get("content") or "").replace("|", "\\|")

        # 輸出 Markdown 表格
        if grid:
            header_row = "| " + " | ".join(grid[0]) + " |"
            sep_row    = "| " + " | ".join(["---"] * col_count) + " |"
            parts.append(header_row)
            parts.append(sep_row)
            for row in grid[1:]:
                parts.append("| " + " | ".join(row) + " |")

    # 條碼：Azure DI 解碼結果（比 OCR 更可靠，適用於設備耗材標籤）
    for page in raw.get("pages", []):
        for bc in page.get("barcodes", []):
            kind = bc.get("kind", "barcode")
            value = (bc.get("value") or "").strip()
            if value:
                parts.append(f"[{kind}] {value}")

    return "\n\n".join(parts)


# ── 主入口 ───────────────────────────────────────────────────────────────────

def convert_image_azure_di(
    img_path: Path,
    category: str = "",
    output_dir: Path | None = None,
    keywords: list[str] | None = None,
) -> dict:
    """Azure DI v4.0 照片 path：圖片 → structured JSON（schema-v3.0）。

    Azure DI 原始輸出最小轉換後直接保留。
    per-cell confidence 明確設為 null（image mode 不提供）。
    原始圖片複製至 output_dir/figures/ 作為 page_images[1]。

    Args:
        img_path   : 圖片路徑（.jpg/.jpeg/.png/.tiff/.heif）
        category   : 文件類別（用於 metadata.classification.document_type）
        output_dir : 輸出目錄（有值時存原始圖片）
        keywords   : 手動指定關鍵字（None 時由上游 keyword_extractor 處理）

    Returns:
        schema-v3.0 dict
    """
    suffix = img_path.suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise ValueError(
            f"azure_di_extractor 不支援格式 {suffix}。"
            f"支援：{sorted(_SUPPORTED_SUFFIXES)}"
        )

    img_bytes = img_path.read_bytes()

    raw: dict              = {}
    _di_api_error: str | None = None

    # ── 1. 呼叫 Azure DI v4.0 API ────────────────────────────────────────
    try:
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
        from azure.core.credentials import AzureKeyCredential
        from azure.identity import DefaultAzureCredential
        from config import settings

        endpoint = settings.azure_di_endpoint
        api_key  = settings.azure_di_api_key
        credential = (
            AzureKeyCredential(api_key)
            if api_key and api_key != "unused"
            else DefaultAzureCredential()
        )

        client = DocumentIntelligenceClient(
            endpoint=endpoint,
            credential=credential,
        )

        # bytes_source 接受原始 bytes；MIME type 由 DI API 自動偵測
        poller = client.begin_analyze_document(
            model_id="prebuilt-layout",
            body=AnalyzeDocumentRequest(bytes_source=img_bytes),
        )
        result = poller.result()

        if result is not None:
            # azure-ai-documentintelligence SDK 回傳 AnalyzeResult 物件
            # 轉為 dict 保留原生欄位
            if hasattr(result, "as_dict"):
                raw = result.as_dict()
            else:
                # 手動組合必要欄位
                raw = {
                    "pages":      [p.as_dict() if hasattr(p, "as_dict") else p
                                   for p in (result.pages or [])],
                    "tables":     [t.as_dict() if hasattr(t, "as_dict") else t
                                   for t in (result.tables or [])],
                    "paragraphs": [p.as_dict() if hasattr(p, "as_dict") else p
                                   for p in (result.paragraphs or [])],
                    "figures":    [f.as_dict() if hasattr(f, "as_dict") else f
                                   for f in (getattr(result, "figures", None) or [])],
                }
        else:
            _di_api_error = "empty result"

    except ImportError as e:
        _di_api_error = f"SDK not installed: {e}"
    except Exception as e:
        _di_api_error = str(e)

    # ── 2. confidence ────────────────────────────────────────────────────
    confidence = _compute_confidence(raw) if raw and not _di_api_error else {
        "source":    "ocr_azure_di",
        "available": False,
        "note":      _di_api_error or "Azure DI API unavailable",
    }

    # ── 3. 修補欄位 ──────────────────────────────────────────────────────
    raw_tables  = raw.get("tables", [])
    tables      = _patch_tables(raw_tables)
    styles      = _patch_styles(raw.get("styles", []))

    # ── 3b. Figure 裁切（需 Pillow + output_dir）────────────────────────
    raw_figures = raw.get("figures", [])
    if output_dir and raw_figures and not _di_api_error:
        raw_figures = _crop_figures(raw_figures, raw.get("pages", []), img_path, output_dir)

    # ── 4. Markdown 生成 ─────────────────────────────────────────────────
    raw_with_patched = dict(raw)
    raw_with_patched["tables"] = tables
    raw_with_patched["figures"] = raw_figures
    markdown = _build_markdown(raw_with_patched)

    # ── 5. page_images：永遠儲存原始圖片 ────────────────────────────────
    page_images: dict[int, dict] = {}
    if output_dir:
        try:
            page_images[1] = _save_original_image(img_path, output_dir)
        except Exception:
            pass  # 存圖失敗不中斷主流程

    # ── 6. QC 量化 ───────────────────────────────────────────────────────
    total_text_chars = sum(
        len(p.get("content") or "")
        for p in raw.get("paragraphs", [])
    )
    qc = _compute_qc(raw, confidence, _di_api_error, total_text_chars)

    # ── 7. metadata ──────────────────────────────────────────────────────
    from metadata_builder import build_metadata

    # 照片通常只有 1 頁
    page_count = len(raw.get("pages", [])) or 1

    # 嘗試從 paragraphs 取得文件標題
    title_para = None
    for p in raw.get("paragraphs", []):
        if p.get("role") == "title":
            content = (p.get("content") or "").strip()
            if content:
                title_para = content
                break

    metadata = build_metadata(
        pdf_path=img_path,
        category=category,
        extractor="azure_di",
        page_count=page_count,
        title=title_para,
        keywords=keywords,
    )

    metadata["confidence"] = confidence
    metadata["qc"]         = qc

    # ── 8. Vision LLM 描述（照片主體文件，供 B 層 embedding_text 使用）────
    vision_description = (
        _generate_vision_description(img_path)
        if VISION_DESCRIPTION_ENABLED and not _di_api_error
        else ""
    )

    # P3: 若 OCR 和 Vision 雙雙為空，升為 danger（靜默空白文件）
    if total_text_chars < 50 and not vision_description:
        qc = dict(qc)  # 避免 mutate 原始 dict
        qc["qc_level"] = "danger"
        warnings = list(qc.get("warnings", []))
        if "VISION_AND_OCR_BOTH_EMPTY" not in warnings:
            warnings.append("VISION_AND_OCR_BOTH_EMPTY")
        qc["warnings"] = warnings

    # ── 9. 組合 schema-v3.0 output ───────────────────────────────────────
    return {
        "schema_version": "v3.0",
        "metadata": metadata,
        "data": {
            "markdown":    markdown if not _di_api_error else f"[Azure DI error: {_di_api_error}]",
            "paragraphs":  raw.get("paragraphs", []),
            "tables":      tables,
            "figures":     raw_figures,
            "pages":       raw.get("pages",      []),
            "styles":      styles,
            "hyperlinks":  raw.get("keyValuePairs", []),  # DI 圖片模式無 hyperlinks，改存 keyValuePairs
            "page_images": page_images,
        },
        "page_count": page_count,
        "vision_description": vision_description,
        "extractor_metadata": {
            "tool":                "azure_document_intelligence",
            "analyzer_id":         "prebuilt-layout",
            "api_version":         "2024-11-30",
            "parsing_mode":        None,
            "version":             "1.0.2",
            "confidence_source": (
                "pages[].words[].confidence（word-level，彙總為 per-page 統計）"
                if not _di_api_error else None
            ),
            "per_cell_confidence": False,
            "confidence_note": (
                "Azure DI image mode 不暴露 per-cell confidence；"
                "tables[].cells[].confidence 為 null。"
                "word-level confidence 仍可用。"
            ),
            "header_flags": (
                "Azure DI 不輸出 column_header / row_header 布林旗標；"
                "由 Structure-aware Layer Input Adapter 以 heuristic（rowIndex==0）推斷"
            ),
            "bounding_box_format": (
                "boundingRegions[].polygon（inches，八點多邊形格式）"
            ),
            "known_limitation": (
                "image mode 無 per-cell confidence；"
                "header 識別需 heuristic；"
                "真實醫療文件 TEDS 約 0.699（JBI 2024）；"
                "手寫欄位與印章覆蓋文字辨識準確率低"
            ),
            "fallback_reason": "azure_di_failed" if _di_api_error else None,
        },
    }
