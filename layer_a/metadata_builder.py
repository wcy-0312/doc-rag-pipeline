"""
metadata_builder.py — 標準 metadata 建構工具

所有 extractor 使用相同的 metadata 結構：
  source         → 檔案事實（與 extractor 無關）
  classification → 是否文件、類型、語言
  document       → 文件管理資訊（標題、版本、修訂日期等）
  processing     → 轉換過程資訊

doc_id 解析規則（護理 SOP 文件編號格式）：
  A31000-Q05-W-C06  → 從檔名解析
  N12345-A01        → 護理 SOP 規格編號
"""

from __future__ import annotations
import re
from datetime import datetime, timezone
from pathlib import Path


# ── doc_id 解析 ───────────────────────────────────────────────────────────

_DOC_ID_RE = re.compile(
    r'[A-Z]\d{4,6}-[A-Z]\d{2}-P-\d{3}(?:-\d+)?'   # A31000-Q09-P-001(-1) 管理辦法（含子附件）
    r'|[A-Z]\d{4,6}-[A-Z]\d{2}-[A-Z]-[A-Z]\d{2}'   # A31000-Q05-W-A30 / Q06-F-A03
    r'|[A-Z]\d{4,6}-[A-Z]\d{2}'                      # N12345-A01 短格式
    r'|[TE]-[A-Za-z]+-\d{3}'                          # T-Adult-055 / E-Adult-081 衛教單張
)

def _extract_doc_id(file_stem: str) -> str | None:
    m = _DOC_ID_RE.search(file_stem)
    return m.group(0) if m else None


# ── document_type 自動偵測 ────────────────────────────────────────────────

# 具體 type code 先比對，避免過寬的 regex 錯誤分類
_FORM_F_RE = re.compile(r'[A-Z]\d{4,6}-[A-Z]\d{2}-F-[A-Z]\d{2}')           # A31000-Q06-F-A03 評核表
_MGMT_P_RE = re.compile(r'[A-Z]\d{4,6}-[A-Z]\d{2}-P-\d{3}(?:-\d+)?')       # A31000-Q09-P-001 管理辦法
_SOP_W_RE  = re.compile(r'[A-Z]\d{4,6}-[A-Z]\d{2}-W-[A-Z]\d{2}')           # A31000-Q05-W-A30 SOP 主文件

_TYPE_RULES: list[tuple] = [
    # (pattern, document_type)
    # 具體 type code 先比對（-F- / -P- / -W- 各自獨立，避免舊的寬泛 regex 錯誤命中）
    (_FORM_F_RE, "評核表"),
    (_MGMT_P_RE, "管理辦法"),
    (_SOP_W_RE,  "護理SOP"),
    # 其他文件類型
    (re.compile(r'診療指引|治療指引|癌.*指引|指引.*癌'), "癌症診療指引"),
    (re.compile(r'衛教|T-Adult|T-Ped|T-TCM|E-Adult|E-Ped'), "衛教單張"),
    (re.compile(r'BIOSTD|PAC.*訪視|CT報告|MRI|放射治療|檢驗'), "HIS病人檔案管理"),
]

def _infer_document_type(file_stem: str) -> str | None:
    """從檔名推斷 document_type，推斷不到回傳 None。"""
    for pattern, doc_type in _TYPE_RULES:
        if pattern.search(file_stem):
            return doc_type
    return None


# ── version / revision_date 解析（從標題或段落）──────────────────────────

_VERSION_RE = re.compile(r'[Vv](?:ersion\s*)?(\d+(?:\.\d+)*)', re.IGNORECASE)
_DATE_RE    = re.compile(
    r'\d{4}[/-]\d{1,2}[/-]\d{1,2}'     # 2024-03-19
    r'|\d{2,3}[.]\d{1,2}[.]\d{1,2}'    # 113.3.5（民國）
    r'|\d{2,3}年\d{1,2}月\d{1,2}日'    # 113年3月19日
)

def _extract_version(text: str) -> str | None:
    m = _VERSION_RE.search(text)
    return m.group(1) if m else None

def _extract_date(text: str) -> str | None:
    m = _DATE_RE.search(text)
    return m.group(0) if m else None


# ── 主函式 ────────────────────────────────────────────────────────────────

def build_metadata(
    pdf_path: Path,
    category: str,
    extractor: str,
    page_count: int,
    title: str | None = None,
    version: str | None = None,
    revision_date: str | None = None,
    effective_date: str | None = None,
    department: str | None = None,
    markdown: str = "",
    llm=None,
) -> dict:
    """標準 4 層 metadata。

    Args:
        pdf_path       : PDF 路徑（取 file_name、file_size、doc_id）
        category       : 文件類別（如「癌症診療指引」）
        extractor      : 使用的 extractor（"azure_cu" / "docling" / "llm"）
        page_count     : 頁數
        title          : 文件標題（None 時從檔名取）
        version        : 版本號（None 時從標題解析）
        revision_date  : 修訂日期（None 時嘗試從標題解析）
        effective_date : 生效日期（公布日期，None 時留空）
        department     : 部門（None 時留空）
        markdown       : 文件 markdown 全文（供 LLM 關鍵字萃取）
        llm            : LLM 實例（None 時跳過關鍵字萃取，回傳 []）
    """
    stem = pdf_path.stem

    # source：純檔案事實
    folder_name = pdf_path.parent.name
    source = {
        "file_name":       pdf_path.name,
        "file_type":       pdf_path.suffix.lstrip(".").lower(),
        "file_size_bytes": pdf_path.stat().st_size,
        "page_count":      page_count,
        "patient_id":      folder_name if folder_name.isdigit() else None,
    }

    # classification：category 明確傳入優先，否則從檔名自動偵測
    resolved_type   = category or _infer_document_type(stem) or "unknown"
    detection_method = "explicit" if category else ("rule_based" if resolved_type != "unknown" else "unknown")

    classification = {
        "is_document":   True,
        "document_type": resolved_type,
        "language":      "zh-TW",
        "method":        detection_method,
    }

    # document：從標題或檔名萃取
    resolved_title = title or stem
    resolved_ver   = version or _extract_version(resolved_title) or _extract_version(stem)
    resolved_date  = revision_date or _extract_date(resolved_title)
    doc_id         = _extract_doc_id(stem)

    from layer_a.keyword_extractor import extract_keywords
    resolved_keywords = extract_keywords(markdown, llm) if (markdown and llm is not None) else []

    document = {
        "title":          resolved_title,
        "doc_id":         doc_id,
        "version":        resolved_ver,
        "version_status": "active",   # 預設現行有效；多版本共存時由外部邏輯更新
        "revision_date":  resolved_date,
        "effective_date": effective_date,
        "department":     department,
        "keywords":       resolved_keywords,
    }

    # processing
    processing = {
        "extractor":      extractor,
        "processed_at":   datetime.now(tz=timezone.utc).isoformat(),
        "schema_version": "v3.0",
    }

    return {
        "source":         source,
        "classification": classification,
        "document":       document,
        "processing":     processing,
    }
