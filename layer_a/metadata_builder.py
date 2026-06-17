"""
metadata_builder.py — 標準 metadata 建構工具

所有 extractor 使用相同的 flat metadata 結構：
  file_name       → 檔案名稱
  file_type       → 副檔名（小寫，不含點）
  page_count      → 頁數
  patient_id      → 父目錄為純數字時填入，否則 null
  document_type   → 識別不到填 None
  keywords        → LLM 萃取關鍵字
  processed_at    → UTC ISO 8601
"""

from __future__ import annotations
import re
from datetime import datetime, timezone
from pathlib import Path


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


# ── 主函式 ────────────────────────────────────────────────────────────────

def _extract_pdf_keywords(pdf_path: Path) -> list[str] | None:
    """從 PDF 檔案 metadata 讀取 keywords（通常空白，作為 fallback）。"""
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        kw_str = (doc.metadata.get("keywords") or "").strip()
        doc.close()
        if not kw_str:
            return None
        return [k.strip() for k in re.split(r'[,;、，；]', kw_str) if k.strip()]
    except Exception:
        return None


def build_metadata(
    pdf_path: Path,
    category: str,
    page_count: int,
    markdown: str | None = None,
    llm=None,
    keywords: list[str] | None = None,
    # 以下參數保留相容性但不使用
    extractor: str | None = None,
    title: str | None = None,
    version: str | None = None,
    revision_date: str | None = None,
    effective_date: str | None = None,
    department: str | None = None,
) -> dict:
    """Flat metadata 結構。

    Args:
        pdf_path   : 檔案路徑（取 file_name、patient_id、document_type）
        category   : 文件類別（如「癌症診療指引」）
        page_count : 頁數
        markdown   : 文件 markdown 內容（供關鍵字萃取使用）
        llm        : LLM 物件（供關鍵字萃取使用）
        keywords   : 手動指定關鍵字（優先使用）
    """
    stem = pdf_path.stem
    folder_name = pdf_path.parent.name

    # document_type：category 明確傳入優先，否則從檔名自動偵測，識別不到回傳 None
    resolved_type = category or _infer_document_type(stem) or None

    # keywords
    resolved_keywords = keywords if keywords is not None else (_extract_pdf_keywords(pdf_path) or [])

    return {
        "file_name":     pdf_path.name,
        "file_type":     pdf_path.suffix.lstrip(".").lower(),
        "page_count":    page_count,
        "patient_id":    folder_name if folder_name.isdigit() else None,
        "document_type": resolved_type,
        "keywords":      resolved_keywords,
        "processed_at":  datetime.now(tz=timezone.utc).isoformat(),
    }
