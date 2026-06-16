from __future__ import annotations
import re
from pathlib import Path as _Path
from layer_b.adapters import adapt, get_source_tool
from layer_b.normalizers.merger import merge_cross_page, expand_spans
from layer_b.normalizers.header_path import build_header_paths
from layer_b.formatters.formatter import linearize_kv, to_json, to_markdown, document_to_retrieval_units
from layer_b.models import IRDocument, RetrievalUnit

SHORT_DOC_THRESHOLD = 500

_SOURCE_PAGE_RE = re.compile(r'^D\((\d+),')
_VERSION_HISTORY_RE = re.compile(r'^\d{4}/\d{2}/\d{2}\s+Version\s+[\d.]+\s*$')


def _parse_source_page(source_str: str) -> int | None:
    m = _SOURCE_PAGE_RE.match(str(source_str or ""))
    return int(m.group(1)) if m else None


def _build_doc_metadata(raw: dict) -> dict:
    meta = raw.get("metadata", {})
    doc = meta.get("document", {})
    src = meta.get("source", {})
    cls = meta.get("classification", {})
    return {
        "doc_title":     doc.get("title"),
        "doc_version":   doc.get("version"),
        "doc_id":        doc.get("doc_id"),
        "document_type": cls.get("document_type"),
        "patient_id":    src.get("patient_id"),
        "keywords":      doc.get("keywords", []),
    }


def _doc_prefix(raw: dict) -> str:
    """Generate a safe doc-level prefix from file_name stem."""
    file_name = raw.get("metadata", {}).get("source", {}).get("file_name", "")
    stem = _Path(file_name).stem if file_name else ""
    return re.sub(r'[^\w\-]', '_', stem) if stem else "doc"

_FLAG = {"high": "ok", "medium": "ok", "low": "low"}


def _quality(level: str) -> tuple[str, float]:
    """Map confidence level to (quality_flag, retrieval_weight). Kept for tests."""
    _WEIGHT = {"high": 1.0, "medium": 0.7, "low": 0.4}
    return _FLAG[level], _WEIGHT[level]


def _continuous_weight(info_loss: float | None) -> float:
    """Continuous retrieval weight: 1.0 - estimated_info_loss_rate.

    null info_loss (vision_llm, or missing data) → 0.7 default.
    Clipped to [0.0, 1.0].
    """
    if info_loss is None:
        return 0.7
    return round(max(0.0, min(1.0, 1.0 - info_loss)), 4)


_INFO_LOSS_LOW = 0.03    # qc=good 門檻
_INFO_LOSS_HIGH = 0.10   # qc=danger 門檻
_WORD_AVG_HIGH = 0.90    # word-level confidence 高門檻
_EMPTY_CELL_RATE_THRESHOLD = 0.30


def assess(table) -> dict:
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


def _is_placeholder(val: str) -> bool:
    """Return True if val is a fill-in placeholder (e.g. 'XX/XX/XXXX').

    Matches strings whose non-separator characters are all uppercase X,
    indicating an unfilled template slot. Does not match real values that
    contain digits or lowercase letters (e.g. '03/24/2023', 'RV', 'BIPL').
    """
    stripped = val.replace("/", "").replace("-", "").replace(".", "").replace(" ", "")
    return bool(stripped) and stripped == "X" * len(stripped)


def _row_to_text(row: dict) -> str:
    """Convert a structured_json row to a natural language string.

    Newlines inside cell values are collapsed to a single space so the
    output is always a clean single-line natural-language sentence ready
    for embedding.
    """
    parts = []
    if row.get("row_header_path"):
        parts.append("，".join(" ".join(h.split()) for h in row["row_header_path"]))
    for cell in row.get("cells", []):
        key_raw = cell["col_header_path"][-1] if cell.get("col_header_path") else ""
        key = " ".join(key_raw.split())
        if len(key) > 20:
            key = key[:20]
        val = " ".join(cell.get("value", "").split())
        if not val or _is_placeholder(val):
            continue
        if key:
            parts.append(f"{key}為{val}")
        else:
            parts.append(val)
    return "，".join(parts)


def _para_has_handwriting(
    para_spans: list[dict],
    styles: list[dict],
    min_confidence: float = 0.5,
) -> bool:
    """Return True if any para span overlaps with a handwritten style span."""
    for style in styles:
        if not style.get("is_handwritten"):
            continue
        if style.get("confidence", 0) < min_confidence:
            continue
        for sstyle in style.get("spans", []):
            s_start = sstyle["offset"]
            s_end = s_start + sstyle["length"]
            for pspan in para_spans:
                p_start = pspan["offset"]
                p_end = p_start + pspan["length"]
                if s_start < p_end and s_end > p_start:
                    return True
    return False


def _process_checkbox_content(content: str) -> tuple[str, list[str]]:
    """Handle Azure DI checkbox markers.

    :unselected: TEXT → TEXT stored in excluded_items, removed from cleaned.
    :selected: TEXT  → marker removed, TEXT kept.
    Returns (cleaned_content, excluded_items).
    """
    excluded = []
    for m in re.finditer(r':unselected:\s*([^:]+?)(?=\s*:(?:un)?selected:|$)', content):
        item = m.group(1).strip()
        if item:
            excluded.append(item)
    cleaned = re.sub(r':unselected:\s*[^:]*?(?=:(?:un)?selected:|$)', '', content)
    cleaned = re.sub(r':selected:\s*', '', cleaned).strip()
    return cleaned, excluded


def _is_formatting_artifact(content: str) -> bool:
    """Return True if content is an empty string or a non-Chinese short artifact."""
    stripped = content.strip()
    if len(stripped) == 0:
        return True
    if len(stripped) < 3 and not re.search(r'[一-鿿]', stripped):
        return True
    return False


MIN_PARA_LEN = 12


def _extract_azure_cu_paragraphs(data: dict) -> list[dict]:
    """Extract paragraph candidates from azure_cu data.paragraphs[]."""
    EXCLUDED_ROLES = {"pageHeader", "pageFooter", "pageNumber"}
    candidates = []
    current_heading = None

    for para in data.get("paragraphs", []):
        role = para.get("role")
        content = para.get("content", "").strip()

        if role in EXCLUDED_ROLES:
            continue

        page = _parse_source_page(para.get("source", ""))
        spans = para.get("spans", [])

        if role == "sectionHeading":
            current_heading = content
            if content:
                candidates.append({
                    "content": content,
                    "page": page,
                    "role": "sectionHeading",
                    "label": None,
                    "heading_breadcrumb": None,
                    "excluded_items": [],
                    "spans": spans,
                })
        else:
            # Filter out form-field labels and navigation artifacts.
            # sectionHeading is exempt; body text requires MIN_PARA_LEN chars.
            if len(content) < MIN_PARA_LEN:
                continue
            if _VERSION_HISTORY_RE.match(content):
                continue
            candidates.append({
                "content": content,
                "page": page,
                "role": None,
                "label": None,
                "heading_breadcrumb": current_heading,
                "excluded_items": [],
                "spans": spans,
            })

    return candidates


_SKIP_ROLES_DI = {"footnote", "pageFooter", "pageNumber"}
_DEMO_PATTERN = re.compile(r'[一-鿿\w]{1,8}[：:]\s*.+')


def _extract_azure_di_paragraphs(data: dict) -> list[dict]:
    """Extract paragraph candidates from azure_di data.paragraphs[].

    Skips footnote/pageFooter/pageNumber roles and short non-heading paragraphs.
    Checkbox-marked paragraphs are always kept (even if short or empty) to preserve
    excluded_items for unselected choices that the caller accumulates.
    """
    candidates = []

    for para in data.get("paragraphs", []):
        role = para.get("role")
        if role in _SKIP_ROLES_DI:
            continue

        content_raw = para.get("content", "")
        had_checkboxes = ":selected:" in content_raw or ":unselected:" in content_raw

        bounding = para.get("boundingRegions", [])
        page = bounding[0]["pageNumber"] if bounding else None
        spans = para.get("spans", [])

        cleaned, excluded = _process_checkbox_content(content_raw)

        # Filter short non-heading paragraphs that aren't checkbox items.
        # Checkbox paragraphs kept even when short: they carry meaningful selected
        # content or excluded_items for unselected options.
        if (role != "sectionHeading"
                and not had_checkboxes
                and cleaned
                and len(cleaned) < MIN_PARA_LEN):
            continue

        candidates.append({
            "content": cleaned,
            "page": page,
            "role": "sectionHeading" if role == "sectionHeading" else None,
            "label": None,
            "heading_breadcrumb": None,
            "excluded_items": excluded,
            "spans": spans,
        })

    return candidates


def _aggregate_header_cluster(candidates: list[dict]) -> list[dict]:
    """Merge consecutive short key:value lines at page start into one demographic unit.

    Lab report photos often have patient info split across many 1-line paragraphs.
    Merging them improves retrieval recall for patient-level queries.
    Requires at least 3 matching lines to trigger (avoids false positives on normal docs).
    """
    if not candidates:
        return candidates

    first_page = candidates[0].get("page")
    header_end = 0
    for c in candidates:
        if c.get("page") != first_page or c.get("role") == "sectionHeading":
            break
        content = c.get("content", "")
        if len(content) <= 80 and _DEMO_PATTERN.search(content):
            header_end += 1
        else:
            break

    if header_end < 3:
        return candidates

    header_cands = candidates[:header_end]
    merged_content = "\n".join(c["content"] for c in header_cands)
    all_excluded = [item for c in header_cands for item in c.get("excluded_items", [])]

    merged = {
        "content": merged_content,
        "page": first_page,
        "role": None,
        "label": "patient_demographics",
        "heading_breadcrumb": None,
        "excluded_items": all_excluded,
        "spans": [],
    }

    return [merged] + candidates[header_end:]


def _extract_docling_paragraphs(data: dict) -> list[dict]:
    """Extract paragraph candidates from docling data.texts[]."""
    ALLOWED_LABELS = {"text", "list_item", "section_header"}
    candidates = []
    current_heading = None

    for text_item in data.get("texts", []):
        label = text_item.get("label", "text")
        content = text_item.get("text", "").strip()

        if label not in ALLOWED_LABELS:
            continue

        if _is_formatting_artifact(content):
            continue

        prov = text_item.get("prov", [])
        # Docling DOCX (xml_native) does not populate prov; page=None is correct,
        # not a bug. Do not fallback to page=1 — it misleads retrieval on multi-page docs.
        page = prov[0].get("page_no") if prov else None

        if label == "section_header":
            current_heading = content
            candidates.append({
                "content": content,
                "page": page,
                "role": None,
                "label": "section_header",
                "heading_breadcrumb": None,
                "excluded_items": [],
                "spans": [],
            })
        else:
            candidates.append({
                "content": content,
                "page": page,
                "role": None,
                "label": label,
                "heading_breadcrumb": current_heading,
                "excluded_items": [],
                "spans": [],
            })

    return candidates


_TOOL_SHORT = {
    "azure_content_understanding": "azure_cu",
    "azure_document_intelligence": "azure_di",
}


def _normalize_source_tool(tool: str) -> str:
    """Normalize full tool names to short canonical names used in RetrievalUnit."""
    return _TOOL_SHORT.get(tool, tool)


def _doc_confidence(raw: dict) -> tuple[str, str, float]:
    """Return (confidence_level, quality_flag, retrieval_weight) for paragraph units.

    Reads estimated_info_loss_rate from metadata.qc (primary) or
    extractor_metadata (fallback for future schema changes).
    """
    info_loss = None
    try:
        info_loss = raw["metadata"]["qc"]["estimated_info_loss_rate"]
    except (KeyError, TypeError):
        pass
    if info_loss is None:
        try:
            info_loss = raw["extractor_metadata"]["estimated_info_loss_rate"]
        except (KeyError, TypeError):
            pass
    weight = _continuous_weight(info_loss)
    # 全掃描文件降權（OCR 準確率低於正常文件）
    if raw.get("extractor_metadata", {}).get("is_fully_scanned"):
        weight = weight * 0.8
    if weight >= 0.97:
        level = "high"
    elif weight >= 0.80:
        level = "medium"
    else:
        level = "low"
    flag = "low" if level == "low" else "ok"
    return level, flag, weight


def _resolve_page_image(page_images: dict, page: int | None) -> str:
    """Resolve image path for a given page from page_images dict."""
    if page is None:
        return ""
    v = page_images.get(str(page)) or page_images.get(page)
    if v is None:
        return ""
    if isinstance(v, dict):
        return v.get("path", "")
    return v or ""


def _paragraph_path(raw: dict, source_tool: str, doc_prefix: str = "", doc_metadata: dict | None = None) -> list[RetrievalUnit]:
    """Extract paragraph-path RetrievalUnits from raw document data.

    Runs in parallel with _table_path(); called by process_document().
    """
    data = raw["data"]
    styles = data.get("styles", [])
    page_images = data.get("page_images", {})
    source_tool = _normalize_source_tool(source_tool)

    if source_tool == "azure_cu":
        candidates = _extract_azure_cu_paragraphs(data)
    elif source_tool == "azure_di":
        candidates = _extract_azure_di_paragraphs(data)
        candidates = _aggregate_header_cluster(candidates)
    elif source_tool == "docling":
        candidates = _extract_docling_paragraphs(data)
    else:
        return []

    # Accumulate excluded_items from all candidates (including empty-content ones)
    all_excluded_items: list[str] = []
    for c in candidates:
        all_excluded_items.extend(c.get("excluded_items", []))

    # Filter out empty-content candidates (azure_di may produce them after checkbox processing).
    # Short-but-non-empty candidates from checkbox items are kept — they may carry
    # meaningful selected content even when under MIN_PARA_LEN.
    candidates = [c for c in candidates if c["content"].strip()]

    # Remove sectionHeading candidates whose content already appears as heading_breadcrumb
    # in a body paragraph — the heading is embedded there and the standalone copy adds no value.
    if source_tool == "azure_cu":
        headings_with_body = {c["heading_breadcrumb"] for c in candidates if c.get("heading_breadcrumb")}
        candidates = [
            c for c in candidates
            if not (c.get("role") == "sectionHeading" and c["content"] in headings_with_body)
        ]

    if not candidates:
        return []

    total_len = sum(len(c["content"]) for c in candidates)

    confidence_level, quality_flag, retrieval_weight = _doc_confidence(raw)

    if total_len < SHORT_DOC_THRESHOLD:
        any_hw = any(
            _para_has_handwriting(c.get("spans", []), styles)
            for c in candidates
        )
        markdown_content = data.get("markdown", "")
        # For azure_di, data.markdown contains raw checkbox markers; use cleaned candidates instead.
        if not markdown_content.strip() or (
            source_tool == "azure_di"
            and (":selected:" in markdown_content or ":unselected:" in markdown_content)
        ):
            markdown_content = "\n".join(c["content"] for c in candidates)
        source_pages = sorted(set(
            c["page"] for c in candidates if c.get("page") is not None
        ))
        short_doc_id = f"{doc_prefix}_p_{source_tool}_doc_001" if doc_prefix else f"p_{source_tool}_doc_001"
        return [RetrievalUnit(
            retrieval_unit_id=short_doc_id,
            source_tool=source_tool,
            embedding_text=markdown_content,
            structured_json={
                "type": "document",
                "page": None,
                "role": None,
                "label": None,
                "heading_breadcrumb": None,
                "content": markdown_content,
                "excluded_items": all_excluded_items,
                "has_handwriting": any_hw,
            },
            display_markdown=markdown_content,
            row_texts=[],
            confidence_level=confidence_level,
            quality_flag=quality_flag,
            retrieval_weight=retrieval_weight,
            source_pages=source_pages,
            page_image_refs={
                str(p): path
                for p in source_pages
                if (path := _resolve_page_image(page_images, p))
            },
            doc_metadata=doc_metadata or {},
        )]

    units: list[RetrievalUnit] = []
    for idx, cand in enumerate(candidates):
        page = cand.get("page")
        page_str = f"{page:03d}" if page is not None else "000"
        has_hw = _para_has_handwriting(cand.get("spans", []), styles)
        hb = cand.get("heading_breadcrumb")
        content = cand["content"]
        embedding_text = f"{hb}\n{content}" if hb else content
        display_markdown = f"**{hb}**\n\n{content}" if hb else content
        img_path = _resolve_page_image(page_images, page)
        para_id = f"{doc_prefix}_p_{source_tool}_{page_str}_{idx:03d}" if doc_prefix else f"p_{source_tool}_{page_str}_{idx:03d}"
        units.append(RetrievalUnit(
            retrieval_unit_id=para_id,
            source_tool=source_tool,
            embedding_text=embedding_text,
            structured_json={
                "type": "paragraph",
                "page": page,
                "role": cand.get("role"),
                "label": cand.get("label"),
                "heading_breadcrumb": hb,
                "content": content,
                "excluded_items": cand.get("excluded_items", []),
                "has_handwriting": has_hw,
            },
            display_markdown=display_markdown,
            row_texts=[],
            confidence_level=confidence_level,
            quality_flag=quality_flag,
            retrieval_weight=retrieval_weight,
            source_pages=[page] if page is not None else [],
            page_image_refs={str(page): img_path} if img_path else {},
            doc_metadata=doc_metadata or {},
        ))
    return units


def _table_path(raw: dict, source_tool: str, doc_prefix: str = "", doc_metadata: dict | None = None) -> list[RetrievalUnit]:
    norm_source = _normalize_source_tool(source_tool)
    tables = adapt(raw, source_tool)
    tables = merge_cross_page(tables)

    # Compute merge_rate BEFORE expand_spans — expansion resets all spans to 1.
    # merge_rate = fraction of cells with row_span > 1 or col_span > 1.
    merge_rates: list[float] = []
    for t in tables:
        total = len(t.cells)
        merged = sum(1 for c in t.cells if c.row_span > 1 or c.col_span > 1)
        merge_rates.append(round(merged / total, 4) if total else 0.0)

    tables = [expand_spans(t) for t in tables]
    labelled = [build_header_paths(t) for t in tables]

    units: list[RetrievalUnit] = []
    for (lt, t), merge_rate in zip(zip(labelled, tables), merge_rates):
        conf = assess(t)
        level = conf["level"]
        flag = "low" if level == "low" else "ok"
        weight = _continuous_weight(conf["score"])
        json_out = to_json(lt)
        json_out["merge_rate"] = merge_rate

        row_texts = [_row_to_text(r) for r in json_out.get("rows", []) if _row_to_text(r)]
        table_id = f"{doc_prefix}_{t.table_id}" if doc_prefix else t.table_id
        markdown_out = to_markdown(lt)
        # azure_di: col-header heuristics make KV linearization unreliable — use markdown always.
        # Other sources: markdown for heavily merged tables (>30% spanning cells).
        embedding_text = markdown_out if (norm_source == "azure_di" or merge_rate > 0.3) else linearize_kv(lt)
        units.append(RetrievalUnit(
            retrieval_unit_id=table_id,
            source_tool=t.source_tool,
            embedding_text=embedding_text,
            structured_json=json_out,
            display_markdown=markdown_out,
            confidence_level=level,
            quality_flag=flag,
            retrieval_weight=weight,
            source_pages=t.source_pages,
            page_image_refs=conf["page_image_refs"],
            row_texts=row_texts,
            doc_metadata=doc_metadata or {},
        ))
    return units


def _build_page_context_map(data: dict, source_tool: str) -> dict:
    """Build page_no → short paragraph summary for figure embedding fallback.

    When a figure has no caption, nearby paragraph text provides semantic context
    so the figure chunk can be retrieved by content-related queries.
    """
    page_texts: dict = {}
    if source_tool == "azure_cu":
        for para in data.get("paragraphs", []):
            content = para.get("content", "").strip()
            if not content or len(content) < MIN_PARA_LEN:
                continue
            page = _parse_source_page(para.get("source", ""))
            if page is None:
                continue
            page_texts.setdefault(page, []).append(content)
    elif source_tool == "azure_di":
        for para in data.get("paragraphs", []):
            bounding = para.get("boundingRegions", [])
            page = bounding[0]["pageNumber"] if bounding else None
            if page is None:
                continue
            content = para.get("content", "").strip()
            if not content or len(content) < MIN_PARA_LEN:
                continue
            page_texts.setdefault(page, []).append(content)
    return {
        page: " ".join(texts[:5])[:300]
        for page, texts in page_texts.items()
        if texts
    }


def _figure_path(raw: dict, source_tool: str, doc_prefix: str, doc_metadata: dict | None = None) -> list[RetrievalUnit]:
    """Generate RetrievalUnit for each meaningful figure (has_image=True, area_sqin >= 0.5)."""
    data = raw.get("data", {})
    figures = data.get("figures", [])
    page_images = data.get("page_images", {})
    confidence_level, quality_flag, retrieval_weight = _doc_confidence(raw)
    norm_tool = _normalize_source_tool(source_tool)
    page_context_map = _build_page_context_map(data, norm_tool)

    units = []
    seq = 0
    for fig in figures:
        if not fig.get("has_image") or fig.get("area_sqin", 0.0) < 0.5:
            continue

        page = _parse_source_page(fig.get("source", ""))

        cap_raw = fig.get("caption") or {}
        caption_text = (cap_raw.get("content", "") if isinstance(cap_raw, dict) else "").strip()

        if caption_text:
            embedding_text = caption_text
        elif page is not None and page_context_map.get(page):
            embedding_text = f"[圖表 第{page}頁] {page_context_map[page]}"
        else:
            embedding_text = f"[圖表 第{page}頁]" if page else "[圖表]"

        img_path = _resolve_page_image(page_images, page)
        page_image_refs = {str(page): img_path} if img_path else {}

        unit_id = f"{doc_prefix}_f_{seq:03d}" if doc_prefix else f"f_{seq:03d}"
        display = f"![figure]({fig.get('path', '')})"
        if caption_text:
            display += f"\n\n{caption_text}"

        units.append(RetrievalUnit(
            retrieval_unit_id=unit_id,
            source_tool=norm_tool,
            embedding_text=embedding_text,
            structured_json={
                "type": "figure",
                "page": page,
                "caption": caption_text,
                "path": fig.get("path"),
                "area_sqin": fig.get("area_sqin"),
            },
            display_markdown=display,
            confidence_level=confidence_level,
            quality_flag=quality_flag,
            retrieval_weight=retrieval_weight,
            source_pages=[page] if page is not None else [],
            page_image_refs=page_image_refs,
            row_texts=[],
            doc_metadata=doc_metadata or {},
        ))
        seq += 1
    return units


def _high_graphics_path(
    raw: dict,
    doc_prefix: str,
    norm_tool: str,
    doc_metadata: dict | None = None,
) -> list[RetrievalUnit]:
    """為有 page_image 但無任何 figure RetrievalUnit 的頁面生成文件級 RetrievalUnit。

    目的：讓癌症診療指引的流程圖/決策樹頁面可被 RAG 查詢命中。
    """
    data = raw.get("data", {})
    page_images = data.get("page_images", {})
    figures = data.get("figures", [])

    if not page_images:
        return []

    # 找出有明確 figure 的頁面（這些頁面 _figure_path 已處理）
    pages_with_figures: set[int] = set()
    for fig in figures:
        if fig.get("has_image"):
            page = _parse_source_page(fig.get("source", ""))
            if page is not None:
                pages_with_figures.add(page)

    confidence_level, quality_flag, retrieval_weight = _doc_confidence(raw)

    page_context_map = _build_page_context_map(data, norm_tool)

    units: list[RetrievalUnit] = []
    for page_key, img_info in page_images.items():
        try:
            page = int(page_key)
        except (ValueError, TypeError):
            continue

        if page in pages_with_figures:
            continue  # 已由 _figure_path 處理

        # img_info may be a dict {"path": ..., "has_image": ...} or a plain str path
        if isinstance(img_info, dict):
            if not img_info.get("has_image"):
                continue
            img_path = img_info.get("path", "")
        elif isinstance(img_info, str):
            img_path = img_info
        else:
            continue

        # 用鄰近段落文字作為 embedding_text（讓查詢可命中）
        context = page_context_map.get(page, "")
        embedding_text = f"[流程圖/圖表 第{page}頁]" + (f" {context}" if context else "")

        unit_id = f"{doc_prefix}_hg_{page:03d}" if doc_prefix else f"hg_{page:03d}"

        units.append(RetrievalUnit(
            retrieval_unit_id=unit_id,
            source_tool=norm_tool,
            embedding_text=embedding_text,
            structured_json={
                "type": "document",
                "label": "high_graphics",
                "page": page,
                "content": embedding_text,
            },
            display_markdown=f"![page_{page}]({img_path})" if img_path else "",
            confidence_level=confidence_level,
            quality_flag=quality_flag,
            retrieval_weight=retrieval_weight * 0.9,  # 輕微降權，因無直接文字
            source_pages=[page],
            page_image_refs={str(page): img_path} if img_path else {},
            row_texts=[],
            doc_metadata=doc_metadata or {},
        ))

    return units


def _document_path(raw: dict) -> list[RetrievalUnit]:
    docs = adapt(raw, "vision_llm")
    units: list[RetrievalUnit] = []
    for doc in docs:
        for section in doc.sections:
            for elem in section.elements:
                content = elem.get("content") or ""
                title = section.title
                display = f"## {title}\n\n{content}" if title else content
                units.append(RetrievalUnit(
                    retrieval_unit_id=f"{doc.doc_id}_{elem.get('element_id', '')}",
                    source_tool="vision_llm",
                    embedding_text=content,
                    structured_json=elem,
                    display_markdown=display,
                    confidence_level="medium",
                    quality_flag="ok",
                    retrieval_weight=_continuous_weight(None),
                    source_pages=[],
                    page_image_refs={},
                    doc_id=doc.doc_id,
                    section_id=section.section_id,
                    section_title=section.title,
                    semantic_type=section.semantic_type,
                    page_no=elem.get("page_no"),
                    reading_order=elem.get("reading_order"),
                    element_type=elem.get("type", "text"),
                    entities=elem.get("entities", {}),
                    document_signals=elem.get("document_signals", []),
                ))
    return units


def process_document(raw: dict) -> list[RetrievalUnit]:
    """Unified entry point: Conversion Layer JSON → list[RetrievalUnit].

    Routes by extractor_metadata.tool:
      - vision_llm → document path (IRDocument → element-per-unit)
      - others     → table path + paragraph path
    """
    source_tool = get_source_tool(raw)
    if source_tool == "vision_llm":
        return _document_path(raw)
    doc_prefix = _doc_prefix(raw)
    doc_metadata = _build_doc_metadata(raw)
    units = []
    units.extend(_table_path(raw, source_tool, doc_prefix, doc_metadata))
    units.extend(_paragraph_path(raw, source_tool, doc_prefix, doc_metadata))
    norm_tool = _normalize_source_tool(source_tool)
    units.extend(_figure_path(raw, source_tool, doc_prefix, doc_metadata))
    units.extend(_high_graphics_path(raw, doc_prefix, norm_tool, doc_metadata))
    vision_desc = raw.get("vision_description", "")
    if vision_desc:
        from dataclasses import replace
        units = [
            replace(u, embedding_text=vision_desc + "\n" + u.embedding_text)
            if u.embedding_text
            else replace(u, embedding_text=vision_desc)
            for u in units
        ]
    return units
