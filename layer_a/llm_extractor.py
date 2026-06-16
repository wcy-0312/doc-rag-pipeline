"""
llm_extractor.py — LLM Extractor（Beta，schema-v3.0）
設計：Element-local semantic annotation

架構：
  1. 載入 playbook（YAML）定義語意類型和訊號規則
  2. fitz + playbook pattern 偵測 section 邊界
  3. 每個 section 送圖片給 Vision LLM
  4. LLM 輸出 elements，並在每個 element 內直接標注 entities / document_signals
  5. 後處理：element_id 生成、element-local validation、QC

注意：
  - entities/signals 的 evidence 是所在 element.content（LLM 生成）
  - element.content ≠ PDF OCR ground truth，無法驗證 LLM 是否幻覺
  - QC 只能驗證標注與 element.content 的內部一致性
"""

from __future__ import annotations
import sys, re, json, base64, hashlib, yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

_PLAYBOOK_DIR = Path(__file__).parent / "playbooks"

VALID_TYPES = {"text", "table", "figure", "list", "formula"}


# ── Playbook 載入 ─────────────────────────────────────────────────────────────

def _load_playbook(document_type: str) -> dict:
    for name in [document_type, "_default"]:
        path = _PLAYBOOK_DIR / f"{name}.yaml"
        if path.exists():
            return yaml.safe_load(path.read_text(encoding="utf-8"))
    return {}


# ── Section 邊界偵測（Regex path）────────────────────────────────────────────

def _detect_sections_regex(
    pdf_path: Path, patterns: list, total_pages: int
) -> list[dict]:
    """Regex-based section detection（有 playbook patterns 時使用）。"""
    import fitz
    doc = fitz.open(str(pdf_path))
    headings: list[tuple[int, str]] = []

    for page_idx in range(total_pages):
        page_no = page_idx + 1
        text = doc[page_idx].get_text()
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            for pat in patterns:
                if pat.match(line):
                    headings.append((page_no, line))
                    break

    doc.close()

    if not headings:
        return []

    seen_pages: set[int] = set()
    deduped: list[tuple[int, str]] = []
    for page_no, title in headings:
        if page_no not in seen_pages:
            seen_pages.add(page_no)
            deduped.append((page_no, title))

    sections = []
    for i, (page_no, title) in enumerate(deduped):
        page_end = (deduped[i + 1][0] - 1) if i + 1 < len(deduped) else total_pages
        page_end = max(page_no, page_end)
        sections.append({"title": title, "level": 1, "page_start": page_no, "page_end": page_end})

    if deduped[0][0] > 1:
        sections.insert(0, {
            "title": "[前置內容]", "level": 0,
            "page_start": 1, "page_end": deduped[0][0] - 1,
        })

    return sections


# ── Section 邊界偵測（LLM first-pass path）───────────────────────────────────

def _detect_sections_llm(pdf_path: Path, total_pages: int, llm) -> list[dict]:
    """LLM first-pass：text-only，識別文件邏輯段落邊界。

    只傳每頁前幾行文字（不傳圖片），token 成本低。
    對無文字頁（JPG 轉 PDF）直接回傳 1 section，不呼叫 LLM。
    """
    import fitz
    from langchain_core.messages import HumanMessage

    doc = fitz.open(str(pdf_path))
    page_previews: list[str] = []
    for page_idx in range(total_pages):
        lines = [
            ln.strip()
            for ln in doc[page_idx].get_text().split("\n")
            if ln.strip()
        ][:8]
        page_previews.append(
            f"[第{page_idx + 1}頁] " + " / ".join(lines)
            if lines else f"[第{page_idx + 1}頁] (無文字)"
        )
    doc.close()

    # 所有頁都無文字（JPG 轉 PDF 等）→ 直接回傳 1 section
    if all("(無文字)" in p for p in page_previews):
        return [{"title": pdf_path.stem, "level": 1, "page_start": 1, "page_end": total_pages}]

    pages_text = "\n".join(page_previews)
    prompt = f"""以下是一份文件各頁的前幾行文字。請識別文件的邏輯段落，輸出段落列表。

規則：
1. 每個段落有標題（從文件中取，不要自創）、起始頁、結束頁
2. 段落數量 1～{min(total_pages, 20)} 個（視文件結構決定，不要過度細分）
3. 段落必須覆蓋所有頁面（第一個 page_start=1，最後一個 page_end={total_pages}）
4. 若文件沒有明顯段落結構（如單頁表格、衛教單張），輸出 1 個段落
5. 只輸出 JSON，不要說明文字

格式（嚴格遵守）：
[{{"title": "段落標題", "page_start": 1, "page_end": 3}}, ...]

文件各頁預覽：
{pages_text}"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r'^```(?:json)?\n?', '', raw)
            raw = re.sub(r'\n?```$', '', raw)
        parsed = json.loads(raw)
        if not isinstance(parsed, list) or not parsed:
            raise ValueError("empty or non-list response")

        sections = []
        for item in parsed:
            title      = str(item.get("title") or "").strip() or "[未命名段落]"
            page_start = max(1, min(int(item.get("page_start", 1)), total_pages))
            page_end   = max(page_start, min(int(item.get("page_end", total_pages)), total_pages))
            sections.append({"title": title, "level": 1, "page_start": page_start, "page_end": page_end})

        if sections and sections[0]["page_start"] > 1:
            sections.insert(0, {
                "title": "[前置內容]", "level": 0,
                "page_start": 1, "page_end": sections[0]["page_start"] - 1,
            })

        return sections

    except Exception:
        # LLM first-pass 失敗 → 回傳 1 section
        return [{"title": pdf_path.stem, "level": 1, "page_start": 1, "page_end": total_pages}]


# ── Section 邊界偵測（統一入口）──────────────────────────────────────────────

def _detect_sections(pdf_path: Path, playbook: dict, llm=None) -> list[dict]:
    """統一入口：有 playbook patterns → regex；無 patterns → LLM first-pass。

    playbook.boundary_detection.patterns 為 optional：
    - 有 patterns：regex 偵測（現有 playbook 行為不變）
    - 無 patterns 且 regex 找不到 heading，且有 llm：LLM text-only first-pass
    - 否則：整份文件視為 1 section
    """
    import fitz
    doc = fitz.open(str(pdf_path))
    total_pages = doc.page_count
    doc.close()

    raw_patterns = playbook.get("boundary_detection", {}).get("patterns", [])
    compiled = [re.compile(p) for p in raw_patterns if p]

    if compiled:
        sections = _detect_sections_regex(pdf_path, compiled, total_pages)
        if sections:
            return sections
        # regex 有 patterns 但找不到 heading → fallthrough 到 LLM

    if llm is not None:
        return _detect_sections_llm(pdf_path, total_pages, llm)

    return [{"title": pdf_path.stem, "level": 1, "page_start": 1, "page_end": total_pages}]


# ── Prompt Schema ─────────────────────────────────────────────────────────────

def _build_prompt_schema(playbook: dict) -> str:
    """新版 schema：entities/signals 直接標注在 element 內，不需要 source_ref。"""
    se = playbook.get("semantic_extraction", {})
    section_types = se.get("section_types", ["purpose", "warning", "procedure", "other"])
    entity_types  = se.get("entity_types",  ["medication", "symptom", "measurement"])
    signal_types  = list((se.get("document_signal_rules") or {}).keys()) or ["warning", "emergency", "dosage", "contact"]

    entity_fields = ", ".join(f'"{t}s": [{{"text": "...", "certainty": "high | medium"}}]' for t in entity_types[:4])

    return f"""{{
  "section_analysis": {{
    "semantic_type": "{' | '.join(section_types)}"
  }},
  "elements": [
    {{
      "type": "text | table | figure | list | formula",
      "page_no": <整數>,
      "reading_order": <此頁內從上到下的順序，從 1 開始>,
      "content": "<文字內容>",
      "entities": {{
        {entity_fields}
      }},
      "document_signals": [
        {{
          "signal_type": "{' | '.join(signal_types)} | other",
          "basis": "explicit_phrase | explicit_pattern",
          "markers": ["<此 element 中實際出現的文字>"]
        }}
      ]
    }}
  ]
}}"""


# ── 後處理：element_id 生成 ────────────────────────────────────────────────────

def _assign_element_ids(section_index: int, elements: list[dict]) -> list[dict]:
    """為每個 element 生成穩定的 element_id（程式生成，不靠 LLM）。"""
    for i, elem in enumerate(elements, 1):
        elem["element_id"] = f"s{section_index:03d}_e{i:03d}"
    return elements


# ── Section 萃取（Vision LLM）────────────────────────────────────────────────

def _extract_section(
    pdf_path: Path,
    section: dict,
    section_index: int,
    playbook: dict,
    llm,
    output_dir: Path | None,
) -> tuple[dict, list[dict]]:
    """呼叫 Vision LLM 萃取一個 section。

    Returns:
        (section_analysis, elements)
    """
    import fitz
    from langchain_core.messages import HumanMessage

    hints   = playbook.get("prompt_hints", {})
    ignore  = playbook.get("ignore_regions", [])
    institution = hints.get("institution_name", "")
    domain_note = hints.get("note", "")
    ignore_str  = "、".join(ignore) if ignore else "無"

    # 頁面圖片
    doc = fitz.open(str(pdf_path))
    page_images: list[tuple[int, str]] = []
    for page_no in range(section["page_start"], section["page_end"] + 1):
        idx = page_no - 1
        if idx >= len(doc):
            continue
        pixmap = doc[idx].get_pixmap(dpi=120)
        b64    = base64.b64encode(pixmap.tobytes("png")).decode()
        page_images.append((page_no, b64))
    doc.close()

    # 圖片存放（figure elements 用）
    _figures_root = Path(output_dir) if output_dir else (
        Path(__file__).parent.parent / "output" / "llm_pages"
    )
    img_dir = _figures_root / "figures"
    img_dir.mkdir(parents=True, exist_ok=True)
    page_b64: dict[int, str] = {pn: b64 for pn, b64 in page_images}
    # 同頁多個 figure → 儲存一次，共用路徑與 sha256
    saved_page_images: dict[int, tuple[str, str]] = {}  # page_no → (image_path, sha256)

    if not page_images:
        return {"semantic_type": "other"}, []

    # 建立 prompt schema（從 playbook 動態生成）
    schema_str = _build_prompt_schema(playbook)

    # 語意類型列表（for prompt）
    se = playbook.get("semantic_extraction", {})
    section_types_str = " | ".join(se.get("section_types", ["other"]))
    signal_rules = se.get("document_signal_rules") or {}
    signal_rules_str = ""
    for stype, rules in signal_rules.items():
        markers = rules.get("explicit_markers", []) + rules.get("explicit_patterns", [])
        if markers:
            signal_rules_str += f"\n  {stype}：{', '.join(markers[:4])}"

    system_prompt = f"""你是醫療文件語意萃取器（非一般 OCR）。

文件類型：{playbook.get('document_type', '未知')}
機構名稱：{institution}（若圖中出現其他機構名稱請以此取代）
忽略區域：{ignore_str}

{domain_note}

請分析以下頁面，輸出結構化 JSON。

規則：
1. section_analysis.semantic_type 只能是：{section_types_str}
2. 每個 element 的 entities 只記錄這個 element 中明確出現的實體，不確定就留空陣列
3. 每個 element 的 document_signals 只記錄這個 element 中明確出現的訊號（如：{signal_rules_str}）
4. entities 和 document_signals 直接標注在對應的 element 內，無需 source_ref
5. certainty 只能是 high（清楚讀到）或 medium（有輕微疑慮）
6. type=figure 且無可讀文字時，content 設為 null；不要自行生成圖片描述
7. 不要產生：摘要、問題、搜尋文字、答案、chunking 建議

嚴格禁止：
- 新增文件沒有的醫療建議或數值
- 猜測模糊文字（不確定就不輸出，不要猜）
- 把頁首、頁尾、頁碼當作正文

格式：
{schema_str}

只輸出 JSON，不要任何說明文字。"""

    content_parts: list = [{"type": "text", "text": system_prompt}]
    for page_no, b64 in page_images:
        content_parts.append({"type": "text", "text": f"\n--- 第 {page_no} 頁 ---"})
        content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

    msg = HumanMessage(content=content_parts)

    default_analysis = {"semantic_type": "other"}

    try:
        response = llm.invoke([msg])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r'^```(?:json)?\n?', '', raw)
            raw = re.sub(r'\n?```$', '', raw)
        parsed = json.loads(raw)
    except Exception as e:
        error_elem = [{
            "type": "text", "page_no": section["page_start"],
            "reading_order": 1, "content": f"[萃取失敗: {e}]", "_error": str(e),
        }]
        return default_analysis, error_elem

    section_analysis = parsed.get("section_analysis", default_analysis)
    elements_raw = parsed.get("elements", [])

    # 後處理 elements：type 驗證 + 存圖
    # 同頁多個 figure 只存一次圖（LLM 無 bbox，無法裁切，全頁圖共用）
    elements = []
    for elem in elements_raw:
        if elem.get("type", "text") not in VALID_TYPES:
            elem["type"] = "text"

        if elem.get("type") == "figure":
            page_no  = elem.get("page_no", section["page_start"])
            b64_data = page_b64.get(page_no)
            if b64_data:
                if page_no not in saved_page_images:
                    png_bytes = base64.b64decode(b64_data)
                    sha256    = hashlib.sha256(png_bytes).hexdigest()
                    fname     = f"{pdf_path.stem}_p{page_no}.png"
                    fpath     = img_dir / fname
                    if not fpath.exists():
                        fpath.write_bytes(png_bytes)
                    image_path = (str(fpath.relative_to(_figures_root))
                                  if output_dir else str(fpath))
                    saved_page_images[page_no] = (image_path, sha256)
                image_path, sha256 = saved_page_images[page_no]
                elem["image_path"] = image_path
                elem["sha256"]     = sha256
                elem["has_image"]  = True
            else:
                elem["image_path"] = None
                elem["has_image"]  = False
        else:
            elem.pop("has_image", None)

        elements.append(elem)

    # element_id 由程式生成
    elements = _assign_element_ids(section_index, elements)

    return section_analysis, elements


# ── QC 計算 ──────────────────────────────────────────────────────────────────

def _compute_llm_qc(sections: list[dict]) -> dict:
    """Element-local QC：驗證 entity/signal 標注是否出現在同一 element 的 content 中。

    entity_local_match_rate：entity.text 出現在所在 element.content 的比率
    signal_local_match_rate：signal.markers 至少一個出現在所在 element.content 的比率
    無文字的 element（figure content=None）跳過不計入分母。
    """
    elements_total    = sum(len(s.get("elements", [])) for s in sections)
    extraction_errors = sum(1 for s in sections for e in s.get("elements", []) if "_error" in e)

    entity_total = entity_match = 0
    signal_total = signal_match = 0

    for s in sections:
        for elem in s.get("elements", []):
            content = elem.get("content") or ""
            if not content:
                continue  # figure 等無文字 element 跳過
            # 空白正規化：tab/換行/多空格統一為單一空格，避免 tab vs space 造成 false negative
            content_norm = re.sub(r'\s+', ' ', content)
            for elist in elem.get("entities", {}).values():
                if isinstance(elist, list):
                    for ent in elist:
                        text = ent.get("text", "")
                        if text:
                            entity_total += 1
                            text_norm = re.sub(r'\s+', ' ', text)
                            if text_norm in content_norm:
                                entity_match += 1
            for sig in elem.get("document_signals", []):
                markers = sig.get("markers", [])
                if markers:
                    signal_total += 1
                    if any(re.sub(r'\s+', ' ', m) in content_norm for m in markers):
                        signal_match += 1

    entity_local_match_rate = round(entity_match / entity_total, 3) if entity_total else 1.0
    signal_local_match_rate = round(signal_match / signal_total, 3) if signal_total else 1.0

    if extraction_errors > 0 and elements_total == 0:
        qc_level = "danger"
    elif signal_local_match_rate < 0.95:
        qc_level = "warning"
    elif entity_local_match_rate < 0.85:
        qc_level = "warning"
    elif extraction_errors > elements_total * 0.30:
        qc_level = "warning"
    else:
        qc_level = "good"

    return {
        "entity_local_match_rate": entity_local_match_rate,
        "signal_local_match_rate": signal_local_match_rate,
        "extraction_errors":       extraction_errors,
        "qc_level":                qc_level,
        "warnings": ["LLM_EXTRACTOR_BETA", "LLM_EVIDENCE_IS_INTERNAL_REFERENCE_ONLY"],
        "errors":   [],
    }


# ── 主入口 ───────────────────────────────────────────────────────────────────

def convert_pdf_llm(
    pdf_path: Path,
    llm,
    category: str = "",
    output_dir: Path | None = None,
    keywords: list[str] | None = None,
) -> dict:
    """LLM Extractor（Beta）：PDF → section-centric JSON（schema-v3.0）。

    輸出結構：
      sections[].semantic_type         — section 語意分類（playbook enum）
      sections[].elements[].entities   — 此 element 中明確出現的實體
      sections[].elements[].document_signals — 此 element 中明確出現的語意訊號
      sections[].elements[].element_id — 穩定 ID（程式生成）

    QC 說明：
      entity_local_match_rate / signal_local_match_rate 驗證標注與
      所在 element.content 的內部一致性。
      element.content 是 LLM 生成，不是外部 OCR ground truth。
    """
    from metadata_builder import build_metadata, _infer_document_type
    import fitz as _fitz

    # 1. playbook
    effective_category = category or _infer_document_type(pdf_path.stem) or ""
    playbook = _load_playbook(effective_category)

    # 2. section 邊界（有 playbook patterns → regex；無 patterns → LLM first-pass）
    sections_meta = _detect_sections(pdf_path, playbook, llm=llm)

    # 3. 逐 section 萃取
    sections_output = []
    for idx, sec in enumerate(sections_meta, 1):
        section_analysis, elements = _extract_section(
            pdf_path, sec, idx, playbook, llm, output_dir
        )
        sections_output.append({
            "section_id":    f"s{idx:03d}",
            "title":         sec["title"],
            "level":         sec["level"],
            "page_start":    sec["page_start"],
            "page_end":      sec["page_end"],
            "semantic_type": section_analysis.get("semantic_type", "other"),
            "elements":      elements,  # entities/signals 在 element 內
        })

    # 4. page count
    doc = _fitz.open(str(pdf_path))
    page_count = doc.page_count
    doc.close()

    # 5. metadata
    metadata = build_metadata(
        pdf_path=pdf_path, category=category,
        extractor="llm", page_count=page_count,
        keywords=keywords,
    )

    # 6. QC
    llm_qc = _compute_llm_qc(sections_output)

    # QC 讀取完畢，移除內部 _error 欄位（不應出現在輸出中）
    for sec in sections_output:
        for elem in sec.get("elements", []):
            elem.pop("_error", None)

    metadata["confidence"] = {
        "source":    "vision_llm",
        "available": False,
        "note":      "LLM 生成內容，無 OCR 信心分數，有幻覺風險",
    }
    metadata["qc"] = {
        "estimated_info_loss_rate": None,   # LLM extractor 不計算此值
        "qc_level":                 llm_qc["qc_level"],
        "warnings":                 llm_qc["warnings"],
        "errors":                   llm_qc["errors"],
        "llm_qc":                   llm_qc,
    }

    # 從 sections 合成 markdown，供 server 端 keyword 萃取使用
    md_parts = []
    for sec in sections_output:
        if sec.get("title"):
            md_parts.append(f"## {sec['title']}")
        for elem in sec.get("elements", []):
            if elem.get("content"):
                md_parts.append(elem["content"])
    markdown = "\n\n".join(md_parts)

    # 從 figure elements 收集 page_images（每頁已去重）
    page_images: dict[int, dict] = {}
    for sec in sections_output:
        for elem in sec.get("elements", []):
            if elem.get("type") == "figure" and elem.get("has_image"):
                pn = elem.get("page_no")
                if pn and pn not in page_images:
                    page_images[pn] = {
                        "path":      elem.get("image_path"),
                        "sha256":    elem.get("sha256"),
                        "has_image": True,
                    }

    return {
        "schema_version": "v3.0",
        "metadata":       metadata,
        "data": {
            "sections":    sections_output,
            "markdown":    markdown,
            "page_images": page_images,
        },
        "page_count": page_count,
        "extractor_metadata": {
            "tool":                "vision_llm",
            "model":               getattr(llm, "model_name", None) or getattr(llm, "model", None),
            "playbook":            effective_category or "_default",
            "version":             "1.0.0",
            "per_cell_confidence": False,
            "confidence_source":   None,
            "known_limitation": (
                "LLM 生成內容，無 OCR 信心分數，有幻覺風險；"
                "entity/signal evidence 為 internal reference（element.content），非 PDF OCR ground truth"
            ),
            "fallback_reason":     None,
        },
    }


# ── DOCX LLM Extractor ───────────────────────────────────────────────────────

def _detect_sections_from_markdown(markdown: str) -> list[dict]:
    """從 Docling markdown 解析 ## / ### heading 作為 section 邊界。

    Docling export_to_markdown() 輸出的 heading 格式為 ## 或 ###。
    無 heading 時整份文件視為 1 section。
    """
    sections: list[dict] = []
    lines = markdown.split("\n")
    heading_lines: list[tuple[int, str]] = []  # (line_idx, title)

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("## ") or stripped.startswith("### "):
            title = re.sub(r'^#{2,3}\s+', '', stripped).strip()
            if title:
                heading_lines.append((i, title))

    if not heading_lines:
        # 嘗試從第一個非空行取文件標題
        first_line = next((l.strip() for l in lines if l.strip()), "文件內容")
        return [{"title": first_line[:60], "line_start": 0, "line_end": len(lines) - 1, "level": 1}]

    result = []
    for i, (line_idx, title) in enumerate(heading_lines):
        line_end = (heading_lines[i + 1][0] - 1) if i + 1 < len(heading_lines) else len(lines) - 1
        result.append({"title": title, "line_start": line_idx, "line_end": line_end, "level": 1})

    if heading_lines[0][0] > 0:
        result.insert(0, {
            "title": "[前置內容]", "line_start": 0,
            "line_end": heading_lines[0][0] - 1, "level": 0,
        })

    return result


def _extract_section_text(
    section_text: str,
    section: dict,
    section_index: int,
    playbook: dict,
    llm,
) -> tuple[dict, list[dict]]:
    """文字 prompt 萃取一個 section（DOCX path 用）。"""
    from langchain_core.messages import HumanMessage

    if not section_text.strip():
        return {"semantic_type": "other"}, []

    schema_str = _build_prompt_schema(playbook)
    se = playbook.get("semantic_extraction", {})
    section_types_str = " | ".join(se.get("section_types", ["other"]))
    signal_rules = se.get("document_signal_rules") or {}
    signal_rules_str = ""
    for stype, rules in signal_rules.items():
        markers = rules.get("explicit_markers", []) + rules.get("explicit_patterns", [])
        if markers:
            signal_rules_str += f"\n  {stype}：{', '.join(markers[:4])}"

    hints = playbook.get("prompt_hints", {})
    domain_note = hints.get("note", "")

    system_prompt = f"""你是醫療文件語意萃取器。以下是文件的 Markdown 文字內容（已保留表格結構）。

{domain_note}

請分析文字，輸出結構化 JSON。

規則：
1. section_analysis.semantic_type 只能是：{section_types_str}
2. 每個 element 的 entities 只記錄此段落中明確出現的實體，不確定就留空陣列
3. 每個 element 的 document_signals 只記錄此段落中明確出現的訊號（如：{signal_rules_str}）
4. certainty 只能是 high（清楚讀到）或 medium（有輕微疑慮）
5. 不要產生：摘要、問題、搜尋文字、答案
6. page_no 設為 1（文字輸入無頁碼資訊）

嚴格禁止：
- 新增文件沒有的醫療建議或數值
- 猜測模糊文字

格式：
{schema_str}

只輸出 JSON，不要任何說明文字。

文件內容：
{section_text[:6000]}"""

    default_analysis = {"semantic_type": "other"}
    try:
        response = llm.invoke([HumanMessage(content=system_prompt)])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r'^```(?:json)?\n?', '', raw)
            raw = re.sub(r'\n?```$', '', raw)
        parsed = json.loads(raw)
    except Exception as e:
        error_elem = [{
            "type": "text", "page_no": 1,
            "reading_order": 1, "content": f"[萃取失敗: {e}]", "_error": str(e),
        }]
        return default_analysis, error_elem

    section_analysis = parsed.get("section_analysis", default_analysis)
    elements_raw = parsed.get("elements", [])
    elements = []
    for elem in elements_raw:
        if elem.get("type", "text") not in VALID_TYPES:
            elem["type"] = "text"
        elem.pop("has_image", None)
        elements.append(elem)

    elements = _assign_element_ids(section_index, elements)
    return section_analysis, elements


def convert_docx_llm(
    docx_path: Path,
    llm,
    category: str = "",
    output_dir: Path | None = None,
    keywords: list[str] | None = None,
) -> dict:
    """LLM Extractor（Beta）：DOCX → Docling markdown → 文字 prompt → schema-v3.0。

    Docling XML-native parsing 已保留表格/標題結構於 markdown，
    直接餵文字給 LLM 比圖片 token 成本低，且結構資訊不損失。
    """
    from metadata_builder import build_metadata, _infer_document_type
    from docling.document_converter import DocumentConverter

    # 1. Docling 轉換
    try:
        conv = DocumentConverter()
        doc = conv.convert(docx_path).document
        markdown = doc.export_to_markdown()
        page_count = len(doc.pages) if hasattr(doc, "pages") and doc.pages else 1
    except Exception as exc:
        markdown = ""
        page_count = 1

    lines = markdown.split("\n")

    # 2. playbook
    effective_category = category or _infer_document_type(docx_path.stem) or ""
    playbook = _load_playbook(effective_category)

    # 3. section 邊界（從 markdown heading 解析）
    sections_meta = _detect_sections_from_markdown(markdown)

    # 4. 逐 section 萃取（文字 prompt）
    sections_output = []
    for idx, sec in enumerate(sections_meta, 1):
        sec_lines = lines[sec["line_start"]: sec["line_end"] + 1]
        sec_text = "\n".join(sec_lines)
        section_analysis, elements = _extract_section_text(
            sec_text, sec, idx, playbook, llm
        )
        sections_output.append({
            "section_id":    f"s{idx:03d}",
            "title":         sec["title"],
            "level":         sec["level"],
            "page_start":    1,
            "page_end":      page_count,
            "semantic_type": section_analysis.get("semantic_type", "other"),
            "elements":      elements,
        })

    # 5. metadata / QC
    effective_category = category or _infer_document_type(docx_path.stem) or ""
    metadata = build_metadata(
        pdf_path=docx_path, category=category,
        extractor="llm", page_count=page_count,
        keywords=keywords,
    )
    llm_qc = _compute_llm_qc(sections_output)
    for sec in sections_output:
        for elem in sec.get("elements", []):
            elem.pop("_error", None)

    metadata["confidence"] = {
        "source":    "text_llm",
        "available": False,
        "note":      "Docling markdown 轉換後由 LLM 語意萃取，無 OCR 信心分數",
    }
    metadata["qc"] = {
        "estimated_info_loss_rate": None,
        "qc_level":  llm_qc["qc_level"],
        "warnings":  llm_qc["warnings"],
        "errors":    llm_qc["errors"],
        "llm_qc":    llm_qc,
    }

    md_parts = []
    for sec in sections_output:
        if sec.get("title"):
            md_parts.append(f"## {sec['title']}")
        for elem in sec.get("elements", []):
            if elem.get("content"):
                md_parts.append(elem["content"])

    return {
        "schema_version": "v3.0",
        "metadata":       metadata,
        "data": {
            "sections":    sections_output,
            "markdown":    "\n\n".join(md_parts),
            "page_images": {},
        },
        "page_count": page_count,
        "extractor_metadata": {
            "tool":                "vision_llm",
            "model":               getattr(llm, "model_name", None) or getattr(llm, "model", None),
            "playbook":            effective_category or "_default",
            "version":             "1.0.0",
            "per_cell_confidence": False,
            "confidence_source":   None,
            "known_limitation": (
                "LLM 生成內容，無 OCR 信心分數，有幻覺風險；"
                "輸入為 Docling markdown，非原始視覺版面"
            ),
            "fallback_reason": None,
        },
    }


# ── Image LLM Extractor ──────────────────────────────────────────────────────

def convert_image_llm(
    img_path: Path,
    llm,
    category: str = "",
    output_dir: Path | None = None,
    keywords: list[str] | None = None,
) -> dict:
    """LLM Extractor（Beta）：JPG/PNG → 圖片 base64 → vision LLM → schema-v3.0。

    整張圖片視為 1 section，直接送 vision LLM。
    重用 _extract_section() 的圖片傳送邏輯，不需要 fitz 開 PDF。
    """
    from metadata_builder import build_metadata, _infer_document_type
    from langchain_core.messages import HumanMessage

    # 1. 讀圖片
    img_bytes = img_path.read_bytes()
    suffix = img_path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    b64 = base64.b64encode(img_bytes).decode()

    # 2. playbook
    effective_category = category or _infer_document_type(img_path.stem) or ""
    playbook = _load_playbook(effective_category)

    # 3. 整張圖 = 1 section，直接萃取
    schema_str = _build_prompt_schema(playbook)
    se = playbook.get("semantic_extraction", {})
    section_types_str = " | ".join(se.get("section_types", ["other"]))
    signal_rules = playbook.get("semantic_extraction", {}).get("document_signal_rules") or {}
    signal_rules_str = ""
    for stype, rules in signal_rules.items():
        markers = rules.get("explicit_markers", []) + rules.get("explicit_patterns", [])
        if markers:
            signal_rules_str += f"\n  {stype}：{', '.join(markers[:4])}"

    hints = playbook.get("prompt_hints", {})
    institution = hints.get("institution_name", "")
    domain_note = hints.get("note", "")
    ignore_str = "、".join(playbook.get("ignore_regions", [])) or "無"

    system_prompt = f"""你是醫療文件語意萃取器（非一般 OCR）。
文件類型：{playbook.get('document_type', '未知')}
機構名稱：{institution}
忽略區域：{ignore_str}

{domain_note}

請分析以下圖片，輸出結構化 JSON。

規則：
1. section_analysis.semantic_type 只能是：{section_types_str}
2. 每個 element 的 entities 只記錄此圖中明確出現的實體，不確定就留空陣列
3. 每個 element 的 document_signals 只記錄此圖中明確出現的訊號（如：{signal_rules_str}）
4. certainty 只能是 high 或 medium
5. type=figure 且無可讀文字時，content 設為 null
6. page_no 設為 1

嚴格禁止：新增圖片沒有的醫療建議或數值、猜測模糊文字。

格式：
{schema_str}

只輸出 JSON，不要任何說明文字。"""

    section_analysis = {"semantic_type": "other"}
    elements: list[dict] = []

    try:
        content_parts = [
            {"type": "text", "text": system_prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]
        response = llm.invoke([HumanMessage(content=content_parts)])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r'^```(?:json)?\n?', '', raw)
            raw = re.sub(r'\n?```$', '', raw)
        parsed = json.loads(raw)
        section_analysis = parsed.get("section_analysis", section_analysis)
        elements_raw = parsed.get("elements", [])
        for elem in elements_raw:
            if elem.get("type", "text") not in VALID_TYPES:
                elem["type"] = "text"
            elem.pop("has_image", None)
            elements.append(elem)
    except Exception as e:
        elements = [{
            "type": "text", "page_no": 1,
            "reading_order": 1, "content": f"[萃取失敗: {e}]",
        }]

    elements = _assign_element_ids(1, elements)

    sections_output = [{
        "section_id":    "s001",
        "title":         img_path.stem,
        "level":         1,
        "page_start":    1,
        "page_end":      1,
        "semantic_type": section_analysis.get("semantic_type", "other"),
        "elements":      elements,
    }]

    # 4. 儲存原始圖片到 page_images
    page_images: dict[int, dict] = {}
    if output_dir:
        try:
            import hashlib, shutil
            img_dir = Path(output_dir) / "figures"
            img_dir.mkdir(parents=True, exist_ok=True)
            dest = img_dir / f"{img_path.stem}_p1_full{suffix}"
            shutil.copy2(str(img_path), str(dest))
            sha256 = hashlib.sha256(img_bytes).hexdigest()
            page_images[1] = {
                "path":      str(dest.relative_to(output_dir)),
                "sha256":    sha256,
                "has_image": True,
            }
        except Exception:
            pass

    # 5. metadata / QC
    metadata = build_metadata(
        pdf_path=img_path, category=category,
        extractor="llm", page_count=1,
        keywords=keywords,
    )
    llm_qc = _compute_llm_qc(sections_output)

    metadata["confidence"] = {
        "source":    "vision_llm",
        "available": False,
        "note":      "LLM 生成內容，無 OCR 信心分數，有幻覺風險",
    }
    metadata["qc"] = {
        "estimated_info_loss_rate": None,
        "qc_level":  llm_qc["qc_level"],
        "warnings":  llm_qc["warnings"],
        "errors":    llm_qc["errors"],
        "llm_qc":    llm_qc,
    }

    md_parts = []
    for elem in elements:
        if elem.get("content"):
            md_parts.append(elem["content"])

    return {
        "schema_version": "v3.0",
        "metadata":       metadata,
        "data": {
            "sections":    sections_output,
            "markdown":    "\n\n".join(md_parts),
            "page_images": page_images,
        },
        "page_count": 1,
        "extractor_metadata": {
            "tool":                "vision_llm",
            "model":               getattr(llm, "model_name", None) or getattr(llm, "model", None),
            "playbook":            effective_category or "_default",
            "version":             "1.0.0",
            "per_cell_confidence": False,
            "confidence_source":   None,
            "known_limitation": (
                "LLM 生成內容，無 OCR 信心分數，有幻覺風險；"
                "整張圖片視為單一 section，無 section 邊界偵測"
            ),
            "fallback_reason": None,
        },
    }
