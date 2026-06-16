from layer_b.models import IRDocument, IRSection, QC


def adapt(raw: dict) -> list[IRDocument]:
    """將 vision_llm schema-v3.0 輸出轉為 list[IRDocument]。"""
    # 從 raw 取 sections
    sections_raw = raw.get("data", {}).get("sections", [])

    # 從 metadata 或 extractor_metadata 取 doc_id
    doc_id = raw.get("metadata", {}).get("doc_id", "doc_001")

    # 建立 IRSection list
    sections = []
    for s in sections_raw:
        sections.append(IRSection(
            section_id=s.get("section_id", ""),
            title=s.get("title", ""),
            level=s.get("level", 1),
            page_start=s.get("page_start", 1),
            page_end=s.get("page_end", 1),
            semantic_type=s.get("semantic_type", "other"),
            elements=s.get("elements", []),
        ))

    # QC：從 metadata.qc 取，若無則 default
    qc_raw = raw.get("metadata", {}).get("qc", {})
    qc = QC(
        qc_level=qc_raw.get("qc_level", "ok"),
        warnings=qc_raw.get("warnings", []),
    )

    doc = IRDocument(
        doc_id=doc_id,
        source_tool="vision_llm",
        sections=sections,
        qc=qc,
    )
    return [doc]
