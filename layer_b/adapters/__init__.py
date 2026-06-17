from layer_b.adapters import azure_cu_adapter, docling_adapter, azure_di_adapter, llm_adapter

_FALLBACK_WARNING = "AZURE_DI_FALLBACK_TO_DOCLING"

_ADAPTERS = {
    "azure_content_understanding": azure_cu_adapter.adapt,
    "docling": docling_adapter.adapt,
    "azure_document_intelligence": azure_di_adapter.adapt,
    "vision_llm": llm_adapter.adapt,
}


def get_source_tool(raw: dict) -> str:
    """Extract source_tool from metadata.extractor_metadata.tool (flat schema).

    Falls back to top-level extractor_metadata for test payloads,
    then to 'azure_content_understanding' if field is absent.
    """
    tool = raw.get("metadata", {}).get("extractor_metadata", {}).get("tool")
    if tool:
        return tool
    # fallback for test payloads that use top-level extractor_metadata
    return raw.get("extractor_metadata", {}).get("tool", "azure_content_understanding")


def adapt(raw: dict, source_tool: str) -> list:
    """Route to the correct adapter, handling Azure DI → Docling fallback."""
    if source_tool == "azure_document_intelligence":
        meta = raw.get("metadata", {})
        # flat schema: warnings in extractor_metadata; legacy: in qc.warnings
        warnings = (
            meta.get("extractor_metadata", {}).get("warnings")
            or meta.get("qc", {}).get("warnings", [])
        )
        if _FALLBACK_WARNING in warnings:
            return docling_adapter.adapt(raw)

    fn = _ADAPTERS.get(source_tool)
    if fn is None:
        raise ValueError(f"Unknown source_tool: {source_tool!r}")
    return fn(raw)
