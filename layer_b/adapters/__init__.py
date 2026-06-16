from layer_b.adapters import azure_cu_adapter, docling_adapter, azure_di_adapter, llm_adapter

_FALLBACK_WARNING = "AZURE_DI_FALLBACK_TO_DOCLING"

_ADAPTERS = {
    "azure_content_understanding": azure_cu_adapter.adapt,
    "docling": docling_adapter.adapt,
    "azure_document_intelligence": azure_di_adapter.adapt,
    "vision_llm": llm_adapter.adapt,
}


def get_source_tool(raw: dict) -> str:
    """Extract source_tool from extractor_metadata.tool (Conversion Layer v3.0).

    Falls back to 'azure_content_understanding' if field is absent.
    """
    return raw.get("extractor_metadata", {}).get("tool", "azure_content_understanding")


def adapt(raw: dict, source_tool: str) -> list:
    """Route to the correct adapter, handling Azure DI → Docling fallback."""
    if source_tool == "azure_document_intelligence":
        warnings = raw.get("metadata", {}).get("qc", {}).get("warnings", [])
        if _FALLBACK_WARNING in warnings:
            return docling_adapter.adapt(raw)

    fn = _ADAPTERS.get(source_tool)
    if fn is None:
        raise ValueError(f"Unknown source_tool: {source_tool!r}")
    return fn(raw)
