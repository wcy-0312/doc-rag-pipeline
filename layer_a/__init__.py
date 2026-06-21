def get_extractor(tool: str):
    """Return the extractor callable for the given tool name.

    tool: "azure_cu" | "azure_di" | "docling" | "llm"
    """
    if tool == "azure_cu":
        from layer_a.azure_cu_extractor import convert_pdf_azure_cu
        return convert_pdf_azure_cu
    elif tool == "azure_di":
        from layer_a.azure_di_extractor import convert_image_azure_di
        return convert_image_azure_di
    elif tool == "docling":
        from layer_a.docling_extractor import convert_word_docling
        return convert_word_docling
    elif tool == "llm":
        from layer_a.llm_extractor import convert_pdf_llm
        return convert_pdf_llm
    else:
        raise ValueError(f"Unknown extractor tool: {tool!r}. Choose: azure_cu, azure_di, docling, llm")


def get_extractor_for_file(path) -> str:
    """根據副檔名自動回傳應使用的 extractor 名稱。

    Returns a tool name string suitable for get_extractor().
    """
    from pathlib import Path
    suffix = Path(path).suffix.lower()
    _ROUTING = {
        ".pdf":  "azure_cu",
        ".docx": "docling",
        ".doc":  "docling",
        ".jpg":  "azure_di",
        ".jpeg": "azure_di",
        ".png":  "azure_di",
        ".tiff": "azure_di",
        ".tif":  "azure_di",
    }
    tool = _ROUTING.get(suffix)
    if tool is None:
        raise ValueError(
            f"Unsupported file extension {suffix!r}. "
            f"Supported: {sorted(_ROUTING.keys())}"
        )
    return tool
