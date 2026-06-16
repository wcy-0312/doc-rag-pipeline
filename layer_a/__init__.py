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
        from layer_a.docling_extractor import convert_pdf_docling
        return convert_pdf_docling
    elif tool == "llm":
        from layer_a.llm_extractor import convert_pdf_llm
        return convert_pdf_llm
    else:
        raise ValueError(f"Unknown extractor tool: {tool!r}. Choose: azure_cu, azure_di, docling, llm")
