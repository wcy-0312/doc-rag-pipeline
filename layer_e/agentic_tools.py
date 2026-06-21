from layer_e.pdf_tools import get_full_page_image

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_page_image",
            "description": "取得 PDF 指定頁碼的整頁截圖，適合閱讀流程圖、算法圖表、治療路徑圖等視覺內容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "page_no": {
                        "type": "integer",
                        "description": "頁碼（從 1 開始）",
                    },
                    "reason": {
                        "type": "string",
                        "description": "為何需要查看此頁（audit log 用）",
                    },
                },
                "required": ["page_no", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_more",
            "description": "在文件中搜尋與問題相關的更多段落或表格。當初始 evidence 缺少某個關鍵資訊時使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜尋關鍵字或語句",
                    },
                    "reason": {
                        "type": "string",
                        "description": "為何需要搜尋此內容（audit log 用）",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "傳回結果數量（預設 3）",
                        "default": 3,
                    },
                },
                "required": ["query", "reason"],
            },
        },
    },
]


def execute_tool(
    tool_call: dict,
    pdf_path: str,
    retriever,
    doc_stem: str,
) -> tuple:
    """Execute a single tool call.

    Returns:
        (text_result, None)         for retrieve_more
        (text_result, base64_png)   for get_page_image
    """
    name = tool_call["name"]
    args = tool_call["arguments"]

    if name == "get_page_image":
        page_no = int(args["page_no"])
        b64 = get_full_page_image(pdf_path, page_no)
        return f"已截取第 {page_no} 頁截圖。", b64

    if name == "retrieve_more":
        top_k = int(args.get("top_k", 3))
        results = retriever.search_text(
            args["query"],
            top_k=top_k,
            doc_ids=[doc_stem],
            rerank=False,
        )
        if not results:
            return "未找到相關段落。", None
        lines = []
        for i, r in enumerate(results, start=1):
            pages = "、".join(f"第{p}頁" for p in r.source_pages)
            lines.append(f"[新增 {i}] {pages}\n{r.display_markdown}")
        return "\n\n".join(lines), None

    return f"未知工具：{name}", None
