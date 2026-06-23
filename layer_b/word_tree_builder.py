from __future__ import annotations

from typing import TYPE_CHECKING

from layer_f.tree_models import TreeNode

if TYPE_CHECKING:
    from layer_e.llm_client import LLMClient

_SUMMARY_SYSTEM = "你是醫療文件助理。"
_SUMMARY_USER_TEMPLATE = (
    "用一句繁體中文（50字以內）摘要以下章節群的主要內容：\n\n{content}"
)

_BODY_LABELS = {"text", "list_item"}


def _get_page(item: dict) -> int | None:
    prov = item.get("prov", [])
    return prov[0].get("page_no") if prov else None


def _make_summary(node: TreeNode, llm_client) -> str:
    context_parts: list[str] = []
    for child in node.children:
        line = f"【{child.title}】" if child.title else ""
        detail = child.summary or child.content[:200]
        if line or detail:
            context_parts.append(f"{line} {detail}".strip())
    if not context_parts:
        return ""
    prompt = _SUMMARY_USER_TEMPLATE.format(content="\n".join(context_parts))
    return llm_client.generate_text(prompt, system=_SUMMARY_SYSTEM)


def build_word_tree(raw: dict, llm_client: "LLMClient | None" = None) -> TreeNode | None:
    """Build a TreeNode hierarchy from a Docling Word raw document.

    Parses data.texts[] produced by docling's export_to_dict().
    section_header items define the tree structure via their `level` field (int 1-6).
    text / list_item items are body content for the current heading node.
    Returns None if texts[] is absent or produces no usable nodes.
    """
    texts = raw.get("data", {}).get("texts", [])
    if not texts:
        return None

    node_counter = [0]

    def _new_id() -> str:
        node_counter[0] += 1
        return f"word_{node_counter[0]}"

    # Stack entries: {"level": int, "node": TreeNode | None, "body": list[str], "pages": list[int]}
    # Index 0 is virtual root sentinel (level=0, node=None)
    stack: list[dict] = [{"level": 0, "node": None, "body": [], "pages": []}]
    roots: list[TreeNode] = []

    def _finalise(entry: dict) -> None:
        node = entry["node"]
        if node is None:
            return
        all_p = [p for p in [node.start_page] + entry["pages"] if p is not None]
        if node.is_leaf:
            node.content = "\n".join(entry["body"])
        node.start_page = min(all_p) if all_p else None
        node.end_page = max(all_p) if all_p else None

    for item in texts:
        label = item.get("label", "")
        text = (item.get("text") or "").strip()
        if not text:
            continue

        if label == "section_header":
            level = int(item.get("level") or 1)
            page = _get_page(item)

            while len(stack) > 1 and stack[-1]["level"] >= level:
                _finalise(stack[-1])
                stack.pop()

            new_node = TreeNode(
                node_id=_new_id(),
                title=text,
                start_page=page,
                end_page=page,
                summary="",
                content="",
                children=[],
            )

            parent = stack[-1]["node"]
            if parent is not None:
                parent.children.append(new_node)
            else:
                roots.append(new_node)

            pages = [page] if page is not None else []
            stack.append({"level": level, "node": new_node, "body": [], "pages": pages})

        elif label in _BODY_LABELS:
            if len(stack) > 1:
                entry = stack[-1]
                entry["body"].append(text)
                page = _get_page(item)
                if page is not None:
                    entry["pages"].append(page)

    while len(stack) > 1:
        _finalise(stack[-1])
        stack.pop()

    if not roots:
        return None

    def _propagate(node: TreeNode) -> None:
        for child in node.children:
            _propagate(child)

        if node.children:
            node.content = ""
            child_pages: list[int] = []
            for child in node.children:
                if child.start_page is not None:
                    child_pages.append(child.start_page)
                if child.end_page is not None:
                    child_pages.append(child.end_page)
            if child_pages:
                node.start_page = min(
                    [p for p in [node.start_page] + child_pages if p is not None],
                    default=None,
                )
                node.end_page = max(
                    [p for p in [node.end_page] + child_pages if p is not None],
                    default=None,
                )
            if llm_client is not None:
                node.summary = _make_summary(node, llm_client)

    for root in roots:
        _propagate(root)

    if len(roots) == 1:
        return roots[0]

    file_name = raw.get("metadata", {}).get("file_name", "文件")
    all_pages: list[int] = []
    for r in roots:
        if r.start_page is not None:
            all_pages.append(r.start_page)
        if r.end_page is not None:
            all_pages.append(r.end_page)
    return TreeNode(
        node_id="root",
        title=file_name,
        start_page=min(all_pages) if all_pages else None,
        end_page=max(all_pages) if all_pages else None,
        summary="",
        content="",
        children=roots,
    )
