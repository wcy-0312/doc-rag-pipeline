from __future__ import annotations

import re
from typing import TYPE_CHECKING

from layer_f.tree_models import TreeNode

if TYPE_CHECKING:
    from layer_e.llm_client import LLMClient

_SOURCE_PAGE_RE = re.compile(r'^D\((\d+),')

_SUMMARY_SYSTEM = "你是醫療文件助理。"
_SUMMARY_USER_TEMPLATE = (
    "用一句繁體中文（50字以內）摘要以下章節群的主要內容：\n\n{content}"
)


def _parse_page(source_str: str) -> int | None:
    m = _SOURCE_PAGE_RE.match(str(source_str or ""))
    return int(m.group(1)) if m else None


def build_tree(raw: dict, llm_client: LLMClient | None = None) -> TreeNode | None:
    """Build a PageIndexTree from an Azure CU raw document dict.

    Returns None if sections[] is absent or produces no usable nodes.
    llm_client: if provided, generates a one-sentence summary for each non-leaf node.
    """
    data = raw.get("data", {})
    sections = data.get("sections", [])
    paragraphs = data.get("paragraphs", [])

    if not sections:
        return None

    section_by_idx = {i: s for i, s in enumerate(sections)}
    visited: set[int] = set()

    # Find sections that are NOT referenced as children by any other section
    child_indices: set[int] = set()
    for sec in sections:
        for elem_ref in sec.get("elements", []):
            parts = str(elem_ref).strip("/").split("/")
            try:
                if parts[-2] == "sections":
                    child_indices.add(int(parts[-1]))
            except (IndexError, ValueError):
                pass

    root_indices = [i for i in range(len(sections)) if i not in child_indices]

    def _build_node(sec_idx: int) -> TreeNode | None:
        if sec_idx in visited:
            return None
        visited.add(sec_idx)
        sec = section_by_idx.get(sec_idx)
        if sec is None:
            return None

        title = (sec.get("title") or "").strip()
        heading_page: int | None = None          # page of the sectionHeading paragraph
        body_paras: list[tuple[str, int | None]] = []   # (content, page)
        child_sec_indices: list[int] = []

        for elem_ref in sec.get("elements", []):
            parts = str(elem_ref).strip("/").split("/")
            try:
                kind, idx = parts[-2], int(parts[-1])
            except (IndexError, ValueError):
                continue

            if kind == "paragraphs" and idx < len(paragraphs):
                para = paragraphs[idx]
                content = (para.get("content") or "").strip()
                page = _parse_page(para.get("source", ""))
                role = para.get("role")
                if role == "sectionHeading":
                    heading_page = page   # always capture for page range
                    if not title:
                        title = content   # only set title if not already from sec["title"]
                elif content:
                    body_paras.append((content, page))
            elif kind == "sections":
                child_sec_indices.append(idx)

        children = [
            n for idx in child_sec_indices
            if (n := _build_node(idx)) is not None
        ]

        # Skip structurally empty nodes
        if not title and not body_paras and not children:
            return None

        # Page range: collect all pages in this subtree (heading + body + children)
        all_pages: list[int] = []
        if heading_page is not None:
            all_pages.append(heading_page)
        all_pages.extend(p for _, p in body_paras if p is not None)
        for child in children:
            if child.start_page is not None:
                all_pages.append(child.start_page)
            if child.end_page is not None:
                all_pages.append(child.end_page)
        start_page = min(all_pages) if all_pages else None
        end_page = max(all_pages) if all_pages else None

        # Leaf: aggregate body text; non-leaf: body text is folded into children
        content = "\n".join(text for text, _ in body_paras) if not children else ""

        # Summary for non-leaf nodes
        summary = ""
        if children and llm_client is not None:
            context_parts: list[str] = []
            for child in children:
                line = f"【{child.title}】" if child.title else ""
                detail = child.summary or child.content[:200]
                if line or detail:
                    context_parts.append(f"{line} {detail}".strip())
            if context_parts:
                prompt = _SUMMARY_USER_TEMPLATE.format(content="\n".join(context_parts))
                summary = llm_client.generate_text(prompt, system=_SUMMARY_SYSTEM)

        return TreeNode(
            node_id=f"sec_{sec_idx}",
            title=title or f"Section {sec_idx}",
            start_page=start_page,
            end_page=end_page,
            summary=summary,
            content=content,
            children=children,
        )

    root_nodes = [n for i in root_indices if (n := _build_node(i)) is not None]

    if not root_nodes:
        return None

    if len(root_nodes) == 1:
        return root_nodes[0]

    # Multiple roots: wrap in a virtual root named after the file
    file_name = raw.get("metadata", {}).get("file_name", "文件")
    all_pages: list[int] = []
    for n in root_nodes:
        if n.start_page is not None:
            all_pages.append(n.start_page)
        if n.end_page is not None:
            all_pages.append(n.end_page)
    return TreeNode(
        node_id="root",
        title=file_name,
        start_page=min(all_pages) if all_pages else None,
        end_page=max(all_pages) if all_pages else None,
        summary="",
        content="",
        children=root_nodes,
    )
