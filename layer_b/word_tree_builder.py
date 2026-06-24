from __future__ import annotations

import re as _re
from enum import Enum
from typing import TYPE_CHECKING

from layer_f.tree_models import TreeNode

if TYPE_CHECKING:
    from layer_e.llm_client import LLMClient

# ── Table type classifier ─────────────────────────────────────────────────────

class _TableType(Enum):
    MATRIX       = "matrix"       # col=0: unique meaningful labels → one leaf per row
    LONGITUDINAL = "longitudinal" # col=0: repeated categories → one leaf per category
    RECORD       = "record"       # col=0: placeholders or field-labels → single leaf
    CHART        = "chart"        # col=0: numeric scale → single leaf from body texts
    INDEX        = "index"        # col=0: sequential integers → chunked leaves


# Matches field-descriptor labels common in transposed-record tables:
#   "日期/班別/", "不符合項目", "處理對策", "再確認", "稽核人員\n簽名"
_FIELD_LABEL_RE = _re.compile(
    r'[/／]|[：:]$|項目$|對策$|確認$|簽名$|編號$|記錄$|說明$', _re.MULTILINE
)


def _data_rows(grid: list[list[dict]]) -> list[list[dict]]:
    """Non-header rows from table grid."""
    return [row for row in grid if row and not row[0].get("column_header", False)]


def _col0_texts(grid: list[list[dict]]) -> list[str]:
    """col=0 text values from non-header rows, stripped."""
    return [row[0].get("text", "").strip() for row in _data_rows(grid) if row]


def _is_numeric(s: str) -> bool:
    return bool(_re.match(r'^-?\d+\.?\d*$', s.strip()))


def _is_sequential_int(values: list[str]) -> bool:
    """True if values are sequential integers starting near 1 (e.g. 1,2,3,4,...)."""
    nums = []
    for v in values:
        v = v.strip()
        if not v:
            continue
        if not _re.match(r'^\d+$', v):
            return False
        nums.append(int(v))
    if len(nums) < 3:
        return False
    return nums == list(range(nums[0], nums[0] + len(nums)))


def _is_nav_label(text: str) -> bool:
    """True if text could serve as a navigation label (meaningful noun, not a placeholder)."""
    s = text.strip()
    if not s or len(s) < 2:
        return False
    # Pure symbol/whitespace
    if _re.match(r'^[/：:\s，、\-\n]+$', s):
        return False
    # Pure number
    if _is_numeric(s):
        return False
    return True


def _classify_table(grid: list[list[dict]]) -> _TableType:
    """Classify a docling table grid into one of five navigation strategies."""
    if not grid:
        return _TableType.RECORD

    col0 = _col0_texts(grid)
    non_empty = [v for v in col0 if v]

    if not non_empty:
        return _TableType.RECORD

    # INDEX: sequential integers (form directory, numbered lists) — check BEFORE CHART
    if _is_sequential_int(non_empty):
        return _TableType.INDEX

    # CHART: all col=0 are numeric (temperature scale, etc.)
    if all(_is_numeric(v) for v in non_empty):
        return _TableType.CHART

    nav = [v for v in non_empty if _is_nav_label(v)]
    nav_ratio = len(nav) / len(non_empty)

    if nav_ratio < 0.5:
        return _TableType.RECORD

    # RECORD: even if nav-looking, if majority are field descriptors
    # (transposed-record tables like 不符合處理單)
    field_count = sum(1 for v in nav if _FIELD_LABEL_RE.search(v))
    if field_count / max(len(nav), 1) >= 0.5:
        return _TableType.RECORD

    # LONGITUDINAL: duplicate col=0 values (categories with sub-rows)
    if len(set(nav)) < len(nav):
        return _TableType.LONGITUDINAL

    return _TableType.MATRIX


# ── Table → TreeNode conversion ──────────────────────────────────────────────

_INDEX_CHUNK = 30  # rows per INDEX leaf


def _table_to_markdown(grid: list[list[dict]]) -> str:
    """Convert docling table grid to GitHub-flavored markdown table string."""
    if not grid:
        return ""
    lines: list[str] = []
    for r, row in enumerate(grid):
        cells = [cell.get("text", "").replace("\n", " ").strip() for cell in row]
        lines.append("| " + " | ".join(cells) + " |")
        if r == 0:
            lines.append("|" + "|".join(["---"] * len(row)) + "|")
    return "\n".join(lines)


def _table_to_nodes(grid: list[list[dict]], doc_title: str) -> list[TreeNode]:
    """Convert a table grid to TreeNodes according to its classified type.

    CHART returns [] — its semantic content lives in surrounding body texts.
    All returned nodes have start_page=None and summary="" (Word has no page numbers).
    """
    ttype = _classify_table(grid)
    data = _data_rows(grid)
    _id = [0]

    def _nid() -> str:
        _id[0] += 1
        return f"tbl_{_id[0]}"

    if ttype == _TableType.CHART:
        return []

    if ttype == _TableType.RECORD:
        return [TreeNode(
            node_id=_nid(),
            title=doc_title,
            start_page=None, end_page=None,
            summary="",
            content=_table_to_markdown(grid),
            children=[],
        )]

    if ttype == _TableType.INDEX:
        nodes: list[TreeNode] = []
        for i in range(0, len(data), _INDEX_CHUNK):
            chunk = data[i:i + _INDEX_CHUNK]
            seq_start = chunk[0][0].get("text", "").strip()
            seq_end = chunk[-1][0].get("text", "").strip()
            nodes.append(TreeNode(
                node_id=_nid(),
                title=f"{doc_title} ({seq_start}–{seq_end})",
                start_page=None, end_page=None,
                summary="",
                content=_table_to_markdown([grid[0]] + chunk),
                children=[],
            ))
        return nodes

    if ttype == _TableType.MATRIX:
        nodes = []
        for row in data:
            c0 = row[0].get("text", "").strip()
            if not c0:
                continue
            # Skip placeholder-like symbols in MATRIX (e.g., "/" or "：")
            if _re.match(r'^[/：:\s，、\-\n]+$', c0):
                continue
            content_parts = [
                cell.get("text", "").strip()
                for cell in row[1:]
                if cell.get("text", "").strip()
            ]
            nodes.append(TreeNode(
                node_id=_nid(),
                title=c0,
                start_page=None, end_page=None,
                summary="",
                content="\n".join(content_parts),
                children=[],
            ))
        return nodes

    if ttype == _TableType.LONGITUDINAL:
        categories: dict[str, list[str]] = {}
        for row in data:
            c0 = row[0].get("text", "").strip()
            c1 = row[1].get("text", "").strip() if len(row) > 1 else ""
            if not c0:
                continue
            if c0 not in categories:
                categories[c0] = []
            if c1:
                categories[c0].append(c1)
        nodes = []
        for category, sub_items in categories.items():
            nodes.append(TreeNode(
                node_id=_nid(),
                title=category,
                start_page=None, end_page=None,
                summary="",
                content="\n".join(sub_items),
                children=[],
            ))
        return nodes

    return []  # fallback (should not reach here)

# ─────────────────────────────────────────────────────────────────────────────

_CN_HEADING_RE = _re.compile(r'^[一二三四五六七八九十百千]+[、.）)：]\s*\S')


def _is_cn_heading(text: str) -> bool:
    """True if text looks like a Chinese-numbered section heading (一、二、三、)."""
    return bool(_CN_HEADING_RE.match(text.strip()))


def _flatten_body_refs(refs: list[str], ref_map: dict) -> list[tuple[str, dict]]:
    """Resolve body.children refs into an ordered list of (kind, item) tuples.

    Skips furniture items. Recursively expands groups.
    kind is one of: "text", "table", "picture".
    """
    result: list[tuple[str, dict]] = []
    for ref in refs:
        entry = ref_map.get(ref)
        if entry is None:
            continue
        kind, item = entry
        if item.get("content_layer") == "furniture":
            continue
        if kind == "group":
            child_refs = [c["$ref"] for c in item.get("children", [])]
            result.extend(_flatten_body_refs(child_refs, ref_map))
        else:
            result.append((kind, item))
    return result


# ── Summary constants and helpers ────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────

def _build_from_section_headers(
    items: list[tuple[str, dict]],
    doc_title: str,
    llm_client,
) -> list[TreeNode]:
    """Stack-based algorithm for documents with section_header labels (existing behavior)."""
    node_counter = [0]

    def _nid() -> str:
        node_counter[0] += 1
        return f"word_{node_counter[0]}"

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

    for kind, item in items:
        if kind != "text":
            continue
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
                node_id=_nid(), title=text,
                start_page=page, end_page=page,
                summary="", content="", children=[],
            )
            parent = stack[-1]["node"]
            if parent is not None:
                parent.children.append(new_node)
            else:
                roots.append(new_node)
            stack.append({"level": level, "node": new_node,
                          "body": [], "pages": [page] if page is not None else []})

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

    def _propagate(node: TreeNode) -> None:
        for child in node.children:
            _propagate(child)
        if node.children:
            node.content = ""
            child_pages = [p for c in node.children
                           for p in [c.start_page, c.end_page] if p is not None]
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

    return roots


def _build_from_cn_headings(
    items: list[tuple[str, dict]],
    doc_title: str,
    llm_client,
) -> list[TreeNode]:
    """Build leaf nodes from Chinese-numbered headings (一、二、三、).

    Text and inline tables between headings become each section's content.
    Content before the first heading is collected as a pre-section; if non-empty
    it becomes the first leaf titled with the document name.
    """
    sections: list[tuple[str | None, list[tuple[str, dict]]]] = []
    current_heading: str | None = None
    current_items: list[tuple[str, dict]] = []

    for kind, item in items:
        if kind == "text" and _is_cn_heading(item.get("text", "")):
            sections.append((current_heading, current_items))
            current_heading = item["text"].strip()
            current_items = []
        else:
            current_items.append((kind, item))
    sections.append((current_heading, current_items))

    nodes: list[TreeNode] = []
    _id = [0]

    def _nid() -> str:
        _id[0] += 1
        return f"cn_{_id[0]}"

    for heading, section_items in sections:
        parts: list[str] = []
        for kind, item in section_items:
            if kind == "text":
                t = (item.get("text") or "").strip()
                if t and not _is_cn_heading(t):
                    parts.append(t)
            elif kind == "table":
                grid = item.get("data", {}).get("grid", [])
                if grid:
                    parts.append(_table_to_markdown(grid))
        content = "\n".join(parts)
        title = heading if heading is not None else doc_title
        if not content and heading is None:
            continue  # skip empty pre-section
        nodes.append(TreeNode(
            node_id=_nid(), title=title,
            start_page=None, end_page=None,
            summary="", content=content, children=[],
        ))

    return nodes


def _build_from_tables(
    items: list[tuple[str, dict]],
    doc_title: str,
) -> list[TreeNode]:
    """Build tree from table structure when no heading signals exist.

    CHART tables are skipped; their semantic content comes from surrounding body texts.
    Multiple non-CHART tables each contribute their own nodes.
    Surrounding body texts are prepended to the content of a single-leaf result.
    """
    pre_texts: list[str] = []
    all_nodes: list[TreeNode] = []
    found_table = False
    post_texts: list[str] = []

    for kind, item in items:
        if kind == "text":
            t = (item.get("text") or "").strip()
            if not t:
                continue
            if not found_table:
                pre_texts.append(t)
            else:
                post_texts.append(t)
        elif kind == "table":
            grid = item.get("data", {}).get("grid", [])
            if not grid:
                continue
            found_table = True
            tbl_nodes = _table_to_nodes(grid, doc_title)
            all_nodes.extend(tbl_nodes)

    surrounding_text = "\n".join(pre_texts + post_texts)

    if not all_nodes:
        # Tables were CHART (returned []) or absent — use body texts only
        if not surrounding_text:
            return []
        return [TreeNode(
            node_id="leaf_0", title=doc_title,
            start_page=None, end_page=None,
            summary="", content=surrounding_text, children=[],
        )]

    if len(all_nodes) == 1 and surrounding_text:
        # Merge surrounding text into the single leaf
        leaf = all_nodes[0]
        merged = (surrounding_text + "\n" + leaf.content).strip()
        all_nodes[0] = TreeNode(
            node_id=leaf.node_id, title=leaf.title,
            start_page=None, end_page=None,
            summary="", content=merged, children=leaf.children,
        )

    return all_nodes


def build_word_tree(raw: dict, llm_client: "LLMClient | None" = None) -> TreeNode | None:
    """Build a TreeNode hierarchy from a Docling Word raw document.

    Walks body.children in document order (falls back to texts[] for legacy test data).
    Dispatches to one of three strategies:
      1. section_header stack  — when label="section_header" items exist
      2. Chinese-number headings — when 一、二、三、 patterns exist
      3. Table-type-based       — for pure form documents
    Returns None if no usable content is found.
    """
    data = raw.get("data", {})
    texts = data.get("texts", [])
    tables = data.get("tables", [])
    pictures = data.get("pictures", [])
    groups_list = data.get("groups", [])
    body = data.get("body", {})
    body_children_refs = [c["$ref"] for c in body.get("children", [])]

    file_name = raw.get("metadata", {}).get("file_name", "文件")
    doc_title = file_name.rsplit(".", 1)[0] if "." in file_name else file_name

    # Build ref_map for O(1) lookups
    ref_map: dict[str, tuple[str, dict]] = {}
    for i, t in enumerate(texts):
        ref_map[f"#/texts/{i}"] = ("text", t)
    for i, tbl in enumerate(tables):
        ref_map[f"#/tables/{i}"] = ("table", tbl)
    for i, pic in enumerate(pictures):
        ref_map[f"#/pictures/{i}"] = ("picture", pic)
    for i, grp in enumerate(groups_list):
        ref_map[f"#/groups/{i}"] = ("group", grp)

    if body_children_refs:
        ordered_items = _flatten_body_refs(body_children_refs, ref_map)
    else:
        # Legacy path: body.children absent (older test data without body structure)
        ordered_items = [
            ("text", t) for t in texts
            if t.get("content_layer") != "furniture"
        ]

    if not ordered_items:
        return None

    # Dispatch to strategy
    has_section_headers = any(
        kind == "text" and item.get("label") == "section_header"
        for kind, item in ordered_items
    )
    has_cn_headings = any(
        kind == "text" and _is_cn_heading(item.get("text", ""))
        for kind, item in ordered_items
    )

    if has_section_headers:
        nodes = _build_from_section_headers(ordered_items, doc_title, llm_client)
    elif has_cn_headings:
        nodes = _build_from_cn_headings(ordered_items, doc_title, llm_client)
    else:
        nodes = _build_from_tables(ordered_items, doc_title)

    if not nodes:
        return None

    if len(nodes) == 1:
        return nodes[0]

    # Multiple roots → wrap in virtual root
    all_pages = [p for n in nodes for p in [n.start_page, n.end_page] if p is not None]
    virtual_root = TreeNode(
        node_id="root",
        title=file_name,
        start_page=min(all_pages) if all_pages else None,
        end_page=max(all_pages) if all_pages else None,
        summary="",
        content="",
        children=nodes,
    )
    if llm_client is not None:
        virtual_root.summary = _make_summary(virtual_root, llm_client)
    return virtual_root
