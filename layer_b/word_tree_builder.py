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
    r'[/／]|項目$|對策$|確認$|簽名$|編號$|記錄$|說明$', _re.MULTILINE
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

    # If MATRIX but contains placeholder symbols in col0, treat as RECORD
    # (indicates form-like table structure)
    if ttype == _TableType.MATRIX and data:
        has_placeholders = False
        for row in data:
            c0 = row[0].get("text", "").strip()
            if _re.match(r'^[/：:\s，、\-\n]+$', c0):
                has_placeholders = True
                break
        if has_placeholders:
            ttype = _TableType.RECORD

    if ttype == _TableType.CHART:
        return []

    if ttype in (_TableType.RECORD,):
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
        from collections import OrderedDict
        categories: OrderedDict[str, list[str]] = OrderedDict()
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
