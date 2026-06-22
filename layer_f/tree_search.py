# layer_f/tree_search.py
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from layer_f.tree_models import CrossTreeResult, TreeNode, TreeSearchResult

if TYPE_CHECKING:
    from layer_e.llm_client import LLMClient

_JSON_RE = re.compile(r'\{[^{}]*"relevant"[^{}]*\}')

_SELECT_SYSTEM = "你是醫療文件導航助理，協助定位與查詢最相關的章節。"

_SELECT_USER_TEMPLATE = """\
查詢：{query}

以下是文件章節列表：
{children_text}

請選出與查詢直接相關的章節編號（可多選）。
只回傳 JSON，格式：{{"relevant": [0, 2]}}
若無相關章節，回傳：{{"relevant": []}}"""

_SYNTHESIS_SYSTEM = "你是醫療文件分析助理，協助比對指引條件與病人資料。"

_SYNTHESIS_USER_TEMPLATE = """\
查詢：{query}

【治療指引】相關章節：
{guideline_content}

【病人資料】相關章節：
{patient_content}

請根據以上資訊，用繁體中文回答查詢。直接回答結論，不需重述資料。"""


def _format_children(children: list[TreeNode]) -> str:
    parts = []
    for i, child in enumerate(children):
        desc = (child.summary or child.content)[:150]
        parts.append(f"[{i}] {child.title}\n    {desc}")
    return "\n\n".join(parts)


def _parse_relevant_indices(text: str, max_idx: int) -> list[int]:
    m = _JSON_RE.search(text or "")
    if not m:
        return []
    try:
        data = json.loads(m.group())
        return [i for i in data.get("relevant", []) if 0 <= i < max_idx]
    except (json.JSONDecodeError, TypeError):
        return []


class TreeSearcher:
    def __init__(self, llm_client: LLMClient, max_depth: int = 5) -> None:
        self._llm = llm_client
        self._max_depth = max_depth

    def search(self, query: str, root: TreeNode) -> TreeSearchResult:
        """Top-down traversal: returns the most relevant leaf (or terminal) nodes."""
        matched_nodes: list[TreeNode] = []
        traversal_paths: list[list[str]] = []
        self._traverse(query, root, [root.title], matched_nodes, traversal_paths, depth=0)
        return TreeSearchResult(
            query=query,
            matched_nodes=matched_nodes,
            traversal_path=traversal_paths,
        )

    def _traverse(
        self,
        query: str,
        node: TreeNode,
        path: list[str],
        matched: list[TreeNode],
        paths: list[list[str]],
        depth: int,
    ) -> None:
        if node.is_leaf or depth >= self._max_depth:
            matched.append(node)
            paths.append(list(path))
            return

        children_text = _format_children(node.children)
        prompt = _SELECT_USER_TEMPLATE.format(query=query, children_text=children_text)
        response = self._llm.generate_text(prompt, system=_SELECT_SYSTEM)
        indices = _parse_relevant_indices(response, len(node.children))

        if not indices:
            # No relevant children — treat this node as terminal
            matched.append(node)
            paths.append(list(path))
            return

        for i in indices:
            child = node.children[i]
            self._traverse(query, child, path + [child.title], matched, paths, depth + 1)

    def search_cross(
        self,
        query: str,
        guideline_tree: TreeNode,
        patient_tree: TreeNode,
    ) -> CrossTreeResult:
        """Search both trees independently, then synthesize a cross-tree answer."""
        guideline_result = self.search(query, guideline_tree)
        patient_result = self.search(query, patient_tree)

        guideline_content = "\n\n".join(
            f"【{n.title}】\n{n.content}"
            for n in guideline_result.matched_nodes
            if n.content
        ) or "（無相關章節）"

        patient_content = "\n\n".join(
            f"【{n.title}】\n{n.content}"
            for n in patient_result.matched_nodes
            if n.content
        ) or "（無相關章節）"

        synthesis_prompt = _SYNTHESIS_USER_TEMPLATE.format(
            query=query,
            guideline_content=guideline_content,
            patient_content=patient_content,
        )
        synthesis = self._llm.generate_text(synthesis_prompt, system=_SYNTHESIS_SYSTEM)

        return CrossTreeResult(
            query=query,
            guideline_nodes=guideline_result.matched_nodes,
            patient_nodes=patient_result.matched_nodes,
            synthesis=synthesis,
        )
