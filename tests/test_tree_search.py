# tests/test_tree_search.py
import pytest
from layer_e.llm_client import _StubLLMClient
from layer_f.tree_models import TreeNode
from layer_f.tree_search import TreeSearcher


def _make_tree() -> TreeNode:
    """Build a 3-level test tree: root → 治療原則 → {第III期, 第IV期}"""
    stage3 = TreeNode(
        node_id="sec_1", title="第III期",
        start_page=15, end_page=16, summary="",
        content="同步化放療為標準治療方案，可手術者考慮新輔助化療。",
        children=[],
    )
    stage4 = TreeNode(
        node_id="sec_2", title="第IV期",
        start_page=20, end_page=20, summary="",
        content="系統性治療為主，標靶或化療依基因結果選擇。",
        children=[],
    )
    treatment = TreeNode(
        node_id="sec_0", title="治療原則",
        start_page=10, end_page=20,
        summary="依分期決定治療方式",
        content="",
        children=[stage3, stage4],
    )
    return treatment


class _SelectFirstLLM:
    """Always selects child index 0."""
    def generate_text(self, user: str, system: str = "") -> str:
        return '{"relevant": [0]}'


class _SelectAllLLM:
    """Always selects all children."""
    def generate_text(self, user: str, system: str = "") -> str:
        import re
        # Count [N] patterns in user prompt to know how many children
        count = len(re.findall(r'^\[\d+\]', user, re.MULTILINE))
        return f'{{"relevant": {list(range(count))}}}'


class _SelectNoneLLM:
    """Never selects any child."""
    def generate_text(self, user: str, system: str = "") -> str:
        return '{"relevant": []}'


def test_search_reaches_leaf():
    tree = _make_tree()
    result = TreeSearcher(_SelectFirstLLM()).search("第III期治療", tree)
    assert len(result.matched_nodes) == 1
    assert result.matched_nodes[0].is_leaf is True


def test_search_selected_content_is_correct():
    tree = _make_tree()
    result = TreeSearcher(_SelectFirstLLM()).search("第III期治療", tree)
    assert "同步化放療" in result.matched_nodes[0].content


def test_search_traversal_path_recorded():
    tree = _make_tree()
    result = TreeSearcher(_SelectFirstLLM()).search("第III期治療", tree)
    assert len(result.traversal_path) == 1
    assert "第III期" in result.traversal_path[0]


def test_search_no_relevant_returns_current_node():
    """When LLM selects no children, current node is returned as terminal."""
    tree = _make_tree()
    result = TreeSearcher(_SelectNoneLLM()).search("任意查詢", tree)
    assert len(result.matched_nodes) == 1
    assert result.matched_nodes[0].node_id == tree.node_id


def test_search_all_selects_both_leaves():
    tree = _make_tree()
    result = TreeSearcher(_SelectAllLLM()).search("所有分期治療", tree)
    assert len(result.matched_nodes) == 2


def test_search_leaf_node_returns_immediately():
    leaf = TreeNode(
        node_id="leaf", title="直接葉節點",
        start_page=1, end_page=1, summary="",
        content="葉節點內容", children=[],
    )
    result = TreeSearcher(_SelectFirstLLM()).search("任意查詢", leaf)
    assert len(result.matched_nodes) == 1
    assert result.matched_nodes[0].content == "葉節點內容"


def test_search_cross_returns_cross_result():
    class _SynthesisLLM:
        call_count = 0
        def generate_text(self, user: str, system: str = "") -> str:
            self.call_count += 1
            if "relevant" in user:        # child selection calls
                return '{"relevant": [0]}'
            return "病人 PD-L1 60%，符合給付條件（≥50%）"   # synthesis call

    guideline = TreeNode(
        node_id="g0", title="給付條件",
        start_page=5, end_page=5, summary="",
        content="PD-L1 ≥ 50%，無 EGFR 突變", children=[],
    )
    patient = TreeNode(
        node_id="p0", title="檢驗報告",
        start_page=1, end_page=1, summary="",
        content="PD-L1 = 60%，EGFR 野生型", children=[],
    )

    llm = _SynthesisLLM()
    result = TreeSearcher(llm).search_cross(
        "病人是否符合免疫治療給付？", guideline, patient
    )
    assert "60%" in result.synthesis
    assert len(result.guideline_nodes) >= 1
    assert len(result.patient_nodes) >= 1
