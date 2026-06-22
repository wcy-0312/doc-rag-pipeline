from layer_f.tree_models import TreeNode, TreeSearchResult, CrossTreeResult


def _leaf(title: str, content: str = "some content", page: int = 1) -> TreeNode:
    return TreeNode(
        node_id=f"n_{title}",
        title=title,
        start_page=page,
        end_page=page,
        summary="",
        content=content,
        children=[],
    )


def _branch(title: str, children: list) -> TreeNode:
    pages = [c.start_page for c in children if c.start_page is not None]
    return TreeNode(
        node_id=f"n_{title}",
        title=title,
        start_page=min(pages) if pages else None,
        end_page=max(pages) if pages else None,
        summary="a summary",
        content="",
        children=children,
    )


def test_leaf_is_leaf():
    assert _leaf("第一節").is_leaf is True


def test_branch_is_not_leaf():
    assert _branch("治療原則", [_leaf("手術"), _leaf("化療")]).is_leaf is False


def test_page_range():
    node = TreeNode(
        node_id="n1", title="T", start_page=5, end_page=10,
        summary="", content="", children=[],
    )
    assert node.page_range == (5, 10)


def test_to_dict_and_from_dict_roundtrip():
    leaf = _leaf("葉節點", content="內容文字", page=3)
    restored = TreeNode.from_dict(leaf.to_dict())
    assert restored.node_id == leaf.node_id
    assert restored.title == leaf.title
    assert restored.content == leaf.content
    assert restored.is_leaf is True


def test_nested_roundtrip():
    tree = _branch("根節點", [_leaf("子A", page=1), _leaf("子B", page=2)])
    restored = TreeNode.from_dict(tree.to_dict())
    assert len(restored.children) == 2
    assert restored.children[0].title == "子A"


def test_tree_search_result():
    node = _leaf("第III期", content="同步化放療")
    result = TreeSearchResult(
        query="cT3N2M0 治療",
        matched_nodes=[node],
        traversal_path=[["治療原則", "依分期", "第III期"]],
    )
    assert len(result.matched_nodes) == 1
    assert result.matched_nodes[0].content == "同步化放療"


def test_cross_tree_result():
    result = CrossTreeResult(
        query="是否符合免疫治療給付？",
        guideline_nodes=[_leaf("給付條件", content="PD-L1 ≥ 50%")],
        patient_nodes=[_leaf("檢驗報告", content="PD-L1 = 60%")],
        synthesis="病人 PD-L1 60%，符合給付條件（≥50%）",
    )
    assert "60%" in result.synthesis
