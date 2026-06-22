import pytest
from layer_b.tree_builder import build_tree


# 最小 Azure CU 資料：根節點「治療原則」下有兩個子節點
_RAW_CU = {
    "metadata": {"file_name": "lung_guide.pdf"},
    "data": {
        "sections": [
            # sections[0]: 根（title=治療原則）
            {
                "elements": [
                    "/paragraphs/0",   # sectionHeading "治療原則"
                    "/sections/1",
                    "/sections/2",
                ]
            },
            # sections[1]: 子節點（第III期）
            {
                "elements": [
                    "/paragraphs/1",   # sectionHeading "第III期"
                    "/paragraphs/2",   # body
                    "/paragraphs/3",   # body
                ]
            },
            # sections[2]: 子節點（第IV期）
            {
                "elements": [
                    "/paragraphs/4",   # sectionHeading "第IV期"
                    "/paragraphs/5",   # body
                ]
            },
        ],
        "paragraphs": [
            {"role": "sectionHeading", "content": "治療原則", "source": "D(10,0,0,1,0,1,1,0,1)", "spans": []},
            {"role": "sectionHeading", "content": "第III期", "source": "D(15,0,0,1,0,1,1,0,1)", "spans": []},
            {"role": None, "content": "同步化放療為標準治療方案", "source": "D(15,0,0,1,0,1,1,0,1)", "spans": []},
            {"role": None, "content": "可手術者考慮術前新輔助化療", "source": "D(16,0,0,1,0,1,1,0,1)", "spans": []},
            {"role": "sectionHeading", "content": "第IV期", "source": "D(20,0,0,1,0,1,1,0,1)", "spans": []},
            {"role": None, "content": "系統性治療為主要方向", "source": "D(20,0,0,1,0,1,1,0,1)", "spans": []},
        ],
    }
}


def test_returns_none_when_no_sections():
    raw = {"metadata": {}, "data": {"paragraphs": []}}
    assert build_tree(raw) is None


def test_root_title():
    tree = build_tree(_RAW_CU)
    assert tree is not None
    assert tree.title == "治療原則"


def test_children_count():
    tree = build_tree(_RAW_CU)
    assert len(tree.children) == 2


def test_children_titles():
    tree = build_tree(_RAW_CU)
    titles = [c.title for c in tree.children]
    assert "第III期" in titles
    assert "第IV期" in titles


def test_leaf_content_aggregated():
    tree = build_tree(_RAW_CU)
    stage3 = next(c for c in tree.children if c.title == "第III期")
    assert "同步化放療" in stage3.content
    assert "新輔助化療" in stage3.content


def test_leaf_page_range():
    tree = build_tree(_RAW_CU)
    stage3 = next(c for c in tree.children if c.title == "第III期")
    assert stage3.start_page == 15
    assert stage3.end_page == 16


def test_root_page_range_spans_children():
    tree = build_tree(_RAW_CU)
    assert tree.start_page == 10
    assert tree.end_page == 20


def test_leaves_have_no_children():
    tree = build_tree(_RAW_CU)
    for child in tree.children:
        assert child.is_leaf is True


def test_summary_generated_for_nonleaf():
    call_count = {"n": 0}

    def mock_llm_client():
        class _Mock:
            def generate_text(self, user, system=""):
                call_count["n"] += 1
                return f"摘要{call_count['n']}"
        return _Mock()

    tree = build_tree(_RAW_CU, llm_client=mock_llm_client())
    # root node is non-leaf → should have a summary
    assert tree.summary != ""
    assert call_count["n"] >= 1


def test_no_summary_when_llm_is_none():
    tree = build_tree(_RAW_CU, llm_client=None)
    assert tree.summary == ""


def test_page_range_includes_heading_page_when_section_has_title():
    """Heading page must be in range even when sec already has a title field."""
    raw = {
        "metadata": {"file_name": "doc.pdf"},
        "data": {
            "sections": [
                {
                    "title": "預設標題",          # pre-set title
                    "elements": [
                        "/paragraphs/0",           # sectionHeading at page 5
                        "/paragraphs/1",           # body at page 6
                    ]
                }
            ],
            "paragraphs": [
                {"role": "sectionHeading", "content": "章節標題", "source": "D(5,0,0,1,0,1,1,0,1)", "spans": []},
                {"role": None, "content": "章節內文", "source": "D(6,0,0,1,0,1,1,0,1)", "spans": []},
            ],
        }
    }
    tree = build_tree(raw)
    assert tree is not None
    assert tree.start_page == 5   # heading page must be included
    assert tree.end_page == 6
