"""Tests that build_tree() includes figures' paragraph text in node content."""
import pytest
from layer_b.tree_builder import build_tree


def _raw_with_figure():
    """Minimal Azure CU JSON that has one section referencing one figure,
    which itself references two paragraphs (simulating a flowchart caption)."""
    return {
        "metadata": {"file_name": "test.pdf"},
        "data": {
            "sections": [
                {
                    "title": "HER2+ 治療路徑",
                    "elements": [
                        "/figures/0",
                    ],
                }
            ],
            "paragraphs": [
                {
                    "content": "先導性化療+trastuzumab+/-pertuzumab",
                    "role": "paragraph",
                    "source": "D(22, 100, 200, 300, 400)",
                },
                {
                    "content": "病理完全反應",
                    "role": "paragraph",
                    "source": "D(22, 100, 500, 300, 600)",
                },
            ],
            "figures": [
                {
                    "elements": [
                        "/paragraphs/0",
                        "/paragraphs/1",
                    ],
                    "boundingRegions": [],
                }
            ],
        },
    }


def test_figure_paragraphs_included_in_node_content():
    """A section referencing a figure must include the figure's paragraph text."""
    raw = _raw_with_figure()
    tree = build_tree(raw)

    assert tree is not None
    assert "先導性化療+trastuzumab+/-pertuzumab" in tree.content
    assert "病理完全反應" in tree.content


def test_figure_page_included_in_node_page_range():
    """Figure paragraphs contribute their page numbers to start_page/end_page."""
    raw = _raw_with_figure()
    tree = build_tree(raw)

    assert tree is not None
    assert tree.start_page == 22
    assert tree.end_page == 22


def test_node_without_figures_unaffected():
    """Sections with only /paragraphs/ elements still work as before."""
    raw = {
        "metadata": {"file_name": "test.pdf"},
        "data": {
            "sections": [
                {
                    "title": "一般段落",
                    "elements": ["/paragraphs/0"],
                }
            ],
            "paragraphs": [
                {
                    "content": "Normal text",
                    "role": "paragraph",
                    "source": "D(5, 0, 0, 100, 100)",
                }
            ],
            "figures": [],
        },
    }
    tree = build_tree(raw)

    assert tree is not None
    assert tree.content == "Normal text"
    assert tree.start_page == 5


def test_figure_with_no_elements_does_not_crash():
    """A figure with no elements list should not raise."""
    raw = {
        "metadata": {"file_name": "test.pdf"},
        "data": {
            "sections": [
                {
                    "title": "圖表章節",
                    "elements": ["/figures/0"],
                }
            ],
            "paragraphs": [],
            "figures": [{"boundingRegions": []}],   # no "elements" key
        },
    }
    tree = build_tree(raw)
    # Node has a title but empty content — still valid (not None)
    assert tree is not None
    assert tree.title == "圖表章節"


def test_figure_out_of_range_does_not_crash():
    """An /figures/99 reference when figures list is short must be silently skipped."""
    raw = {
        "metadata": {"file_name": "test.pdf"},
        "data": {
            "sections": [
                {
                    "title": "章節",
                    "elements": ["/figures/99"],
                }
            ],
            "paragraphs": [],
            "figures": [],
        },
    }
    # Should not raise; node has a title but no content
    tree = build_tree(raw)
    assert tree is not None
