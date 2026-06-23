import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams

from layer_f.tree_models import TreeNode
from layer_f.tree_store import TreeStore

_COLLECTION = "test_trees"


@pytest.fixture
def qdrant():
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=_COLLECTION,
        vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
        },
    )
    return client


@pytest.fixture
def sample_tree() -> TreeNode:
    return TreeNode(
        node_id="sec_0",
        title="治療原則",
        start_page=10,
        end_page=20,
        summary="第III/IV期肺癌治療方針",
        content="",
        children=[
            TreeNode(
                node_id="sec_1",
                title="第III期",
                start_page=10,
                end_page=15,
                summary="",
                content="同步化放療為標準治療",
                children=[],
            ),
        ],
    )


def test_store_and_load_static(qdrant, sample_tree):
    store = TreeStore()
    store.store_static("lung_guide", sample_tree, qdrant, _COLLECTION)
    loaded = store.load_static("lung_guide", qdrant, _COLLECTION)
    assert loaded is not None
    assert loaded.title == "治療原則"
    assert len(loaded.children) == 1
    assert loaded.children[0].content == "同步化放療為標準治療"


def test_store_static_populates_cache(qdrant, sample_tree):
    store = TreeStore()
    store.store_static("guide_pdf", sample_tree, qdrant, _COLLECTION)
    assert "guide_pdf" in store._static_cache
    assert store._static_cache["guide_pdf"].title == "治療原則"


def test_load_static_returns_none_for_missing(qdrant):
    store = TreeStore()
    result = store.load_static("nonexistent", qdrant, _COLLECTION)
    assert result is None


def test_store_and_load_dynamic(sample_tree):
    store = TreeStore()
    store.store_dynamic("session_abc", "patient_doc", sample_tree)
    loaded = store.load_dynamic("session_abc", "patient_doc")
    assert loaded is not None
    assert loaded.title == "治療原則"


def test_load_dynamic_returns_none_for_missing():
    store = TreeStore()
    assert store.load_dynamic("session_xyz", "missing") is None


def test_clear_session_removes_dynamic(sample_tree):
    store = TreeStore()
    store.store_dynamic("session_123", "doc_a", sample_tree)
    store.store_dynamic("session_123", "doc_b", sample_tree)
    store.clear_session("session_123")
    assert store.load_dynamic("session_123", "doc_a") is None
    assert store.load_dynamic("session_123", "doc_b") is None


def test_clear_session_does_not_affect_other_sessions(sample_tree):
    store = TreeStore()
    store.store_dynamic("session_keep", "doc", sample_tree)
    store.store_dynamic("session_delete", "doc", sample_tree)
    store.clear_session("session_delete")
    assert store.load_dynamic("session_keep", "doc") is not None


def test_preload_static_populates_cache(qdrant, sample_tree):
    store = TreeStore()
    store.store_static("lung_guide_pdf", sample_tree, qdrant, _COLLECTION)
    loaded = store.preload_static(["lung_guide_pdf"], qdrant, _COLLECTION)
    assert loaded == ["lung_guide_pdf"]
    assert "lung_guide_pdf" in store._static_cache
    assert store._static_cache["lung_guide_pdf"].title == "治療原則"


def test_load_static_uses_cache_skips_qdrant(sample_tree):
    store = TreeStore()
    store._static_cache["cached_doc"] = sample_tree
    # Pass client=None — must not raise, cache should be hit before any Qdrant call
    result = store.load_static("cached_doc", client=None, collection_name="ignored")
    assert result is not None
    assert result.title == "治療原則"


def test_list_loaded_stems(sample_tree):
    store = TreeStore()
    store._static_cache["doc_a"] = sample_tree
    store._static_cache["doc_b"] = sample_tree
    assert set(store.list_loaded_stems()) == {"doc_a", "doc_b"}


def test_get_static_summaries_fallback_chain():
    from layer_f.tree_models import TreeNode
    store = TreeStore()
    store._static_cache["with_summary"] = TreeNode(
        node_id="r", title="指引A", start_page=1, end_page=1,
        summary="完整摘要", content="", children=[],
    )
    store._static_cache["title_only"] = TreeNode(
        node_id="r", title="指引B", start_page=1, end_page=1,
        summary="", content="", children=[],
    )
    store._static_cache["neither"] = TreeNode(
        node_id="r", title="", start_page=1, end_page=1,
        summary="", content="", children=[],
    )
    s = store.get_static_summaries()
    assert s["with_summary"] == "完整摘要"
    assert s["title_only"] == "指引B"
    assert s["neither"] == "neither"
