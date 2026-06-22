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
