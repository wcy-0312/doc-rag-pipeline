import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams

from layer_f.tree_models import TreeNode
from layer_f.tree_store import TreeStore

_COLLECTION = "test_agentic"


class _AwareStubLLM:
    """Stub LLM that returns appropriate JSON for routing, traversal, and generation."""

    def generate_text(self, user: str, system: str = "") -> str:
        if "relevant_guidelines" in user or "可用的治療指引" in user:
            return '{"need_patient_context": false, "relevant_guidelines": [0]}'
        if "章節列表" in user or '"relevant"' in user:
            return '{"relevant": [0]}'
        return "測試合成答案"

    def generate(self, system: str, user: str) -> dict:
        return {
            "answer": "stub answer",
            "claims": [{"text": "stub", "citations": ["E1"]}],
            "abstain": False,
            "abstain_reason": None,
        }

    def generate_with_tools(self, messages, tools):
        import json
        return ([], json.dumps({
            "answer": "stub", "claims": [], "abstain": False, "abstain_reason": None,
        }))


@pytest.fixture
def pipeline_with_tree():
    from pipeline.runner import RAGPipeline

    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=_COLLECTION,
        vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
        },
    )
    stub_llm = _AwareStubLLM()
    pipeline = RAGPipeline(
        embedding_provider=None,
        qdrant_client=client,
        collection_name=_COLLECTION,
        llm_client=stub_llm,
        abstention_threshold=0.0,
        reranker=None,
    )
    # Build and store a static tree manually
    tree = TreeNode(
        node_id="root", title="肺癌診療指引", start_page=1, end_page=50,
        summary="肺癌診療指引摘要", content="",
        children=[
            TreeNode(
                node_id="sec_1", title="非小細胞肺癌治療", start_page=10, end_page=30,
                summary="", content="Stage III NSCLC 標準治療為同步化放療。",
                children=[],
            )
        ],
    )
    pipeline._tree_store.store_static("肺癌診療指引v20_1_pdf", tree, client, _COLLECTION)
    return pipeline, client, stub_llm


def test_preload_trees_loads_into_cache(pipeline_with_tree):
    pipeline, client, _ = pipeline_with_tree
    loaded = pipeline.preload_trees(["肺癌診療指引v20_1_pdf"])
    assert loaded == ["肺癌診療指引v20_1_pdf"]
    assert "肺癌診療指引v20_1_pdf" in pipeline._tree_store._static_cache


def test_query_tree_agentic_no_patient_context(pipeline_with_tree):
    pipeline, _, stub_llm = pipeline_with_tree
    pipeline.preload_trees(["肺癌診療指引v20_1_pdf"])
    result = pipeline.query_tree_agentic(
        "Stage III 非小細胞肺癌的標準治療是什麼？",
        session_id=None,
        llm_client=stub_llm,
    )
    assert result.abstain is False
    assert result.answer != ""


def test_query_tree_agentic_abstains_when_no_trees_preloaded():
    from pipeline.runner import RAGPipeline
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="empty",
        vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
        },
    )
    pipeline = RAGPipeline(
        embedding_provider=None,
        qdrant_client=client,
        collection_name="empty",
        llm_client=_AwareStubLLM(),
        abstention_threshold=0.0,
        reranker=None,
    )
    result = pipeline.query_tree_agentic("任何問題", session_id=None)
    assert result.abstain is True


def test_query_tree_agentic_with_patient_context(pipeline_with_tree):
    pipeline, _, stub_llm = pipeline_with_tree

    # Preload static tree
    pipeline.preload_trees(["肺癌診療指引v20_1_pdf"])

    # Set up a dynamic patient tree
    patient_tree = TreeNode(
        node_id="p_root", title="病人病歷", start_page=1, end_page=3,
        summary="", content="",
        children=[
            TreeNode(
                node_id="p_1", title="診斷", start_page=1, end_page=2,
                summary="", content="病人確診為肺腺癌，cT2N1M0，EGFR 陰性。",
                children=[],
            )
        ],
    )
    pipeline._tree_store.store_dynamic("session_test", "病歷_pdf", patient_tree)

    # Override stub to request patient context
    class _PatientAwareLLM(_AwareStubLLM):
        def generate_text(self, user: str, system: str = "") -> str:
            if "relevant_guidelines" in user or "可用的治療指引" in user:
                return '{"need_patient_context": true, "relevant_guidelines": [0]}'
            if "章節列表" in user or '"relevant"' in user:
                return '{"relevant": [0]}'
            return "根據病人診斷及指引，建議進行同步化放療。"

    result = pipeline.query_tree_agentic(
        "這位病人應接受什麼治療？",
        session_id="session_test",
        llm_client=_PatientAwareLLM(),
    )
    assert result.abstain is False
    assert result.answer != ""
