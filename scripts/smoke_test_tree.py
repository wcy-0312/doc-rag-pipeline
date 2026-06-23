"""
End-to-end smoke test for Tree RAG — Scenario 1 (static tree query) and
the new query_tree_agentic() flow.

Run:
    python scripts/smoke_test_tree.py

Requires:
    - Gemma3 endpoint reachable at http://172.31.6.3:8080/gemma3/v1
    - output/layer_b/raw_乳癌診療指引-2026年.json exists
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams

from layer_b.tree_builder import build_tree
from layer_e.llm_client import Gemma3Client, _StubLLMClient
from pipeline.runner import RAGPipeline

RAW_PATH = Path("output/layer_b/raw_乳癌診療指引-2026年.json")
DOC_ID = "乳癌診療指引-2026年.pdf"
COLLECTION = "smoke_test_trees"


def make_pipeline(llm_client=None):
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
        },
    )
    return RAGPipeline(
        embedding_provider=None,
        qdrant_client=client,
        collection_name=COLLECTION,
        llm_client=llm_client or _StubLLMClient(),
        abstention_threshold=0.0,
        reranker=None,
    ), client


def print_section(title: str):
    print(f"\n{'='*60}\n{title}\n{'='*60}")


def main():
    print("載入 raw document...")
    with open(RAW_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    gemma = Gemma3Client()

    # ── Setup: build and store static tree ──────────────────────────────
    print_section("建立靜態樹")
    pipeline, qdrant_client = make_pipeline(llm_client=gemma)
    tree = pipeline.build_tree(raw, doc_id=DOC_ID, static=True)
    if tree is None:
        print("ERROR: build_tree 回傳 None，請確認 raw JSON 包含 sections[]")
        sys.exit(1)

    def count_nodes(n, depth=0):
        return 1 + sum(count_nodes(c) for c in n.children)

    print(f"樹建立成功：root='{tree.title}'，共 {count_nodes(tree)} 個節點")

    # ── Scenario 1: query_tree() ─────────────────────────────────────────
    print_section("情境一：query_tree() — 靜態樹直接查詢")

    doc_stem = "乳癌診療指引-2026年_pdf"
    queries_s1 = [
        "三陰性晚期乳癌的治療選項有哪些？",
        "HER-2 陽性晚期乳癌的一線治療建議？",
    ]
    for q in queries_s1:
        print(f"\n查詢: {q}")
        result = pipeline.query_tree(q, doc_ids=[doc_stem], llm_client=gemma)
        status = "abstained" if result.abstain else "answered"
        print(f"狀態: {status}")
        print(f"答案: {result.answer[:300]}")

    # ── Scenario 2: query_tree_agentic() without patient context ─────────
    print_section("情境二：query_tree_agentic() — 無病人資料，LLM 自主路由")

    loaded = pipeline.preload_trees([DOC_ID])
    print(f"預載靜態樹: {loaded}")

    queries_s2 = [
        "如果病人是 ER+/HER-2- 第 II 期乳癌，術後建議哪些輔助治療？",
        "乳癌放射線治療的劑量原則為何？",
    ]
    for q in queries_s2:
        print(f"\n查詢: {q}")
        result = pipeline.query_tree_agentic(q, session_id=None, llm_client=gemma)
        status = "abstained" if result.abstain else "answered"
        reason = f" ({result.abstain_reason})" if result.abstain else ""
        print(f"狀態: {status}{reason}")
        print(f"答案: {result.answer[:400]}")

    print_section("Smoke test 完成")


if __name__ == "__main__":
    main()
