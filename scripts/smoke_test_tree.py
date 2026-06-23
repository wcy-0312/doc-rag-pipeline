"""
End-to-end smoke test for Tree RAG — Scenario 1 (static tree query).

Measures:
  - Tree build time
  - Per-query tree-search time for 5 clinical questions

Run:
    python scripts/smoke_test_tree.py

Requires:
    - Gemma3 endpoint reachable at http://172.31.6.3:8080/gemma3/v1
    - output/layer_b/raw_乳癌診療指引-2026年.json exists
"""
import glob
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams

from layer_e.llm_client import Gemma3Client, _StubLLMClient
from pipeline.runner import RAGPipeline

RAW_PATH = Path("output/layer_b/raw_乳癌診療指引-2026年.json")
DOC_ID = "乳癌診療指引-2026年.pdf"

_pdf_candidates = sorted(glob.glob("**/*乳癌*2026*.pdf", recursive=True))
PDF_PATH = Path(_pdf_candidates[0]) if _pdf_candidates else Path("docs/癌症診療指引/乳癌診療指引-2026年.pdf")


def _make_stem(s: str) -> str:
    """Mirror build_tree()'s doc_stem derivation (NFKC + sanitize)."""
    return re.sub(r'[^\w\-]', '_', unicodedata.normalize('NFKC', s))
COLLECTION = "smoke_test_trees"

QUERIES = [
    "晚期三陰性乳癌（Advanced TNBC）的化學治療選項有哪些？",
    "HER-2 陽性晚期乳癌的第一線治療建議是什麼？",
    "CDK4/6 inhibitor 適用於哪種乳癌亞型？",
    "全乳房放射治療（WBI）的劑量標準是什麼？",
    "如果病人是 cT2N1M0，可以就這份乳癌指引給出治療建議嗎？",
]


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


def count_nodes(node):
    return 1 + sum(count_nodes(c) for c in node.children)


def hr(char="─", width=64):
    print(char * width)


def main():
    print("載入 raw document...")
    with open(RAW_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    gemma = Gemma3Client()
    pipeline, _ = make_pipeline(llm_client=gemma)

    # Derive stem the same way build_tree() does
    file_name = raw.get("metadata", {}).get("file_name") or DOC_ID
    doc_stem = _make_stem(file_name)

    # ── Build static tree ────────────────────────────────────────────────
    hr("═")
    print("建立靜態樹")
    hr("═")
    print(f"doc_stem   : {doc_stem!r}")
    t0 = time.perf_counter()
    tree = pipeline.build_tree(raw, doc_id=DOC_ID, static=True, pdf_path=str(PDF_PATH.resolve()))
    build_elapsed = time.perf_counter() - t0

    if tree is None:
        print("ERROR: build_tree 回傳 None，請確認 raw JSON 包含 sections[]")
        sys.exit(1)

    total_nodes = count_nodes(tree)
    print(f"root title : {tree.title!r}")
    print(f"node count : {total_nodes}")
    print(f"build time : {build_elapsed:.1f}s")

    # ── Scenario 1: query_tree_agentic() — 5 questions ───────────────────────
    hr("═")
    print("情境一：query_tree_agentic() — 靜態樹自動路由查詢（共 5 題）")
    hr("═")

    pipeline.preload_trees([DOC_ID])

    timings = []
    for i, query in enumerate(QUERIES, start=1):
        hr()
        print(f"Q{i}: {query}")
        hr()
        t0 = time.perf_counter()
        result = pipeline.query_tree_agentic(query, llm_client=gemma)
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)

        if result.abstain:
            print(f"⚠ abstained: {result.abstain_reason}")
        else:
            print("▶ 回答：")
            print()
            print(result.answer)
        print()
        print(f"⏱  {elapsed:.1f}s")

    hr("═")
    print(f"平均耗時：{sum(timings)/len(timings):.1f}s")


if __name__ == "__main__":
    main()
