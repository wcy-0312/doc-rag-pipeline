"""
Vision synthesis smoke test — Q5 (cT2N1M0 treatment).

Tests vision synthesis via query_tree_agentic():
  build_tree(pdf_path=...) → preload_trees() → query_tree_agentic() → vision LLM reads page images

Run:
    python scripts/smoke_test_vision_q5.py
    python scripts/smoke_test_vision_q5.py --gpt41   # use GPT41 instead of Gemma3
"""
import argparse
import glob as _glob
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams

from pipeline.runner import RAGPipeline

RAW_PATH  = Path("output/layer_b/raw_乳癌診療指引-2026年.json")
# Use glob to resolve the actual on-disk path (filename may contain CJK compatibility chars)
_pdf_candidates = sorted(_glob.glob("**/*乳癌*2026*.pdf", recursive=True))
PDF_PATH  = Path(_pdf_candidates[0]) if _pdf_candidates else Path("docs/癌症診療指引/乳癌診療指引-2026年.pdf")
DOC_ID    = "乳癌診療指引-2026年.pdf"
COLLECTION = "smoke_vision_q5"

QUERY = "如果病人是 cT2N1M0，可以就這份乳癌指引給出治療建議嗎？"


def make_pipeline(llm_client):
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
        llm_client=llm_client,
        abstention_threshold=0.0,
        reranker=None,
    )


def hr(char="─", width=64):
    print(char * width)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt41", action="store_true", help="Use GPT41Client instead of Gemma3")
    args = parser.parse_args()

    if args.gpt41:
        from layer_e.llm_client import GPT41Client
        llm = GPT41Client()
        llm_name = "GPT-4.1"
    else:
        from layer_e.llm_client import Gemma3Client
        llm = Gemma3Client()
        llm_name = "Gemma3"

    print(f"LLM: {llm_name}")
    print(f"Query: {QUERY}")

    # ── Load raw document ────────────────────────────────────────────────
    hr("═")
    print("載入 raw document 並建立靜態樹 (with pdf_path)...")
    with open(RAW_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    pipeline = make_pipeline(llm)

    t0 = time.perf_counter()
    tree = pipeline.build_tree(
        raw,
        doc_id=DOC_ID,
        static=True,
        pdf_path=str(PDF_PATH.resolve()),
    )
    build_elapsed = time.perf_counter() - t0

    if tree is None:
        print("ERROR: build_tree 回傳 None")
        sys.exit(1)

    print(f"✓ 建樹完成 ({build_elapsed:.1f}s)，{sum(1 for _ in _iter_nodes(tree))} 個節點")

    # Confirm pdf_path registered
    stem = re.sub(r'[^\w\-]', '_', unicodedata.normalize('NFKC', DOC_ID))
    registered = pipeline._tree_pdf_paths.get(stem)
    print(f"✓ pdf_path 已登錄 stem={stem!r}")
    print(f"  → {registered}")

    # ── Run query_tree_agentic() with vision synthesis ───────────────────────
    hr("═")
    print("執行 query_tree_agentic() — vision synthesis 路徑")
    hr("═")

    pipeline.preload_trees([DOC_ID])

    t0 = time.perf_counter()
    result = pipeline.query_tree_agentic(QUERY, llm_client=llm, vision_dpi=50)
    elapsed = time.perf_counter() - t0

    hr()
    if result.abstain:
        print(f"⚠ abstained: {result.abstain_reason}")
    else:
        print("▶ 回答：")
        print()
        print(result.answer)

    hr()
    print(f"⏱  {elapsed:.1f}s")
    hr("═")


def _iter_nodes(node):
    yield node
    for c in node.children:
        yield from _iter_nodes(c)


if __name__ == "__main__":
    main()
