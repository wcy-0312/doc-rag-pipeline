"""
端對端測試：Agentic RAG Loop 回答 cT2N1M0 治療建議問題

執行：
    conda run -n hospital-rag python3 scripts/test_agentic_query.py

流程：
    1. Layer A: Azure CU 萃取乳癌診療指引 PDF（或讀取快取）
    2. Layer B: 結構化為 RetrievalUnit
    3. Layer C: BGE-M3 embedding
    4. Layer D: 存入 in-memory Qdrant
    5. Layer E: AgenticPipeline + GPT-4.1 tool calling
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "layer_a"))

# 檔案名含 CJK 相容字元，用目錄掃描找實際路徑
def _find_pdf() -> Path:
    d = _ROOT / "docs" / "癌症診療指引"
    for f in d.iterdir():
        if "乳癌" in f.name and f.suffix == ".pdf":
            return f
    raise FileNotFoundError(f"找不到乳癌 PDF in {d}")

PDF_PATH = _find_pdf()
LAYER_B_CACHE = _ROOT / "output" / "layer_b" / f"retrieval_units_{PDF_PATH.stem}.json"
LAYER_B_ENRICHED_CACHE = _ROOT / "output" / "layer_b" / f"retrieval_units_{PDF_PATH.stem}_enriched.json"
COLLECTION = PDF_PATH.stem
QUERY = "如果病人是cT2N1M0，醫點家可以就這份PDF指引給出治療建議嗎?"


def step(msg: str):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def run_layer_a() -> dict:
    """Layer A: 用 Azure CU 萃取 PDF。有快取則跳過。"""
    raw_cache = _ROOT / "output" / "layer_b" / "raw_乳癌診療指引-2026年.json"
    if raw_cache.exists():
        print(f"  [快取] 讀取 {raw_cache}")
        return json.loads(raw_cache.read_text())

    step("Layer A: Azure CU 萃取 PDF（可能需要 1–3 分鐘）")
    from layer_a import get_extractor
    extractor = get_extractor("azure_cu")
    raw = extractor(PDF_PATH)  # requires Path object
    raw_cache.parent.mkdir(parents=True, exist_ok=True)
    raw_cache.write_text(json.dumps(raw, ensure_ascii=False, indent=2))
    print(f"  [完成] 已快取至 {raw_cache}")
    return raw


def run_layer_b(raw: dict) -> list:
    """Layer B: 結構化。有快取則跳過。"""
    if LAYER_B_CACHE.exists():
        print(f"  [快取] 讀取 {LAYER_B_CACHE}")
        from layer_b.pipeline import RetrievalUnit
        data = json.loads(LAYER_B_CACHE.read_text())
        return [RetrievalUnit(**u) for u in data]

    step("Layer B: 結構化 RetrievalUnit")
    from layer_b.pipeline import process_document
    units = process_document(raw)
    print(f"  [完成] {len(units)} 個 RetrievalUnit")
    # 快取
    LAYER_B_CACHE.parent.mkdir(parents=True, exist_ok=True)
    LAYER_B_CACHE.write_text(
        json.dumps([u.__dict__ if hasattr(u, "__dict__") else dict(u) for u in units],
                   ensure_ascii=False, indent=2)
    )
    return units


def main():
    if not PDF_PATH.exists():
        print(f"[錯誤] PDF 不存在: {PDF_PATH}")
        sys.exit(1)

    print(f"\n  PDF:  {PDF_PATH}")
    print(f"  問題: {QUERY}")

    # ── Layer A ──────────────────────────────────────────────────────────────
    raw = run_layer_a()

    # ── Layer B ──────────────────────────────────────────────────────────────
    step("Layer B: 結構化 RetrievalUnit")
    from layer_b.pipeline import process_document
    if LAYER_B_CACHE.exists():
        print(f"  [快取] 讀取 {LAYER_B_CACHE}")
        units_raw = json.loads(LAYER_B_CACHE.read_text())
        # 直接傳 dict list 給 process_and_embed 用 namedtuple/dataclass
        import dataclasses
        from layer_b.models import RetrievalUnit
        _valid = {f.name for f in dataclasses.fields(RetrievalUnit)}
        units = [RetrievalUnit(**{k: v for k, v in u.items() if k in _valid}) for u in units_raw]
    else:
        units = process_document(raw)
        print(f"  [完成] {len(units)} 個 RetrievalUnit")
        LAYER_B_CACHE.parent.mkdir(parents=True, exist_ok=True)
        # 序列化（dataclass → dict）
        import dataclasses
        units_raw = [dataclasses.asdict(u) if dataclasses.is_dataclass(u) else u.__dict__ for u in units]
        LAYER_B_CACHE.write_text(json.dumps(units_raw, ensure_ascii=False, indent=2))

    # ── Layer B.5: Semantic Enrichment ───────────────────────────────────────
    step("Layer B.5: Semantic Enrichment（LLM 標注 [適用] 對象）")
    import dataclasses as _dc
    from layer_b.enrichment import enrich_units
    from layer_e.llm_client import GPT41Client

    if LAYER_B_ENRICHED_CACHE.exists():
        print(f"  [快取] 讀取 {LAYER_B_ENRICHED_CACHE}")
        from layer_b.models import RetrievalUnit as _RU
        _valid = {f.name for f in _dc.fields(_RU)}
        units = [_RU(**{k: v for k, v in u.items() if k in _valid})
                 for u in json.loads(LAYER_B_ENRICHED_CACHE.read_text())]
    else:
        llm = GPT41Client()
        units = enrich_units(units, llm)
        enriched_raw = [_dc.asdict(u) for u in units]
        LAYER_B_ENRICHED_CACHE.write_text(json.dumps(enriched_raw, ensure_ascii=False, indent=2))
        enriched_count = sum(
            1 for u in units
            if u.embedding_text.startswith("[適用]") or u.embedding_text.startswith("[摘要]")
        )
        print(f"  [完成] {len(units)} 個 unit，{enriched_count} 個已標注")

    # ── Layer C: BGE-M3 Embedding ─────────────────────────────────────────
    step("Layer C: BGE-M3 Embedding（首次需下載模型，約 1 分鐘）")
    from layer_c.providers.bge_m3 import BGEm3Provider
    from layer_c.pipeline import process_and_embed
    provider = BGEm3Provider()
    t0 = time.time()
    chunks = process_and_embed(units, provider)
    print(f"  [完成] {len(chunks)} 個 EmbeddedChunk，耗時 {time.time()-t0:.1f}s")

    # ── Layer D: In-memory Qdrant ─────────────────────────────────────────
    step("Layer D: 建立 in-memory Qdrant 並 ingest")
    from qdrant_client import QdrantClient
    from layer_d.ingestion import DocumentIngester
    from layer_d.retrieval import HybridRetriever
    from layer_d.reranker import BGEReranker

    qdrant = QdrantClient(":memory:")
    ingester = DocumentIngester(client=qdrant, collection_name=COLLECTION)
    ingester.create_collection_if_not_exists()
    n = ingester.ingest(chunks)
    print(f"  [完成] {n} 個 chunk 已 ingest")

    reranker = BGEReranker()
    retriever = HybridRetriever(client=qdrant, collection_name=COLLECTION, reranker=reranker)

    # ── Layer E: Agentic Query ───────────────────────────────────────────
    step("Layer E: Agentic RAG Loop（GPT-4.1 tool calling）")
    from layer_e.agentic_pipeline import AgenticPipeline
    from layer_e.llm_client import GPT41Client

    doc_stem = PDF_PATH.stem  # "乳癌診療指引-2026年"
    ranked = retriever.search_text(QUERY, top_k=5, prefetch_k=20, rerank=True)
    print(f"  [檢索] 取得 {len(ranked)} 筆 ranked results")
    for i, r in enumerate(ranked[:3]):
        print(f"    [{i+1}] page={r.source_pages} score={r.rerank_score:.3f} | {r.display_markdown[:60]}...")

    agentic = AgenticPipeline(
        llm_client=GPT41Client(),
        retriever=retriever,
        pdf_path=str(PDF_PATH),
        doc_stem=doc_stem,
        abstention_threshold=0.0,  # 測試時停用 abstention，讓 agentic loop 運作
    )

    t0 = time.time()
    result = agentic.run(QUERY, ranked)
    elapsed = time.time() - t0

    # ── 結果輸出 ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  結果（耗時 {elapsed:.1f}s）")
    print(f"{'='*60}")
    print(f"\n  abstain:        {result.abstain}")
    print(f"  safety_verdict: {result.safety_verdict}")
    print(f"  tool calls:     {len(result.steps_log)} 次")
    if result.steps_log:
        for s in result.steps_log:
            print(f"    step {s['step_no']}: {s['tool']} — {s.get('reason','')}")
    print(f"\n  ── 答案 ──")
    print(result.answer)
    if result.unsupported_claims:
        print(f"\n  ── 無支持 claims ({len(result.unsupported_claims)}) ──")
        for c in result.unsupported_claims:
            print(f"    [unsupported] {c}")
    print(f"\n  ── Evidence Map ──")
    for eid, meta in result.evidence_map.items():
        print(f"    {eid}: page={meta.get('source_pages')} {meta.get('chunk_id','')[:40]}")

    # ── 輸出 Markdown 報告 ──────────────────────────────────────────────
    out_dir = _ROOT / "output" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "query_result.md"

    lines = [
        f"# RAG Pipeline 回答報告",
        f"",
        f"**問題：** {QUERY}",
        f"",
        f"**文件：** {PDF_PATH.name}",
        f"",
        f"---",
        f"",
        f"## 答案",
        f"",
        result.answer,
        f"",
        f"---",
        f"",
        f"## Evidence Map",
        f"",
    ]
    for eid, meta in result.evidence_map.items():
        pages = meta.get("source_pages", [])
        chunk_id = meta.get("chunk_id", "")
        content = meta.get("content", "")
        lines.append(f"### {eid}  (page {pages})")
        lines.append(f"")
        lines.append(f"**chunk_id:** `{chunk_id}`")
        lines.append(f"")
        lines.append(content)
        lines.append(f"")

    if result.unsupported_claims:
        lines += [
            f"---",
            f"",
            f"## 無支持 Claims",
            f"",
        ]
        for c in result.unsupported_claims:
            lines.append(f"- {c}")
        lines.append(f"")

    lines += [
        f"---",
        f"",
        f"**abstain:** {result.abstain}  ",
        f"**safety_verdict:** {result.safety_verdict}  ",
        f"**tool calls:** {len(result.steps_log)} 次  ",
        f"**耗時:** {elapsed:.1f}s",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  [報告] 已寫入 {out_path}")


if __name__ == "__main__":
    main()
