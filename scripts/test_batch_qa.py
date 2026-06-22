"""
批次 QA 測試：乳癌診療指引-2026年.pdf
分批執行並輸出 Markdown 報告

執行：
    conda run -n hospital-rag python3 scripts/test_batch_qa.py

輸出：
    output/eval/batch_qa_乳癌診療指引.md
"""
from __future__ import annotations

import dataclasses
import json
import re as _re
import sys
import time
from pathlib import Path


def _build_refs(claims: list, evidence_map: dict, summaries: dict) -> list:
    """按 claims[].citations 出現順序收集被引用的 evidence。
    回傳 [(編號, eid, meta, summary), ...]
    """
    seen: dict[str, int] = {}
    ordered: list[tuple[int, str]] = []
    for claim in claims:
        for eid in claim.get("citations", []):
            if eid not in seen and eid in evidence_map:
                seen[eid] = len(seen) + 1
                ordered.append((seen[eid], eid))
    refs = []
    for num, eid in ordered:
        meta = evidence_map[eid]
        summary = summaries.get(eid, "").strip()
        if not summary:
            summary = (meta.get("content") or "")[:120]
        refs.append((num, eid, meta, summary))
    return refs

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# ── 問題清單 ─────────────────────────────────────────────────────────────────

BATCHES = [
    {
        "name": "批次 A：基礎查找",
        "questions": [
            "HER2陽性早期乳癌，術後輔助治療建議使用哪些靶向藥物？",
            "乳癌亞型中的「Luminal A」如何定義？需滿足哪些條件？",
            "保乳手術後全乳放射治療（whole breast irradiation）的標準劑量與分次數是多少？",
        ],
    },
    {
        "name": "批次 B：多段落推理",
        "questions": [
            "cT3N2M0 的 HER2 陽性乳癌，術前是否建議先做新輔助治療（neoadjuvant therapy）？",
            "CDK4/6 抑制劑在轉移性乳癌的使用適應症是什麼？應限定哪種乳癌亞型？",
            "乳癌前哨淋巴結切片（sentinel lymph node biopsy）的適應症與禁忌症為何？",
        ],
    },
    {
        "name": "批次 C：跨章節推理",
        "questions": [
            "BRCA1/2 胚系突變的三陰性乳癌（TNBC），指引中有哪些額外治療選項？",
            "炎性乳癌（inflammatory breast cancer）的初始治療原則是什麼？和一般乳癌有何不同？",
            "停經後 ER+/HER2- 的 pT2N1M0 病人，術後輔助治療應優先選擇內分泌治療還是化療？依據指引的判斷條件是什麼？",
        ],
    },
    {
        "name": "壓軸：邊界測試",
        "questions": [
            "乳癌病人懷孕時，化療對胎兒的安全性數據為何？",
            "如果病人是cT2N1M0，是否能根據指引給出治療建議?",
        ],
    },
]

# ── PDF / 快取路徑 ─────────────────────────────────────────────────────────

def _find_pdf() -> Path:
    d = _ROOT / "docs" / "癌症診療指引"
    for f in d.iterdir():
        if "乳癌" in f.name and f.suffix == ".pdf":
            return f
    raise FileNotFoundError(f"找不到乳癌 PDF in {d}")

PDF_PATH = _find_pdf()
_STEM = PDF_PATH.stem
_DOC_STEM = _re.sub(r'[^\w\-]', '_', _STEM) if _STEM else "doc"

RAW_CACHE     = _ROOT / "output" / "layer_b" / f"raw_{_STEM}.json"
LB_CACHE      = _ROOT / "output" / "layer_b" / f"retrieval_units_{_STEM}.json"
ENRICHED_CACHE= _ROOT / "output" / "layer_b" / f"retrieval_units_{_STEM}_enriched.json"
OUT_DIR       = _ROOT / "output" / "eval"
OUT_PATH      = OUT_DIR / f"batch_qa_{_STEM}.md"

COLLECTION = _STEM


def step(msg: str):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


# ── Layer A ───────────────────────────────────────────────────────────────────

def load_raw() -> dict:
    if RAW_CACHE.exists():
        print(f"  [快取] Layer A: {RAW_CACHE.name}")
        return json.loads(RAW_CACHE.read_text())
    step("Layer A: Azure CU 萃取（需 1–3 分鐘）")
    sys.path.insert(0, str(_ROOT / "layer_a"))
    from layer_a import get_extractor
    raw = get_extractor("azure_cu")(PDF_PATH)
    RAW_CACHE.parent.mkdir(parents=True, exist_ok=True)
    RAW_CACHE.write_text(json.dumps(raw, ensure_ascii=False, indent=2))
    return raw


# ── Layer B ───────────────────────────────────────────────────────────────────

def _deserialize_units(records: list) -> list:
    from layer_b.models import RetrievalUnit
    _valid = {f.name for f in dataclasses.fields(RetrievalUnit)}
    return [RetrievalUnit(**{k: v for k, v in u.items() if k in _valid}) for u in records]


def load_units() -> list:
    # 1. Enriched cache hit — fastest path
    if ENRICHED_CACHE.exists():
        print(f"  [快取] Layer B (enriched): {ENRICHED_CACHE.name}")
        return _deserialize_units(json.loads(ENRICHED_CACHE.read_text()))

    # 2. Base units: load from cache or rebuild from Layer B
    if LB_CACHE.exists():
        print(f"  [快取] Layer B: {LB_CACHE.name}")
        base_units = _deserialize_units(json.loads(LB_CACHE.read_text()))
    else:
        step("Layer B: 結構化 RetrievalUnit")
        from layer_b.pipeline import process_document
        raw = load_raw()
        base_units = process_document(raw)
        print(f"  [完成] {len(base_units)} 個 RetrievalUnit")
        LB_CACHE.parent.mkdir(parents=True, exist_ok=True)
        LB_CACHE.write_text(
            json.dumps([dataclasses.asdict(u) for u in base_units], ensure_ascii=False, indent=2)
        )

    # 3. Run enrichment (table + figure only) and cache result
    step("Layer B: Enrichment（table / figure）")
    from layer_b.enrichment import enrich_units
    from layer_e.llm_client import GPT41Client
    t0 = time.time()
    enriched = enrich_units(base_units, GPT41Client())
    enriched_count = sum(
        1 for a, b in zip(base_units, enriched) if a.embedding_text != b.embedding_text
    )
    print(f"  [完成] {enriched_count} 個 unit 已 enrich，耗時 {time.time()-t0:.1f}s")
    ENRICHED_CACHE.write_text(
        json.dumps([dataclasses.asdict(u) for u in enriched], ensure_ascii=False, indent=2)
    )
    return enriched


# ── Layer C ───────────────────────────────────────────────────────────────────

def embed_units(units: list):
    step("Layer C: BGE-M3 Embedding")
    from layer_c.providers.bge_m3 import BGEm3Provider
    from layer_c.pipeline import process_and_embed
    t0 = time.time()
    chunks = process_and_embed(units, BGEm3Provider())
    print(f"  [完成] {len(chunks)} 個 EmbeddedChunk，耗時 {time.time()-t0:.1f}s")
    return chunks


# ── Layer D ───────────────────────────────────────────────────────────────────

def build_qdrant(chunks, raw: dict):
    step("Layer D: In-memory Qdrant")
    from qdrant_client import QdrantClient
    from layer_d.ingestion import DocumentIngester
    from layer_d.retrieval import HybridRetriever
    from layer_d.reranker import BGEReranker
    from layer_b.pipeline import extract_document_index

    qdrant = QdrantClient(":memory:")
    ingester = DocumentIngester(client=qdrant, collection_name=COLLECTION)
    ingester.create_collection_if_not_exists()
    n = ingester.ingest(chunks)
    print(f"  [完成] {n} 個 chunk ingest")

    doc_index = extract_document_index(raw)
    if doc_index:
        ingester.store_document_index(_DOC_STEM, doc_index)
        sections = doc_index.get("sections", [])
        print(f"  [document_index] 已儲存，頂層章節 {len(sections)} 個")
    else:
        print("  [document_index] 無 sections[]，未儲存")

    retriever = HybridRetriever(
        client=qdrant, collection_name=COLLECTION, reranker=BGEReranker()
    )
    return retriever


# ── Layer E（單題）────────────────────────────────────────────────────────────

def run_question(query: str, retriever, idx: int) -> dict:
    from layer_e.agentic_pipeline import AgenticPipeline
    from layer_e.llm_client import GPT41Client

    ranked = retriever.search_text(query, top_k=5, prefetch_k=20, rerank=True)
    agentic = AgenticPipeline(
        llm_client=GPT41Client(),
        retriever=retriever,
        pdf_path=str(PDF_PATH),
        doc_stem=_DOC_STEM,
    )
    t0 = time.time()
    result = agentic.run(query, ranked)
    elapsed = time.time() - t0

    print(f"  Q{idx}: {'abstain' if result.abstain else '✅'} | "
          f"{len(result.steps_log)} tool calls | {elapsed:.1f}s")

    return {
        "query": query,
        "answer": result.answer,
        "abstain": result.abstain,
        "safety_verdict": result.safety_verdict,
        "tool_calls": len(result.steps_log),
        "elapsed": elapsed,
        "claims": [{"text": c.text, "citations": c.citations} for c in result.claims],
        "evidence_map": result.evidence_map,
        "evidence_summaries": result.evidence_summaries,
        "top_hits": [
            {
                "rank": i + 1,
                "pages": r.source_pages,
                "score": round(r.rerank_score, 3),
                "preview": r.display_markdown[:80],
            }
            for i, r in enumerate(ranked[:3])
        ],
    }


# ── Markdown 輸出 ─────────────────────────────────────────────────────────────

def write_md(batch_results: list[dict], total_elapsed: float):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# 批次 QA 測試報告",
        f"",
        f"**文件：** {PDF_PATH.name}  ",
        f"**總耗時：** {total_elapsed:.1f}s  ",
        f"**題數：** {sum(len(b['results']) for b in batch_results)} 題",
        f"",
        f"---",
        f"",
    ]

    q_idx = 0
    for batch in batch_results:
        lines += [f"## {batch['name']}", f""]
        for res in batch["results"]:
            q_idx += 1
            abstain_tag = " *（無法回答）*" if res["abstain"] else ""

            if res["abstain"]:
                lines += [
                    f"### Q{q_idx}：{res['query']}{abstain_tag}",
                    f"",
                    f"*系統判定資訊不足，拒絕回答。原因：{res.get('abstain_reason', '')}*",
                    f"",
                    f"---",
                    f"",
                ]
                continue

            summaries = res.get("evidence_summaries", {})
            refs = _build_refs(res.get("claims", []), res["evidence_map"], summaries)

            lines += [
                f"### Q{q_idx}：{res['query']}",
                f"",
                res["answer"],
                f"",
            ]

            for num, eid, meta, summary in refs:
                pages = meta.get("source_pages", [])
                source_doc = meta.get("source_doc") or meta.get("chunk_id", "").split("_p_")[0].split("_t_")[0]
                pages_str = "、".join(f"第{p}頁" for p in pages) if pages else ""
                source_str = f"來源：{source_doc} {pages_str}".strip()
                lines.append(f"[{num}] {summary} ({source_str})")

            lines += [f"", f"---", f""]

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  [報告] 已寫入 {OUT_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n  PDF:  {PDF_PATH}")
    print(f"  出題: {sum(len(b['questions']) for b in BATCHES)} 題（{len(BATCHES)} 批）")

    raw   = load_raw()
    units = load_units()
    chunks = embed_units(units)
    retriever = build_qdrant(chunks, raw)

    total_t0 = time.time()
    batch_results = []
    q_idx = 0

    for batch in BATCHES:
        step(batch["name"])
        results = []
        for q in batch["questions"]:
            q_idx += 1
            print(f"\n  [{q_idx}] {q}")
            res = run_question(q, retriever, q_idx)
            results.append(res)
        batch_results.append({"name": batch["name"], "results": results})

    write_md(batch_results, time.time() - total_t0)


if __name__ == "__main__":
    main()
