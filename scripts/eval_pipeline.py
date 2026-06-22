"""
eval_pipeline.py — 完整 RAG pipeline 自動評估腳本

用法：
    python scripts/eval_pipeline.py --input-dir docs/ --output-dir output/eval/ [--dry-run]

功能：
    對指定目錄下的 PDF / Word / 照片逐一執行 A→E pipeline，
    生成合成問題並以 LLM-as-judge 評估：
      - NDCG@5 (檢索品質)
      - 幻覺率 (unsupported claims)
      - Citation pass rate (格式正確性)
      - Faithfulness (答案忠實度)

    每份文件結果存入 output_dir/<safe_name>/result.json，
    整體摘要存入 output_dir/summary.json。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# 設定 sys.path，讓 scripts/ 目錄下的腳本可以 import 專案模組
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# layer_a 內部使用相對 import（如 metadata_builder），需要加入 layer_a 目錄
_LAYER_A_DIR = str(_PROJECT_ROOT / "layer_a")
if _LAYER_A_DIR not in sys.path:
    sys.path.insert(0, _LAYER_A_DIR)

# ---------------------------------------------------------------------------
# Judge LLM：在 import layer_d.evaluation 之前先 patch JUDGE_MODEL env
# ---------------------------------------------------------------------------
os.environ.setdefault("JUDGE_MODEL", "gpt-4.1")

from azure.identity import DefaultAzureCredential, get_bearer_token_provider  # noqa: E402
from openai import AzureOpenAI  # noqa: E402

from config import settings  # noqa: E402
import layer_d.evaluation as _eval_mod  # noqa: E402

_credential = DefaultAzureCredential()
_token_provider = get_bearer_token_provider(
    _credential, "https://cognitiveservices.azure.com/.default"
)

JUDGE_CLIENT = AzureOpenAI(
    azure_ad_token_provider=_token_provider,
    # settings.gpt41_endpoint 可能帶 /openai/v1 後綴，AzureOpenAI 只需要 base URL
    azure_endpoint=settings.gpt41_endpoint.split("/openai")[0] + "/",
    api_version="2025-01-01-preview",
)
JUDGE_DEPLOYMENT: str = settings.gpt41_deployment  # "gpt-4.1"
_eval_mod.JUDGE_MODEL = JUDGE_DEPLOYMENT  # patch module-level var

# ---------------------------------------------------------------------------
# 共用資源（lazy-load；BGEm3Provider 第一次 embed 才真正載入模型）
# ---------------------------------------------------------------------------
from layer_c.providers.bge_m3 import BGEm3Provider  # noqa: E402
from layer_d.reranker import BGEReranker  # noqa: E402

PROVIDER = BGEm3Provider()

try:
    RERANKER: Optional[BGEReranker] = BGEReranker()
except Exception as _e:
    print(f"[warn] BGEReranker 初始化失敗（{_e}），rerank 功能停用。")
    RERANKER = None

# ---------------------------------------------------------------------------
# Extractor 函式簽名統一包裝
# ---------------------------------------------------------------------------
from layer_a import get_extractor_for_file, get_extractor  # noqa: E402


def _call_extractor(tool: str, file_path: Path, images_dir: Path) -> dict:
    extractor = get_extractor(tool)
    images_dir.mkdir(parents=True, exist_ok=True)

    if tool in {"azure_cu", "azure_di", "docling"}:
        return extractor(file_path)

    else:
        return extractor(str(file_path))


# ---------------------------------------------------------------------------
# Judge LLM 直接呼叫（不走 evaluation.py 的 _get_llm_client()）
# ---------------------------------------------------------------------------

def _judge_call(prompt: str, max_tokens: int = 10) -> str:
    """直接呼叫 Azure OpenAI judge client。"""
    resp = JUDGE_CLIENT.chat.completions.create(
        model=JUDGE_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# 輔助：找出 SyntheticQuery 對應 chunk 的 embedding_text
# ---------------------------------------------------------------------------

def _get_relevant_chunk_text(query, chunks) -> str:
    """依 source_chunk_id 找出對應 chunk 的 embedding_text。"""
    for chunk in chunks:
        if chunk.chunk_id == query.source_chunk_id:
            return chunk.embedding_text[:800]
    # fallback：用第一個 chunk
    if chunks:
        return chunks[0].embedding_text[:800]
    return ""


# ---------------------------------------------------------------------------
# 品質檢查函式
# ---------------------------------------------------------------------------

def _check_citation_pass_rate(gen_results: list) -> float:
    """計算 citation ID 在 evidence_map 中實際存在的比率。"""
    total, passed = 0, 0
    for r in gen_results:
        for claim in r["claims"]:
            for cid in claim["citations"]:
                total += 1
                if cid in r["evidence_map"]:
                    passed += 1
    return passed / total if total > 0 else 1.0


def _compute_hallucination_rate(gen_results: list) -> float:
    """計算 unsupported_claims 佔所有 claims 的比率。"""
    total_claims = sum(len(r["claims"]) for r in gen_results)
    hallucinated = sum(len(r["unsupported_claims"]) for r in gen_results)
    return hallucinated / total_claims if total_claims > 0 else 0.0


def _compute_faithfulness(gen_results: list) -> float:
    """以 Judge LLM 對每條答案的忠實度打分（0-3 均值）。"""
    faithfulness_scores = []
    for r in gen_results:
        if r["abstain"] or not r["answer"]:
            continue
        evidence_text = "\n\n".join(r["evidence_map"].values())[:2000]
        prompt = (
            f"【Evidence】\n{evidence_text}\n\n"
            f"【Answer】\n{r['answer']}\n\n"
            "這個答案是否只使用了 Evidence 中的內容，沒有引入文件以外的知識？\n"
            "評分 0-3：3=完全忠實，2=大致忠實，1=有少量引入外部知識，0=主要依賴外部知識\n"
            "只輸出一個整數（0-3）。"
        )
        score_str = _judge_call(prompt, max_tokens=5)
        m = re.search(r"[0-3]", score_str)
        faithfulness_scores.append(int(m.group()) if m else 0)

    return sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0


# ---------------------------------------------------------------------------
# 目錄名稱 sanitize
# ---------------------------------------------------------------------------

def _sanitize(file_path: Path) -> str:
    """建立安全的目錄名稱：保留中文、英文、數字，替換其餘特殊字元為底線。"""
    name = file_path.stem
    # \w 已包含 ASCII 字母數字底線；加上 CJK Unicode 區段
    safe = re.sub(r'[^\w一-鿿㐀-䶿　-〿＀-￯]', '_', name)
    return safe[:100]


# ---------------------------------------------------------------------------
# 單份文件處理
# ---------------------------------------------------------------------------

def run_file(file_path: Path, output_dir: Path) -> dict:
    """執行 A→E pipeline 並回傳評估結果 dict。

    發生任何 exception 時，記錄到 result["error"] 並回傳（不中斷整體流程）。
    """
    from qdrant_client import QdrantClient
    from layer_b.pipeline import process_document
    from layer_c.pipeline import process_and_embed
    from layer_d.ingestion import DocumentIngester
    from layer_d.retrieval import HybridRetriever
    from layer_d.evaluation import SyntheticQueryGenerator, RelevanceJudge, NDCGEvaluator
    from layer_e.pipeline import GenerationPipeline
    from layer_e.llm_client import Gemma3Client

    result_base: dict = {
        "file": str(file_path.name),
        "file_path": str(file_path),
        "issues_found": [],
    }

    # ── Step 0：Gemma3 endpoint 可達性檢查 ─────────────────────────────────
    import urllib.request
    try:
        urllib.request.urlopen(
            "http://172.31.6.3:8080/gemma3/v1",
            timeout=5,
        )
    except Exception as _conn_err:
        # endpoint 無法連線，不中止，但 LLM 呼叫會失敗時才真正報錯
        # 這裡只是 warn，讓後續流程自然失敗並被 except 捕捉
        pass

    try:
        # ── Step 1：A→E pipeline 建立索引 ──────────────────────────────────
        tool = get_extractor_for_file(file_path)
        images_dir = output_dir / "images"

        raw = _call_extractor(tool, file_path, images_dir)

        units = process_document(raw)
        chunks = process_and_embed(units, PROVIDER)

        qdrant = QdrantClient(":memory:")
        collection_name = "eval_doc"
        ingester = DocumentIngester(client=qdrant, collection_name=collection_name)
        ingester.create_collection_if_not_exists()
        ingested_count = ingester.ingest(chunks)

        if ingested_count == 0:
            r = {
                **result_base,
                "tool": tool,
                "warning": "沒有任何 chunk 被 ingest，跳過此文件",
                "ingested_chunks": 0,
            }
            return r

        use_rerank = RERANKER is not None
        retriever = HybridRetriever(
            client=qdrant,
            collection_name=collection_name,
            reranker=RERANKER,
        )

        # abstention_threshold=0.0 讓評估時不因 score 低而 abstain
        gen_pipeline = GenerationPipeline(
            llm_client=Gemma3Client(),
            abstention_threshold=0.0,
        )

        # ── Step 2：生成問題集並驗證 ───────────────────────────────────────
        sq_gen = SyntheticQueryGenerator(llm_client=JUDGE_CLIENT)
        candidates = sq_gen.generate(chunks, target_count=8)

        valid_questions = []
        for q in candidates:
            chunk_text = _get_relevant_chunk_text(q, chunks)
            prompt = (
                f"以下是從文件中提取的內容：\n"
                f"{chunk_text}\n\n"
                f"問題：{q.query_text}\n\n"
                "這個問題能否從以上文件內容中找到明確答案？只回答 yes 或 no。"
            )
            answer = _judge_call(prompt, max_tokens=10)
            if answer.lower().strip().startswith("yes"):
                valid_questions.append(q)

        if len(valid_questions) < 3:
            r = {
                **result_base,
                "tool": tool,
                "ingested_chunks": ingested_count,
                "warning": f"有效問題不足 3 題（只有 {len(valid_questions)} 題），跳過",
                "questions_generated": len(candidates),
                "questions_valid": len(valid_questions),
            }
            return r

        # ── Step 3：執行 query() 並收集結果 ──────────────────────────────
        gen_results = []
        for q in valid_questions:
            ranked = retriever.search_text(
                q.query_text, top_k=5, prefetch_k=20, rerank=use_rerank
            )
            result_obj = gen_pipeline.run(q.query_text, ranked)
            gen_results.append({
                "query": q.query_text,
                "answer": result_obj.answer,
                "claims": [
                    {"text": c.text, "citations": c.citations}
                    for c in result_obj.claims
                ],
                "evidence_map": {
                    k: (
                        v.get("display_markdown", "")
                        if isinstance(v, dict)
                        else str(v)
                    )
                    for k, v in result_obj.evidence_map.items()
                },
                "unsupported_claims": result_obj.unsupported_claims,
                "abstain": result_obj.abstain,
                "ranked_results": ranked,
            })

        # ── Step 4：品質檢查 ──────────────────────────────────────────────
        # (a) Citation 格式驗證
        citation_pass_rate = _check_citation_pass_rate(gen_results)

        # (b) 幻覺率（直接使用 generate() 內部已呼叫 detect_unsupported_claims 的結果）
        hallucination_rate = _compute_hallucination_rate(gen_results)

        # (c) 答案忠實度
        faithfulness_avg = _compute_faithfulness(gen_results)

        # (d) NDCG@5
        judge = RelevanceJudge(llm_client=JUDGE_CLIENT)
        evaluator = NDCGEvaluator(judge=judge)
        eval_result = evaluator.evaluate(
            valid_questions, retriever, reranker=None, top_k=5
        )
        ndcg_at_5 = eval_result.ndcg_at_10  # evaluate() 使用 top_k=5，回傳值存在 ndcg_at_10

        # ── Step 5：判斷 issues ──────────────────────────────────────────
        issues = []
        if citation_pass_rate < 1.0:
            issues.append({
                "type": "citation_missing",
                "layer": "E",
                "value": citation_pass_rate,
            })
        if hallucination_rate > 0.1:
            issues.append({
                "type": "hallucination",
                "layer": "E",
                "value": hallucination_rate,
            })
        if faithfulness_avg < 2.0:
            issues.append({
                "type": "low_faithfulness",
                "layer": "E",
                "value": faithfulness_avg,
            })
        if ndcg_at_5 < 0.7:
            issues.append({
                "type": "low_ndcg",
                "layer": "D",
                "value": ndcg_at_5,
            })

        # ── Step 6：組合 result ──────────────────────────────────────────
        r = {
            "file": str(file_path.name),
            "file_path": str(file_path),
            "tool": tool,
            "ingested_chunks": ingested_count,
            "questions_generated": len(candidates),
            "questions_valid": len(valid_questions),
            "ndcg_at_5": ndcg_at_5,
            "hallucination_rate": hallucination_rate,
            "citation_pass_rate": citation_pass_rate,
            "faithfulness_avg": faithfulness_avg,
            "issues_found": issues,
            "layers_modified": [],
            "answers": [
                {
                    "query": r2["query"],
                    "answer": r2["answer"],
                    "abstain": r2["abstain"],
                }
                for r2 in gen_results
            ],
        }

        # 存到 output_dir/result.json
        (output_dir / "result.json").write_text(
            json.dumps(r, ensure_ascii=False, indent=2)
        )
        return r

    except Exception as e:
        import traceback
        r = {
            **result_base,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        (output_dir / "result.json").write_text(
            json.dumps(r, ensure_ascii=False, indent=2)
        )
        return r


# ---------------------------------------------------------------------------
# 摘要計算
# ---------------------------------------------------------------------------

def _compute_summary(results: list) -> dict:
    valid = [
        r for r in results
        if "error" not in r and "warning" not in r
    ]
    return {
        "total_files": len(results),
        "completed": len(valid),
        "errors": len([r for r in results if "error" in r]),
        "warnings": len([r for r in results if "warning" in r and "error" not in r]),
        "avg_ndcg_at_5": (
            sum(r.get("ndcg_at_5", 0) for r in valid) / len(valid)
            if valid else 0.0
        ),
        "avg_hallucination_rate": (
            sum(r.get("hallucination_rate", 0) for r in valid) / len(valid)
            if valid else 0.0
        ),
        "avg_faithfulness": (
            sum(r.get("faithfulness_avg", 0) for r in valid) / len(valid)
            if valid else 0.0
        ),
        "files_with_issues": [
            r["file"] for r in valid if r.get("issues_found")
        ],
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

_SUPPORTED_SUFFIXES = {
    ".pdf", ".docx", ".doc", ".jpg", ".jpeg", ".png"
}


def run_all(
    input_dir: Path,
    output_dir: Path,
    resume: bool = True,
    dry_run: bool = False,
) -> dict:
    files = sorted(
        f for f in input_dir.rglob("*")
        if f.suffix.lower() in _SUPPORTED_SUFFIXES
    )

    if dry_run:
        print(f"[dry-run] 共找到 {len(files)} 個支援的文件：")
        for f in files:
            print(f"  {f.relative_to(input_dir)}")
        return {"total_files": len(files), "dry_run": True}

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    results: list = []

    for i, f in enumerate(files):
        safe_name = _sanitize(f)
        # 加上原始副檔名避免不同格式的相同 stem 互相覆蓋
        dir_name = f"{safe_name}{f.suffix.lower().replace('.', '_')}"
        file_out = output_dir / dir_name
        file_out.mkdir(parents=True, exist_ok=True)
        result_path = file_out / "result.json"

        if resume and result_path.exists():
            print(f"[{i+1}/{len(files)}] 跳過（已完成）: {f.name}")
            try:
                results.append(json.loads(result_path.read_text()))
            except Exception:
                pass
            continue

        print(f"[{i+1}/{len(files)}] 處理: {f.name}")
        try:
            result = run_file(f, file_out)
        except Exception as e:
            import traceback
            result = {
                "file": f.name,
                "file_path": str(f),
                "error": str(e),
                "traceback": traceback.format_exc(),
                "issues_found": [],
            }
            result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))

        results.append(result)

        # 每份文件完成後立即更新 summary
        summary = _compute_summary(results)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

        if "error" in result:
            print(f"   ERROR: {result['error']}")
        elif "warning" in result:
            print(f"   WARN: {result['warning']}")
        else:
            ndcg_val = result.get("ndcg_at_5", 0.0)
            hall_val = result.get("hallucination_rate", 0.0)
            issues_count = len(result.get("issues_found", []))
            print(
                f"   NDCG@5={ndcg_val:.3f}  "
                f"幻覺率={hall_val:.1%}  "
                f"issues={issues_count}"
            )

    return _compute_summary(results)


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG pipeline 自動評估腳本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="輸入目錄（含 PDF / Word / 照片）",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="評估結果輸出目錄",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="只列出要處理的文件，不實際執行",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        default=False,
        help="不跳過已有 result.json 的文件（預設 resume=True）",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    resume = not args.no_resume

    if not input_dir.exists():
        print(f"[error] input-dir 不存在：{input_dir}")
        sys.exit(1)

    print(f"input_dir  : {input_dir}")
    print(f"output_dir : {output_dir}")
    print(f"resume     : {resume}")
    print(f"dry_run    : {args.dry_run}")
    print(f"judge      : {JUDGE_DEPLOYMENT} @ {settings.gpt41_endpoint}")
    print()

    summary = run_all(
        input_dir=input_dir,
        output_dir=output_dir,
        resume=resume,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        print()
        print("=== 評估完成 ===")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
