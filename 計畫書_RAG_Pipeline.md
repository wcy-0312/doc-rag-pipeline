# doc-rag-pipeline 整合計畫書

**版本：** 1.0  
**日期：** 2026-06-16  
**狀態：** 已實作  
**Repo：** `doc-rag-pipeline`

---

## 一、計畫背景與目標

### 1.1 背景

本系統為醫院內部 RAG（Retrieval-Augmented Generation）Pipeline，由五個獨立 Repo 分層開發：

| Layer | 原始 Repo | 職責 |
|-------|-----------|------|
| A — Conversion | `doc-convert-api` | PDF / Word / 照片 → 原生 JSON |
| B — Structure-aware | `doc-structure-layer` | 原生 JSON → RetrievalUnit[] |
| C — Chunk + Embed | `doc-chunk-embed-layer` | RetrievalUnit[] → EmbeddedChunk[]（含 dense vector） |
| D — Retrieval & Index | `doc-retrieval-layer` | EmbeddedChunk[] → Qdrant 索引 → RankedResult[] |
| E — Generation | `doc-generation-layer` | RankedResult[] → grounded answer + citation + safety verdict |

各 Repo 獨立開發、獨立測試，但在實際應用中需要串接為完整 A→E Pipeline。跨 Repo 的串接存在三個工程問題：

1. **Python 命名空間衝突**：各 Repo 皆有 `src/pipeline.py`，同時加入 `sys.path` 會造成模組遮蔽（module shadowing）
2. **序列化邊界複雜**：各層 output 為自訂 dataclass（`RetrievalUnit`、`EmbeddedChunk`、`RankedResult`），跨 Repo pickle 反序列化易出錯
3. **設定分散**：Azure 金鑰、Qdrant endpoint、LLM 路徑散落在各 Repo 的 config.env

### 1.2 目標

1. 建立統一的 Monorepo，將五層邏輯整合為單一可安裝套件集
2. 消除命名空間衝突：各層 package 改為唯一命名（`layer_b`、`layer_c` 等）
3. 提供 `RAGPipeline` 統一入口，一行呼叫完成 ingest 與 query
4. **保持各層獨立可用**：可直接 `from layer_b.pipeline import process_document` 單獨使用任一層
5. 維持原始測試套件，確保各層功能不因整合而退化

### 1.3 設計原則

- **A-layer（extractors）不整合 Flask API server**：extractors 是純 Python 函式，server.py 屬部署細節，不應耦合進 Pipeline 邏輯
- **程式碼完整複製（非 git submodule）**：新 repo 為獨立可運作，不依賴原始 Repo 路徑
- **不新增業務邏輯**：整合層（`pipeline/runner.py`）只負責串接，所有業務邏輯保留在各層

---

## 二、Repo 結構設計

### 2.1 目錄結構

```
doc-rag-pipeline/
├── layer_a/                    # Conversion Layer（extractors only）
│   ├── __init__.py
│   ├── azure_cu_extractor.py   # PDF → JSON（Azure Content Understanding）
│   ├── azure_di_extractor.py   # 照片 → JSON（Azure Document Intelligence）
│   ├── docling_extractor.py    # Word / fallback
│   ├── llm_extractor.py        # Vision LLM path
│   ├── keyword_extractor.py
│   └── metadata_builder.py
│
├── layer_b/                    # Structure-aware Layer
│   ├── __init__.py
│   ├── pipeline.py             # process_document(raw) → list[RetrievalUnit]
│   ├── models.py               # IRTable, RetrievalUnit, etc.
│   ├── adapters/               # 各格式 adapter（azure_cu / azure_di / docling / llm）
│   ├── normalizers/            # merge_cross_page, expand_spans, header_path, confidence
│   └── formatters/             # to_markdown, to_json, linearize_kv
│
├── layer_c/                    # Chunk + Embed Layer
│   ├── __init__.py
│   ├── pipeline.py             # process_and_embed(units, provider) → list[EmbeddedChunk]
│   ├── chunker.py              # retrieval_unit_to_chunks（table / paragraph / figure 三路由）
│   ├── models.py               # EmbeddedChunk
│   └── providers/              # BGEm3Provider, Qwen3EmbedProvider, OpenAIEmbedProvider
│
├── layer_d/                    # Retrieval & Index Layer
│   ├── __init__.py
│   ├── ingestion.py            # DocumentIngester（Qdrant upsert）
│   ├── retrieval.py            # HybridRetriever（dense + sparse RRF）
│   ├── reranker.py             # BGEReranker（cross-encoder）
│   ├── evaluation.py           # Reverse RAG + LLM-as-judge
│   └── models.py               # EmbeddedChunk, RankedResult
│
├── layer_e/                    # Generation Layer
│   ├── __init__.py
│   ├── pipeline.py             # generate(query, ranked) → GenerationResult
│   ├── guardrail.py            # check_abstention, compute_safety_verdict
│   ├── attribution.py          # validate_citations, detect_unsupported_claims
│   ├── context_packer.py       # pack(ranked) → evidence_list, collect_image_paths
│   ├── prompt_builder.py       # build(evidence_list, query)
│   ├── llm_client.py           # Gemma3Client, get_llm_client()
│   └── models.py               # EvidenceItem, ClaimCitation, GenerationResult
│
├── pipeline/                   # 統一 Runner
│   ├── __init__.py
│   └── runner.py               # RAGPipeline（ingest + query）
│
├── tests/
│   ├── test_layer_b/           # 移植自 doc-structure-layer/tests/（113 tests）
│   ├── test_layer_c/           # 移植自 doc-chunk-embed-layer/tests/（15 tests）
│   ├── test_layer_d/           # 移植自 doc-retrieval-layer/tests/（57 tests）
│   └── test_layer_e/           # 移植自 doc-generation-layer/tests/（77 tests）
│
├── output/
│   └── layer_b/                # 測試用 RetrievalUnit JSON（3 份文件）
│
├── config.py                   # Shared Settings（pydantic-settings）
├── config.env.example          # 設定範本
├── conftest.py                 # pytest sys.path 設定
├── pytest.ini
└── requirements.txt
```

### 2.2 命名空間設計

各原始 Repo 的 `src/` 在新 Repo 中重命名為 `layer_x/`：

| 原始 | 新 Repo | 原始 import | 新 import |
|------|---------|-----------|---------|
| `doc-structure-layer/src/` | `layer_b/` | `from src.pipeline import process_document` | `from layer_b.pipeline import process_document` |
| `doc-chunk-embed-layer/src/` | `layer_c/` | `from src.pipeline import process_and_embed` | `from layer_c.pipeline import process_and_embed` |
| `doc-retrieval-layer/src/` | `layer_d/` | `from src.ingestion import DocumentIngester` | `from layer_d.ingestion import DocumentIngester` |
| `doc-generation-layer/src/` | `layer_e/` | `from src.pipeline import generate` | `from layer_e.pipeline import generate` |

Layer A（`doc-convert-api/lib/`）：原始程式碼使用 `sys.path.insert(0, parent.parent)` 指向 `config.py`。新 Repo 中 `config.py` 置於根目錄，`layer_a/` 的相對路徑相同（`parent.parent = doc-rag-pipeline/`），不需修改。

Layer E 使用 Python relative imports（`from .models import ...`），移到 `layer_e/` 後相對路徑不變，無需修改。

### 2.3 pytest 設定

各層有同名測試檔（`test_pipeline.py`、`test_models.py`），pytest 預設 import mode（prepend）無法處理。解法：`pytest.ini` 中設 `--import-mode=importlib`（pytest 6.0+），使 pytest 以 importlib 為每個 test file 產生唯一 module key，避免名稱衝突。

測試目錄命名為 `test_layer_x`（非 `layer_x`），防止 `tests/layer_e/` 遮蔽頂層 `layer_e/` 套件。

---

## 三、各層獨立使用指引

### 3.1 Layer A：Conversion（A-layer standalone）

```python
from layer_a.azure_cu_extractor import convert_pdf_azure_cu

raw = convert_pdf_azure_cu(
    pdf_path="口腔癌指引.pdf",
    category="癌症診療指引",
    output_dir="./output/layer_a",
)
# raw: dict（schema v3.0）
```

所需環境變數（via `config.env` 或 `.env`）：
```
AZURE_CU_ENDPOINT=https://<resource>.services.ai.azure.com/
AZURE_CU_API_KEY=<key>
```

### 3.2 Layer B：Structure-aware（B-layer standalone）

```python
from layer_b.pipeline import process_document

units = process_document(raw)
# units: list[RetrievalUnit]
# unit.retrieval_unit_id, unit.embedding_text, unit.source_pages, ...
```

無需任何外部 API 或模型，純 Python。

### 3.3 Layer C：Chunk + Embed（C-layer standalone）

```python
from layer_c.pipeline import process_and_embed
from layer_c.providers.bge_m3 import BGEm3Provider

provider = BGEm3Provider()  # 需 GPU + FlagEmbedding
chunks = process_and_embed(units, provider)
# chunks: list[EmbeddedChunk], chunk.vector 為 1024-dim dense vector
```

### 3.4 Layer D：Retrieval & Index（D-layer standalone）

```python
from qdrant_client import QdrantClient
from layer_d.ingestion import DocumentIngester
from layer_d.retrieval import HybridRetriever

client = QdrantClient(":memory:")  # 或指定 host/port
ingester = DocumentIngester(client, "my_collection")
ingester.create_collection_if_not_exists()
ingester.ingest(chunks)

retriever = HybridRetriever(client, "my_collection")
ranked = retriever.search_text("治療方式", top_k=5)
# ranked: list[RankedResult]
```

### 3.5 Layer E：Generation（E-layer standalone）

```python
from layer_e.pipeline import generate
from layer_e.llm_client import get_llm_client

result = generate(
    query="第一期口腔癌的治療方式？",
    ranked_results=ranked,
    llm_client=get_llm_client(),
)
# result.answer, result.claims, result.evidence_map, result.safety_verdict
```

---

## 四、統一 Pipeline 使用（RAGPipeline）

```python
from qdrant_client import QdrantClient
from layer_c.providers.bge_m3 import BGEm3Provider
from layer_a.azure_cu_extractor import convert_pdf_azure_cu
from pipeline import RAGPipeline

# 初始化
provider = BGEm3Provider()
client = QdrantClient("localhost", port=6333)
pipeline = RAGPipeline(
    embedding_provider=provider,
    qdrant_client=client,
    collection_name="oral_cancer_guideline",
)

# Ingest（A → B → C → D）
raw = convert_pdf_azure_cu("口腔癌指引.pdf", category="癌症診療指引")
count = pipeline.ingest(raw)
print(f"Ingested {count} chunks")

# Query（D → E）
result = pipeline.query("第一期口腔癌的治療方式？")
print(result.answer)
for claim in result.claims:
    print(f"  {claim.text} [{', '.join(claim.citations)}]")
```

`ingest()` 接受 A-layer extractor 輸出的 raw dict，pipeline 內部依序執行 B → C → D。A-layer 保持獨立呼叫，使外部呼叫方可自由選擇 extractor（azure_cu / azure_di / docling / llm）。

---

## 五、資料流與介面規格

### 5.1 層間資料類型

```
A → B:  dict（schema v3.0，含 extractor_metadata.tool）
B → C:  list[RetrievalUnit]
          - retrieval_unit_id: str
          - source_tool: str（azure_cu / azure_di / docling）
          - embedding_text: str
          - display_markdown: str
          - structured_json: dict
          - source_pages: list[int]
          - page_image_refs: dict[str, str]
          - retrieval_weight: float
C → D:  list[EmbeddedChunk]
          - chunk_id: str
          - chunk_type: str（table / row / paragraph / document / figure）
          - embedding_text: str
          - vector: list[float]（1024-dim, BGE-M3）
          - metadata: dict（含 source_pages, page_image_refs, retrieval_weight）
D → E:  list[RankedResult]
          - chunk_id: str
          - final_score: float（RRF fusion score）
          - rerank_score: float（0.0 if no reranker）
          - display_markdown: str
          - page_image_refs: dict[str, str]
E output: GenerationResult
          - answer: str
          - claims: list[ClaimCitation]
          - evidence_map: dict[str, EvidenceItem]
          - abstain: bool
          - safety_verdict: str（"safe" | "needs_review" | "abstained"）
```

### 5.2 Layer A routing 規則（原始 server.py 邏輯保留於 doc-convert-api）

| 副檔名 | extractor |
|--------|-----------|
| `.pdf` | `DEFAULT_EXTRACTOR`（預設 `azure_cu`，可設 env） |
| `.docx` / `.pptx` / `.xlsx` | `docling`（強制） |
| `.jpg` / `.png` / `.jpeg` / `.tiff` | `azure_di` |
| any（手動指定） | `llm`（vision LLM path） |

---

## 六、各層已知問題與優化狀態

已在整合前確認並修正以下問題（詳見各層計畫書）：

| # | 層 | 問題 | 修正狀態 |
|---|---|------|---------|
| P1 | B | Figure embedding_text = 9 chars，召回率 0% | ✅ 已修正：加入 page context fallback |
| P2 | C | BGE-M3 max_length=512 截斷長表格 | ✅ 已修正：改為 8192 |
| P3 | E | all rerank_score=0.0 → 全部 abstain | ✅ 已修正：偵測無 reranker 情況 |
| P4 | B | Figure area 閾值 2.0 sqin 過濾 80% 圖片 | ✅ 已修正：改為 0.5 sqin |
| P5 | B | 版本記錄段落造成 retrieval noise | ✅ 已修正：`_VERSION_HISTORY_RE` 過濾 |
| P7 | C | Cross-page cosine dedup 誤刪跨頁合法內容 | ✅ 已修正：改為 per-page dedup |
| P9 | E | Empty citations claim 未標記為 unsupported | ✅ 已修正：自動加入 unsupported_claims |

---

## 七、測試覆蓋率

移植後共 **314 個單元測試**通過（不含整合測試 `test_integration_*.py`）：

| 層 | 測試數 | 說明 |
|---|--------|------|
| B | 113 | paragraph path、adapter、normalizer、formatter、pipeline |
| C | 15 | chunker、pipeline（regression chunk count）、providers |
| D | 57 | ingestion、retrieval、reranker、evaluation、e2e |
| E | 77 | guardrail、pipeline、attribution、context_packer、prompt_builder、llm_client |
| **Total** | **314** | |

---

## 八、部署注意事項

### 8.1 環境變數

複製 `config.env.example` 為 `.env`（不進 git），填入 Azure 金鑰與 Qdrant 設定。

```bash
cp config.env.example .env
# 編輯 .env，填入 AZURE_CU_ENDPOINT、AZURE_CU_API_KEY 等
```

### 8.2 依賴安裝

```bash
pip install -r requirements.txt
```

BGE-M3（Layer C）需要 CUDA 環境。若在 CPU 環境，改用 `BGEm3Provider(use_fp16=False)` 或使用 `OpenAIEmbedProvider`。

### 8.3 Qdrant

本地測試可使用 in-memory client（`QdrantClient(":memory:")`）。Production 建議使用 Docker：

```bash
docker run -p 6333:6333 qdrant/qdrant
```

---

## 九、未來升級路徑

| 優先 | 項目 | 所在層 | 說明 |
|------|------|--------|------|
| High | 部署 BGE-Reranker-v2-m3 | D | 消除 rerank_score=0.0 問題，提升排序品質 |
| High | 表格長 chunk 分拆（row-level chunking for >512 tokens） | C | 根本解決截斷問題 |
| Medium | Figure caption 生成（describe_visuals=True） | A | 取代 page context fallback，提供精確圖片語意 |
| Medium | 照片 path 完整測試 | A+B | 現有實測為 lab report，需擴展到更多文件類型 |
| Low | pyproject.toml 各層獨立安裝 | 全層 | 讓各層可 `pip install -e ./layer_c` 單獨使用 |
