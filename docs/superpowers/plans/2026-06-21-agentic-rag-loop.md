# Agentic RAG Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 將 Layer E 的 one-shot generation 改為 agentic loop，讓 GPT-4.1 能透過 tool calling 主動請求 PDF 頁面截圖或追加檢索，解決流程圖、跨 chunk 推理無法回答的問題。

**Architecture:** 查詢時先做標準 hybrid retrieval（現有機制），取得 top-k chunks 作為初始 evidence。接著進入 agentic loop：GPT-4.1 以 tool calling 決定是否要截圖某頁或再搜尋文件；每次 tool 執行結果加進對話歷史，直到 LLM 輸出最終答案（所有 claim 有 citation）或達到軟上限為止。

**Tech Stack:** PyMuPDF (fitz)、OpenAI Python SDK（Azure）、Qdrant（已有）、pytest

## Global Constraints

- Python 3.11，conda env `hospital-rag`
- 所有新 module 放在 `layer_e/` 下；測試放在 `tests/test_layer_e/`
- Agentic loop 只用 `GPT41Client`；Gemma3/4 不支援 tool calling（vLLM 尚未啟用），保持原有 `generate()` 路徑不動
- `layer_e/pipeline.py` 的 `generate()` / `GenerationPipeline` 維持不變（非 agentic 查詢繼續用原有路徑）
- Temperature 固定 `0.0`（已有）
- `page_image_refs` 格式不在此次變更範圍，agentic loop 使用 `source_pages` 決定截哪頁
- 軟上限 `soft_limit=8` tool calls；硬上限 `hard_limit=12` iterations
- 測試不得呼叫真實 API（mock GPT41Client 與 retriever）

---

## File Map

| 狀態 | 路徑 | 職責 |
|------|------|------|
| 新增 | `layer_e/pdf_tools.py` | PDF 頁面截圖，回傳 base64 PNG |
| 新增 | `layer_e/agentic_tools.py` | Tool 定義（OpenAI format）+ 執行器 |
| 新增 | `layer_e/agentic_pipeline.py` | Agentic loop 主邏輯 |
| 修改 | `layer_e/llm_client.py` | `LLMClient` + `GPT41Client` + `_StubLLMClient` 加 `generate_with_tools` |
| 修改 | `pipeline/runner.py` | `RAGPipeline` 加 `query_agentic` |
| 新增 | `tests/test_layer_e/test_pdf_tools.py` | |
| 新增 | `tests/test_layer_e/test_agentic_tools.py` | |
| 新增 | `tests/test_layer_e/test_agentic_pipeline.py` | |
| 修改 | `tests/test_layer_e/test_llm_client.py` | 加 `generate_with_tools` 測試 |

---

## Task 1: PDF Screenshot Utility

**Files:**
- Create: `layer_e/pdf_tools.py`
- Create: `tests/test_layer_e/test_pdf_tools.py`

**Interfaces:**
- Produces: `get_full_page_image(pdf_path: str, page_no: int, dpi: int = 150) -> str`  → base64 PNG 字串

---

- [ ] **Step 1: 寫失敗測試**

```python
# tests/test_layer_e/test_pdf_tools.py
import base64
import pytest
import fitz
from layer_e.pdf_tools import get_full_page_image


@pytest.fixture
def sample_pdf(tmp_path):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Test page 1")
    pdf_path = str(tmp_path / "sample.pdf")
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_returns_valid_base64_png(sample_pdf):
    result = get_full_page_image(sample_pdf, page_no=1)
    assert isinstance(result, str)
    decoded = base64.b64decode(result)
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


def test_invalid_page_raises(sample_pdf):
    with pytest.raises(ValueError, match="out of range"):
        get_full_page_image(sample_pdf, page_no=999)


def test_zero_page_raises(sample_pdf):
    with pytest.raises(ValueError):
        get_full_page_image(sample_pdf, page_no=0)


def test_missing_file_raises():
    with pytest.raises(Exception):
        get_full_page_image("/nonexistent/file.pdf", page_no=1)
```

- [ ] **Step 2: 確認測試失敗**

```bash
conda run -n hospital-rag pytest tests/test_layer_e/test_pdf_tools.py -v
```
Expected: `ImportError: cannot import name 'get_full_page_image'`

- [ ] **Step 3: 實作**

```python
# layer_e/pdf_tools.py
import base64
import fitz


def get_full_page_image(pdf_path: str, page_no: int, dpi: int = 150) -> str:
    """Return base64-encoded PNG of a single PDF page (1-based page_no)."""
    doc = fitz.open(pdf_path)
    try:
        if page_no < 1 or page_no > len(doc):
            raise ValueError(f"page_no {page_no} out of range 1..{len(doc)}")
        pixmap = doc[page_no - 1].get_pixmap(dpi=dpi)
        png_bytes = pixmap.tobytes("png")
    finally:
        doc.close()
    return base64.b64encode(png_bytes).decode()
```

- [ ] **Step 4: 確認測試通過**

```bash
conda run -n hospital-rag pytest tests/test_layer_e/test_pdf_tools.py -v
```
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add layer_e/pdf_tools.py tests/test_layer_e/test_pdf_tools.py
git commit -m "feat(layer_e): add pdf_tools.get_full_page_image for on-demand page screenshots"
```

---

## Task 2: GPT41Client Tool Calling

**Files:**
- Modify: `layer_e/llm_client.py`
- Modify: `tests/test_layer_e/test_llm_client.py`

**Interfaces:**
- Consumes: 無新依賴
- Produces: `LLMClient.generate_with_tools(messages: list, tools: list) -> tuple[list, str | None]`
  - 回傳 `(tool_calls, None)` 若 LLM 要呼叫工具
  - 回傳 `([], content_str)` 若 LLM 給最終答案
  - `tool_calls` 格式：`[{"id": str, "name": str, "arguments": dict}]`

---

- [ ] **Step 1: 寫失敗測試**

在 `tests/test_layer_e/test_llm_client.py` 末尾加入：

```python
def test_stub_generate_with_tools_returns_final_answer():
    import json
    stub = _StubLLMClient()
    tool_calls, content = stub.generate_with_tools(
        messages=[{"role": "user", "content": "test"}],
        tools=[],
    )
    assert tool_calls == []
    assert content is not None
    parsed = json.loads(content)
    assert "answer" in parsed
    assert "claims" in parsed


def test_stub_generate_with_tools_no_tool_calls():
    stub = _StubLLMClient()
    tool_calls, _ = stub.generate_with_tools(messages=[], tools=[])
    assert isinstance(tool_calls, list)
    assert len(tool_calls) == 0
```

- [ ] **Step 2: 確認測試失敗**

```bash
conda run -n hospital-rag pytest tests/test_layer_e/test_llm_client.py::test_stub_generate_with_tools_returns_final_answer -v
```
Expected: `AttributeError: '_StubLLMClient' object has no attribute 'generate_with_tools'`

- [ ] **Step 3: 實作**

修改 `layer_e/llm_client.py`，在三個地方加入 `generate_with_tools`：

**a) `LLMClient` 抽象類別加預設實作（非抽象，子類可覆寫）：**

```python
class LLMClient(abc.ABC):
    @abc.abstractmethod
    def generate(self, system: str, user: str) -> dict:
        ...

    def generate_multimodal(self, messages: list) -> dict:
        # ... 現有程式碼不動 ...

    def generate_with_tools(self, messages: list, tools: list) -> tuple:
        """Override in subclasses that support function calling.

        Returns:
            (tool_calls, None)  — LLM wants to call tools
            ([], content_str)   — LLM gives final answer

        tool_calls: list of {"id": str, "name": str, "arguments": dict}
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support tool calling. Use GPT41Client."
        )
```

**b) `_StubLLMClient` 加覆寫：**

```python
class _StubLLMClient(LLMClient):
    def generate(self, system: str, user: str) -> dict:
        # ... 現有程式碼不動 ...

    def generate_with_tools(self, messages: list, tools: list) -> tuple:
        import json
        content = json.dumps({
            "answer": "stub",
            "claims": [{"text": "stub claim", "citations": ["E1"]}],
            "abstain": False,
            "abstain_reason": None,
        })
        return ([], content)
```

**c) `GPT41Client` 加覆寫：**

在 `GPT41Client` class 的 `generate` 方法後加入：

```python
    def generate_with_tools(self, messages: list, tools: list) -> tuple:
        response = self._client.chat.completions.create(
            model=self._deployment,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.0,
        )
        msg = response.choices[0].message
        if msg.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                }
                for tc in msg.tool_calls
            ]
            return (tool_calls, None)
        return ([], msg.content)
```

- [ ] **Step 4: 確認測試通過**

```bash
conda run -n hospital-rag pytest tests/test_layer_e/test_llm_client.py -v
```
Expected: 全部通過（包含原有 8 個測試 + 新增 2 個）

- [ ] **Step 5: Commit**

```bash
git add layer_e/llm_client.py tests/test_layer_e/test_llm_client.py
git commit -m "feat(layer_e): add generate_with_tools to LLMClient, _StubLLMClient, GPT41Client"
```

---

## Task 3: Agentic Tool Definitions & Executors

**Files:**
- Create: `layer_e/agentic_tools.py`
- Create: `tests/test_layer_e/test_agentic_tools.py`

**Interfaces:**
- Consumes: `get_full_page_image` from `layer_e.pdf_tools`
- Produces:
  - `TOOL_DEFINITIONS: list[dict]` — OpenAI function 格式的工具列表
  - `execute_tool(tool_call: dict, pdf_path: str, retriever, doc_stem: str) -> tuple[str, str | None]`
    - 回傳 `(text_result, base64_png_or_None)`

---

- [ ] **Step 1: 寫失敗測試**

```python
# tests/test_layer_e/test_agentic_tools.py
import json
import pytest
from unittest.mock import MagicMock, patch
from layer_e.agentic_tools import TOOL_DEFINITIONS, execute_tool


def test_tool_definitions_valid_openai_format():
    assert isinstance(TOOL_DEFINITIONS, list)
    assert len(TOOL_DEFINITIONS) == 2
    names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
    assert names == {"get_page_image", "retrieve_more"}
    for t in TOOL_DEFINITIONS:
        assert t["type"] == "function"
        fn = t["function"]
        assert "name" in fn
        assert "description" in fn
        assert "parameters" in fn
        assert "required" in fn["parameters"]


def test_execute_get_page_image(tmp_path):
    import fitz
    doc = fitz.open()
    doc.new_page()
    pdf_path = str(tmp_path / "test.pdf")
    doc.save(pdf_path)
    doc.close()

    tool_call = {"id": "call_1", "name": "get_page_image", "arguments": {"page_no": 1, "reason": "test"}}
    text, b64 = execute_tool(tool_call, pdf_path=pdf_path, retriever=None, doc_stem="test")
    assert "1" in text
    assert b64 is not None
    import base64
    decoded = base64.b64decode(b64)
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


def test_execute_retrieve_more():
    mock_result = MagicMock()
    mock_result.source_pages = [5]
    mock_result.display_markdown = "相關段落內容"

    mock_retriever = MagicMock()
    mock_retriever.search_text.return_value = [mock_result]

    tool_call = {"id": "call_2", "name": "retrieve_more", "arguments": {"query": "T2N1 分期", "reason": "test"}}
    text, b64 = execute_tool(tool_call, pdf_path="", retriever=mock_retriever, doc_stem="doc")
    assert "相關段落內容" in text
    assert b64 is None
    mock_retriever.search_text.assert_called_once_with(
        "T2N1 分期", top_k=3, doc_ids=["doc"], rerank=False
    )


def test_execute_retrieve_more_empty():
    mock_retriever = MagicMock()
    mock_retriever.search_text.return_value = []

    tool_call = {"id": "call_3", "name": "retrieve_more", "arguments": {"query": "不存在的內容", "reason": "test"}}
    text, b64 = execute_tool(tool_call, pdf_path="", retriever=mock_retriever, doc_stem="doc")
    assert "未找到" in text
    assert b64 is None


def test_execute_unknown_tool():
    tool_call = {"id": "call_x", "name": "nonexistent", "arguments": {}}
    text, b64 = execute_tool(tool_call, pdf_path="", retriever=None, doc_stem="doc")
    assert "未知工具" in text
    assert b64 is None
```

- [ ] **Step 2: 確認測試失敗**

```bash
conda run -n hospital-rag pytest tests/test_layer_e/test_agentic_tools.py -v
```
Expected: `ImportError: cannot import name 'TOOL_DEFINITIONS'`

- [ ] **Step 3: 實作**

```python
# layer_e/agentic_tools.py
from layer_e.pdf_tools import get_full_page_image

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_page_image",
            "description": "取得 PDF 指定頁碼的整頁截圖，適合閱讀流程圖、算法圖表、治療路徑圖等視覺內容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "page_no": {
                        "type": "integer",
                        "description": "頁碼（從 1 開始）",
                    },
                    "reason": {
                        "type": "string",
                        "description": "為何需要查看此頁（audit log 用）",
                    },
                },
                "required": ["page_no", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_more",
            "description": "在文件中搜尋與問題相關的更多段落或表格。當初始 evidence 缺少某個關鍵資訊時使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜尋關鍵字或語句",
                    },
                    "reason": {
                        "type": "string",
                        "description": "為何需要搜尋此內容（audit log 用）",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "傳回結果數量（預設 3）",
                        "default": 3,
                    },
                },
                "required": ["query", "reason"],
            },
        },
    },
]


def execute_tool(
    tool_call: dict,
    pdf_path: str,
    retriever,
    doc_stem: str,
) -> tuple:
    """Execute a single tool call.

    Returns:
        (text_result, None)         for retrieve_more
        (text_result, base64_png)   for get_page_image
    """
    name = tool_call["name"]
    args = tool_call["arguments"]

    if name == "get_page_image":
        page_no = int(args["page_no"])
        b64 = get_full_page_image(pdf_path, page_no)
        return f"已截取第 {page_no} 頁截圖。", b64

    if name == "retrieve_more":
        top_k = int(args.get("top_k", 3))
        results = retriever.search_text(
            args["query"],
            top_k=top_k,
            doc_ids=[doc_stem],
            rerank=False,
        )
        if not results:
            return "未找到相關段落。", None
        lines = []
        for i, r in enumerate(results, start=1):
            pages = "、".join(f"第{p}頁" for p in r.source_pages)
            lines.append(f"[新增 {i}] {pages}\n{r.display_markdown}")
        return "\n\n".join(lines), None

    return f"未知工具：{name}", None
```

- [ ] **Step 4: 確認測試通過**

```bash
conda run -n hospital-rag pytest tests/test_layer_e/test_agentic_tools.py -v
```
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add layer_e/agentic_tools.py tests/test_layer_e/test_agentic_tools.py
git commit -m "feat(layer_e): add agentic_tools — TOOL_DEFINITIONS and execute_tool"
```

---

## Task 4: Agentic Pipeline

**Files:**
- Create: `layer_e/agentic_pipeline.py`
- Create: `tests/test_layer_e/test_agentic_pipeline.py`

**Interfaces:**
- Consumes:
  - `TOOL_DEFINITIONS, execute_tool` from `layer_e.agentic_tools`
  - `context_packer.pack` from `layer_e.context_packer`
  - `_format_evidence_block` from `layer_e.prompt_builder`（需 export）
  - `guardrail.check_abstention` from `layer_e.guardrail`
  - `GenerationResult, ClaimCitation` from `layer_e.models`
  - `LLMClient` from `layer_e.llm_client`
- Produces:
  - `class AgenticPipeline`
  - `AgenticPipeline(llm_client, retriever, pdf_path, doc_stem, soft_limit=8, hard_limit=12, abstention_threshold=0.10)`
  - `AgenticPipeline.run(query: str, ranked_results: list) -> GenerationResult`

**System prompt 設計：**
- 初始 evidence block 嵌入 system prompt
- 說明工具用途與停止規則（所有 claim 有 citation 才輸出最終 JSON）
- 超過 soft_limit 後在 system prompt 加入「請根據現有資訊給出最佳答案」提示

**圖片傳遞方式：**
- `get_page_image` tool 執行後，tool result 傳文字（"已截取第 N 頁"）
- 緊接著加一條 `role=user` 訊息，content 為 `image_url` list
- GPT-4.1 multimodal 支援此 pattern

---

- [ ] **Step 1: 在 `layer_e/prompt_builder.py` export `_format_evidence_block`**

`_format_evidence_block` 目前是 module-private。將函式名稱去掉底線前綴（rename to `format_evidence_block`），並在 `build()` 內改用新名稱：

```python
# layer_e/prompt_builder.py — 修改兩處

# 1. 函式名稱改為 format_evidence_block（去掉底線）
def format_evidence_block(evidence_list: list) -> str:   # ← 原 _format_evidence_block
    lines = []
    for idx, item in enumerate(evidence_list, start=1):
        ...
    return "\n".join(lines).rstrip()


# 2. build() 內部呼叫改名
def build(evidence_list: list, query: str) -> dict:
    evidence_block = format_evidence_block(evidence_list)   # ← 原 _format_evidence_block
    ...
```

確認現有測試仍通過：

```bash
conda run -n hospital-rag pytest tests/test_layer_e/test_prompt_builder.py -v
```

- [ ] **Step 2: 寫失敗測試**

```python
# tests/test_layer_e/test_agentic_pipeline.py
import json
import pytest
from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import MagicMock

from layer_e.agentic_pipeline import AgenticPipeline
from layer_e.llm_client import _StubLLMClient
from layer_e.models import GenerationResult


@dataclass
class _FakeRankedResult:
    chunk_id: str
    chunk_type: str = "paragraph"
    parent_chunk_id: Optional[str] = None
    retrieval_unit_id: str = "unit_1"
    final_score: float = 0.9
    rrf_score: float = 0.9
    retrieval_weight: float = 1.0
    display_markdown: str = "cT2N1M0 建議新輔助化療後手術"
    metadata: dict = field(default_factory=dict)
    source_tool: str = "azure_cu"
    source_pages: List[int] = field(default_factory=lambda: [5])
    embedding_text: str = "cT2N1M0"
    rerank_score: float = 0.8
    page_image_refs: dict = field(default_factory=dict)


def _make_pipeline(llm_client=None, retriever=None):
    return AgenticPipeline(
        llm_client=llm_client or _StubLLMClient(),
        retriever=retriever or MagicMock(),
        pdf_path="/fake/path.pdf",
        doc_stem="乳癌診療指引-2026年",
    )


def test_abstain_when_no_results():
    pipeline = _make_pipeline()
    result = pipeline.run("query", [])
    assert result.abstain is True
    assert result.safety_verdict == "abstained"


def test_abstain_when_low_rerank_score():
    pipeline = _make_pipeline()
    r = _FakeRankedResult(chunk_id="c1", rerank_score=0.01)
    result = pipeline.run("query", [r])
    assert result.abstain is True


def test_normal_flow_returns_generation_result():
    pipeline = _make_pipeline()
    r = _FakeRankedResult(chunk_id="c1")
    result = pipeline.run("cT2N1M0 治療建議", [r])
    assert isinstance(result, GenerationResult)
    assert result.abstain is False
    assert result.answer == "stub"


def test_steps_log_is_recorded():
    call_count = [0]

    class _ToolCallingStub(_StubLLMClient):
        def generate_with_tools(self, messages, tools):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: request a page image
                return (
                    [{"id": "call_1", "name": "get_page_image", "arguments": {"page_no": 5, "reason": "確認流程圖"}}],
                    None,
                )
            # Second call: final answer
            return ([], json.dumps({
                "answer": "cT2N1M0 建議先做新輔助化療",
                "claims": [{"text": "cT2N1M0 建議先做新輔助化療", "citations": ["E1"]}],
                "abstain": False,
                "abstain_reason": None,
            }))

    with pytest.raises(Exception):
        # pdf_path is fake → get_full_page_image will raise
        # This test validates that tool was CALLED (step log), not that image succeeds
        pass

    # Use a retriever mock and a patched pdf_tools to avoid real PDF access
    from unittest.mock import patch
    with patch("layer_e.agentic_tools.get_full_page_image", return_value="base64fakepng"):
        pipeline = AgenticPipeline(
            llm_client=_ToolCallingStub(),
            retriever=MagicMock(),
            pdf_path="/fake/path.pdf",
            doc_stem="doc",
        )
        r = _FakeRankedResult(chunk_id="c1")
        result = pipeline.run("query", [r])

    assert result.answer == "cT2N1M0 建議先做新輔助化療"
    assert len(result.steps_log) == 1
    assert result.steps_log[0]["tool"] == "get_page_image"
    assert result.steps_log[0]["step_no"] == 1


def test_hard_limit_prevents_infinite_loop():
    """LLM that always calls tools must be stopped by hard_limit."""
    class _AlwaysToolStub(_StubLLMClient):
        def generate_with_tools(self, messages, tools):
            return (
                [{"id": "call_x", "name": "retrieve_more", "arguments": {"query": "x", "reason": "x"}}],
                None,
            )

    mock_retriever = MagicMock()
    mock_retriever.search_text.return_value = []

    pipeline = AgenticPipeline(
        llm_client=_AlwaysToolStub(),
        retriever=mock_retriever,
        pdf_path="/fake/path.pdf",
        doc_stem="doc",
        hard_limit=3,
    )
    r = _FakeRankedResult(chunk_id="c1")
    result = pipeline.run("query", [r])
    # Must return some result (abstain or answer), never loop forever
    assert isinstance(result, GenerationResult)
```

- [ ] **Step 3: 確認測試失敗**

```bash
conda run -n hospital-rag pytest tests/test_layer_e/test_agentic_pipeline.py -v
```
Expected: `ImportError: cannot import name 'AgenticPipeline'`

- [ ] **Step 4: 實作**

```python
# layer_e/agentic_pipeline.py
import json
import logging
from typing import Optional

from layer_e import context_packer, guardrail
from layer_e.agentic_tools import TOOL_DEFINITIONS, execute_tool
from layer_e.models import ClaimCitation, GenerationResult
from layer_e.prompt_builder import format_evidence_block

logger = logging.getLogger(__name__)

_SYSTEM_TEMPLATE = """\
你是醫療知識庫助理，協助醫護人員查詢文件。

已為你提供以下從文件中檢索到的 Evidence，每筆格式為 [EN]《文件名》第X頁：
{evidence_block}

你可以呼叫工具取得更多資訊：
- get_page_image: 取得 PDF 頁面整頁截圖，適合閱讀流程圖、治療路徑圖
- retrieve_more: 在文件中搜尋更多相關段落或表格

停止規則：
1. 答案中每一個 medical claim 必須引用 Evidence 編號（如 [E1]）
2. 無法找到支持的 claim 標記 [unsupported]
3. 當所有 claim 都有 citation（或標記 [unsupported]）時，輸出最終 JSON

最終答案必須是以下 JSON 格式，不加 markdown fence：
{{"answer": "完整答案", "claims": [{{"text": "claim", "citations": ["E1"]}}], "abstain": false, "abstain_reason": null}}
"""

_SOFT_LIMIT_NOTICE = (
    "\n\n[系統提示] 你已使用多次工具。請根據目前已取得的所有資訊給出最佳答案。"
    "對無法找到 citation 的 claim 標記 [unsupported]。請立即輸出最終 JSON。"
)

_UNSUPPORTED_MARKER = "unsupported"


class AgenticPipeline:
    def __init__(
        self,
        llm_client,
        retriever,
        pdf_path: str,
        doc_stem: str,
        soft_limit: int = 8,
        hard_limit: int = 12,
        abstention_threshold: float = 0.10,
    ):
        self._llm = llm_client
        self._retriever = retriever
        self._pdf_path = pdf_path
        self._doc_stem = doc_stem
        self._soft_limit = soft_limit
        self._hard_limit = hard_limit
        self._abstention_threshold = abstention_threshold

    def run(self, query: str, ranked_results: list) -> GenerationResult:
        # Guardrail: abstain if evidence is insufficient
        should_abstain, reason = guardrail.check_abstention(
            ranked_results, threshold=self._abstention_threshold
        )
        if should_abstain:
            return GenerationResult(
                answer="",
                claims=[],
                evidence_map={},
                unsupported_claims=[],
                abstain=True,
                abstain_reason=reason,
                safety_verdict="abstained",
                steps_log=[],
            )

        evidence_list, evidence_map = context_packer.pack(ranked_results)
        evidence_block = format_evidence_block(evidence_list)
        system_content = _SYSTEM_TEMPLATE.format(evidence_block=evidence_block)

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]

        steps_log = []
        tool_call_count = 0

        for iteration in range(self._hard_limit):
            # Inject soft-limit notice into system message
            if tool_call_count >= self._soft_limit:
                msgs = list(messages)
                msgs[0] = {
                    "role": "system",
                    "content": system_content + _SOFT_LIMIT_NOTICE,
                }
            else:
                msgs = messages

            tool_calls, final_content = self._llm.generate_with_tools(msgs, TOOL_DEFINITIONS)

            if not tool_calls:
                # LLM gave final answer
                return self._parse_final(final_content, evidence_map, steps_log)

            # Execute each tool call
            # Add assistant message with tool_calls to history
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"]),
                        },
                    }
                    for tc in tool_calls
                ],
            })

            pending_images = []
            for tc in tool_calls:
                tool_call_count += 1
                step = {
                    "step_no": len(steps_log) + 1,
                    "tool": tc["name"],
                    "arguments": tc["arguments"],
                    "reason": tc["arguments"].get("reason", ""),
                }
                steps_log.append(step)
                logger.info("Agentic tool call %d: %s(%s)", tool_call_count, tc["name"], tc["arguments"])

                try:
                    text_result, b64_image = execute_tool(
                        tc, self._pdf_path, self._retriever, self._doc_stem
                    )
                except Exception as exc:
                    text_result = f"工具執行失敗：{exc}"
                    b64_image = None

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": text_result,
                })

                if b64_image:
                    pending_images.append(b64_image)

            # Append images as a user message so GPT-4.1 can see them
            if pending_images:
                content = [{"type": "text", "text": f"以上工具附帶 {len(pending_images)} 張截圖："}]
                for b64 in pending_images:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    })
                messages.append({"role": "user", "content": content})

        # Hard limit reached — force conclusion via one final call without tools
        logger.warning("Agentic loop hit hard_limit=%d, forcing conclusion", self._hard_limit)
        messages[0] = {"role": "system", "content": system_content + _SOFT_LIMIT_NOTICE}
        _, final_content = self._llm.generate_with_tools(messages, [])
        return self._parse_final(final_content, evidence_map, steps_log)

    def _parse_final(self, content: str, evidence_map: dict, steps_log: list) -> GenerationResult:
        try:
            data = json.loads(content or "{}")
        except (json.JSONDecodeError, TypeError):
            data = {"answer": content or "", "claims": [], "abstain": False, "abstain_reason": None}

        if data.get("abstain"):
            return GenerationResult(
                answer="",
                claims=[],
                evidence_map=evidence_map,
                unsupported_claims=[],
                abstain=True,
                abstain_reason=data.get("abstain_reason"),
                safety_verdict="abstained",
                steps_log=steps_log,
            )

        claims = []
        unsupported = []
        for c in data.get("claims", []):
            raw_citations = c.get("citations", [])
            clean = [eid for eid in raw_citations if eid != _UNSUPPORTED_MARKER]
            if _UNSUPPORTED_MARKER in raw_citations or not raw_citations:
                unsupported.append(c.get("text", ""))
            claims.append(ClaimCitation(text=c.get("text", ""), citations=clean))

        safety_verdict = "needs_review" if unsupported else "safe"
        return GenerationResult(
            answer=data.get("answer", ""),
            claims=claims,
            evidence_map=evidence_map,
            unsupported_claims=unsupported,
            abstain=False,
            abstain_reason=None,
            safety_verdict=safety_verdict,
            steps_log=steps_log,
        )
```

- [ ] **Step 5: `GenerationResult` 加 `steps_log` 欄位**

修改 `layer_e/models.py`：

```python
@dataclass
class GenerationResult:
    answer: str
    claims: List[ClaimCitation]
    evidence_map: dict
    unsupported_claims: List[str]
    abstain: bool
    abstain_reason: Optional[str]
    safety_verdict: str
    steps_log: List[dict] = field(default_factory=list)  # ← 新增，有預設值不破壞現有呼叫
```

確認 `field` 已 import：`from dataclasses import dataclass, field`

- [ ] **Step 6: 確認所有 layer_e 測試通過**

```bash
conda run -n hospital-rag pytest tests/test_layer_e/ -v
```
Expected: 全部通過（原有測試不受影響，`steps_log` 有預設值）

- [ ] **Step 7: Commit**

```bash
git add layer_e/agentic_pipeline.py layer_e/models.py layer_e/prompt_builder.py tests/test_layer_e/test_agentic_pipeline.py
git commit -m "feat(layer_e): add AgenticPipeline with tool-calling loop and steps_log"
```

---

## Task 5: RAGPipeline Integration

**Files:**
- Modify: `pipeline/runner.py`

**Interfaces:**
- Consumes: `AgenticPipeline` from `layer_e.agentic_pipeline`
- Produces: `RAGPipeline.query_agentic(query_text, pdf_path, top_k=5, prefetch_k=20, rerank=True) -> GenerationResult`

---

- [ ] **Step 1: 寫失敗測試**

在 `pipeline/runner.py` 同目錄下沒有現成的 runner 測試，在 `tests/` 根目錄下新增（或直接 import 測試）：

```python
# 在 tests/test_layer_e/test_agentic_pipeline.py 末尾加入
from unittest.mock import patch, MagicMock
from pipeline.runner import RAGPipeline


def test_rag_pipeline_query_agentic_returns_result(tmp_path):
    import fitz
    doc = fitz.open()
    doc.new_page()
    pdf_path = str(tmp_path / "test.pdf")
    doc.save(pdf_path)
    doc.close()

    mock_qdrant = MagicMock()
    mock_provider = MagicMock()
    mock_provider.embed.return_value = [[0.1] * 1024]

    with patch("pipeline.runner.HybridRetriever") as MockRetriever, \
         patch("pipeline.runner.DocumentIngester"), \
         patch("pipeline.runner.BGEReranker"), \
         patch("layer_e.agentic_pipeline.AgenticPipeline.run") as mock_run:

        from layer_e.models import GenerationResult
        mock_run.return_value = GenerationResult(
            answer="agentic answer",
            claims=[],
            evidence_map={},
            unsupported_claims=[],
            abstain=False,
            abstain_reason=None,
            safety_verdict="safe",
            steps_log=[{"step_no": 1, "tool": "get_page_image"}],
        )
        MockRetriever.return_value.search_text.return_value = []

        pipeline = RAGPipeline(mock_provider, mock_qdrant, "test_collection")
        result = pipeline.query_agentic("cT2N1M0 治療建議", pdf_path=pdf_path)

    assert result.answer == "agentic answer"
    assert len(result.steps_log) == 1
```

- [ ] **Step 2: 確認測試失敗**

```bash
conda run -n hospital-rag pytest tests/test_layer_e/test_agentic_pipeline.py::test_rag_pipeline_query_agentic_returns_result -v
```
Expected: `AttributeError: 'RAGPipeline' object has no attribute 'query_agentic'`

- [ ] **Step 3: 實作**

在 `pipeline/runner.py` 加入 import 和 `query_agentic` 方法：

```python
# pipeline/runner.py — 在現有 import 後加入
from layer_e.agentic_pipeline import AgenticPipeline
from layer_e.llm_client import GPT41Client
```

在 `RAGPipeline` class 的 `query` 方法後加入：

```python
    def query_agentic(
        self,
        query_text: str,
        pdf_path: str,
        top_k: int = 5,
        prefetch_k: int = 20,
        rerank: bool = True,
    ):
        """Retrieve evidence then run the agentic loop with GPT-4.1 tool calling.

        Parameters
        ----------
        query_text:
            Natural language question.
        pdf_path:
            Absolute path to the original PDF file (for on-demand screenshots).
        top_k, prefetch_k, rerank:
            Same as query().

        Returns
        -------
        GenerationResult
            Same structure as query(), plus .steps_log with the agentic trace.
        """
        ranked = self._retriever.search_text(
            query_text, top_k=top_k, prefetch_k=prefetch_k, rerank=rerank
        )
        doc_stem = self._ingester.collection_name  # collection name doubles as doc stem
        agentic = AgenticPipeline(
            llm_client=GPT41Client(),
            retriever=self._retriever,
            pdf_path=pdf_path,
            doc_stem=doc_stem,
        )
        return agentic.run(query_text, ranked)
```

- [ ] **Step 4: 確認所有測試通過**

```bash
conda run -n hospital-rag pytest tests/ -v --tb=short
```
Expected: 全部通過

- [ ] **Step 5: Commit**

```bash
git add pipeline/runner.py
git commit -m "feat(pipeline): add RAGPipeline.query_agentic with GPT-4.1 tool-calling loop"
```

---

## Self-Review

### Spec Coverage

| 設計決策 | 對應 Task |
|---------|---------|
| On-demand PDF 截圖（不預存圖片） | Task 1 `get_full_page_image` |
| GPT-4.1 tool calling | Task 2 `generate_with_tools` |
| get_page_image tool | Task 3 |
| retrieve_more tool（單文件 doc_ids 限制） | Task 3 |
| Citation-grounded stopping（[unsupported] 機制） | Task 4 `_parse_final` |
| 軟上限（`soft_limit=8`）注入提示 | Task 4 agentic loop |
| 硬上限（`hard_limit=12`）防無窮迴圈 | Task 4 agentic loop |
| steps_log（記錄 tool 呼叫次數與原因） | Task 4 + `GenerationResult` |
| 單文件模式（`doc_ids=[doc_stem]`） | Task 3 `execute_tool` + Task 5 |
| 原有 `generate()` / `query()` 不受影響 | 無修改 `layer_e/pipeline.py` |

### Placeholder Scan

無 TBD 或 TODO。所有步驟均含完整程式碼。

### Type Consistency

- `generate_with_tools` 回傳 `tuple[list, str | None]`：Task 2 定義，Task 4 消費，一致
- `execute_tool` 回傳 `tuple[str, str | None]`：Task 3 定義，Task 4 消費，一致
- `GenerationResult.steps_log: List[dict]`：Task 4 定義，Task 5 測試，一致
- `format_evidence_block`（無底線）：Task 4 Step 1 改名，Task 4 主程式呼叫，一致
