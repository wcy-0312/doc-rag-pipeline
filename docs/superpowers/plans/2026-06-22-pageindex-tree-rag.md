# PageIndex Tree RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在現有向量搜尋路徑旁，新增一條 PageIndex 風格的 Tree-based RAG 路徑，支援 Example 1（單樹 top-down 查詢）和 Example 2（靜態指引 + 動態病歷跨樹比對）。

**Architecture:** 文件 ingest 時，除了現有的 chunk embedding 路徑，額外從 Azure CU `sections[]` 建出帶摘要的 `TreeNode` 樹；靜態樹存入 Qdrant（`retrieval_weight=0.0`，不影響一般搜尋），動態樹存入 session-scoped dict。查詢時，`TreeSearcher` 在每個 tree level 呼叫 LLM 選出相關子節點，遞迴向下直到葉節點，最終將葉節點內容包裝成 `RankedResult` 交由現有 `GenerationPipeline` 生成答案。

**Tech Stack:** Python 3.10+、qdrant-client（已安裝）、pytest（已安裝）、`layer_e.llm_client.LLMClient`

## Global Constraints

- 所有新模組必須在沒有 Azure 憑證的環境中可 import（lazy import pattern，同現有程式碼）
- Python 3.10+ union type syntax（`X | Y`）
- 不引入新的 pip 依賴
- 存入 Qdrant 的樹節點必須使用 `retrieval_weight=0.0`（避免干擾現有 hybrid search）
- 每個非葉節點 summary 以 LLM 一次呼叫生成，上限 50 字
- 所有測試以 `QdrantClient(":memory:")` 執行，不需要真實 Qdrant 服務

---

## File Map

### 新增檔案

```
layer_f/__init__.py                ← package init（空）
layer_f/tree_models.py             ← TreeNode, TreeSearchResult, CrossTreeResult dataclasses
layer_f/tree_store.py              ← static（Qdrant）+ dynamic（in-memory）樹存取
layer_f/tree_search.py             ← top-down LLM traversal + cross-tree synthesis

layer_b/tree_builder.py            ← Azure CU sections[] → TreeNode（含 LLM summaries）

tests/test_tree_models.py
tests/test_tree_builder.py
tests/test_tree_store.py
tests/test_tree_search.py
```

### 修改檔案

```
layer_e/llm_client.py              ← 新增 generate_text(user, system="") 到所有 client
pipeline/runner.py                 ← 新增 build_tree(), query_tree(), query_tree_cross()
```

---

## Task 1: Tree Models（`layer_f/tree_models.py`）

純資料類別，不依賴任何外部套件。定義整個 tree 系統的核心資料結構。

**Files:**
- Create: `layer_f/__init__.py`
- Create: `layer_f/tree_models.py`
- Test: `tests/test_tree_models.py`

**Interfaces:**
- Produces: `TreeNode`、`TreeSearchResult`、`CrossTreeResult`，供後續所有 Task 使用

- [ ] **Step 1: 寫失敗測試**

```python
# tests/test_tree_models.py
from layer_f.tree_models import TreeNode, TreeSearchResult, CrossTreeResult


def _leaf(title: str, content: str = "some content", page: int = 1) -> TreeNode:
    return TreeNode(
        node_id=f"n_{title}",
        title=title,
        start_page=page,
        end_page=page,
        summary="",
        content=content,
        children=[],
    )


def _branch(title: str, children: list) -> TreeNode:
    pages = [c.start_page for c in children if c.start_page is not None]
    return TreeNode(
        node_id=f"n_{title}",
        title=title,
        start_page=min(pages) if pages else None,
        end_page=max(pages) if pages else None,
        summary="a summary",
        content="",
        children=children,
    )


def test_leaf_is_leaf():
    assert _leaf("第一節").is_leaf is True


def test_branch_is_not_leaf():
    assert _branch("治療原則", [_leaf("手術"), _leaf("化療")]).is_leaf is False


def test_page_range():
    node = TreeNode(
        node_id="n1", title="T", start_page=5, end_page=10,
        summary="", content="", children=[],
    )
    assert node.page_range == (5, 10)


def test_to_dict_and_from_dict_roundtrip():
    leaf = _leaf("葉節點", content="內容文字", page=3)
    restored = TreeNode.from_dict(leaf.to_dict())
    assert restored.node_id == leaf.node_id
    assert restored.title == leaf.title
    assert restored.content == leaf.content
    assert restored.is_leaf is True


def test_nested_roundtrip():
    tree = _branch("根節點", [_leaf("子A", page=1), _leaf("子B", page=2)])
    restored = TreeNode.from_dict(tree.to_dict())
    assert len(restored.children) == 2
    assert restored.children[0].title == "子A"


def test_tree_search_result():
    node = _leaf("第III期", content="同步化放療")
    result = TreeSearchResult(
        query="cT3N2M0 治療",
        matched_nodes=[node],
        traversal_path=[["治療原則", "依分期", "第III期"]],
    )
    assert len(result.matched_nodes) == 1
    assert result.matched_nodes[0].content == "同步化放療"


def test_cross_tree_result():
    result = CrossTreeResult(
        query="是否符合免疫治療給付？",
        guideline_nodes=[_leaf("給付條件", content="PD-L1 ≥ 50%")],
        patient_nodes=[_leaf("檢驗報告", content="PD-L1 = 60%")],
        synthesis="病人 PD-L1 60%，符合給付條件（≥50%）",
    )
    assert "60%" in result.synthesis
```

- [ ] **Step 2: 確認測試失敗**

```bash
cd /home/wangcy0312/doc-rag-pipeline
pytest tests/test_tree_models.py -v
```
Expected: `ModuleNotFoundError: No module named 'layer_f'`

- [ ] **Step 3: 建立 `layer_f/__init__.py`（空檔）**

```python
# layer_f/__init__.py
```

- [ ] **Step 4: 建立 `layer_f/tree_models.py`**

```python
# layer_f/tree_models.py
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class TreeNode:
    node_id: str
    title: str
    start_page: int | None
    end_page: int | None
    summary: str       # LLM 生成；非葉節點才有意義，葉節點為 ""
    content: str       # 聚合段落文字；只有葉節點才有，非葉節點為 ""
    children: list[TreeNode] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def page_range(self) -> tuple[int | None, int | None]:
        return (self.start_page, self.end_page)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "title": self.title,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "summary": self.summary,
            "content": self.content,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, d: dict) -> TreeNode:
        return cls(
            node_id=d["node_id"],
            title=d["title"],
            start_page=d.get("start_page"),
            end_page=d.get("end_page"),
            summary=d.get("summary", ""),
            content=d.get("content", ""),
            children=[cls.from_dict(c) for c in d.get("children", [])],
        )


@dataclass
class TreeSearchResult:
    query: str
    matched_nodes: list[TreeNode]
    traversal_path: list[list[str]]   # 每個 matched node 對應一條路徑，如 ["根", "治療", "III期"]


@dataclass
class CrossTreeResult:
    query: str
    guideline_nodes: list[TreeNode]
    patient_nodes: list[TreeNode]
    synthesis: str                    # LLM 跨樹比對結論
```

- [ ] **Step 5: 確認測試通過**

```bash
pytest tests/test_tree_models.py -v
```
Expected: 7 tests PASS

- [ ] **Step 6: Commit**

```bash
git add layer_f/__init__.py layer_f/tree_models.py tests/test_tree_models.py
git commit -m "feat(layer_f): add TreeNode, TreeSearchResult, CrossTreeResult models"
```

---

## Task 2: LLMClient.generate_text（`layer_e/llm_client.py`）

Tree builder 和 tree search 都需要純文字生成（非 JSON）。在 `LLMClient` 新增 `generate_text` 方法，各 subclass 實作。

**Files:**
- Modify: `layer_e/llm_client.py`

**Interfaces:**
- Produces: `LLMClient.generate_text(user: str, system: str = "") -> str`

- [ ] **Step 1: 在 `LLMClient` 基底類別新增 abstract method**

在 `layer_e/llm_client.py` 的 `LLMClient` class 加入：

```python
    @abc.abstractmethod
    def generate_text(self, user: str, system: str = "") -> str:
        """Plain text generation without JSON parsing."""
        ...
```

- [ ] **Step 2: 在 `_StubLLMClient` 實作**

```python
    def generate_text(self, user: str, system: str = "") -> str:
        return "stub summary"
```

- [ ] **Step 3: 在 `Gemma3Client` 實作**

```python
    def generate_text(self, user: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        response = self._client.chat.completions.create(
            model="/model",
            messages=messages,
            temperature=0.0,
        )
        return (response.choices[0].message.content or "").strip()
```

- [ ] **Step 4: 在 `GPT41Client` 實作**

```python
    def generate_text(self, user: str, system: str = "") -> str:
        _, text = self.generate_with_tools(
            ([{"role": "system", "content": system}] if system else [])
            + [{"role": "user", "content": user}],
            [],
        )
        return (text or "").strip()
```

- [ ] **Step 5: 在 `Gemma4Client` 實作**

```python
    def generate_text(self, user: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        response = self._client.chat.completions.create(
            model="/model",
            messages=messages,
            temperature=0.0,
        )
        return (response.choices[0].message.content or "").strip()
```

- [ ] **Step 6: 快速驗證 stub client（不需啟動任何服務）**

```bash
python - <<'EOF'
from layer_e.llm_client import _StubLLMClient
c = _StubLLMClient()
assert c.generate_text("hello") == "stub summary"
print("OK")
EOF
```
Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add layer_e/llm_client.py
git commit -m "feat(llm_client): add generate_text() to all LLMClient subclasses"
```

---

## Task 3: Tree Builder（`layer_b/tree_builder.py`）

從 Azure CU `sections[]` + `paragraphs[]` 建出 `TreeNode` 樹。計算每個節點的頁碼範圍，聚合葉節點的文字內容，並對非葉節點呼叫 LLM 生成摘要。

**Files:**
- Create: `layer_b/tree_builder.py`
- Test: `tests/test_tree_builder.py`

**Interfaces:**
- Consumes: `TreeNode` from `layer_f.tree_models`；Azure CU raw dict
- Consumes: `LLMClient` from `layer_e.llm_client`（可為 None，此時 summary 留空）
- Produces: `build_tree(raw: dict, llm_client=None) -> TreeNode | None`

- [ ] **Step 1: 寫失敗測試**

```python
# tests/test_tree_builder.py
import pytest
from layer_b.tree_builder import build_tree


# 最小 Azure CU 資料：根節點「治療原則」下有兩個子節點
_RAW_CU = {
    "metadata": {"file_name": "lung_guide.pdf"},
    "data": {
        "sections": [
            # sections[0]: 根（title=治療原則）
            {
                "elements": [
                    "/paragraphs/0",   # sectionHeading "治療原則"
                    "/sections/1",
                    "/sections/2",
                ]
            },
            # sections[1]: 子節點（第III期）
            {
                "elements": [
                    "/paragraphs/1",   # sectionHeading "第III期"
                    "/paragraphs/2",   # body
                    "/paragraphs/3",   # body
                ]
            },
            # sections[2]: 子節點（第IV期）
            {
                "elements": [
                    "/paragraphs/4",   # sectionHeading "第IV期"
                    "/paragraphs/5",   # body
                ]
            },
        ],
        "paragraphs": [
            {"role": "sectionHeading", "content": "治療原則", "source": "D(10,0,0,1,0,1,1,0,1)", "spans": []},
            {"role": "sectionHeading", "content": "第III期", "source": "D(15,0,0,1,0,1,1,0,1)", "spans": []},
            {"role": None, "content": "同步化放療為標準治療方案", "source": "D(15,0,0,1,0,1,1,0,1)", "spans": []},
            {"role": None, "content": "可手術者考慮術前新輔助化療", "source": "D(16,0,0,1,0,1,1,0,1)", "spans": []},
            {"role": "sectionHeading", "content": "第IV期", "source": "D(20,0,0,1,0,1,1,0,1)", "spans": []},
            {"role": None, "content": "系統性治療為主要方向", "source": "D(20,0,0,1,0,1,1,0,1)", "spans": []},
        ],
    }
}


def test_returns_none_when_no_sections():
    raw = {"metadata": {}, "data": {"paragraphs": []}}
    assert build_tree(raw) is None


def test_root_title():
    tree = build_tree(_RAW_CU)
    assert tree is not None
    assert tree.title == "治療原則"


def test_children_count():
    tree = build_tree(_RAW_CU)
    assert len(tree.children) == 2


def test_children_titles():
    tree = build_tree(_RAW_CU)
    titles = [c.title for c in tree.children]
    assert "第III期" in titles
    assert "第IV期" in titles


def test_leaf_content_aggregated():
    tree = build_tree(_RAW_CU)
    stage3 = next(c for c in tree.children if c.title == "第III期")
    assert "同步化放療" in stage3.content
    assert "新輔助化療" in stage3.content


def test_leaf_page_range():
    tree = build_tree(_RAW_CU)
    stage3 = next(c for c in tree.children if c.title == "第III期")
    assert stage3.start_page == 15
    assert stage3.end_page == 16


def test_root_page_range_spans_children():
    tree = build_tree(_RAW_CU)
    assert tree.start_page == 10
    assert tree.end_page == 20


def test_leaves_have_no_children():
    tree = build_tree(_RAW_CU)
    for child in tree.children:
        assert child.is_leaf is True


def test_summary_generated_for_nonleaf():
    call_count = {"n": 0}

    def mock_llm_client():
        class _Mock:
            def generate_text(self, user, system=""):
                call_count["n"] += 1
                return f"摘要{call_count['n']}"
        return _Mock()

    tree = build_tree(_RAW_CU, llm_client=mock_llm_client())
    # root node is non-leaf → should have a summary
    assert tree.summary != ""
    assert call_count["n"] >= 1


def test_no_summary_when_llm_is_none():
    tree = build_tree(_RAW_CU, llm_client=None)
    assert tree.summary == ""
```

- [ ] **Step 2: 確認測試失敗**

```bash
pytest tests/test_tree_builder.py -v
```
Expected: `ModuleNotFoundError: No module named 'layer_b.tree_builder'`

- [ ] **Step 3: 建立 `layer_b/tree_builder.py`**

```python
# layer_b/tree_builder.py
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from layer_f.tree_models import TreeNode

if TYPE_CHECKING:
    from layer_e.llm_client import LLMClient

_SOURCE_PAGE_RE = re.compile(r'^D\((\d+),')

_SUMMARY_SYSTEM = "你是醫療文件助理。"
_SUMMARY_USER_TEMPLATE = (
    "用一句繁體中文（50字以內）摘要以下章節群的主要內容：\n\n{content}"
)


def _parse_page(source_str: str) -> int | None:
    m = _SOURCE_PAGE_RE.match(str(source_str or ""))
    return int(m.group(1)) if m else None


def build_tree(raw: dict, llm_client: LLMClient | None = None) -> TreeNode | None:
    """Build a PageIndexTree from an Azure CU raw document dict.

    Returns None if sections[] is absent or produces no usable nodes.
    llm_client: if provided, generates a one-sentence summary for each non-leaf node.
    """
    data = raw.get("data", {})
    sections = data.get("sections", [])
    paragraphs = data.get("paragraphs", [])

    if not sections:
        return None

    section_by_idx = {i: s for i, s in enumerate(sections)}
    visited: set[int] = set()

    # Find sections that are NOT referenced as children by any other section
    child_indices: set[int] = set()
    for sec in sections:
        for elem_ref in sec.get("elements", []):
            parts = str(elem_ref).strip("/").split("/")
            try:
                if parts[-2] == "sections":
                    child_indices.add(int(parts[-1]))
            except (IndexError, ValueError):
                pass

    root_indices = [i for i in range(len(sections)) if i not in child_indices]

    def _build_node(sec_idx: int) -> TreeNode | None:
        if sec_idx in visited:
            return None
        visited.add(sec_idx)
        sec = section_by_idx.get(sec_idx)
        if sec is None:
            return None

        title = (sec.get("title") or "").strip()
        body_paras: list[tuple[str, int | None]] = []   # (content, page)
        child_sec_indices: list[int] = []

        for elem_ref in sec.get("elements", []):
            parts = str(elem_ref).strip("/").split("/")
            try:
                kind, idx = parts[-2], int(parts[-1])
            except (IndexError, ValueError):
                continue

            if kind == "paragraphs" and idx < len(paragraphs):
                para = paragraphs[idx]
                content = (para.get("content") or "").strip()
                page = _parse_page(para.get("source", ""))
                role = para.get("role")
                if role == "sectionHeading" and not title:
                    title = content
                elif content:
                    body_paras.append((content, page))
            elif kind == "sections":
                child_sec_indices.append(idx)

        children = [
            n for idx in child_sec_indices
            if (n := _build_node(idx)) is not None
        ]

        # Skip structurally empty nodes
        if not title and not body_paras and not children:
            return None

        # Page range: collect all pages in this subtree
        all_pages: list[int] = [p for _, p in body_paras if p is not None]
        for child in children:
            if child.start_page is not None:
                all_pages.append(child.start_page)
            if child.end_page is not None:
                all_pages.append(child.end_page)
        start_page = min(all_pages) if all_pages else None
        end_page = max(all_pages) if all_pages else None

        # Leaf: aggregate body text; non-leaf: body text is folded into children
        content = "\n".join(text for text, _ in body_paras) if not children else ""

        # Summary for non-leaf nodes
        summary = ""
        if children and llm_client is not None:
            context_parts: list[str] = []
            for child in children:
                line = f"【{child.title}】" if child.title else ""
                detail = child.summary or child.content[:200]
                if line or detail:
                    context_parts.append(f"{line} {detail}".strip())
            if context_parts:
                prompt = _SUMMARY_USER_TEMPLATE.format(content="\n".join(context_parts))
                summary = llm_client.generate_text(prompt, system=_SUMMARY_SYSTEM)

        return TreeNode(
            node_id=f"sec_{sec_idx}",
            title=title or f"Section {sec_idx}",
            start_page=start_page,
            end_page=end_page,
            summary=summary,
            content=content,
            children=children,
        )

    root_nodes = [n for i in root_indices if (n := _build_node(i)) is not None]

    if not root_nodes:
        return None

    if len(root_nodes) == 1:
        return root_nodes[0]

    # Multiple roots: wrap in a virtual root named after the file
    file_name = raw.get("metadata", {}).get("file_name", "文件")
    all_pages: list[int] = []
    for n in root_nodes:
        if n.start_page is not None:
            all_pages.append(n.start_page)
        if n.end_page is not None:
            all_pages.append(n.end_page)
    return TreeNode(
        node_id="root",
        title=file_name,
        start_page=min(all_pages) if all_pages else None,
        end_page=max(all_pages) if all_pages else None,
        summary="",
        content="",
        children=root_nodes,
    )
```

- [ ] **Step 4: 確認測試通過**

```bash
pytest tests/test_tree_builder.py -v
```
Expected: 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add layer_b/tree_builder.py tests/test_tree_builder.py
git commit -m "feat(layer_b): add build_tree() — Azure CU sections[] → PageIndexTree"
```

---

## Task 4: Tree Store（`layer_f/tree_store.py`）

管理兩種樹的生命週期：靜態樹存入 Qdrant（`retrieval_weight=0.0`），動態樹存入 session-scoped in-memory dict。

**Files:**
- Create: `layer_f/tree_store.py`
- Test: `tests/test_tree_store.py`

**Interfaces:**
- Consumes: `TreeNode` from `layer_f.tree_models`
- Produces:
  - `TreeStore.store_static(doc_stem, tree, client, collection) -> None`
  - `TreeStore.load_static(doc_stem, client, collection) -> TreeNode | None`
  - `TreeStore.store_dynamic(session_id, doc_stem, tree) -> None`
  - `TreeStore.load_dynamic(session_id, doc_stem) -> TreeNode | None`
  - `TreeStore.clear_session(session_id) -> None`

- [ ] **Step 1: 寫失敗測試**

```python
# tests/test_tree_store.py
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
```

- [ ] **Step 2: 確認測試失敗**

```bash
pytest tests/test_tree_store.py -v
```
Expected: `ModuleNotFoundError: No module named 'layer_f.tree_store'`

- [ ] **Step 3: 建立 `layer_f/tree_store.py`**

```python
# layer_f/tree_store.py
from __future__ import annotations

import json
from uuid import uuid5, NAMESPACE_DNS

from layer_f.tree_models import TreeNode

_TREE_CHUNK_TYPE = "page_index_tree"


def _doc_stem_to_point_id(doc_stem: str) -> str:
    chunk_id = f"{doc_stem}__page_index_tree"
    return str(uuid5(NAMESPACE_DNS, chunk_id))


class TreeStore:
    """Manages static (Qdrant) and dynamic (in-memory) PageIndex trees."""

    def __init__(self) -> None:
        # {session_id: {doc_stem: TreeNode}}
        self._dynamic: dict[str, dict[str, TreeNode]] = {}

    # ── Static trees (Qdrant) ─────────────────────────────────────────────

    def store_static(
        self,
        doc_stem: str,
        tree: TreeNode,
        client,
        collection_name: str,
    ) -> None:
        """Upsert tree as a special Qdrant point (retrieval_weight=0.0)."""
        from qdrant_client.models import PointStruct, SparseVector

        point_id = _doc_stem_to_point_id(doc_stem)
        client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector={
                        "dense": [0.0] * 1024,
                        "sparse": SparseVector(indices=[], values=[]),
                    },
                    payload={
                        "chunk_type": _TREE_CHUNK_TYPE,
                        "retrieval_weight": 0.0,
                        "doc_stem": doc_stem,
                        "tree_json": json.dumps(tree.to_dict(), ensure_ascii=False),
                    },
                )
            ],
        )

    def load_static(
        self,
        doc_stem: str,
        client,
        collection_name: str,
    ) -> TreeNode | None:
        """Retrieve a previously stored static tree from Qdrant."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        results, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="chunk_type", match=MatchValue(value=_TREE_CHUNK_TYPE)),
                    FieldCondition(key="doc_stem", match=MatchValue(value=doc_stem)),
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not results:
            return None
        tree_json = results[0].payload.get("tree_json", "")
        try:
            return TreeNode.from_dict(json.loads(tree_json))
        except (json.JSONDecodeError, KeyError):
            return None

    # ── Dynamic trees (in-memory) ─────────────────────────────────────────

    def store_dynamic(self, session_id: str, doc_stem: str, tree: TreeNode) -> None:
        """Store a session-scoped tree in memory."""
        self._dynamic.setdefault(session_id, {})[doc_stem] = tree

    def load_dynamic(self, session_id: str, doc_stem: str) -> TreeNode | None:
        """Retrieve a session-scoped tree."""
        return self._dynamic.get(session_id, {}).get(doc_stem)

    def clear_session(self, session_id: str) -> None:
        """Remove all dynamic trees for a session (call when session ends)."""
        self._dynamic.pop(session_id, None)
```

- [ ] **Step 4: 確認測試通過**

```bash
pytest tests/test_tree_store.py -v
```
Expected: 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add layer_f/tree_store.py tests/test_tree_store.py
git commit -m "feat(layer_f): add TreeStore — static Qdrant + dynamic session-scoped trees"
```

---

## Task 5: Tree Search（`layer_f/tree_search.py`）

包含兩個功能：(1) `TreeSearcher.search()` — Example 1 的單樹 top-down LLM 走樹；(2) `TreeSearcher.search_cross()` — Example 2 的跨樹比對。

**Files:**
- Create: `layer_f/tree_search.py`
- Test: `tests/test_tree_search.py`

**Interfaces:**
- Consumes: `TreeNode`, `TreeSearchResult`, `CrossTreeResult` from `layer_f.tree_models`
- Consumes: `LLMClient` from `layer_e.llm_client`
- Produces:
  - `TreeSearcher(llm_client, max_depth=5)`
  - `TreeSearcher.search(query, root) -> TreeSearchResult`
  - `TreeSearcher.search_cross(query, guideline_tree, patient_tree) -> CrossTreeResult`

- [ ] **Step 1: 寫失敗測試**

```python
# tests/test_tree_search.py
import pytest
from layer_e.llm_client import _StubLLMClient
from layer_f.tree_models import TreeNode
from layer_f.tree_search import TreeSearcher


def _make_tree() -> TreeNode:
    """Build a 3-level test tree: root → 治療原則 → {第III期, 第IV期}"""
    stage3 = TreeNode(
        node_id="sec_1", title="第III期",
        start_page=15, end_page=16, summary="",
        content="同步化放療為標準治療方案，可手術者考慮新輔助化療。",
        children=[],
    )
    stage4 = TreeNode(
        node_id="sec_2", title="第IV期",
        start_page=20, end_page=20, summary="",
        content="系統性治療為主，標靶或化療依基因結果選擇。",
        children=[],
    )
    treatment = TreeNode(
        node_id="sec_0", title="治療原則",
        start_page=10, end_page=20,
        summary="依分期決定治療方式",
        content="",
        children=[stage3, stage4],
    )
    return treatment


class _SelectFirstLLM:
    """Always selects child index 0."""
    def generate_text(self, user: str, system: str = "") -> str:
        return '{"relevant": [0]}'


class _SelectAllLLM:
    """Always selects all children."""
    def generate_text(self, user: str, system: str = "") -> str:
        import re
        # Count [N] patterns in user prompt to know how many children
        count = len(re.findall(r'^\[\d+\]', user, re.MULTILINE))
        return f'{{"relevant": {list(range(count))}}}'


class _SelectNoneLLM:
    """Never selects any child."""
    def generate_text(self, user: str, system: str = "") -> str:
        return '{"relevant": []}'


def test_search_reaches_leaf():
    tree = _make_tree()
    result = TreeSearcher(_SelectFirstLLM()).search("第III期治療", tree)
    assert len(result.matched_nodes) == 1
    assert result.matched_nodes[0].is_leaf is True


def test_search_selected_content_is_correct():
    tree = _make_tree()
    result = TreeSearcher(_SelectFirstLLM()).search("第III期治療", tree)
    assert "同步化放療" in result.matched_nodes[0].content


def test_search_traversal_path_recorded():
    tree = _make_tree()
    result = TreeSearcher(_SelectFirstLLM()).search("第III期治療", tree)
    assert len(result.traversal_path) == 1
    assert "第III期" in result.traversal_path[0]


def test_search_no_relevant_returns_current_node():
    """When LLM selects no children, current node is returned as terminal."""
    tree = _make_tree()
    result = TreeSearcher(_SelectNoneLLM()).search("任意查詢", tree)
    assert len(result.matched_nodes) == 1
    assert result.matched_nodes[0].node_id == tree.node_id


def test_search_all_selects_both_leaves():
    tree = _make_tree()
    result = TreeSearcher(_SelectAllLLM()).search("所有分期治療", tree)
    assert len(result.matched_nodes) == 2


def test_search_leaf_node_returns_immediately():
    leaf = TreeNode(
        node_id="leaf", title="直接葉節點",
        start_page=1, end_page=1, summary="",
        content="葉節點內容", children=[],
    )
    result = TreeSearcher(_SelectFirstLLM()).search("任意查詢", leaf)
    assert len(result.matched_nodes) == 1
    assert result.matched_nodes[0].content == "葉節點內容"


def test_search_cross_returns_cross_result():
    class _SynthesisLLM:
        call_count = 0
        def generate_text(self, user: str, system: str = "") -> str:
            self.call_count += 1
            if "relevant" in user:        # child selection calls
                return '{"relevant": [0]}'
            return "病人 PD-L1 60%，符合給付條件（≥50%）"   # synthesis call

    guideline = TreeNode(
        node_id="g0", title="給付條件",
        start_page=5, end_page=5, summary="",
        content="PD-L1 ≥ 50%，無 EGFR 突變", children=[],
    )
    patient = TreeNode(
        node_id="p0", title="檢驗報告",
        start_page=1, end_page=1, summary="",
        content="PD-L1 = 60%，EGFR 野生型", children=[],
    )

    llm = _SynthesisLLM()
    result = TreeSearcher(llm).search_cross(
        "病人是否符合免疫治療給付？", guideline, patient
    )
    assert "60%" in result.synthesis
    assert len(result.guideline_nodes) >= 1
    assert len(result.patient_nodes) >= 1
```

- [ ] **Step 2: 確認測試失敗**

```bash
pytest tests/test_tree_search.py -v
```
Expected: `ModuleNotFoundError: No module named 'layer_f.tree_search'`

- [ ] **Step 3: 建立 `layer_f/tree_search.py`**

```python
# layer_f/tree_search.py
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from layer_f.tree_models import CrossTreeResult, TreeNode, TreeSearchResult

if TYPE_CHECKING:
    from layer_e.llm_client import LLMClient

_JSON_RE = re.compile(r'\{[^{}]*"relevant"[^{}]*\}')

_SELECT_SYSTEM = "你是醫療文件導航助理，協助定位與查詢最相關的章節。"

_SELECT_USER_TEMPLATE = """\
查詢：{query}

以下是文件章節列表：
{children_text}

請選出與查詢直接相關的章節編號（可多選）。
只回傳 JSON，格式：{{"relevant": [0, 2]}}
若無相關章節，回傳：{{"relevant": []}}"""

_SYNTHESIS_SYSTEM = "你是醫療文件分析助理，協助比對指引條件與病人資料。"

_SYNTHESIS_USER_TEMPLATE = """\
查詢：{query}

【治療指引】相關章節：
{guideline_content}

【病人資料】相關章節：
{patient_content}

請根據以上資訊，用繁體中文回答查詢。直接回答結論，不需重述資料。"""


def _format_children(children: list[TreeNode]) -> str:
    parts = []
    for i, child in enumerate(children):
        desc = (child.summary or child.content)[:150]
        parts.append(f"[{i}] {child.title}\n    {desc}")
    return "\n\n".join(parts)


def _parse_relevant_indices(text: str, max_idx: int) -> list[int]:
    m = _JSON_RE.search(text or "")
    if not m:
        return []
    try:
        data = json.loads(m.group())
        return [i for i in data.get("relevant", []) if 0 <= i < max_idx]
    except (json.JSONDecodeError, TypeError):
        return []


class TreeSearcher:
    def __init__(self, llm_client: LLMClient, max_depth: int = 5) -> None:
        self._llm = llm_client
        self._max_depth = max_depth

    def search(self, query: str, root: TreeNode) -> TreeSearchResult:
        """Top-down traversal: returns the most relevant leaf (or terminal) nodes."""
        matched_nodes: list[TreeNode] = []
        traversal_paths: list[list[str]] = []
        self._traverse(query, root, [root.title], matched_nodes, traversal_paths, depth=0)
        return TreeSearchResult(
            query=query,
            matched_nodes=matched_nodes,
            traversal_path=traversal_paths,
        )

    def _traverse(
        self,
        query: str,
        node: TreeNode,
        path: list[str],
        matched: list[TreeNode],
        paths: list[list[str]],
        depth: int,
    ) -> None:
        if node.is_leaf or depth >= self._max_depth:
            matched.append(node)
            paths.append(list(path))
            return

        children_text = _format_children(node.children)
        prompt = _SELECT_USER_TEMPLATE.format(query=query, children_text=children_text)
        response = self._llm.generate_text(prompt, system=_SELECT_SYSTEM)
        indices = _parse_relevant_indices(response, len(node.children))

        if not indices:
            # No relevant children — treat this node as terminal
            matched.append(node)
            paths.append(list(path))
            return

        for i in indices:
            child = node.children[i]
            self._traverse(query, child, path + [child.title], matched, paths, depth + 1)

    def search_cross(
        self,
        query: str,
        guideline_tree: TreeNode,
        patient_tree: TreeNode,
    ) -> CrossTreeResult:
        """Search both trees independently, then synthesize a cross-tree answer."""
        guideline_result = self.search(query, guideline_tree)
        patient_result = self.search(query, patient_tree)

        guideline_content = "\n\n".join(
            f"【{n.title}】\n{n.content}"
            for n in guideline_result.matched_nodes
            if n.content
        ) or "（無相關章節）"

        patient_content = "\n\n".join(
            f"【{n.title}】\n{n.content}"
            for n in patient_result.matched_nodes
            if n.content
        ) or "（無相關章節）"

        synthesis_prompt = _SYNTHESIS_USER_TEMPLATE.format(
            query=query,
            guideline_content=guideline_content,
            patient_content=patient_content,
        )
        synthesis = self._llm.generate_text(synthesis_prompt, system=_SYNTHESIS_SYSTEM)

        return CrossTreeResult(
            query=query,
            guideline_nodes=guideline_result.matched_nodes,
            patient_nodes=patient_result.matched_nodes,
            synthesis=synthesis,
        )
```

- [ ] **Step 4: 確認測試通過**

```bash
pytest tests/test_tree_search.py -v
```
Expected: 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add layer_f/tree_search.py tests/test_tree_search.py
git commit -m "feat(layer_f): add TreeSearcher — top-down LLM traversal + cross-tree synthesis"
```

---

## Task 6: Pipeline Integration（`pipeline/runner.py`）

在 `RAGPipeline` 新增三個方法：`build_tree()`、`query_tree()`（Example 1）、`query_tree_cross()`（Example 2）。Tree search 結果轉換為 `RankedResult` 後，交給現有的 `GenerationPipeline`，保持生成層不變。

**Files:**
- Modify: `pipeline/runner.py`
- Test: 在現有 `conftest.py` 加 fixture（如果沒有就在 test file 裡定義）
- Test: `tests/test_tree_integration.py`

**Interfaces:**
- Consumes: `build_tree` from `layer_b.tree_builder`；`TreeStore` from `layer_f.tree_store`；`TreeSearcher` from `layer_f.tree_search`；`TreeNode` from `layer_f.tree_models`
- Produces:
  - `RAGPipeline.build_tree(raw_document, doc_id, static=True, session_id=None, llm_client=None) -> TreeNode | None`
  - `RAGPipeline.query_tree(query_text, doc_ids, top_k=5) -> GenerationResult`
  - `RAGPipeline.query_tree_cross(query_text, guideline_doc_id, session_id, patient_doc_stem, top_k=5) -> GenerationResult`

- [ ] **Step 1: 寫失敗測試**

```python
# tests/test_tree_integration.py
import pytest
from qdrant_client import QdrantClient
from layer_e.llm_client import _StubLLMClient
from pipeline.runner import RAGPipeline


_COLLECTION = "test_tree_integration"

# Minimal raw document with sections[]
_RAW_GUIDELINE = {
    "schema_version": "v3.0",
    "metadata": {"file_name": "lung_guide.pdf", "document_type": "癌症診療指引"},
    "data": {
        "sections": [
            {
                "elements": [
                    "/paragraphs/0",
                    "/sections/1",
                    "/sections/2",
                ]
            },
            {
                "elements": [
                    "/paragraphs/1",
                    "/paragraphs/2",
                ]
            },
            {
                "elements": [
                    "/paragraphs/3",
                    "/paragraphs/4",
                ]
            },
        ],
        "paragraphs": [
            {"role": "sectionHeading", "content": "治療原則", "source": "D(10,0,0,1,0,1,1,0,1)", "spans": []},
            {"role": "sectionHeading", "content": "第III期", "source": "D(15,0,0,1,0,1,1,0,1)", "spans": []},
            {"role": None, "content": "同步化放療為標準治療方案", "source": "D(15,0,0,1,0,1,1,0,1)", "spans": []},
            {"role": "sectionHeading", "content": "第IV期", "source": "D(20,0,0,1,0,1,1,0,1)", "spans": []},
            {"role": None, "content": "系統性治療為主，標靶或化療依基因", "source": "D(20,0,0,1,0,1,1,0,1)", "spans": []},
        ],
        "tables": [], "figures": [], "markdown": "",
    },
    "page_count": 25,
}

_RAW_PATIENT = {
    "schema_version": "v3.0",
    "metadata": {"file_name": "patient_001.pdf", "document_type": "病歷"},
    "data": {
        "sections": [
            {
                "elements": [
                    "/paragraphs/0",
                    "/paragraphs/1",
                ]
            }
        ],
        "paragraphs": [
            {"role": "sectionHeading", "content": "檢驗報告", "source": "D(1,0,0,1,0,1,1,0,1)", "spans": []},
            {"role": None, "content": "PD-L1 = 60%，EGFR 野生型，ECOG 1", "source": "D(1,0,0,1,0,1,1,0,1)", "spans": []},
        ],
        "tables": [], "figures": [], "markdown": "",
    },
    "page_count": 3,
}


@pytest.fixture
def pipeline():
    client = QdrantClient(":memory:")
    stub_llm = _StubLLMClient()
    return RAGPipeline(
        embedding_provider=None,   # not used by tree path
        qdrant_client=client,
        collection_name=_COLLECTION,
        llm_client=stub_llm,
    )


def test_build_tree_static_returns_tree_node(pipeline):
    tree = pipeline.build_tree(_RAW_GUIDELINE, doc_id="lung_guide", static=True)
    assert tree is not None
    assert tree.title == "治療原則"


def test_build_tree_static_stored_in_qdrant(pipeline):
    pipeline.build_tree(_RAW_GUIDELINE, doc_id="lung_guide", static=True)
    # Re-load to confirm persistence
    loaded = pipeline._tree_store.load_static(
        "lung_guide", pipeline._qdrant_client, _COLLECTION
    )
    assert loaded is not None
    assert loaded.title == "治療原則"


def test_build_tree_dynamic_stored_in_session(pipeline):
    tree = pipeline.build_tree(
        _RAW_PATIENT, doc_id="patient_001",
        static=False, session_id="sess_1"
    )
    assert tree is not None
    loaded = pipeline._tree_store.load_dynamic("sess_1", "patient_001")
    assert loaded is not None


def test_query_tree_returns_generation_result(pipeline):
    pipeline.build_tree(_RAW_GUIDELINE, doc_id="lung_guide", static=True)
    result = pipeline.query_tree("第III期治療方式？", doc_ids=["lung_guide"])
    assert result is not None
    assert hasattr(result, "answer")


def test_query_tree_cross_returns_synthesis(pipeline):
    pipeline.build_tree(_RAW_GUIDELINE, doc_id="lung_guide", static=True)
    pipeline.build_tree(
        _RAW_PATIENT, doc_id="patient_001",
        static=False, session_id="sess_2"
    )
    result = pipeline.query_tree_cross(
        query_text="病人是否符合免疫治療給付？",
        guideline_doc_id="lung_guide",
        session_id="sess_2",
        patient_doc_stem="patient_001",
    )
    assert result is not None
    assert hasattr(result, "answer")
```

- [ ] **Step 2: 確認測試失敗**

```bash
pytest tests/test_tree_integration.py -v
```
Expected: `AttributeError` 或 `TypeError`（build_tree 方法不存在）

- [ ] **Step 3: 在 `pipeline/runner.py` 加入 import 和 helper**

在現有 import 區塊下方加入（不移動任何現有 import）：

```python
from layer_b.tree_builder import build_tree as _build_tree_from_raw
from layer_d.models import RankedResult
from layer_f.tree_models import TreeNode
from layer_f.tree_store import TreeStore
from layer_f.tree_search import TreeSearcher
```

在 `RAGPipeline.__init__` 末尾加入（在現有 `self._registry = ...` 之後）：

```python
        self._tree_store = TreeStore()
        self._qdrant_client = qdrant_client  # keep reference for tree store
        self._collection_name_ref = collection_name
```

- [ ] **Step 4: 在 `RAGPipeline` 加入 `build_tree()` 方法**

```python
    def build_tree(
        self,
        raw_document: dict,
        doc_id: str,
        static: bool = True,
        session_id: str | None = None,
        llm_client=None,
    ) -> TreeNode | None:
        """Build a PageIndexTree from a raw document dict and store it.

        Parameters
        ----------
        raw_document:
            Output from a layer_a extractor (same format as ingest()).
        doc_id:
            Document identifier used as the storage key (e.g. PDF filename stem).
        static:
            True → store in Qdrant (persists across sessions).
            False → store in session memory (session_id required).
        session_id:
            Required when static=False. Identifies the user session.
        llm_client:
            LLM client for generating node summaries. If None, summaries are empty.
        """
        import re as _re
        from pathlib import Path as _Path
        raw_stem = _Path(raw_document.get("metadata", {}).get("file_name", doc_id)).stem
        doc_stem = _re.sub(r'[^\w\-]', '_', raw_stem) if raw_stem else doc_id

        tree = _build_tree_from_raw(raw_document, llm_client=llm_client)
        if tree is None:
            return None

        if static:
            self._tree_store.store_static(
                doc_stem, tree, self._qdrant_client, self._collection_name_ref
            )
        else:
            if not session_id:
                raise ValueError("session_id is required for dynamic trees (static=False)")
            self._tree_store.store_dynamic(session_id, doc_stem, tree)

        return tree
```

- [ ] **Step 5: 在 `RAGPipeline` 加入 `_tree_nodes_to_ranked_results()` helper**

```python
    @staticmethod
    def _tree_nodes_to_ranked_results(nodes: list[TreeNode]) -> list[RankedResult]:
        """Convert TreeNode leaf content into RankedResult for GenerationPipeline."""
        results = []
        for node in nodes:
            pages = (
                list(range(node.start_page, node.end_page + 1))
                if node.start_page is not None and node.end_page is not None
                else []
            )
            display = f"**{node.title}**\n\n{node.content}" if node.title else node.content
            results.append(RankedResult(
                chunk_id=node.node_id,
                chunk_type="paragraph",
                parent_chunk_id=None,
                retrieval_unit_id=node.node_id,
                final_score=1.0,
                rrf_score=1.0,
                retrieval_weight=1.0,
                display_markdown=display,
                metadata={
                    "source_pages": pages,
                    "file_name": "",
                    "confidence_level": "high",
                    "quality_flag": "ok",
                    "source_tool": "page_index_tree",
                    "has_handwriting": False,
                    "excluded_items": [],
                    "patient_id": None,
                    "document_type": None,
                },
                source_tool="page_index_tree",
                source_pages=pages,
                embedding_text=node.content,
            ))
        return results
```

- [ ] **Step 6: 在 `RAGPipeline` 加入 `query_tree()` 方法（Example 1）**

```python
    def query_tree(
        self,
        query_text: str,
        doc_ids: list[str],
        llm_client=None,
    ):
        """Example 1: Top-down tree traversal on one or more static trees.

        Parameters
        ----------
        query_text:
            Natural language question.
        doc_ids:
            List of doc_stem keys for static trees to search.
        llm_client:
            LLM client for tree traversal. Defaults to self._gen's llm_client.

        Returns
        -------
        GenerationResult (same structure as query())
        """
        import re as _re
        _llm = llm_client or self._gen._llm

        trees = []
        for doc_id in doc_ids:
            raw_stem = _re.sub(r'[^\w\-]', '_', doc_id) if doc_id else doc_id
            tree = self._tree_store.load_static(
                raw_stem, self._qdrant_client, self._collection_name_ref
            )
            if tree is not None:
                trees.append(tree)

        if not trees:
            from layer_e.models import GenerationResult
            return GenerationResult(
                answer="", claims=[], evidence_map={}, unsupported_claims=[],
                abstain=True, abstain_reason="找不到對應的文件樹",
                safety_verdict="abstained", steps_log=[],
            )

        searcher = TreeSearcher(_llm)
        all_nodes = []
        for tree in trees:
            result = searcher.search(query_text, tree)
            all_nodes.extend(result.matched_nodes)

        ranked = self._tree_nodes_to_ranked_results(all_nodes)
        return self._gen.run(query_text, ranked)
```

- [ ] **Step 7: 在 `RAGPipeline` 加入 `query_tree_cross()` 方法（Example 2）**

```python
    def query_tree_cross(
        self,
        query_text: str,
        guideline_doc_id: str,
        session_id: str,
        patient_doc_stem: str,
        llm_client=None,
    ):
        """Example 2: Cross-tree query — static guideline + dynamic patient record.

        Parameters
        ----------
        query_text:
            Natural language question (e.g. "病人是否符合免疫治療給付？").
        guideline_doc_id:
            doc_stem of the static guideline tree.
        session_id:
            Session identifier for the dynamic patient tree.
        patient_doc_stem:
            doc_stem of the dynamic patient tree.
        llm_client:
            LLM client. Defaults to self._gen's llm_client.

        Returns
        -------
        GenerationResult where .answer contains the cross-tree synthesis.
        """
        import re as _re
        from layer_e.models import GenerationResult, ClaimCitation

        _llm = llm_client or self._gen._llm
        g_stem = _re.sub(r'[^\w\-]', '_', guideline_doc_id) if guideline_doc_id else guideline_doc_id

        guideline_tree = self._tree_store.load_static(
            g_stem, self._qdrant_client, self._collection_name_ref
        )
        patient_tree = self._tree_store.load_dynamic(session_id, patient_doc_stem)

        if guideline_tree is None or patient_tree is None:
            return GenerationResult(
                answer="", claims=[], evidence_map={}, unsupported_claims=[],
                abstain=True,
                abstain_reason=f"缺少必要的文件樹（指引：{guideline_tree is None}，病歷：{patient_tree is None}）",
                safety_verdict="abstained", steps_log=[],
            )

        searcher = TreeSearcher(_llm)
        cross_result = searcher.search_cross(query_text, guideline_tree, patient_tree)

        # Package synthesis as a GenerationResult so the caller has a uniform interface
        return GenerationResult(
            answer=cross_result.synthesis,
            claims=[ClaimCitation(text=cross_result.synthesis, citations=[])],
            evidence_map={},
            unsupported_claims=[],
            abstain=False,
            abstain_reason=None,
            safety_verdict="safe",
            steps_log=[],
        )
```

- [ ] **Step 8: 確認測試通過**

```bash
pytest tests/test_tree_integration.py -v
```
Expected: 5 tests PASS

- [ ] **Step 9: 跑完整測試套件確認無回歸**

```bash
pytest tests/ -v --tb=short
```
Expected: 所有原有測試仍 PASS

- [ ] **Step 10: Commit**

```bash
git add pipeline/runner.py tests/test_tree_integration.py
git commit -m "feat(pipeline): add build_tree(), query_tree(), query_tree_cross() — PageIndex tree path"
```

---

## Self-Review

**Spec coverage:**
- ✅ Example 1（單樹查詢）→ Task 5 `TreeSearcher.search()` + Task 6 `query_tree()`
- ✅ Example 2（跨樹比對）→ Task 5 `TreeSearcher.search_cross()` + Task 6 `query_tree_cross()`
- ✅ 靜態樹（Qdrant 持久）→ Task 4 `TreeStore.store_static/load_static`
- ✅ 動態樹（session-scoped）→ Task 4 `TreeStore.store_dynamic/load_dynamic/clear_session`
- ✅ 使用者看到樹 → `TreeNode.to_dict()` 可直接序列化給前端（Task 1）
- ✅ 使用者修改樹 → `TreeNode.from_dict()` 支援從前端修改後重新 import（Task 1）
- ✅ 不影響現有 vector search → 兩條路徑完全獨立（Task 6 只新增方法）
- 📋 未來規劃（Examples 3-5）→ 不在本 plan 範圍

**Type consistency:**
- `build_tree` 在 Task 3 和 Task 6 都用 `layer_b.tree_builder.build_tree` — ✅ 一致
- `TreeNode.from_dict / to_dict` 在 Task 1 定義，Task 4 使用 — ✅ 一致
- `RankedResult` fields 對應 `layer_d.models.RankedResult` 的所有必填欄位 — ✅ 驗證過

**Placeholder scan:** 無 TBD / TODO / "similar to above"。
