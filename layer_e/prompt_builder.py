import base64
from pathlib import Path
from typing import List
from .models import EvidenceItem

_SYSTEM_INSTRUCTIONS = """你是醫療知識庫助理，專門協助醫護人員查詢醫院文件。
以下是從醫院文件中檢索到的 Evidence，每筆格式為 [EN]《文件名》第X頁。
你只能引用以下 Evidence 中的內容作為答案依據。
對於答案中的每一個 claim，必須在句末以方括號標注 Evidence 編號，例如：「建議每 8 小時換藥。[E1]」
若某個 claim 無法在以下 Evidence 中找到支持，必須標注 [unsupported]。
請以繁體中文回答。

請以以下 JSON 格式輸出答案：
{
  "answer": "完整答案文字（句末含 [E1]、[E2] 等簡短引用標注）",
  "claims": [
    {"text": "claim 文字", "citations": ["E1"]},
    ...
  ],
  "abstain": false,
  "abstain_reason": null
}"""


def _format_evidence_block(evidence_list: list) -> str:
    lines = []
    for idx, item in enumerate(evidence_list, start=1):
        label = f"E{idx}"
        doc = f"《{item.source_doc}》" if item.source_doc else ""
        if item.source_pages:
            pages_str = "、".join(f"第 {p} 頁" for p in item.source_pages)
            header = f"[{label}]{doc}{pages_str}"
        else:
            header = f"[{label}]{doc}"
        lines.append(header)
        lines.append(item.content)
        lines.append("")
    return "\n".join(lines).rstrip()


_MIME_MAP = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}


def build_multimodal_messages(system: str, user: str, image_paths: List[str]) -> List[dict]:
    """Build OpenAI-compatible messages list, embedding images when paths are provided."""
    if not image_paths:
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    content: list = [{"type": "text", "text": user}]
    for path in image_paths[:4]:
        img_bytes = Path(path).read_bytes()
        b64 = base64.b64encode(img_bytes).decode()
        ext = Path(path).suffix.lstrip(".").lower()
        mime = _MIME_MAP.get(ext, "image/png")
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def format_references(evidence_map: dict, cited_eids: set | None = None) -> str:
    """Generate an academic-style References block from evidence_map.

    Only lists evidence IDs that appear in cited_eids (if provided).
    """
    lines = ["**References**", ""]
    for eid, entry in evidence_map.items():
        if cited_eids is not None and eid not in cited_eids:
            continue
        doc = entry.get("source_doc", "")
        pages = entry.get("source_pages", [])
        content = (entry.get("content") or "").strip()
        pages_str = "、".join(f"第{p}頁" for p in pages)
        doc_str = f"《{doc}》" if doc else ""
        lines.append(f"[{eid}] {doc_str}{pages_str}")
        if content:
            excerpt = content[:200] + ("..." if len(content) > 200 else "")
            lines.append(f"    {excerpt}")
        lines.append("")
    return "\n".join(lines).rstrip()


def build(evidence_list: list, query: str) -> dict:
    evidence_block = _format_evidence_block(evidence_list)
    system = _SYSTEM_INSTRUCTIONS + "\n\n" + evidence_block
    return {"system": system, "user": query}
