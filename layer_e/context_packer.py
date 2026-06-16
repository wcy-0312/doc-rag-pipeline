import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
from .models import EvidenceItem


def _extract_source_doc(
    retrieval_unit_id: str | None,
    metadata: dict | None = None,
) -> str | None:
    """Build a human-readable document identifier.

    Priority: patient_id + document_type from metadata > stem from retrieval_unit_id.
    """
    if metadata:
        pid = metadata.get("patient_id")
        dtype = metadata.get("document_type")
        if pid:
            return f"病歷#{pid}_{dtype}" if dtype else f"病歷#{pid}"
    if not retrieval_unit_id:
        return None
    m = re.match(r"^(.+?)_[pt]_", retrieval_unit_id)
    return m.group(1) if m else None


def collect_image_paths(evidence_map: Dict, base_dir: str = "") -> List[str]:
    """Return deduplicated absolute image paths from all evidence page_image_refs."""
    if not base_dir:
        base_dir = os.getenv("EMBEDDED_CHUNKS_DIR", "")
    seen: dict = {}
    for entry in evidence_map.values():
        for page_str, img_path in entry.get("page_image_refs", {}).items():
            p = Path(img_path)
            if not p.is_absolute() and base_dir:
                p = Path(base_dir) / img_path
            abs_str = str(p)
            if abs_str not in seen:
                seen[abs_str] = True
    return list(seen.keys())


def pack(results: list, max_tokens: int = 12000) -> Tuple[List[EvidenceItem], Dict]:
    sorted_results = sorted(results, key=lambda r: r.rerank_score, reverse=True)

    result_chunk_ids = {r.chunk_id for r in results}
    chunk_id_to_result = {r.chunk_id: r for r in results}

    evidence_list: List[EvidenceItem] = []
    evidence_map: Dict = {}
    assigned_chunk_ids: set = set()
    token_budget = 0

    for r in sorted_results:
        if r.chunk_id in assigned_chunk_ids:
            continue

        if r.parent_chunk_id and r.parent_chunk_id in result_chunk_ids:
            parent_id = r.parent_chunk_id

            if parent_id in assigned_chunk_ids:
                assigned_chunk_ids.add(r.chunk_id)
                continue

            parent = chunk_id_to_result[parent_id]
            content = parent.display_markdown
            if r.retrieval_weight < 0.5:
                content = "[低信心] " + content

            if len(content.strip()) < 15:
                assigned_chunk_ids.add(r.chunk_id)
                continue

            tokens = len(content) // 4
            if token_budget + tokens > max_tokens:
                break

            token_budget += tokens
            evidence_id = f"E{len(evidence_list) + 1}"
            parent_unit_id = getattr(parent, "retrieval_unit_id", None)
            parent_meta = getattr(parent, "metadata", None)
            source_doc = _extract_source_doc(parent_unit_id, parent_meta)
            item = EvidenceItem(
                id=evidence_id,
                chunk_id=parent_id,
                content=content,
                retrieval_weight=r.retrieval_weight,
                source_pages=parent.source_pages,
                source_tool=parent.source_tool,
                retrieval_unit_id=parent_unit_id,
                source_doc=source_doc,
            )
            evidence_list.append(item)
            evidence_map[evidence_id] = {
                "chunk_id": parent_id,
                "retrieval_unit_id": parent_unit_id,
                "source_doc": source_doc,
                "source_pages": parent.source_pages,
                "source_tool": parent.source_tool,
                "retrieval_weight": r.retrieval_weight,
                "page_image_refs": getattr(parent, "page_image_refs", {}),
                "content": content,
            }
            assigned_chunk_ids.add(parent_id)
            assigned_chunk_ids.add(r.chunk_id)
        else:
            has_row_child = any(
                c.parent_chunk_id == r.chunk_id
                for c in results
                if c.chunk_id != r.chunk_id
            )
            if has_row_child:
                continue

            content = r.display_markdown
            if r.retrieval_weight < 0.5:
                content = "[低信心] " + content

            if len(content.strip()) < 15:
                assigned_chunk_ids.add(r.chunk_id)
                continue

            tokens = len(content) // 4
            if token_budget + tokens > max_tokens:
                break

            token_budget += tokens
            evidence_id = f"E{len(evidence_list) + 1}"
            unit_id = getattr(r, "retrieval_unit_id", None)
            r_meta = getattr(r, "metadata", None)
            source_doc = _extract_source_doc(unit_id, r_meta)
            item = EvidenceItem(
                id=evidence_id,
                chunk_id=r.chunk_id,
                content=content,
                retrieval_weight=r.retrieval_weight,
                source_pages=r.source_pages,
                source_tool=r.source_tool,
                retrieval_unit_id=unit_id,
                source_doc=source_doc,
            )
            evidence_list.append(item)
            evidence_map[evidence_id] = {
                "chunk_id": r.chunk_id,
                "retrieval_unit_id": unit_id,
                "source_doc": source_doc,
                "source_pages": r.source_pages,
                "source_tool": r.source_tool,
                "retrieval_weight": r.retrieval_weight,
                "page_image_refs": getattr(r, "page_image_refs", {}),
                "content": content,
            }
            assigned_chunk_ids.add(r.chunk_id)

    return evidence_list, evidence_map
