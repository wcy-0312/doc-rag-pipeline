from __future__ import annotations
import json
from pathlib import Path


class DocumentRegistry:
    """Persistent mapping from doc_id to PDF file path and Qdrant collection name.

    Stores data as a JSON file so it survives process restarts.
    doc_id is the PDF filename stem (e.g. "乳癌診療指引-2026年").
    """

    def __init__(self, registry_path: str | Path):
        self._path = Path(registry_path)
        self._data: dict[str, dict] = {}
        if self._path.exists():
            self._data = json.loads(self._path.read_text(encoding="utf-8"))

    def register(self, doc_id: str, pdf_path: str, collection_name: str = "") -> None:
        """Add or update an entry and persist to disk immediately."""
        self._data[doc_id] = {
            "pdf_path": str(pdf_path),
            "collection_name": collection_name,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_pdf_path(self, doc_id: str) -> str | None:
        """Return pdf_path for doc_id, or None if not registered."""
        entry = self._data.get(doc_id)
        return entry["pdf_path"] if entry else None

    def get_collection_name(self, doc_id: str) -> str | None:
        """Return collection_name for doc_id, or None if not registered."""
        entry = self._data.get(doc_id)
        return entry.get("collection_name") if entry else None
