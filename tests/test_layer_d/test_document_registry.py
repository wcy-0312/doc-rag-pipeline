import json
import pytest
from layer_d.document_registry import DocumentRegistry


def test_register_and_get(tmp_path):
    reg = DocumentRegistry(tmp_path / "registry.json")
    reg.register("乳癌診療指引-2026年", "/docs/乳癌.pdf", collection_name="乳癌診療指引-2026年")
    assert reg.get_pdf_path("乳癌診療指引-2026年") == "/docs/乳癌.pdf"


def test_get_unknown_returns_none(tmp_path):
    reg = DocumentRegistry(tmp_path / "registry.json")
    assert reg.get_pdf_path("nonexistent") is None


def test_persists_to_disk(tmp_path):
    path = tmp_path / "registry.json"
    reg = DocumentRegistry(path)
    reg.register("doc_a", "/path/to/a.pdf")
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["doc_a"]["pdf_path"] == "/path/to/a.pdf"


def test_corrupt_registry_starts_fresh(tmp_path):
    """A corrupted registry file should be silently ignored and replaced on next write."""
    path = tmp_path / "registry.json"
    path.write_text("{not valid json}", encoding="utf-8")
    reg = DocumentRegistry(path)
    assert reg.get_pdf_path("anything") is None
    # After a register(), the file should contain valid JSON
    reg.register("doc_x", "/x.pdf")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["doc_x"]["pdf_path"] == "/x.pdf"


def test_loads_existing_data(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps({"doc_b": {"pdf_path": "/b.pdf", "collection_name": "col_b"}}),
        encoding="utf-8",
    )
    reg = DocumentRegistry(path)
    assert reg.get_pdf_path("doc_b") == "/b.pdf"


def test_register_overwrites(tmp_path):
    reg = DocumentRegistry(tmp_path / "registry.json")
    reg.register("doc_c", "/old.pdf")
    reg.register("doc_c", "/new.pdf")
    assert reg.get_pdf_path("doc_c") == "/new.pdf"


def test_get_collection_name(tmp_path):
    reg = DocumentRegistry(tmp_path / "registry.json")
    reg.register("doc_d", "/d.pdf", collection_name="col_d")
    assert reg.get_collection_name("doc_d") == "col_d"


def test_get_collection_name_unknown(tmp_path):
    reg = DocumentRegistry(tmp_path / "registry.json")
    assert reg.get_collection_name("unknown") is None
