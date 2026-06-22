from unittest.mock import MagicMock, patch
from pipeline.runner import RAGPipeline


def _make_pipeline():
    provider = MagicMock()
    qdrant = MagicMock()
    llm_client = MagicMock()
    p = RAGPipeline(provider, qdrant, "test_col", llm_client=llm_client)
    return p


RAW_WITH_SECTIONS = {
    "schema_version": "v3.0",
    "metadata": {
        "file_name": "cancer guide.pdf",
        "extractor_metadata": {"tool": "azure_content_understanding", "is_fully_scanned": False, "warnings": []},
    },
    "data": {
        "paragraphs": [], "tables": [], "figures": [],
        "sections": [{"title": "第一章", "elements": ["/paragraphs/0"], "sections": []}],
        "pages": [],
    },
    "page_count": 1,
}

RAW_NO_SECTIONS = {
    "schema_version": "v3.0",
    "metadata": {
        "file_name": "notes.pdf",
        "extractor_metadata": {"tool": "azure_content_understanding", "is_fully_scanned": False, "warnings": []},
    },
    "data": {"paragraphs": [], "tables": [], "figures": [], "sections": [], "pages": []},
    "page_count": 1,
}


def test_ingest_stores_document_index_when_sections_present():
    p = _make_pipeline()
    store_mock = MagicMock()
    p._ingester.store_document_index = store_mock
    with patch("pipeline.runner.process_document", return_value=[]), \
         patch("pipeline.runner.process_and_embed", return_value=[]), \
         patch("pipeline.runner.extract_document_index", return_value={"title": "第一章", "children": []}) as ei_mock:
        p._ingester.ingest = MagicMock(return_value=0)
        p._ingester.create_collection_if_not_exists = MagicMock()
        p.ingest(RAW_WITH_SECTIONS)
    ei_mock.assert_called_once_with(RAW_WITH_SECTIONS)
    store_mock.assert_called_once()
    call_args = store_mock.call_args[0]
    assert call_args[0] == "cancer_guide", f"expected 'cancer_guide', got '{call_args[0]}'"


def test_ingest_skips_store_when_no_document_index():
    p = _make_pipeline()
    store_mock = MagicMock()
    p._ingester.store_document_index = store_mock
    with patch("pipeline.runner.process_document", return_value=[]), \
         patch("pipeline.runner.process_and_embed", return_value=[]), \
         patch("pipeline.runner.extract_document_index", return_value=None):
        p._ingester.ingest = MagicMock(return_value=0)
        p._ingester.create_collection_if_not_exists = MagicMock()
        p.ingest(RAW_NO_SECTIONS)
    store_mock.assert_not_called()


def test_query_agentic_doc_stem_sanitized():
    p = _make_pipeline()
    with patch("pipeline.runner.AgenticPipeline") as ap_mock:
        ap_instance = MagicMock()
        ap_mock.return_value = ap_instance
        ap_instance.run.return_value = MagicMock()
        p._retriever.search_text = MagicMock(return_value=[])
        p.query_agentic("question", pdf_path="/data/cancer guide.pdf")
    call_kwargs = ap_mock.call_args[1]
    assert call_kwargs["doc_stem"] == "cancer_guide", \
        f"expected 'cancer_guide', got '{call_kwargs['doc_stem']}'"
