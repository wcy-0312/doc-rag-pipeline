import pytest
from qdrant_client import QdrantClient
from layer_e.llm_client import _StubLLMClient
from pipeline.runner import RAGPipeline

import fitz as _fitz

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


@pytest.fixture
def minimal_pdf(tmp_path):
    doc = _fitz.open()
    for _ in range(25):
        doc.new_page()
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(doc.tobytes())
    doc.close()
    return str(pdf_path)


def test_build_tree_requires_pdf_path(pipeline):
    """build_tree() must raise ValueError when pdf_path is not provided."""
    with pytest.raises(ValueError, match="pdf_path is required"):
        pipeline.build_tree(_RAW_GUIDELINE, doc_id="lung_guide.pdf", static=True)


def test_build_tree_static_returns_tree_node(pipeline, minimal_pdf):
    tree = pipeline.build_tree(_RAW_GUIDELINE, doc_id="lung_guide.pdf", static=True, pdf_path=minimal_pdf)
    assert tree is not None
    assert tree.title == "治療原則"


def test_build_tree_static_stored_in_qdrant(pipeline, minimal_pdf):
    pipeline.build_tree(_RAW_GUIDELINE, doc_id="lung_guide.pdf", static=True, pdf_path=minimal_pdf)
    # Re-load to confirm persistence
    loaded = pipeline._tree_store.load_static(
        "lung_guide_pdf", pipeline._qdrant_client, _COLLECTION
    )
    assert loaded is not None
    assert loaded.title == "治療原則"


def test_build_tree_dynamic_stored_in_session(pipeline, minimal_pdf):
    tree = pipeline.build_tree(
        _RAW_PATIENT, doc_id="patient_001.pdf",
        static=False, session_id="sess_1", pdf_path=minimal_pdf,
    )
    assert tree is not None
    loaded = pipeline._tree_store.load_dynamic("sess_1", "patient_001_pdf")
    assert loaded is not None


def test_query_tree_agentic_returns_generation_result(pipeline, minimal_pdf):
    """query_tree_agentic() should return a GenerationResult."""
    pipeline.build_tree(_RAW_GUIDELINE, doc_id="lung_guide.pdf", static=True, pdf_path=minimal_pdf)
    pipeline.preload_trees(["lung_guide.pdf"])
    result = pipeline.query_tree_agentic("第III期治療方式？")
    assert result is not None
    assert hasattr(result, "answer")


def test_query_tree_cross_returns_synthesis(pipeline, minimal_pdf):
    pipeline.build_tree(_RAW_GUIDELINE, doc_id="lung_guide.pdf", static=True, pdf_path=minimal_pdf)
    pipeline.build_tree(
        _RAW_PATIENT, doc_id="patient_001.pdf",
        static=False, session_id="sess_2", pdf_path=minimal_pdf,
    )
    result = pipeline.query_tree_cross(
        query_text="病人是否符合免疫治療給付？",
        guideline_doc_id="lung_guide_pdf",
        session_id="sess_2",
        patient_doc_stem="patient_001_pdf",
    )
    assert result is not None
    assert hasattr(result, "answer")
