"""Tests for DOCX→PDF render path added to docling_extractor."""
import pytest
import importlib.util
from pathlib import Path


_SAMPLE_DOCX = Path(
    "docs/作業常規(SOP)/Q16-品管組/"
    "護理部_A31000-Q16-F-B12_急救護理用品基數效期查核表-2.1版.docx"
)
_DOCLING_AVAILABLE = importlib.util.find_spec("docling") is not None


@pytest.mark.skipif(
    not _SAMPLE_DOCX.exists() or not _DOCLING_AVAILABLE,
    reason="sample file or docling not found"
)
class TestDocxToPdfConversion:
    def test_pdf_path_key_present_in_output(self):
        """convert_word_docling output always has pdf_path_for_render key."""
        from layer_a.docling_extractor import convert_word_docling
        raw = convert_word_docling(_SAMPLE_DOCX)
        assert "pdf_path_for_render" in raw["data"]

    def test_pdf_file_exists_when_conversion_succeeds(self):
        """pdf_path_for_render points to an existing .pdf file."""
        from layer_a.docling_extractor import convert_word_docling
        raw = convert_word_docling(_SAMPLE_DOCX)
        pdf_path = raw["data"]["pdf_path_for_render"]
        if pdf_path is None:
            pytest.skip("LibreOffice conversion failed on this system")
        assert Path(pdf_path).exists()
        assert Path(pdf_path).suffix == ".pdf"

    def test_pdf_renderable_by_pymupdf(self):
        """The produced PDF can be opened and rendered by fitz."""
        from layer_a.docling_extractor import convert_word_docling
        raw = convert_word_docling(_SAMPLE_DOCX)
        pdf_path = raw["data"]["pdf_path_for_render"]
        if pdf_path is None:
            pytest.skip("LibreOffice conversion failed on this system")
        import fitz
        doc = fitz.open(pdf_path)
        assert doc.page_count >= 1
        pix = doc[0].get_pixmap()
        assert pix.width > 0
        doc.close()


class TestTreePipelineWordFallback:
    """tree_pipeline._synthesise renders all pages for Word nodes (start_page=None)."""

    def _make_pipeline(self):
        from unittest.mock import MagicMock, patch
        qdrant = MagicMock()
        llm = MagicMock()
        llm.generate_text_multimodal.return_value = "answer"
        from pipeline.tree_pipeline import TreeRAGPipeline
        return TreeRAGPipeline(qdrant, "col", llm), llm

    def test_build_tree_reads_pdf_path_from_raw(self):
        """build_tree() uses raw_document["data"]["pdf_path_for_render"] when pdf_path absent."""
        from unittest.mock import MagicMock, patch
        from layer_f.tree_models import TreeNode
        from pipeline.tree_pipeline import TreeRAGPipeline

        pipeline, _ = self._make_pipeline()
        fake_tree = TreeNode("n1", "doc", None, None, "", "content")

        raw = {
            "metadata": {"file_name": "test.docx"},
            "data": {"pdf_path_for_render": "/tmp/fake.pdf"},
        }

        with patch("pipeline.tree_pipeline._build_tree_from_raw", return_value=fake_tree), \
             patch.object(pipeline._ingester, "create_collection_if_not_exists"), \
             patch.object(pipeline._tree_store, "store_static"):
            pipeline.build_tree(raw, doc_id="test.docx")  # no pdf_path arg

        assert pipeline._pdf_paths.get("test_docx") == "/tmp/fake.pdf"

    def test_synthesise_renders_all_pages_for_word_nodes(self, tmp_path):
        """When all nodes have start_page=None, render all pages of the PDF."""
        import fitz
        from layer_f.tree_models import TreeNode
        from pipeline.tree_pipeline import TreeRAGPipeline

        # Create a minimal valid 1-page PDF
        pdf_path = str(tmp_path / "test.pdf")
        doc = fitz.open()
        doc.new_page()
        doc.save(pdf_path)
        doc.close()

        pipeline, llm = self._make_pipeline()
        node = TreeNode("n1", "表單", None, None, "", "content")
        pipeline._pdf_paths["表單"] = pdf_path

        result = pipeline._synthesise("query", [node], ["表單"], 72, "")
        llm.generate_text_multimodal.assert_called_once()
        images_arg = llm.generate_text_multimodal.call_args[0][1]
        assert len(images_arg) == 1  # 1-page PDF → 1 image
