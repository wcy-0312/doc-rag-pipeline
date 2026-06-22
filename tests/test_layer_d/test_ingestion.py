from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import List
from unittest.mock import MagicMock, call, patch

import pytest

from layer_d.ingestion import DocumentIngester, _chunk_id_to_point_id
from layer_d.models import EmbeddedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(
    chunk_id: str = "abc00001-0000-0000-0000-000000000001",
    chunk_type: str = "paragraph",
    parent_chunk_id=None,
    retrieval_unit_id: str = "abc00001-0000-0000-0000-000000000001",
    embedding_text: str = "右肺中葉浸潤，建議追蹤",
    vector: List[float] = None,
    retrieval_weight: float = 0.95,
) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk_id=chunk_id,
        chunk_type=chunk_type,
        parent_chunk_id=parent_chunk_id,
        embedding_text=embedding_text,
        structured_json={"type": chunk_type},
        display_markdown=embedding_text,
        metadata={
            "source_tool": "azure_cu",
            "confidence_level": "high",
            "quality_flag": "ok",
            "retrieval_weight": retrieval_weight,
            "source_pages": [1],
            "has_handwriting": False,
        },
        retrieval_unit_id=retrieval_unit_id,
        vector=vector if vector is not None else [0.1] * 1024,
    )


def _make_row_chunk(index: int = 0) -> EmbeddedChunk:
    parent_id = "abc00002-0000-0000-0000-000000000002"
    return _make_chunk(
        chunk_id=f"{parent_id}_row{index}",
        chunk_type="row",
        parent_chunk_id=parent_id,
        retrieval_unit_id=parent_id,
        embedding_text=f"row {index} content",
    )


def _fake_sparse_weights(texts: List[str]) -> List[dict]:
    """Deterministic fake sparse weights for testing."""
    return [{101: 0.5, 2023: 0.3, 5432: 0.8} for _ in texts]


def _mock_qdrant_client(collection_exists: bool = False) -> MagicMock:
    client = MagicMock()
    col_name = "medical_docs"

    if collection_exists:
        mock_col = MagicMock()
        mock_col.name = col_name
        client.get_collections.return_value = SimpleNamespace(
            collections=[mock_col]
        )
    else:
        client.get_collections.return_value = SimpleNamespace(collections=[])

    return client


# ---------------------------------------------------------------------------
# _chunk_id_to_point_id
# ---------------------------------------------------------------------------

class TestChunkIdToPointId:
    def test_standard_uuid_input(self):
        chunk_id = "abc00001-0000-0000-0000-000000000001"
        point_id = _chunk_id_to_point_id(chunk_id)
        parsed = uuid.UUID(point_id)   # must not raise
        assert str(parsed) == point_id

    def test_row_chunk_id(self):
        chunk_id = "abc00002-0000-0000-0000-000000000002_row3"
        point_id = _chunk_id_to_point_id(chunk_id)
        uuid.UUID(point_id)            # must not raise

    def test_deterministic(self):
        chunk_id = "abc00003-0000-0000-0000-000000000003_row7"
        assert _chunk_id_to_point_id(chunk_id) == _chunk_id_to_point_id(chunk_id)

    def test_different_ids_produce_different_point_ids(self):
        id_a = _chunk_id_to_point_id("abc00001-0000-0000-0000-000000000001")
        id_b = _chunk_id_to_point_id("abc00001-0000-0000-0000-000000000001_row0")
        assert id_a != id_b


# ---------------------------------------------------------------------------
# create_collection_if_not_exists
# ---------------------------------------------------------------------------

class TestCreateCollection:
    def test_creates_collection_when_not_exists(self):
        client = _mock_qdrant_client(collection_exists=False)
        ingester = DocumentIngester(client=client, collection_name="medical_docs")
        ingester.create_collection_if_not_exists()

        client.create_collection.assert_called_once()
        call_kwargs = client.create_collection.call_args.kwargs

        assert call_kwargs["collection_name"] == "medical_docs"

        # dense vector config
        dense_cfg = call_kwargs["vectors_config"]["dense"]
        assert dense_cfg.size == 1024

        # sparse vector config exists
        assert "sparse" in call_kwargs["sparse_vectors_config"]

    def test_skips_creation_when_already_exists(self):
        client = _mock_qdrant_client(collection_exists=True)
        ingester = DocumentIngester(client=client, collection_name="medical_docs")
        ingester.create_collection_if_not_exists()

        client.create_collection.assert_not_called()

    def test_payload_indexes_created(self):
        client = _mock_qdrant_client(collection_exists=False)
        ingester = DocumentIngester(client=client, collection_name="medical_docs")
        ingester.create_collection_if_not_exists()

        calls = client.create_payload_index.call_args_list
        indexed_fields = {c.kwargs["field_name"] for c in calls}
        assert "retrieval_weight" in indexed_fields
        assert "chunk_type" in indexed_fields
        assert "source_tool" in indexed_fields


# ---------------------------------------------------------------------------
# ingest – batch behaviour
# ---------------------------------------------------------------------------

class TestIngest:
    @patch("layer_d.ingestion._encode_sparse", side_effect=_fake_sparse_weights)
    def test_single_batch(self, mock_encode):
        client = _mock_qdrant_client()
        ingester = DocumentIngester(client=client, collection_name="medical_docs")

        chunks = [_make_chunk(chunk_id=f"abc0000{i}-0000-0000-0000-000000000001") for i in range(3)]
        total = ingester.ingest(chunks, batch_size=64)

        assert total == 3
        client.upsert.assert_called_once()

    @patch("layer_d.ingestion._encode_sparse", side_effect=_fake_sparse_weights)
    def test_multiple_batches(self, mock_encode):
        client = _mock_qdrant_client()
        ingester = DocumentIngester(client=client, collection_name="medical_docs")

        chunks = [
            _make_chunk(chunk_id=f"abc{i:04d}0000-0000-0000-0000-000000000000")
            for i in range(5)
        ]
        total = ingester.ingest(chunks, batch_size=2)

        assert total == 5
        assert client.upsert.call_count == 3   # ceil(5/2) = 3

    @patch("layer_d.ingestion._encode_sparse", side_effect=_fake_sparse_weights)
    def test_skips_chunks_without_vector(self, mock_encode):
        client = _mock_qdrant_client()
        ingester = DocumentIngester(client=client, collection_name="medical_docs")

        chunks = [
            _make_chunk(chunk_id="abc00001-0000-0000-0000-000000000001"),
            _make_chunk(chunk_id="abc00002-0000-0000-0000-000000000002", vector=[]),
            _make_chunk(chunk_id="abc00003-0000-0000-0000-000000000003"),
        ]
        total = ingester.ingest(chunks, batch_size=64)

        assert total == 2   # chunk with empty vector skipped

    @patch("layer_d.ingestion._encode_sparse", side_effect=_fake_sparse_weights)
    def test_returns_zero_for_empty_input(self, mock_encode):
        client = _mock_qdrant_client()
        ingester = DocumentIngester(client=client, collection_name="medical_docs")
        assert ingester.ingest([], batch_size=64) == 0
        client.upsert.assert_not_called()


# ---------------------------------------------------------------------------
# ingest – point structure
# ---------------------------------------------------------------------------

class TestPointStructure:
    @patch("layer_d.ingestion._encode_sparse", side_effect=_fake_sparse_weights)
    def test_point_id_is_valid_uuid(self, mock_encode):
        client = _mock_qdrant_client()
        ingester = DocumentIngester(client=client, collection_name="medical_docs")

        chunk = _make_chunk()
        ingester.ingest([chunk], batch_size=64)

        upserted_points = client.upsert.call_args.kwargs["points"]
        assert len(upserted_points) == 1
        uuid.UUID(upserted_points[0].id)   # must not raise

    @patch("layer_d.ingestion._encode_sparse", side_effect=_fake_sparse_weights)
    def test_point_id_deterministic(self, mock_encode):
        client = _mock_qdrant_client()
        ingester = DocumentIngester(client=client, collection_name="medical_docs")

        chunk_id = "abc00001-0000-0000-0000-000000000001"
        chunk = _make_chunk(chunk_id=chunk_id)
        ingester.ingest([chunk], batch_size=64)

        point = client.upsert.call_args.kwargs["points"][0]
        assert point.id == _chunk_id_to_point_id(chunk_id)

    @patch("layer_d.ingestion._encode_sparse", side_effect=_fake_sparse_weights)
    def test_payload_fields_present(self, mock_encode):
        client = _mock_qdrant_client()
        ingester = DocumentIngester(client=client, collection_name="medical_docs")

        chunk = _make_chunk(retrieval_weight=0.87)
        ingester.ingest([chunk], batch_size=64)

        payload = client.upsert.call_args.kwargs["points"][0].payload
        assert payload["chunk_id"] == chunk.chunk_id
        assert payload["chunk_type"] == chunk.chunk_type
        assert payload["retrieval_unit_id"] == chunk.retrieval_unit_id
        assert payload["retrieval_weight"] == pytest.approx(0.87)
        assert "source_tool" in payload
        assert "confidence_level" in payload

    @patch("layer_d.ingestion._encode_sparse", side_effect=_fake_sparse_weights)
    def test_sparse_vector_indices_values_equal_length(self, mock_encode):
        client = _mock_qdrant_client()
        ingester = DocumentIngester(client=client, collection_name="medical_docs")

        ingester.ingest([_make_chunk()], batch_size=64)

        sparse_vec = client.upsert.call_args.kwargs["points"][0].vector["sparse"]
        assert len(sparse_vec.indices) == len(sparse_vec.values)
        assert len(sparse_vec.indices) > 0

    @patch("layer_d.ingestion._encode_sparse", side_effect=_fake_sparse_weights)
    def test_sparse_vector_types(self, mock_encode):
        client = _mock_qdrant_client()
        ingester = DocumentIngester(client=client, collection_name="medical_docs")

        ingester.ingest([_make_chunk()], batch_size=64)

        sparse_vec = client.upsert.call_args.kwargs["points"][0].vector["sparse"]
        assert all(isinstance(i, int) for i in sparse_vec.indices)
        assert all(isinstance(v, float) for v in sparse_vec.values)

    @patch("layer_d.ingestion._encode_sparse", side_effect=_fake_sparse_weights)
    def test_dense_vector_present(self, mock_encode):
        client = _mock_qdrant_client()
        ingester = DocumentIngester(client=client, collection_name="medical_docs")

        chunk = _make_chunk()
        ingester.ingest([chunk], batch_size=64)

        dense_vec = client.upsert.call_args.kwargs["points"][0].vector["dense"]
        assert dense_vec == chunk.vector

    @patch("layer_d.ingestion._encode_sparse", side_effect=_fake_sparse_weights)
    def test_row_chunk_point_id_valid_uuid(self, mock_encode):
        client = _mock_qdrant_client()
        ingester = DocumentIngester(client=client, collection_name="medical_docs")

        chunk = _make_row_chunk(index=5)
        ingester.ingest([chunk], batch_size=64)

        point = client.upsert.call_args.kwargs["points"][0]
        uuid.UUID(point.id)   # non-UUID chunk_id must still produce valid UUID point_id
        assert point.payload["parent_chunk_id"] == chunk.parent_chunk_id


# ---------------------------------------------------------------------------
# page_image_refs and patient_id propagation (修正 ⑥)
# ---------------------------------------------------------------------------

def _make_chunk_with_image_refs(
    chunk_id: str = "abc00010-0000-0000-0000-000000000010",
    page_image_refs: dict = None,
    patient_id: str = None,
) -> EmbeddedChunk:
    metadata = {
        "source_tool": "azure_cu",
        "confidence_level": "high",
        "quality_flag": "ok",
        "retrieval_weight": 0.95,
        "source_pages": [1, 2],
        "has_handwriting": False,
        "page_image_refs": page_image_refs or {"1": "images/page_1.png", "2": "images/page_2.png"},
    }
    if patient_id is not None:
        metadata["patient_id"] = patient_id
    return EmbeddedChunk(
        chunk_id=chunk_id,
        chunk_type="paragraph",
        parent_chunk_id=None,
        embedding_text="右肺中葉浸潤，建議追蹤",
        structured_json={"type": "paragraph"},
        display_markdown="右肺中葉浸潤，建議追蹤",
        metadata=metadata,
        retrieval_unit_id=chunk_id,
        vector=[0.1] * 1024,
    )


class TestPageImageRefsAndPatientId:
    @patch("layer_d.ingestion._encode_sparse", side_effect=_fake_sparse_weights)
    def test_patient_id_stored_in_payload(self, mock_encode):
        client = _mock_qdrant_client()
        ingester = DocumentIngester(client=client, collection_name="medical_docs")

        chunk = _make_chunk_with_image_refs(patient_id="PT-20240001")
        ingester.ingest([chunk], batch_size=64)

        payload = client.upsert.call_args.kwargs["points"][0].payload
        assert payload["patient_id"] == "PT-20240001"

    @patch("layer_d.ingestion._encode_sparse", side_effect=_fake_sparse_weights)
    def test_patient_id_none_when_absent(self, mock_encode):
        client = _mock_qdrant_client()
        ingester = DocumentIngester(client=client, collection_name="medical_docs")

        chunk = _make_chunk()
        ingester.ingest([chunk], batch_size=64)

        payload = client.upsert.call_args.kwargs["points"][0].payload
        assert payload["patient_id"] is None

    def test_patient_id_index_created(self):
        client = _mock_qdrant_client(collection_exists=False)
        ingester = DocumentIngester(client=client, collection_name="medical_docs")
        ingester.create_collection_if_not_exists()

        calls = client.create_payload_index.call_args_list
        indexed_fields = {c.kwargs["field_name"] for c in calls}
        assert "patient_id" in indexed_fields


# ---------------------------------------------------------------------------
# store_document_index (Task 4)
# ---------------------------------------------------------------------------

def test_store_document_index():
    """store_document_index upserts a point with chunk_type=document_index."""
    mock_qdrant_client = _mock_qdrant_client()
    ingester = DocumentIngester(client=mock_qdrant_client)
    index = {"sections": [{"title": "第一章"}]}
    ingester.store_document_index("my_doc", index)

    calls = mock_qdrant_client.upsert.call_args_list
    assert len(calls) == 1
    points = calls[0].kwargs["points"]
    assert len(points) == 1
    payload = points[0].payload
    assert payload["chunk_type"] == "document_index"
    assert payload["retrieval_weight"] == 0.0
    assert payload["chunk_id"] == "my_doc__document_index"
    assert payload["document_index"] == index
