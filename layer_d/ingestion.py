from __future__ import annotations

import os
from typing import List, Optional
from uuid import uuid5, NAMESPACE_DNS

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from layer_d.models import EmbeddedChunk

_COLLECTION = os.getenv("QDRANT_COLLECTION", "medical_docs")
_HOST = os.getenv("QDRANT_HOST", "localhost")
_PORT = int(os.getenv("QDRANT_PORT", "6333"))

_sparse_model = None


def _get_sparse_model():
    global _sparse_model
    if _sparse_model is None:
        from FlagEmbedding import BGEM3FlagModel
        _sparse_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    return _sparse_model


def _chunk_id_to_point_id(chunk_id: str) -> str:
    return str(uuid5(NAMESPACE_DNS, chunk_id))


def _encode_sparse(texts: List[str]) -> List[dict]:
    model = _get_sparse_model()
    out = model.encode(
        texts,
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    return out["lexical_weights"]


class DocumentIngester:
    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        collection_name: str = _COLLECTION,
    ) -> None:
        self.client = client or QdrantClient(host=_HOST, port=_PORT)
        self.collection_name = collection_name

    def create_collection_if_not_exists(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name in existing:
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(size=1024, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            },
        )

        from qdrant_client.models import PayloadSchemaType
        for field in ("source_tool", "confidence_level", "quality_flag",
                      "chunk_type", "parent_chunk_id", "retrieval_unit_id",
                      "patient_id", "document_type"):
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="retrieval_weight",
            field_schema=PayloadSchemaType.FLOAT,
        )

    def ingest(
        self,
        chunks: List[EmbeddedChunk],
        batch_size: int = 64,
    ) -> int:
        """Upsert chunks into Qdrant. Returns total number of points upserted."""
        eligible = [c for c in chunks if c.vector]
        total = 0

        for start in range(0, len(eligible), batch_size):
            batch = eligible[start : start + batch_size]
            texts = [c.embedding_text for c in batch]
            sparse_weights = _encode_sparse(texts)

            points = []
            for chunk, lw in zip(batch, sparse_weights):
                payload = {
                    "chunk_id":          chunk.chunk_id,
                    "chunk_type":        chunk.chunk_type,
                    "parent_chunk_id":   chunk.parent_chunk_id,
                    "retrieval_unit_id": chunk.retrieval_unit_id,
                    "source_tool":       chunk.metadata.get("source_tool"),
                    "confidence_level":  chunk.metadata.get("confidence_level"),
                    "quality_flag":      chunk.metadata.get("quality_flag"),
                    "retrieval_weight":  chunk.metadata.get("retrieval_weight", 1.0),
                    "source_pages":      chunk.metadata.get("source_pages", []),
                    "has_handwriting":   chunk.metadata.get("has_handwriting", False),
                    "embedding_text":    chunk.embedding_text,
                    "display_markdown":  chunk.display_markdown,
                    "patient_id":        chunk.metadata.get("patient_id"),
                    "document_type":     chunk.metadata.get("document_type"),
                }
                points.append(
                    PointStruct(
                        id=_chunk_id_to_point_id(chunk.chunk_id),
                        vector={
                            "dense": chunk.vector,
                            "sparse": SparseVector(
                                indices=[int(k) for k in lw.keys()],
                                values=[float(v) for v in lw.values()],
                            ),
                        },
                        payload=payload,
                    )
                )

            self.client.upsert(collection_name=self.collection_name, points=points)
            total += len(points)

        return total
