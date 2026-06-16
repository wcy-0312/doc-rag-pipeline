from __future__ import annotations

from typing import List

from layer_c.chunker import chunk_units
from layer_c.models import EmbeddedChunk
from layer_c.providers import EmbeddingProvider

_DEDUP_THRESHOLD = 0.98


def _deduplicate_by_cosine(chunks: List[EmbeddedChunk], threshold: float = _DEDUP_THRESHOLD) -> List[EmbeddedChunk]:
    """Remove chunks that are near-duplicates (cosine similarity > threshold) of an already-kept chunk."""
    if len(chunks) < 2:
        return chunks

    try:
        import numpy as np
    except ImportError:
        return chunks

    vectors = np.array([c.vector for c in chunks], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors_norm = vectors / norms

    kept_indices: list[int] = []
    kept_vecs: list = []

    for i, vec in enumerate(vectors_norm):
        if not kept_vecs:
            kept_indices.append(i)
            kept_vecs.append(vec)
            continue
        sims = np.array(kept_vecs) @ vec
        if float(sims.max()) < threshold:
            kept_indices.append(i)
            kept_vecs.append(vec)

    return [chunks[i] for i in kept_indices]


def process_and_embed(units: List[dict], provider: EmbeddingProvider) -> List[EmbeddedChunk]:
    """Chunk retrieval units, embed non-empty chunks, and return all EmbeddedChunks.

    Chunks with embedding_text="" are included in the output with vector=[]
    (they were already excluded from chunk production by chunk_units, so this
    path currently produces no skipped chunks — kept explicit for future-proofing).

    Dedup strategy: paragraph/row/document chunks are dedup'd per source page to
    avoid false positives from repeated headings across pages. Tables and figures
    are dedup'd globally since they don't legitimately repeat.
    """
    chunks = chunk_units(units)

    embeddable = [c for c in chunks if c.embedding_text != ""]
    skipped = [c for c in chunks if c.embedding_text == ""]

    if embeddable:
        texts = [c.embedding_text for c in embeddable]
        vectors = provider.embed(texts)
        for chunk, vec in zip(embeddable, vectors):
            chunk.vector = vec

        _PER_PAGE_TYPES = {"paragraph", "row", "document"}
        para_chunks = [c for c in embeddable if c.chunk_type in _PER_PAGE_TYPES]
        other_chunks = [c for c in embeddable if c.chunk_type not in _PER_PAGE_TYPES]

        # Dedup paragraph/row/document chunks within each page separately
        page_groups: dict = {}
        for c in para_chunks:
            page = (c.metadata.get("source_pages") or [0])[0]
            page_groups.setdefault(page, []).append(c)
        deduped_paras: List[EmbeddedChunk] = []
        for page_chunks in page_groups.values():
            deduped_paras.extend(_deduplicate_by_cosine(page_chunks))

        other_chunks = _deduplicate_by_cosine(other_chunks)
        embeddable = deduped_paras + other_chunks

    return embeddable + skipped
