from __future__ import annotations

import json
import pathlib
from typing import List

import pytest

from layer_c.pipeline import process_and_embed
from layer_c.providers import EmbeddingProvider

STRUCTURE_OUTPUT = pathlib.Path(__file__).parent.parent.parent / "output" / "layer_b"


class NullEmbeddingProvider(EmbeddingProvider):
    """Returns zero-length vectors for every text (no model required)."""

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [[] for _ in texts]


def _load(filename: str) -> list:
    with open(STRUCTURE_OUTPUT / filename, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def null_provider():
    return NullEmbeddingProvider()


# ---------------------------------------------------------------------------
# Chunk-count regression tests
# ---------------------------------------------------------------------------

def test_mri_chunk_count(null_provider):
    units = _load("retrieval_units_MRI報告_2024.json")
    chunks = process_and_embed(units, null_provider)
    assert len(chunks) == 276, f"MRI expected 276, got {len(chunks)}"


def test_tcm_chunk_count(null_provider):
    units = _load("retrieval_units_中醫護理衛教指導.json")
    chunks = process_and_embed(units, null_provider)
    assert len(chunks) == 1, f"TCM expected 1, got {len(chunks)}"


def test_nursing_quality_chunk_count(null_provider):
    units = _load("retrieval_units_護理品質監測.json")
    chunks = process_and_embed(units, null_provider)
    assert len(chunks) == 9, f"Nursing expected 9, got {len(chunks)}"


# ---------------------------------------------------------------------------
# Invariant tests across all three documents
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def all_chunks(null_provider):
    all_units = (
        _load("retrieval_units_MRI報告_2024.json")
        + _load("retrieval_units_中醫護理衛教指導.json")
        + _load("retrieval_units_護理品質監測.json")
    )
    return process_and_embed(all_units, null_provider)


def test_retrieval_weight_in_range(all_chunks):
    for chunk in all_chunks:
        w = chunk.metadata.get("retrieval_weight")
        assert w is not None, f"{chunk.chunk_id} missing retrieval_weight"
        assert 0.0 <= w <= 1.0, f"{chunk.chunk_id} retrieval_weight={w} out of [0,1]"


def test_embedding_text_not_modified(null_provider):
    """pipeline must not mutate embedding_text on any chunk."""
    units = _load("retrieval_units_護理品質監測.json")
    # record original texts from chunker directly
    from layer_c.chunker import chunk_units
    original_texts = {c.chunk_id: c.embedding_text for c in chunk_units(units)}

    result = process_and_embed(units, null_provider)
    for chunk in result:
        assert chunk.embedding_text == original_texts[chunk.chunk_id], (
            f"{chunk.chunk_id} embedding_text was mutated by pipeline"
        )


def test_null_provider_vectors_are_empty(all_chunks):
    for chunk in all_chunks:
        assert chunk.vector == [], f"{chunk.chunk_id} vector should be [] from NullProvider"


def test_no_chunk_has_empty_embedding_text(all_chunks):
    """chunk_units skips empty embedding_text; pipeline output must contain none."""
    for chunk in all_chunks:
        assert chunk.embedding_text != "", f"{chunk.chunk_id} has empty embedding_text in output"
