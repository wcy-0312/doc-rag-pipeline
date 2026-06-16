from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


def _make_fake_flag_embedding(vec_dim: int = 8):
    """Build a minimal FlagEmbedding stub so import succeeds without GPU."""
    fake_module = ModuleType("FlagEmbedding")

    class FakeBGEM3FlagModel:
        def __init__(self, model_name, use_fp16=True):
            self.model_name = model_name
            self.use_fp16 = use_fp16

        def encode(self, texts, batch_size=32, max_length=8192):
            import numpy as np  # noqa: PLC0415
            return {"dense_vecs": [np.zeros(vec_dim) for _ in texts]}

    fake_module.BGEM3FlagModel = FakeBGEM3FlagModel
    return fake_module


@pytest.fixture(autouse=True)
def patch_flag_embedding():
    """Inject a fake FlagEmbedding module for every test in this file."""
    fake = _make_fake_flag_embedding()
    sys.modules["FlagEmbedding"] = fake
    yield
    sys.modules.pop("FlagEmbedding", None)


@pytest.fixture()
def provider():
    from layer_c.providers.bge_m3 import BGEm3Provider  # noqa: PLC0415
    return BGEm3Provider(model_name="BAAI/bge-m3", batch_size=32, use_fp16=True)


# ---------------------------------------------------------------------------
# Lazy loading
# ---------------------------------------------------------------------------

def test_model_is_none_before_first_embed(provider):
    assert provider._model is None


def test_model_loaded_after_embed(provider):
    provider.embed(["hello"])
    assert provider._model is not None


# ---------------------------------------------------------------------------
# Return value shape
# ---------------------------------------------------------------------------

def test_embed_returns_list_of_lists(provider):
    result = provider.embed(["text one", "text two", "text three"])
    assert isinstance(result, list)
    assert len(result) == 3
    for vec in result:
        assert isinstance(vec, list)


def test_embed_length_matches_input(provider):
    texts = [f"sample text {i}" for i in range(10)]
    result = provider.embed(texts)
    assert len(result) == len(texts)


# ---------------------------------------------------------------------------
# Batch splitting
# ---------------------------------------------------------------------------

def test_large_input_splits_into_multiple_batches():
    """100 texts with batch_size=32 must call encode() at least 4 times."""
    call_log = []

    class TrackingModel:
        def encode(self, texts, batch_size=32, max_length=8192):
            import numpy as np  # noqa: PLC0415
            call_log.append(len(texts))
            return {"dense_vecs": [np.zeros(8) for _ in texts]}

    from layer_c.providers.bge_m3 import BGEm3Provider  # noqa: PLC0415
    p = BGEm3Provider(batch_size=32)
    p._model = TrackingModel()  # skip lazy loading

    texts = [f"text {i}" for i in range(100)]
    result = p.embed(texts)

    assert len(result) == 100
    assert len(call_log) == 4  # ceil(100/32) = 4 batches
    assert call_log == [32, 32, 32, 4]


def test_input_smaller_than_batch_uses_single_call():
    """5 texts with batch_size=32 must call encode() exactly once."""
    call_log = []

    class TrackingModel:
        def encode(self, texts, batch_size=32, max_length=8192):
            import numpy as np  # noqa: PLC0415
            call_log.append(len(texts))
            return {"dense_vecs": [np.zeros(8) for _ in texts]}

    from layer_c.providers.bge_m3 import BGEm3Provider  # noqa: PLC0415
    p = BGEm3Provider(batch_size=32)
    p._model = TrackingModel()

    result = p.embed([f"text {i}" for i in range(5)])
    assert len(result) == 5
    assert call_log == [5]


# ---------------------------------------------------------------------------
# encode() called with correct kwargs
# ---------------------------------------------------------------------------

def test_encode_called_with_correct_kwargs():
    """encode() must receive batch_size=32 and max_length=8192."""
    encode_kwargs = []

    class InspectingModel:
        def encode(self, texts, **kwargs):
            import numpy as np  # noqa: PLC0415
            encode_kwargs.append(kwargs)
            return {"dense_vecs": [np.zeros(8) for _ in texts]}

    from layer_c.providers.bge_m3 import BGEm3Provider  # noqa: PLC0415
    p = BGEm3Provider(batch_size=32)
    p._model = InspectingModel()

    p.embed(["a", "b"])
    assert len(encode_kwargs) == 1
    assert encode_kwargs[0]["batch_size"] == 32
    assert encode_kwargs[0]["max_length"] == 8192
