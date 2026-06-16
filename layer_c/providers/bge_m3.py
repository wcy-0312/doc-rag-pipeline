from __future__ import annotations

from typing import List

from layer_c.providers import EmbeddingProvider


class BGEm3Provider(EmbeddingProvider):
    """BGE-M3 embedding provider using FlagEmbedding.BGEM3FlagModel.

    Loaded lazily on first call to embed() so that importing this module
    does not require the GPU/FlagEmbedding dependency to be present.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", batch_size: int = 32, use_fp16: bool = True):
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self._model = None

    def _load_model(self):
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel  # noqa: PLC0415
            self._model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16)

    def embed(self, texts: List[str]) -> List[List[float]]:
        self._load_model()
        results = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            output = self._model.encode(batch, batch_size=self.batch_size, max_length=8192)
            dense_vecs = output["dense_vecs"]
            for vec in dense_vecs:
                results.append(vec.tolist())
        return results
