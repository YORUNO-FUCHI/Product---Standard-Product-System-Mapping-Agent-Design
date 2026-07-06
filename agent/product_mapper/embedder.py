"""可插拔 Embedding。

- HashingEmbedder：零依赖占位向量（字符 n-gram 哈希到定长向量）。能立即运行，
  但语义能力弱，仅用于打通向量召回链路；正式实验请切换到 'st'。
- STEmbedder：本地 sentence-transformers（如 bge-small-zh），语义质量好。
  需 `pip install sentence-transformers`，并设 EMBEDDER=st。

DeepSeek API 无 embedding 接口，故向量来源不走 DeepSeek。
"""
import numpy as np

from . import config
from .text import norm


class HashingEmbedder:
    """字符 1/2-gram 哈希嵌入，带符号哈希，L2 归一化。确定性、无需下载。"""

    def __init__(self, dim: int = None):
        self.dim = dim or config.EMBED_DIM

    def _grams(self, s: str):
        s = norm(s)
        for ch in s:                       # unigram
            yield ch
        for i in range(len(s) - 1):        # bigram
            yield s[i:i + 2]

    def encode(self, texts) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, t in enumerate(texts):
            for g in self._grams(t or ""):
                h = hash(g)
                idx = h % self.dim
                sign = 1.0 if (h >> 32) & 1 else -1.0
                vecs[r, idx] += sign
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class STEmbedder:
    """sentence-transformers 后端（可选）。"""

    def __init__(self, model_name: str = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name or config.ST_MODEL)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts) -> np.ndarray:
        return np.asarray(
            self.model.encode(list(texts), normalize_embeddings=True,
                              show_progress_bar=False),
            dtype=np.float32,
        )

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def get_embedder():
    if config.EMBEDDER == "st":
        return STEmbedder()
    return HashingEmbedder()
