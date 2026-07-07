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

# 修复 conda 环境 SSL_CERT_FILE 可能指向不存在路径的问题
import os as _os
_ssl = _os.environ.get("SSL_CERT_FILE", "")
if _ssl and not _os.path.exists(_ssl):
    try:
        import certifi as _certifi
        _os.environ["SSL_CERT_FILE"] = _certifi.where()
    except ImportError:
        _os.environ.pop("SSL_CERT_FILE", None)


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
    """sentence-transformers 后端（可选）。

    模型加载优先级：
    1. 显式路径（如 .env 的 ST_MODEL）且该路径存在 → local_files_only=True
    2. 项目本地 models/<model_id> 目录（免下载、秒切）→ local_files_only=True
    3. HuggingFace Hub 在线下载（首次较慢）
    """

    _AUTO_PATHS = None  # 首次初始化时计算

    @classmethod
    def _local_model_path(cls, model_name: str):
        """若 model_name 是 HF repo id（如 BAAI/bge-small-zh-v1.5），
        查找项目本地 models/ 目录下是否已有副本。"""
        if cls._AUTO_PATHS is None:
            cls._AUTO_PATHS = []
            from . import config as _cfg
            # 项目根同级（agent/）下的 models 目录
            sibling = _cfg.ROOT.parent / "models"
            if sibling.is_dir():
                cls._AUTO_PATHS.append(sibling)
        for base in cls._AUTO_PATHS:
            # 支持完整 repo id（如 BAAI/bge-small-zh-v1.5）或短名
            for name in [model_name, model_name.split("/")[-1] if "/" in model_name else model_name]:
                candidate = base / name
                if candidate.is_dir():
                    return str(candidate)
        return None

    def __init__(self, model_name: str = None):
        self._fix_ssl()
        from sentence_transformers import SentenceTransformer
        import os as _os
        model_path = model_name or config.ST_MODEL

        # 优先级：显式本地路径 > 自动检测 > 在线
        if _os.path.isdir(model_path):
            # 用户配置了本地路径，直接使用
            local_only = True
        elif (auto := self._local_model_path(model_path)) is not None:
            model_path = auto
            local_only = True
        else:
            local_only = False

        self.model = SentenceTransformer(model_path, local_files_only=local_only)
        self.dim = self.model.get_embedding_dimension()

    @staticmethod
    def _fix_ssl():
        """修复 SSL_CERT_FILE 指向不存在路径的问题。"""
        import os
        ssl_path = os.environ.get("SSL_CERT_FILE", "")
        if ssl_path and not os.path.exists(ssl_path):
            try:
                import certifi
                os.environ["SSL_CERT_FILE"] = certifi.where()
            except ImportError:
                os.environ.pop("SSL_CERT_FILE", None)

    def encode(self, texts) -> np.ndarray:
        return np.asarray(
            self.model.encode(list(texts), normalize_embeddings=True,
                              show_progress_bar=False),
            dtype=np.float32,
        )

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


_ST_AVAILABLE = None


def _check_st():
    global _ST_AVAILABLE
    if _ST_AVAILABLE is not None:
        return _ST_AVAILABLE
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        _ST_AVAILABLE = True
    except ImportError:
        _ST_AVAILABLE = False
    return _ST_AVAILABLE


def get_embedder(embedder_type: str = None):
    """返回 embedder 实例。embedder_type 可选 'hash' / 'st'，默认走 config。"""
    t = embedder_type or config.EMBEDDER
    if t == "st" and _check_st():
        return STEmbedder()
    return HashingEmbedder()


def st_available() -> bool:
    return _check_st()
