"""召回层（内存后端）：Route A 只使用向量语义召回。

字面 trigram / pg_trgm 不再参与 Route A 候选召回，避免局部词相似造成误召回。
"""
import time

import numpy as np

from . import config
from .embedder import get_embedder


class Candidate:
    __slots__ = ("node", "trgm", "vec", "fused", "lexical_quality", "core_overlap", "core_terms")

    def __init__(self, node, trgm=0.0, vec=0.0):
        self.node = node
        self.trgm = trgm
        self.vec = vec
        self.fused = 0.0
        self.lexical_quality = ""
        self.core_overlap = False
        self.core_terms = []


class MemoryRecall:
    def __init__(self, nodes, embedder_type: str = None):
        self.nodes = nodes
        self.embedder_type = embedder_type or config.EMBEDDER
        self.embedder = get_embedder(self.embedder_type)
        self._emb_cache = {}  # embedder_type → (embedder, emb_matrix)
        self._build()

    def _build(self):
        t0 = time.time()
        # 向量矩阵
        texts = [n.search_text() for n in self.nodes]
        self._emb = self.embedder.encode(texts)
        self.build_seconds = time.time() - t0

    def rebuild(self, embedder_type: str):
        """切换 embedder 并重建向量矩阵（优先用预计算缓存）。"""
        if embedder_type == self.embedder_type:
            return 0
        t0 = time.time()
        self.embedder_type = embedder_type

        if embedder_type in self._emb_cache:
            self.embedder, self._emb = self._emb_cache[embedder_type]
        else:
            self.embedder = get_embedder(embedder_type)
            texts = [n.search_text() for n in self.nodes]
            self._emb = self.embedder.encode(texts)
        return time.time() - t0

    # ── 向量召回 ────────────────────────────────────────────────
    def _recall_vec(self, query, k):
        qv = self.embedder.encode_one(query)
        sims = self._emb @ qv                      # 余弦（均已归一化）
        if k >= len(sims):
            idxs = np.argsort(-sims)
        else:
            idxs = np.argpartition(-sims, k)[:k]
            idxs = idxs[np.argsort(-sims[idxs])]
        return [(int(i), float(sims[i])) for i in idxs]

    def recall(self, query, k_trgm=None, k_vec=None):
        k_vec = k_vec or config.K_VEC
        merged = {}
        for idx, s in self._recall_vec(query, k_vec):
            merged.setdefault(idx, Candidate(self.nodes[idx]))
            merged[idx].vec = s
        return list(merged.values())


def get_recall(nodes):
    if config.RECALL_BACKEND == "pg":
        from .recall_pg import PgRecall
        return PgRecall(nodes)
    return MemoryRecall(nodes)
