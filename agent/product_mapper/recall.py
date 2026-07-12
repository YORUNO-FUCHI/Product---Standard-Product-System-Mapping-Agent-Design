"""召回层（内存后端）：双路召回 = trigram 字面 ∪ 向量语义。

trigram 只负责扩充候选池；最终排序/兜底不让字面分单独主导。
"""
import time
from collections import defaultdict

import numpy as np

from . import config
from .embedder import get_embedder
from .text import trigrams


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
        # trigram 倒排索引（和 embedder 无关，只建一次）
        self._node_trisets = []
        self._inverted = defaultdict(list)
        for i, n in enumerate(self.nodes):
            names = [n.name] + n.synonyms
            tset = set()
            for nm in names:
                tset |= trigrams(nm)
            self._node_trisets.append(tset)
            for tg in tset:
                self._inverted[tg].append(i)

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

    # ── 字面召回 ────────────────────────────────────────────────
    def _recall_trgm(self, query, k):
        q = trigrams(query)
        if not q:
            return []
        shared = defaultdict(int)
        for tg in q:
            for idx in self._inverted.get(tg, ()):
                shared[idx] += 1
        scored = []
        for idx, inter in shared.items():
            union = len(q | self._node_trisets[idx])
            scored.append((idx, inter / union if union else 0.0))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def recall(self, query, k_trgm=None, k_vec=None):
        k_trgm = k_trgm or config.K_TRGM
        k_vec = k_vec or config.K_VEC
        merged = {}
        for idx, s in self._recall_trgm(query, k_trgm):
            merged.setdefault(idx, Candidate(self.nodes[idx]))
            merged[idx].trgm = s
        for idx, s in self._recall_vec(query, k_vec):
            merged.setdefault(idx, Candidate(self.nodes[idx]))
            merged[idx].vec = s
        return list(merged.values())


def get_recall(nodes):
    if config.RECALL_BACKEND == "pg":
        from .recall_pg import PgRecall
        return PgRecall(nodes)
    return MemoryRecall(nodes)
