"""召回层（内存后端）：双路召回 = trigram 字面 ∪ 向量语义。

- 字面路：字符三元组 + 倒排索引（模拟 pg_trgm 的 GIN 索引，只对含公共
  trigram 的节点打分，避免全量扫描）。
- 向量路：节点 search_text 的 embedding 矩阵，与查询做余弦相似度。

统一返回候选列表 [{node, trgm, vec}]，供精排层融合。
"""
import time
from collections import defaultdict

import numpy as np

from . import config
from .embedder import get_embedder
from .text import trigrams


class Candidate:
    __slots__ = ("node", "trgm", "vec", "fused")

    def __init__(self, node, trgm=0.0, vec=0.0):
        self.node = node
        self.trgm = trgm
        self.vec = vec
        self.fused = 0.0


class MemoryRecall:
    def __init__(self, nodes):
        self.nodes = nodes
        self.embedder = get_embedder()
        self._build()

    def _build(self):
        t0 = time.time()
        # 每个节点的字面文本集合（名称 + 同义词），及其 trigram 集合
        self._node_trisets = []      # index -> set(trigram)
        self._inverted = defaultdict(list)  # trigram -> [node_index]
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
        self._emb = self.embedder.encode(texts)   # (N, dim)，已 L2 归一化
        self.build_seconds = time.time() - t0

    # ── 字面召回 ────────────────────────────────────────────────
    def _recall_trgm(self, query, k):
        q = trigrams(query)
        if not q:
            return []
        # 倒排：累计与查询共享的 trigram 数
        shared = defaultdict(int)
        for tg in q:
            for idx in self._inverted.get(tg, ()):  # 只碰有交集的节点
                shared[idx] += 1
        scored = []
        for idx, inter in shared.items():
            union = len(q | self._node_trisets[idx])
            scored.append((idx, inter / union if union else 0.0))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

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

    # ── 合并 ────────────────────────────────────────────────────
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
