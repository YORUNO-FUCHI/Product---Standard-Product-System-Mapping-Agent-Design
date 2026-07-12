"""召回层（Postgres + pgvector 后端）——装 Docker 后启用。

启用步骤：
  1) cd docker && docker compose up -d          # 起 pgvector 服务
  2) python -m product_mapper.ingest_pg          # 建表 + 写节点 + 建索引
  3) 在 .env 设 RECALL_BACKEND=pg
业务代码（agent/rerank/evaluate）无需改动，接口与内存后端一致。

需要：pip install "psycopg[binary]"
"""
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


class PgRecall:
    def __init__(self, nodes):
        import psycopg
        self.conn = psycopg.connect(config.PG_DSN)
        self.by_id = {n.id: n for n in nodes}
        self.embedder = get_embedder()

    def recall(self, query, k_trgm=None, k_vec=None):
        k_vec = k_vec or config.K_VEC
        qv = self.embedder.encode_one(query).tolist()
        merged = {}

        with self.conn.cursor() as cur:
            # 向量路：pgvector 余弦距离（HNSW 索引），相似度 = 1 - 距离
            cur.execute(
                """
                SELECT category_id, 1 - (embedding <=> %s::vector) AS sim
                FROM product_taxonomy
                ORDER BY embedding <=> %s::vector LIMIT %s
                """,
                (qv, qv, k_vec),
            )
            for cid, sim in cur.fetchall():
                merged.setdefault(cid, Candidate(self.by_id[cid]))
                merged[cid].vec = float(sim)

        return list(merged.values())
