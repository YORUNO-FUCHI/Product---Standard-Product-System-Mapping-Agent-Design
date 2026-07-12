"""把节点写入 Postgres 并建索引（RECALL_BACKEND=pg 前先跑一次）。

需要：pip install "psycopg[binary]"，且 docker 的 pgvector 已启动。
"""
from . import config
from .taxonomy import load_nodes
from .embedder import get_embedder


FEEDBACK_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS product_synonym_feedback (
        task_id TEXT PRIMARY KEY,
        product TEXT NOT NULL,
        node_id INT NOT NULL,
        node_name TEXT,
        node_path TEXT,
        vec REAL,
        trgm REAL,
        status TEXT NOT NULL,
        llm_decision BOOLEAN,
        llm_confidence REAL,
        reason TEXT,
        error TEXT,
        created_at TEXT,
        updated_at TEXT,
        approved_at TEXT
    );
"""


def _load_approved_feedback(cur) -> list[tuple[int, str]]:
    cur.execute(FEEDBACK_TABLE_SQL)
    cur.execute(
        """
        SELECT node_id, product
        FROM product_synonym_feedback
        WHERE status = 'approved'
        """
    )
    rows = []
    for node_id, product in cur.fetchall():
        product = (product or "").strip()
        if product:
            rows.append((int(node_id), product))
    return rows


def _merge_feedback_synonyms(nodes: list, approved: list[tuple[int, str]]) -> int:
    by_id = {n.id: n for n in nodes}
    merged = 0
    for node_id, product in approved:
        node = by_id.get(node_id)
        if node and product not in node.synonyms:
            node.synonyms.append(product)
            merged += 1
    return merged


def main():
    import psycopg

    nodes = load_nodes()
    conn = psycopg.connect(config.PG_DSN)
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        approved = _load_approved_feedback(cur)
        merged_count = _merge_feedback_synonyms(nodes, approved)

        embedder = get_embedder()
        dim = getattr(embedder, "dim", config.EMBED_DIM)
        texts = [n.search_text() for n in nodes]
        embs = embedder.encode(texts)

        cur.execute("DROP TABLE IF EXISTS product_taxonomy;")
        cur.execute(f"""
            CREATE TABLE product_taxonomy (
                category_id INT PRIMARY KEY,
                name        TEXT,
                parent_id   INT,
                depth       INT,
                path_names  TEXT,
                synonyms    TEXT,
                search_text TEXT,
                embedding   vector({dim})
            );
        """)
        rows = [
            (n.id, n.name, n.parent_id, n.depth, n.path_str,
             "、".join(n.synonyms), n.search_text(), list(map(float, embs[i])))
            for i, n in enumerate(nodes)
        ]
        cur.executemany(
            "INSERT INTO product_taxonomy VALUES (%s,%s,%s,%s,%s,%s,%s,%s)", rows)
        cur.execute("CREATE INDEX ON product_taxonomy "
                    "USING gin (search_text gin_trgm_ops);")
        cur.execute("CREATE INDEX ON product_taxonomy "
                    "USING hnsw (embedding vector_cosine_ops);")
    conn.commit()
    print(
        f"已写入 {len(nodes)} 节点并建立 GIN(trgm)+HNSW(vector) 索引，dim={dim}；"
        f"合并已确认同义词 {merged_count} 条"
    )


if __name__ == "__main__":
    main()
