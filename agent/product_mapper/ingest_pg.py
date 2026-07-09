"""把节点写入 Postgres 并建索引（RECALL_BACKEND=pg 前先跑一次）。

需要：pip install "psycopg[binary]"，且 docker 的 pgvector 已启动。
"""
from . import config
from .taxonomy import load_nodes
from .embedder import get_embedder


def main():
    import psycopg

    nodes = load_nodes()
    embedder = get_embedder()
    dim = getattr(embedder, "dim", config.EMBED_DIM)
    texts = [n.search_text() for n in nodes]
    embs = embedder.encode(texts)

    conn = psycopg.connect(config.PG_DSN)
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
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
        cur.execute("""
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
        """)
    conn.commit()
    print(f"已写入 {len(nodes)} 节点并建立 GIN(trgm)+HNSW(vector) 索引，dim={dim}")


if __name__ == "__main__":
    main()
