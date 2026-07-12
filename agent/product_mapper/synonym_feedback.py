"""同义词反馈环：高向量相似、零字面相似时生成可复核的同义词写回任务。"""
from __future__ import annotations

import queue
import threading
import time
import uuid
from datetime import datetime
from typing import Any

from . import config
from .llm import chat_json


SYSTEM = """你是产品标准体系的同义词审核助手。
判断“输入产品名”是否可以作为“候选标准节点”的同义词写入标准体系。
只有当它表达的是同一类产品、同一标准节点，且不是更上位/更下位/无关产品时，才返回 true。
严格输出 JSON：
{"is_synonym": true/false, "confidence": 0~1, "reason": "简短理由"}"""


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _split_synonyms(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [x.strip() for x in str(raw).replace(",", "、").split("、") if x.strip()]


def _join_synonyms(items: list[str]) -> str:
    return "、".join(dict.fromkeys(x.strip() for x in items if x and x.strip()))


class SynonymFeedbackManager:
    def __init__(self):
        self.tasks: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.queue: queue.Queue[str] = queue.Queue()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()
        self.db_ready = False
        self.db_error = ""
        if self.supported:
            self.ensure_table()

    @property
    def supported(self) -> bool:
        return config.SYN_FEEDBACK_ENABLED and config.RECALL_BACKEND == "pg"

    def ensure_table(self) -> None:
        try:
            import psycopg

            with psycopg.connect(config.PG_DSN) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
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
                    )
                conn.commit()
            self.db_ready = True
            self.db_error = ""
        except Exception as e:
            self.db_ready = False
            self.db_error = str(e)

    def maybe_enqueue(self, product: str, candidates: list[dict], mapper=None) -> dict:
        base = {
            "enabled": config.SYN_FEEDBACK_ENABLED,
            "supported": self.supported,
            "triggered": False,
            "status": "not_triggered",
            "message": "",
            "vec_threshold": config.SYN_FEEDBACK_VEC_THRESHOLD,
            "trgm_threshold": config.SYN_FEEDBACK_TRGM_THRESHOLD,
        }
        if not config.SYN_FEEDBACK_ENABLED:
            base.update({"status": "disabled", "message": "同义词反馈环未启用"})
            return base
        if config.RECALL_BACKEND != "pg":
            base.update({"status": "unsupported", "message": "当前为 memory 后端，只有 Postgres 后端支持写回 syn_list"})
            return base
        if not self.db_ready:
            self.ensure_table()
        if not self.db_ready:
            base.update({"status": "failed", "message": f"反馈表不可用：{self.db_error}"})
            return base

        candidate = self._pick_candidate(candidates)
        if not candidate:
            return base

        existing = self._find_existing(product, int(candidate["id"]))
        if existing:
            return self._public(existing)

        task = {
            "task_id": uuid.uuid4().hex[:16],
            "product": product,
            "node_id": int(candidate["id"]),
            "node_name": candidate.get("name", ""),
            "node_path": candidate.get("path", ""),
            "vec": float(candidate.get("vec", 0.0) or 0.0),
            "trgm": float(candidate.get("trgm", 0.0) or 0.0),
            "status": "queued",
            "llm_decision": None,
            "llm_confidence": 0.0,
            "reason": "",
            "error": "",
            "mapper": mapper,
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "approved_at": "",
        }
        with self.lock:
            self.tasks[task["task_id"]] = task
        self._upsert_db(task)
        self.queue.put(task["task_id"])
        return self._public(task)

    def get(self, task_id: str) -> dict:
        with self.lock:
            task = self.tasks.get(task_id)
        if task:
            return self._public(task)
        task = self._load_db(task_id)
        return self._public(task) if task else {"error": "task not found"}

    def approve(self, task_id: str, mapper) -> dict:
        task = self._load_db(task_id)
        if not task:
            return {"error": "task not found"}
        if task["status"] == "approved":
            return self._public(task)
        if task["status"] != "pending_review":
            return {"error": f"task is not pending_review: {task['status']}", **self._public(task)}
        if task.get("llm_decision") is not True:
            return {"error": "LLM 未判断为可写入同义词", **self._public(task)}

        try:
            self._write_back(task, mapper)
            task["status"] = "approved"
            task["approved_at"] = now_iso()
            task["updated_at"] = now_iso()
            self._upsert_db(task)
            with self.lock:
                self.tasks[task_id] = task
            return self._public(task)
        except Exception as e:
            task["status"] = "failed"
            task["error"] = str(e)
            task["updated_at"] = now_iso()
            self._upsert_db(task)
            return {"error": str(e), **self._public(task)}

    def _pick_candidate(self, candidates: list[dict]) -> dict | None:
        eligible = []
        for c in candidates or []:
            vec = float(c.get("vec", 0.0) or 0.0)
            trgm = float(c.get("trgm", 0.0) or 0.0)
            if vec > config.SYN_FEEDBACK_VEC_THRESHOLD and trgm <= config.SYN_FEEDBACK_TRGM_THRESHOLD:
                eligible.append(c)
        eligible.sort(key=lambda x: float(x.get("vec", 0.0) or 0.0), reverse=True)
        return eligible[0] if eligible else None

    def _worker_loop(self):
        while True:
            task_id = self.queue.get()
            try:
                self._process(task_id)
            finally:
                self.queue.task_done()

    def _process(self, task_id: str) -> None:
        with self.lock:
            task = self.tasks.get(task_id)
        if not task:
            return
        task["status"] = "running"
        task["updated_at"] = now_iso()
        self._upsert_db(task)

        if not config.has_llm():
            task["status"] = "failed"
            task["error"] = "未配置 DeepSeek API Key，无法完成 LLM 同义词判断"
            task["updated_at"] = now_iso()
            self._upsert_db(task)
            with self.lock:
                self.tasks[task_id] = task
            return

        user = (
            f"输入产品名：{task['product']}\n"
            f"候选标准节点：{task['node_name']}\n"
            f"候选路径：{task['node_path']}\n"
            f"触发分数：pgvector={task['vec']:.3f}, pg_trgm={task['trgm']:.3f}\n"
            "请判断输入产品名是否适合作为该候选标准节点的同义词。"
        )
        out = chat_json(SYSTEM, user, timeout=45)
        if not out:
            task["status"] = "failed"
            task["error"] = "LLM 调用失败或未返回合法 JSON"
        else:
            decision = bool(out.get("is_synonym"))
            task["llm_decision"] = decision
            task["llm_confidence"] = float(out.get("confidence", 0.0) or 0.0)
            task["reason"] = str(out.get("reason", "") or "")
            if decision and config.SYN_FEEDBACK_AUTO_APPROVE:
                mapper = task.get("mapper")
                if mapper is None:
                    task["status"] = "failed"
                    task["error"] = "自动写回失败：缺少运行时 mapper"
                else:
                    try:
                        self._write_back(task, mapper)
                        task["status"] = "approved"
                        task["approved_at"] = now_iso()
                        task["reason"] = (task["reason"] + "；LLM 通过，已自动写回 syn_list").strip("；")
                    except Exception as e:
                        task["status"] = "failed"
                        task["error"] = f"自动写回失败：{e}"
            else:
                task["status"] = "pending_review" if decision else "rejected"
        task["updated_at"] = now_iso()
        self._upsert_db(task)
        with self.lock:
            self.tasks[task_id] = task

    def _write_back(self, task: dict, mapper) -> None:
        import psycopg

        node_id = int(task["node_id"])
        product = str(task["product"]).strip()
        node = mapper.by_id[node_id]
        runtime_synonyms = list(node.synonyms)
        if product not in runtime_synonyms:
            runtime_synonyms.append(product)
        search_text = " ".join([node.name] + runtime_synonyms + node.path_names[:-1])
        embedding = mapper.recaller.embedder.encode_one(search_text).tolist()

        with psycopg.connect(config.PG_DSN) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT synonyms FROM product_taxonomy WHERE category_id = %s",
                    (node_id,),
                )
                row = cur.fetchone()
                current = _split_synonyms(row[0] if row else "")
                if product not in current:
                    current.append(product)
                cur.execute(
                    """
                    UPDATE product_taxonomy
                    SET synonyms = %s, search_text = %s, embedding = %s
                    WHERE category_id = %s
                    """,
                    (_join_synonyms(current), search_text, list(map(float, embedding)), node_id),
                )
            conn.commit()
        mapper.add_synonym(node_id, product)

    def _find_existing(self, product: str, node_id: int) -> dict | None:
        try:
            import psycopg

            with psycopg.connect(config.PG_DSN) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT task_id, product, node_id, node_name, node_path, vec, trgm, status,
                               llm_decision, llm_confidence, reason, error, created_at, updated_at, approved_at
                        FROM product_synonym_feedback
                        WHERE product = %s AND node_id = %s
                        ORDER BY created_at DESC
                        LIMIT 1
                        """,
                        (product, node_id),
                    )
                    row = cur.fetchone()
            return self._row_to_task(row) if row else None
        except Exception:
            return None

    def _load_db(self, task_id: str) -> dict | None:
        try:
            import psycopg

            with psycopg.connect(config.PG_DSN) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT task_id, product, node_id, node_name, node_path, vec, trgm, status,
                               llm_decision, llm_confidence, reason, error, created_at, updated_at, approved_at
                        FROM product_synonym_feedback
                        WHERE task_id = %s
                        """,
                        (task_id,),
                    )
                    row = cur.fetchone()
            return self._row_to_task(row) if row else None
        except Exception:
            return None

    def _upsert_db(self, task: dict) -> None:
        try:
            import psycopg

            with psycopg.connect(config.PG_DSN) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO product_synonym_feedback
                        (task_id, product, node_id, node_name, node_path, vec, trgm, status,
                         llm_decision, llm_confidence, reason, error, created_at, updated_at, approved_at)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (task_id) DO UPDATE SET
                            status = EXCLUDED.status,
                            llm_decision = EXCLUDED.llm_decision,
                            llm_confidence = EXCLUDED.llm_confidence,
                            reason = EXCLUDED.reason,
                            error = EXCLUDED.error,
                            updated_at = EXCLUDED.updated_at,
                            approved_at = EXCLUDED.approved_at
                        """,
                        (
                            task["task_id"], task["product"], task["node_id"], task["node_name"],
                            task["node_path"], task["vec"], task["trgm"], task["status"],
                            task.get("llm_decision"), task.get("llm_confidence", 0.0),
                            task.get("reason", ""), task.get("error", ""), task.get("created_at", now_iso()),
                            task.get("updated_at", now_iso()), task.get("approved_at", ""),
                        ),
                    )
                conn.commit()
        except Exception as e:
            self.db_ready = False
            self.db_error = str(e)

    def _row_to_task(self, row) -> dict:
        keys = [
            "task_id", "product", "node_id", "node_name", "node_path", "vec", "trgm", "status",
            "llm_decision", "llm_confidence", "reason", "error", "created_at", "updated_at", "approved_at",
        ]
        return dict(zip(keys, row))

    def _public(self, task: dict) -> dict:
        if not task:
            return {"triggered": False, "status": "not_triggered"}
        return {
            "enabled": config.SYN_FEEDBACK_ENABLED,
            "supported": self.supported,
            "triggered": True,
            "task_id": task.get("task_id"),
            "status": task.get("status"),
            "product": task.get("product"),
            "node_id": task.get("node_id"),
            "node_name": task.get("node_name"),
            "node_path": task.get("node_path"),
            "vec": float(task.get("vec", 0.0) or 0.0),
            "trgm": float(task.get("trgm", 0.0) or 0.0),
            "llm_decision": task.get("llm_decision"),
            "llm_confidence": float(task.get("llm_confidence", 0.0) or 0.0),
            "reason": task.get("reason", ""),
            "error": task.get("error", ""),
            "auto_approve": config.SYN_FEEDBACK_AUTO_APPROVE,
            "can_approve": task.get("status") == "pending_review" and task.get("llm_decision") is True,
            "message": task.get("error") or task.get("reason") or "",
        }


MANAGER = SynonymFeedbackManager()
