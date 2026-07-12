"""全局配置：从 .env 读取，集中管理路径 / 模型 / 后端选择。

改这里的少量开关即可在【纯 Python 内存后端】与【Postgres+pgvector 后端】、
【哈希占位向量】与【本地 bge-m3 语义向量】之间切换，业务代码无需改动。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录（本文件的上上级）
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# ── 数据与缓存路径 ────────────────────────────────────────────────
EXCEL_PATH = ROOT / "产品标准体系.xlsx"
CACHE_DIR = ROOT / "cache"
NODES_CACHE = CACHE_DIR / "nodes.json"        # 解析后的节点缓存
EVALSET_PATH = CACHE_DIR / "evalset.json"     # 由同义词自动构造的评测集

# ── 召回后端：'memory'（现在就能跑）| 'pg'（装 Docker 后切换）────────
RECALL_BACKEND = os.getenv("RECALL_BACKEND", "memory")

# ── Embedding：'hash'（零依赖占位）| 'st'（本地 sentence-transformers）─
EMBEDDER = os.getenv("EMBEDDER", "hash")
EMBED_DIM = int(os.getenv("EMBED_DIM", "256"))        # hash 向量维度
ST_MODEL = os.getenv("ST_MODEL", "BAAI/bge-small-zh-v1.5")  # st 模式用的模型

# ── 召回参数 ──────────────────────────────────────────────────────
K_TRGM = int(os.getenv("K_TRGM", "20"))   # 字面召回 Top-K
K_VEC = int(os.getenv("K_VEC", "20"))     # 向量召回 Top-K
K_RERANK = int(os.getenv("K_RERANK", "12"))  # 送给 LLM 精排的候选数

# ── DeepSeek（OpenAI 兼容 chat 接口；DeepSeek 无 embedding 接口）──────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
LLM_PROXY = os.getenv("LLM_PROXY", "")

# ── Postgres 连接（RECALL_BACKEND=pg 时使用）──────────────────────
PG_DSN = os.getenv("PG_DSN", "postgresql://postgres:postgres@localhost:5432/taxonomy")

# ── 同义词反馈环：pgvector 高相似 + pg_trgm 零字面 → LLM 判断 → 人工确认写回 ──
SYN_FEEDBACK_ENABLED = os.getenv("SYN_FEEDBACK_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
SYN_FEEDBACK_VEC_THRESHOLD = float(os.getenv("SYN_FEEDBACK_VEC_THRESHOLD", "0.95"))
SYN_FEEDBACK_TRGM_THRESHOLD = float(os.getenv("SYN_FEEDBACK_TRGM_THRESHOLD", "0.0"))
SYN_FEEDBACK_AUTO_APPROVE = os.getenv("SYN_FEEDBACK_AUTO_APPROVE", "false").lower() in {"1", "true", "yes", "on"}


def has_llm() -> bool:
    """是否已配置 DeepSeek key（未配置时精排自动降级为按分数选 Top-1）。"""
    return bool(DEEPSEEK_API_KEY)
