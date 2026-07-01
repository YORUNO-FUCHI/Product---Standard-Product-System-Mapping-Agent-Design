# 产品映射智能体 · 初步实现（Route A：RAG 召回 + LLM 精排）

把一个具体产品名映射到「标准产品体系树」中唯一准确的节点。本初步版实现了
**双路召回（trigram 字面 ∪ 向量语义）→ 多策略融合 + DeepSeek 精排** 的核心闭环，
并配套「用同义词自动构造评测集」与评测脚本。

## 设计要点
- **召回层可插拔双后端**：`memory`（纯 Python，现在就能跑）/ `pg`（装 Docker 后
  一键切换到真实 `pg_trgm`+`pgvector`，业务代码不变）。
- **trigram 相似度纯 Python 实现**，语义对齐 pg_trgm（前补 2 空格、后补 1 空格、
  长度 3 滑窗、Jaccard），并用**倒排索引**模拟 GIN，避免全量扫描。
- **Embedding 可插拔**：`hash`（零依赖占位）/ `st`（本地 bge-m3 语义向量）。
  DeepSeek 无 embedding 接口，故向量不走 DeepSeek。
- **LLM 精排走 DeepSeek**（OpenAI 兼容，仅用 requests）；未配置 key 时自动降级为
  按融合分选 Top-1，流程照样跑通。

## 目录
```
product_mapper/
  config.py        # 配置开关（后端/embedder/参数/DeepSeek）
  taxonomy.py      # Excel → 节点 + 树，缓存 JSON
  text.py          # trigram 相似度（对齐 pg_trgm）
  embedder.py      # HashingEmbedder / STEmbedder
  recall.py        # 内存双路召回 + 倒排索引
  recall_pg.py     # Postgres 后端（装 Docker 后启用）
  ingest_pg.py     # 写库 + 建 GIN/HNSW 索引
  llm.py           # DeepSeek chat 客户端
  rerank.py        # 融合 + LLM 精排
  agent.py         # ProductMapper.map() 入口
  evalset.py       # 用同义词自动造评测集
  evaluate.py      # Recall@K / Top-1 准确率
docker/            # pgvector 的 docker-compose（后续用）
```

## 快速开始（现在，零新增依赖）
```bash
# 1) 配置（可选：先不配 key 也能跑，精排会降级）
cp .env.example .env        # 填入 DeepSeek key 后可启用真实精排

# 2) 解析体系树（首次会读 Excel 并缓存）
python -m product_mapper.taxonomy

# 3) 造评测集（同义词 → 节点）
python -m product_mapper.evalset

# 4) 跑演示：几个产品的映射结果
python -m product_mapper.agent

# 5) 评测：Recall@K（不花钱）+ Top-1 准确率（配了 key 才跑）
python -m product_mapper.evaluate
```

## 之后升级（无需改业务代码）
- **接 DeepSeek**：`.env` 填 `DEEPSEEK_API_KEY` → 精排从「融合分兜底」变为真实 LLM 判断。
- **切语义向量**：`pip install sentence-transformers`，`.env` 设 `EMBEDDER=st`。
- **切 Postgres**：装 Docker → `cd docker && docker compose up -d` →
  `python -m product_mapper.ingest_pg` → `.env` 设 `RECALL_BACKEND=pg`。

## 尚未包含（后续周次）
Route B（PageIndex 树上推理匹配）、子题 (3) 体系扩展建议与同义词反馈环、
消融实验与调优——见《产品映射智能体_实施计划.md》。
