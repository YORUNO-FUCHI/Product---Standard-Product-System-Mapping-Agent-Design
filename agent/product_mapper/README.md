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
- **精确匹配短路**：`agent.py` 在召回前先查精确索引（产品名 = node.name 或 synonym），
  命中则直接返回 `source=exact_match, confidence=1.0`，跳过召回与 LLM，0ms 延迟。
- **ST 模型本地自动检测**：`embedder.py` 按优先级查找模型：① `.env` 显式路径 →
  ② `agent/models/<model_id>` 自动发现 → ③ HuggingFace Hub 在线下载。
  预置 `bge-small-zh-v1.5` 免下载，加载时 `local_files_only=True`。
- **向量缓存与预热**：`server.py` 启动时预计算 ST 向量矩阵并注入 `MemoryRecall._emb_cache`，
  `rebuild()` 优先用缓存，Hash ↔ ST 切换 < 0.1s（无需重建 21k 节点编码）。
- **前端自动重搜**：切换 Embedder 后自动用当前输入重新查询，结果即时更新。
- **数据预处理流水线**：`preprocess.py` 对原始公司产品表（72 万条）进行 6 步清洗：
  去括号 → 去数字单位（英寸/V/Ah/吨级…）→ 去型号前缀 → 去颜色词 → 去品质形容词 → 残留清理。
  输出清洗对照 Excel + JSON，按 `cleaned/unchanged/empty/spec` 分类标记供审核。

## 目录
```
product_mapper/
  config.py        # 配置开关（后端/embedder/参数/DeepSeek）
  taxonomy.py      # Excel → 节点 + 树，缓存 JSON
  text.py          # trigram 相似度（对齐 pg_trgm）
  embedder.py      # HashingEmbedder / STEmbedder（本地模型自动发现）
  recall.py        # 内存双路召回 + 倒排索引 + 向量缓存
  recall_pg.py     # Postgres 后端（装 Docker 后启用）
  ingest_pg.py     # 写库 + 建 GIN/HNSW 索引
  llm.py           # DeepSeek chat 客户端
  rerank.py        # 融合 + LLM 精排
  agent.py         # ProductMapper.map() 入口（含精确匹配短路）
  evalset.py       # 用同义词自动造评测集
  evaluate.py      # Recall@K / Top-1 准确率
  server.py        # Web 可视化前端 + ST 预热 + 动态切换
  preprocess.py    # 公司产品名清洗：去颜色/尺寸/规格形容词
models/            # 本地 Embedding 模型（免下载）
  bge-small-zh-v1.5/
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

# 3.5) 数据预处理：清洗公司产品名（可选）
python -m product_mapper.preprocess --limit 5000  # 先跑 5000 条看效果
python -m product_mapper.preprocess               # 全量 72 万条

# 4) 跑演示：几个产品的映射结果
python -m product_mapper.agent

# 5) 评测：Recall@K（不花钱）+ Top-1 准确率（配了 key 才跑）
python -m product_mapper.evaluate
```

## 之后升级（无需改业务代码）
- **接 DeepSeek**：`.env` 填 `DEEPSEEK_API_KEY` → 精排从「融合分兜底」变为真实 LLM 判断。
- **切语义向量**：`pip install sentence-transformers`，`.env` 设 `EMBEDDER=st`
  （本地模型自动检测，无需配置路径，首次切换 < 0.1s）。
- **切 Postgres**：装 Docker → `cd docker && docker compose up -d` →
  `python -m product_mapper.ingest_pg` → `.env` 设 `RECALL_BACKEND=pg`。

## 最近更新
- **精确匹配短路**：产品名命中节点名/同义词直接返回，0ms、0 token。
- **ST 本地免下载**：`agent/models/` 预置模型，自动发现、离线加载。
- **向量缓存预热**：服务启动时预计算 ST 矩阵，前后端切换即时生效。
- **前端自动重搜**：切换 Embedder 后自动刷新查询结果。
- **Bug 修复**：Web UI 精确匹配显示为绿色 `⚡ 精确匹配` badge；
  切换提示由误导性的"需下载模型"改为"加载本地模型"。
- **数据预处理**：新增 `preprocess.py`，6 步流水线清洗 72 万条公司产品名，去除
  括号 → 数字单位（英寸/V/Ah/吨…）→ 型号前缀 → 颜色词 → 品质形容词 → 残留标点，
  输出 `cache/cleaned_products.xlsx` 对照表（171,536 已清洗 / 526,705 无变化）。

## 尚未包含（后续周次）
Route B（PageIndex 树上推理匹配）、子题 (3) 体系扩展建议与同义词反馈环、
消融实验与调优——见《产品映射智能体_实施计划.md》。
