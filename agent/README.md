# 产品 - 标准产品体系映射智能体

2026 小学期课程设计（选题 2）：**已知标准产品体系（树形结构），把输入的一个具体产品映射到树中唯一准确的节点。**

## 已实现（Route A：RAG 召回 + LLM 精排）
- **双路召回**：字符三元组字面匹配（对齐 `pg_trgm`，含倒排索引模拟 GIN）∪ 向量语义召回（对齐 `pgvector`）。
- **多策略融合 + DeepSeek 精排**：融合字面/语义分排序，交大模型从候选中判定唯一节点，输出 `node_id / 置信度 / 理由`。
- **自动评测**：用数据集里 6.2 万条同义词自造「产品 → 节点」评测集。实测 Recall@K=0.986，端到端 Top-1=0.96。
- **Web 可视化前端**：零依赖，浏览器演示召回候选打分与精排命中路径。

## 快速开始
```bash
cp .env.example .env          # 填入 DeepSeek key（不填也能跑，精排降级）
python -m product_mapper.taxonomy    # 解析体系树
python -m product_mapper.evalset     # 造评测集
python -m product_mapper.agent       # 演示映射
python -m product_mapper.evaluate    # 评测
python -m product_mapper.server      # Web 前端 → http://localhost:8000
```
详见 [`product_mapper/README.md`](product_mapper/README.md) 与 [`产品映射智能体_实施计划.md`](产品映射智能体_实施计划.md)。

## 目录
- `product_mapper/` — 智能体核心代码（召回 / 精排 / 评测 / 前端 / 双后端）
- `docker/` — Postgres+pgvector 的 docker-compose（切真实 PG 后端时用）
- `产品标准体系.xlsx` — 标准产品体系树数据集（21090 节点）
- `产品映射智能体_实施计划.md` — 四周实施计划与数据分析

## 待实现
Route B（PageIndex 无向量树搜索）、子题 (3) 体系扩展建议与同义词反馈环。

## 依赖与说明
- DeepSeek 仅提供 chat 接口（无 embedding），故向量由本地 Hashing/`bge-m3` 生成。
- 第三方 [PageIndex](https://github.com/VectifyAI/PageIndex)（Route B 参考）未纳入本仓库，可自行 clone。
