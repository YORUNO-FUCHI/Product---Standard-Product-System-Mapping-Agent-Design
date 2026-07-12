"""精排层：双路候选 + LLM 判断，从候选中选出唯一标准节点。

- 候选来自 pg_trgm/trigram 与 pgvector。
- 排序分数以向量为主，trgm 只用于补充候选和展示，不单独主导兜底。
- LLM 判断：把产品 + 候选（含完整路径与同义词）交给 DeepSeek，返回
  {node_id, confidence, reason}；若都不合适返回 node_id=null（供体系扩展子题）。
"""
from . import config
from .llm import chat_json

SYSTEM = (
    "你是产品分类专家。给定一个【待映射产品】和若干【候选标准节点】"
    "（每个含唯一 id、从顶层到叶子的完整路径、同义词、pg_trgm 字面分、pgvector 语义分），"
    "请选出最准确对应的唯一节点。只能从候选 id 中选择；"
    "判断时优先看产品本体和用途，不要只因为 5G、4G、智能、设备、系统等修饰词相似就选择；"
    "pg_trgm 只代表字面召回证据，不能单独作为最终分类依据；"
    "若没有任何候选合适，node_id 返回 null。"
    '严格输出 JSON：{"node_id": <int或null>, "confidence": <0~1小数>, "reason": "<简短理由>"}'
)


def _fuse(cands, product: str = ""):
    """排序以向量为主；纯 trgm 候选排在向量候选之后，供 LLM 参考。"""
    if not cands:
        return []
    for c in cands:
        c.lexical_quality = "vector_only"
        c.core_overlap = ""
        c.core_terms = []
        vec = max(0.0, min(float(c.vec or 0.0), 1.0))
        trgm = max(0.0, min(float(c.trgm or 0.0), 1.0))
        c.fused = vec if vec > 0 else trgm * 0.35
    return sorted(cands, key=lambda c: c.fused, reverse=True)


def _format_candidates(cands):
    lines = []
    for c in cands:
        n = c.node
        syn = ("，同义词：" + "、".join(n.synonyms[:8])) if n.synonyms else ""
        lines.append(
            f"- id={n.id}｜路径：{n.path_str}"
            f"｜pg_trgm={float(c.trgm or 0.0):.3f}"
            f"｜pgvector={float(c.vec or 0.0):.3f}"
            f"｜排序分={float(c.fused or 0.0):.3f}{syn}"
        )
    return "\n".join(lines)


def rerank(product, cands, k_rerank=None):
    """返回 dict：{node_id, name, path, confidence, reason, source}。"""
    k_rerank = k_rerank or config.K_RERANK
    ordered = _fuse(cands, product)[:k_rerank]
    if not ordered:
        return {"node_id": None, "name": None, "path": None,
                "confidence": 0.0, "reason": "无候选", "source": "empty"}

    by_id = {c.node.id: c.node for c in ordered}

    # LLM 精排
    if config.has_llm():
        user = (f"待映射产品：{product}\n\n候选标准节点：\n"
                f"{_format_candidates(ordered)}")
        out = chat_json(SYSTEM, user)
        if out and "node_id" in out:
            nid = out.get("node_id")
            node = by_id.get(nid) if nid is not None else None
            chosen = next((c for c in ordered if c.node.id == nid), None)
            confidence = float(out.get("confidence", 0.0) or 0.0)
            reason = out.get("reason", "")
            return {
                "node_id": nid if node else None,
                "name": node.name if node else None,
                "path": node.path_str if node else None,
                "confidence": confidence,
                "reason": reason,
                "source": "llm",
                "lexical_quality": getattr(chosen, "lexical_quality", ""),
                "core_overlap": getattr(chosen, "core_overlap", False),
            }

    # 兜底：融合分 Top-1
    top = ordered[0]
    return {
        "node_id": top.node.id,
        "name": top.node.name,
        "path": top.node.path_str,
        "confidence": round(max(0.0, min(top.vec, 1.0)), 3),
        "reason": "LLM 未启用，按向量相似度选 Top-1；pg_trgm 仅作为候选补充",
        "source": "fusion",
        "lexical_quality": top.lexical_quality,
        "core_overlap": top.core_overlap,
    }
