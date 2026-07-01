"""精排层：多策略融合 + LLM 判断，从候选中选出唯一标准节点。

- 融合分数 = 归一化(trgm) + 归一化(vec)，用于排序、LLM 缺失时的兜底。
- LLM 判断：把产品 + 候选（含完整路径与同义词）交给 DeepSeek，返回
  {node_id, confidence, reason}；若都不合适返回 node_id=null（供体系扩展子题）。
"""
from . import config
from .llm import chat_json

SYSTEM = (
    "你是产品分类专家。给定一个【待映射产品】和若干【候选标准节点】"
    "（每个含唯一 id、从顶层到叶子的完整路径、同义词），"
    "请选出最准确对应的唯一节点。只能从候选 id 中选择；"
    "若没有任何候选合适，node_id 返回 null。"
    '严格输出 JSON：{"node_id": <int或null>, "confidence": <0~1小数>, "reason": "<简短理由>"}'
)


def _fuse(cands):
    """按归一化融合分排序，返回排序后的候选列表。"""
    if not cands:
        return []
    tmax = max((c.trgm for c in cands), default=0.0) or 1.0
    vmax = max((c.vec for c in cands), default=0.0) or 1.0
    for c in cands:
        c.fused = c.trgm / tmax + c.vec / vmax
    return sorted(cands, key=lambda c: c.fused, reverse=True)


def _format_candidates(cands):
    lines = []
    for c in cands:
        n = c.node
        syn = ("，同义词：" + "、".join(n.synonyms[:8])) if n.synonyms else ""
        lines.append(f"- id={n.id}｜路径：{n.path_str}{syn}")
    return "\n".join(lines)


def rerank(product, cands, k_rerank=None):
    """返回 dict：{node_id, name, path, confidence, reason, source}。"""
    k_rerank = k_rerank or config.K_RERANK
    ordered = _fuse(cands)[:k_rerank]
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
            return {
                "node_id": nid if node else None,
                "name": node.name if node else None,
                "path": node.path_str if node else None,
                "confidence": float(out.get("confidence", 0.0) or 0.0),
                "reason": out.get("reason", ""),
                "source": "llm",
            }

    # 兜底：融合分 Top-1
    top = ordered[0]
    return {
        "node_id": top.node.id,
        "name": top.node.name,
        "path": top.node.path_str,
        "confidence": round(min(top.fused / 2, 1.0), 3),
        "reason": "LLM 未启用，按 trgm+向量融合分选 Top-1",
        "source": "fusion",
    }
