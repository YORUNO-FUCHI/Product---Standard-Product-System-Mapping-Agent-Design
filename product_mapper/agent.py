"""映射智能体入口（Route A）：召回 → 精排 → 唯一节点。

用法：
    from product_mapper.agent import ProductMapper
    mapper = ProductMapper()          # 首次会解析 Excel 并建索引
    print(mapper.map("苞米"))
"""
import time

from . import config
from .taxonomy import load_nodes
from .recall import get_recall
from .rerank import rerank


class ProductMapper:
    def __init__(self, nodes=None):
        self.nodes = nodes or load_nodes()
        self.by_id = {n.id: n for n in self.nodes}
        self.recaller = get_recall(self.nodes)

    def map(self, product: str, topk_candidates: int = None, verbose: bool = False):
        t0 = time.time()
        cands = self.recaller.recall(product)
        result = rerank(product, cands, k_rerank=topk_candidates)
        result["product"] = product
        result["latency_ms"] = round((time.time() - t0) * 1000, 1)
        result["n_candidates"] = len(cands)
        if verbose:
            self._print(result, cands)
        return result

    def explain(self, product: str):
        """给前端用：返回 (最终结果, 候选明细列表)，展示召回+精排全过程。"""
        t0 = time.time()
        cands = self.recaller.recall(product)
        result = rerank(product, cands)   # 会在 cands 上写入 .fused
        result["product"] = product
        result["latency_ms"] = round((time.time() - t0) * 1000, 1)
        result["n_candidates"] = len(cands)
        view = sorted(cands, key=lambda c: c.fused, reverse=True)[:config.K_RERANK]
        cand_list = [{
            "id": c.node.id, "name": c.node.name, "path": c.node.path_str,
            "trgm": round(c.trgm, 3), "vec": round(c.vec, 3),
            "fused": round(c.fused, 3), "synonyms": c.node.synonyms[:6],
            "chosen": (c.node.id == result["node_id"]),
        } for c in view]
        return result, cand_list

    @staticmethod
    def _print(result, cands):
        print(f"\n产品：{result['product']}  （候选 {result['n_candidates']} 个，"
              f"{result['latency_ms']}ms，来源={result['source']}）")
        if result["node_id"] is None:
            print("  → 未找到合适节点（可触发【体系扩展建议】子题）")
        else:
            print(f"  → 命中节点 id={result['node_id']}｜{result['path']}")
            print(f"     置信度={result['confidence']}｜理由：{result['reason']}")


if __name__ == "__main__":
    mapper = ProductMapper()
    print(f"索引就绪：{len(mapper.nodes)} 节点，"
          f"建索引耗时 {getattr(mapper.recaller, 'build_seconds', 0):.2f}s，"
          f"LLM={'已启用' if config.has_llm() else '未启用(降级)'}")
    for p in ["苞米", "独头蒜", "Vigna radiata", "红富士苹果", "笔记本电脑"]:
        mapper.map(p, verbose=True)
