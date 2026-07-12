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
from .rerank import rerank, _fuse


def _norm(s: str) -> str:
    """归一化字符串用于精确匹配：半角+去空白+小写。"""
    return s.strip().lower()


class ProductMapper:
    def __init__(self, nodes=None, embedder_type: str = None):
        self.nodes = nodes or load_nodes()
        self.by_id = {n.id: n for n in self.nodes}
        self.embedder_type = embedder_type or config.EMBEDDER
        self.recaller = get_recall(self.nodes)

        # 精确匹配索引：node name / synonym → node（用于短路，免走 LLM）
        self._exact_index = {}
        for n in self.nodes:
            key = _norm(n.name)
            if key not in self._exact_index:
                self._exact_index[key] = n
            for syn in n.synonyms:
                key = _norm(syn)
                if key not in self._exact_index:
                    self._exact_index[key] = n

    def add_synonym(self, node_id: int, synonym: str) -> bool:
        """同步运行时节点与精确匹配索引，供同义词反馈写回后立即生效。"""
        synonym = (synonym or "").strip()
        node = self.by_id.get(int(node_id))
        if not node or not synonym:
            return False
        if synonym not in node.synonyms:
            node.synonyms.append(synonym)
        key = _norm(synonym)
        if key not in self._exact_index:
            self._exact_index[key] = node
        return True

    def set_embedder(self, embedder_type: str) -> float:
        """动态切换 embedder，返回重建耗时（秒）。trigram 索引不重建。"""
        from .embedder import st_available
        if embedder_type == "st" and not st_available():
            raise RuntimeError("sentence-transformers 未安装，无法切换到 STEmbedder")
        if hasattr(self.recaller, 'rebuild'):
            dt = self.recaller.rebuild(embedder_type)
        else:
            self.recaller = get_recall(self.nodes)
            dt = getattr(self.recaller, 'build_seconds', 0)
        self.embedder_type = embedder_type
        return dt

    def _exact_result(self, product: str, node, t0: float, reason: str) -> dict:
        """构造精确匹配的返回结果。"""
        return {
            "product": product,
            "node_id": node.id,
            "name": node.name,
            "path": node.path_str,
            "confidence": 1.0,
            "reason": reason,
            "source": "exact_match",
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "n_candidates": 1,
            "lexical_quality": "strong",
            "core_overlap": True,
        }

    def map(self, product: str, topk_candidates: int = None, verbose: bool = False):
        t0 = time.time()

        # 精确匹配短路：产品名 == 节点名或同义词 → 直接返回，省去召回+LLM
        exact = self._exact_index.get(_norm(product))
        if exact is not None:
            result = self._exact_result(product, exact, t0,
                                        f"精确匹配：产品名与节点「{exact.name}」一致")
            if verbose:
                self._print(result, [])
            return result

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

        # 精确匹配短路：跳过 LLM，但仍跑召回+融合，展示全部候选供对比
        exact = self._exact_index.get(_norm(product))
        if exact is not None:
            cands = self.recaller.recall(product)
            ordered = _fuse(cands, product)[:config.K_RERANK]
            result = self._exact_result(product, exact, t0,
                                        f"精确匹配：产品名与节点「{exact.name}」一致")
            result["n_candidates"] = len(cands)
            cand_list = [{
                "id": c.node.id, "name": c.node.name, "path": c.node.path_str,
                "trgm": round(c.trgm, 3), "vec": round(c.vec, 3),
                "fused": round(c.fused, 3), "synonyms": c.node.synonyms[:6],
                "lexical_quality": c.lexical_quality,
                "core_overlap": c.core_overlap,
                "core_terms": c.core_terms,
                "chosen": (c.node.id == exact.id),
            } for c in ordered]
            return result, cand_list

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
            "lexical_quality": c.lexical_quality,
            "core_overlap": c.core_overlap,
            "core_terms": c.core_terms,
            "chosen": (c.node.id == result["node_id"]),
        } for c in view]
        return result, cand_list

    @staticmethod
    def _print(result, cands):
        source_tag = {"exact_match": "⚡精确匹配", "llm": "DeepSeek 精排",
                      "fusion": "融合分兜底", "empty": "无候选"}.get(result.get("source"), result.get("source", ""))
        print(f"\n产品：{result['product']}  （候选 {result['n_candidates']} 个，"
              f"{result['latency_ms']}ms，来源={source_tag}）")
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
