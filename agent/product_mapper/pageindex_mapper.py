"""PageIndex 路线（Route B）：无向量、LLM 在体系树上逐层推理搜索。

核心思想（源自 PageIndex 项目）：
  - 不依赖向量库，LLM 直接看当前层的孩子列表
  - 逐层推理选择 → 下钻 → 直到叶子
  - 输出可追溯的完整推理路径

宽节点（>50 孩子）处理：
  - Phase 1：trigram 字面预筛到 Top-30
  - Phase 2：LLM 在 30 个候选中精挑
"""

import time
import re
from collections import Counter

from . import config
from .taxonomy import load_nodes, build_json_tree, Node
from .llm import chat_json
from .text import core_overlap as lexical_core_overlap
from .text import lexical_quality, similarity as trigram_similarity


# ── 系统提示词 ────────────────────────────────────────────────────
SYSTEM_PROMPT = """你是产品分类专家。给你一个【待映射产品】和当前层的若干【候选子节点】（含名称与同义词），
请选出产品最应该归入的那个子节点。

规则：
1. 从候选列表中选出最准确的 1 个子节点
2. 如果产品明显不属于任何候选，返回 selected_index: -1
3. 考虑产品的核心功能/材质/用途，忽略品牌、型号、规格前缀
4. 同一父节点下的子节点之间是互斥的，应选语义最接近的

严格输出 JSON：
{"selected_index": <int, 0-based, -1 表示都不合适>,
 "confidence": <0~1 小数>,
 "reason": "<简短理由>"}"""

LEAF_SYSTEM = """你是产品分类专家。当前已到达叶子节点层，请判断该产品是否应映射到此叶子节点。
如果不是，从候选兄弟节点中另选一个，或返回 -1 表示都不合适。

严格输出 JSON：
{"selected_index": <int, 0-based, -1 表示都不合适>,
 "confidence": <0~1 小数>,
 "reason": "<简短理由>"}"""


class PageIndexMapper:
    """PageIndex 路线：LLM 在体系树上逐层推理搜索。

    用法：
        mapper = PageIndexMapper()
        result = mapper.map("苞米")
        # → {node_id, name, path, confidence, reason, source: "pageindex", trace: [...]}
    """

    def __init__(self, nodes: list = None):
        self.nodes = nodes or load_nodes()
        self.by_id = {n.id: n for n in self.nodes}

        # 构建树形结构
        self.tree = build_json_tree(self.nodes)

        # 节点 → 孩子映射
        self._children_of: dict[int, list] = {}
        for n in self.nodes:
            self._children_of.setdefault(n.parent_id, []).append(n)

        # 节点 → 父节点映射
        self._parent_of: dict[int, int] = {}
        for n in self.nodes:
            self._parent_of[n.id] = n.parent_id

        # 精确匹配索引
        self._exact_index = {}
        for n in self.nodes:
            key = n.name.strip().lower()
            if key not in self._exact_index:
                self._exact_index[key] = n
            for syn in n.synonyms:
                key = syn.strip().lower()
                if key not in self._exact_index:
                    self._exact_index[key] = n

        # 预计算节点 trigram 集合（用于快速相似度计算）
        self._node_trigrams: dict[int, set] = {}
        self._node_name_trigrams: dict[int, set] = {}
        for n in self.nodes:
            from .text import trigrams
            self._node_trigrams[n.id] = trigrams(n.search_text())
            names = [n.name] + n.synonyms
            tset = set()
            for name in names:
                tset |= trigrams(name)
            self._node_name_trigrams[n.id] = tset

    def _get_children(self, node_id: int) -> list:
        """获取某节点的子节点列表（按名称排序）。"""
        children = self._children_of.get(node_id, [])
        return sorted(children, key=lambda n: n.name)

    def _get_root_children(self) -> list:
        """获取虚拟根（-1）的直接子节点 → 体系树的顶层类别。"""
        return self._get_children(-1)

    def _format_options(self, nodes: list, max_syn: int = 5) -> str:
        """格式化候选节点列表供 LLM 阅读。"""
        lines = []
        for i, n in enumerate(nodes):
            syn_text = ""
            if n.synonyms:
                syn_list = n.synonyms[:max_syn]
                syn_text = f"  [同义词: {', '.join(syn_list)}]"
            lines.append(f"{i}. {n.name}{syn_text}")
        return "\n".join(lines)

    def _prefilter(self, product: str, candidates: list, top_k: int = 30) -> list:
        """用 trigram 相似度预筛候选（宽节点时使用）。"""
        if len(candidates) <= top_k:
            return candidates

        from .text import trigrams as get_trigrams
        prod_tri = get_trigrams(product)
        scored = []
        for n in candidates:
            node_tri = self._node_trigrams.get(n.id, set())
            if not prod_tri or not node_tri:
                score = 0.0
            else:
                inter = len(prod_tri & node_tri)
                score = inter / len(prod_tri | node_tri) if inter > 0 else 0.0
            scored.append((score, n))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in scored[:top_k]]

    def _pick_from_options(self, product: str, options: list, is_leaf_level: bool = False) -> dict:
        """让 LLM 从候选中选一个，返回 {index, confidence, reason} 或 None。"""
        if not options:
            return None
        if len(options) == 1:
            return {"selected_index": 0, "confidence": 0.9,
                    "reason": f"唯一候选：{options[0].name}"}

        system = LEAF_SYSTEM if is_leaf_level else SYSTEM_PROMPT
        user = f"待映射产品：{product}\n\n候选子节点：\n{self._format_options(options)}"

        result = chat_json(system, user)
        if result and "selected_index" in result:
            idx = result.get("selected_index", -1)
            if idx is not None and 0 <= idx < len(options):
                return {
                    "selected_index": idx,
                    "confidence": float(result.get("confidence", 0.7) or 0.7),
                    "reason": result.get("reason", ""),
                }
            elif idx == -1 or idx is None:
                return {"selected_index": -1, "confidence": 0.0,
                        "reason": result.get("reason", "LLM 判断无合适候选")}

        # 降级：LLM 不可用，用 trigram 相似度选最优候选
        return self._trigram_pick(product, options)

    def _trigram_pick(self, product: str, options: list) -> dict:
        """LLM 不可用时，用 trigram 相似度从候选中选最优。"""
        from .text import trigrams as get_trigrams
        prod_tri = get_trigrams(product)
        best_idx, best_score = 0, 0.0
        for i, n in enumerate(options):
            node_tri = self._node_trigrams.get(n.id, set())
            if not prod_tri or not node_tri:
                score = 0.0
            else:
                inter = len(prod_tri & node_tri)
                score = inter / len(prod_tri | node_tri) if inter > 0 else 0.0
            if score > best_score:
                best_score = score
                best_idx = i
        return {
            "selected_index": best_idx,
            "confidence": round(best_score, 3),
            "reason": f"trigram 相似度={best_score:.3f}，选「{options[best_idx].name}」",
        }

    def _llm_available(self) -> bool:
        return config.has_llm()

    def explain(self, product: str) -> tuple:
        """给前端用：返回 (result, trace_list)，与 Route A 的 explain 接口对齐。"""
        result = self.map(product)
        trace_list = result.get("trace", [])
        # 转换为前端兼容格式
        cand_list = []
        for t in trace_list:
            n = self.by_id.get(t["node_id"])
            cand_list.append({
                "id": t["node_id"],
                "name": t["name"],
                "path": n.path_str if n else t["name"],
                "confidence": t["confidence"],
                "reason": t["reason"],
                "chosen": True,
            })
        return result, cand_list

    def map(self, product: str, max_depth: int = 10, verbose: bool = False) -> dict:
        """在体系树上逐层推理搜索，返回映射结果。"""
        t0 = time.time()
        trace = []

        # 精确匹配短路
        exact = self._exact_index.get(product.strip().lower())
        if exact is not None:
            return {
                "product": product,
                "node_id": exact.id,
                "name": exact.name,
                "path": exact.path_str,
                "confidence": 1.0,
                "reason": f"精确匹配：产品名与节点「{exact.name}」一致",
                "source": "pageindex_exact",
                "latency_ms": round((time.time() - t0) * 1000, 1),
                "trace": [],
            }

        if not self._llm_available():
            # 无 LLM 时用 trigram 在全量节点中匹配
            return self._fallback_trigram(product, t0)

        # ── 逐层推理搜索 ──
        current_level = self._get_root_children()
        selected_node = None
        path_nodes = []

        for depth in range(max_depth):
            if not current_level:
                break

            if verbose:
                print(f"  [depth={depth}] 当前层 {len(current_level)} 个候选")

            # 宽节点预筛
            options = self._prefilter(product, current_level, top_k=30)
            if verbose and len(current_level) > 30:
                print(f"    trigram 预筛: {len(current_level)} → {len(options)}")

            is_leaf = all(n.is_leaf for n in options)

            pick = self._pick_from_options(product, options, is_leaf_level=is_leaf)

            if pick is None or pick["selected_index"] == -1:
                # LLM 判断都不合适，停在当前层
                if verbose:
                    print(f"    LLM 判断无合适候选，停止下钻")
                break

            selected_node = options[pick["selected_index"]]
            path_nodes.append({
                "node_id": selected_node.id,
                "name": selected_node.name,
                "confidence": pick["confidence"],
                "reason": pick["reason"],
            })
            trace.append(path_nodes[-1])

            if verbose:
                print(f"    → 选择: {selected_node.name} (id={selected_node.id}, "
                      f"conf={pick['confidence']:.2f})")

            # 到达叶子 → 停止
            if selected_node.is_leaf:
                if verbose:
                    print(f"    → 到达叶子节点，搜索完成")
                break

            # 下钻一层
            current_level = self._get_children(selected_node.id)

        # ── 构造结果 ──
        if selected_node is None:
            return {
                "product": product,
                "node_id": None,
                "name": None,
                "path": None,
                "confidence": 0.0,
                "reason": "从根层开始未找到合适类别",
                "source": "pageindex_empty",
                "latency_ms": round((time.time() - t0) * 1000, 1),
                "trace": trace,
            }

        return {
            "product": product,
            "node_id": selected_node.id,
            "name": selected_node.name,
            "path": selected_node.path_str,
            "confidence": round(trace[-1]["confidence"] if trace else 0.5, 3),
            "reason": " > ".join(
                f"[{t['name']}]({t['reason']})" for t in trace
            ),
            "source": "pageindex",
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "n_layers_visited": len(trace),
            "trace": trace,
            "lexical_quality": lexical_quality(product, selected_node.name, trace[-1]["confidence"] if trace else 0.5),
            "core_overlap": lexical_core_overlap(product, selected_node.name),
        }

    def _fallback_trigram(self, product: str, t0: float) -> dict:
        """无 LLM 时的候选式降级方案：全量节点匹配 Top-1。

        返回弱匹配而不是直接置空，便于实验表观察 Route B 的候选覆盖能力。
        """
        from .text import trigrams as get_trigrams
        prod_tri = get_trigrams(product)
        boosted_product = _fallback_alias(product)
        boosted_tri = get_trigrams(boosted_product)
        best_node, best_score = None, 0.0
        for n in self.nodes:
            score = _jaccard(prod_tri, self._node_name_trigrams.get(n.id, set()))
            score = max(score, _jaccard(prod_tri, self._node_trigrams.get(n.id, set())) * 0.85)
            if boosted_product != product:
                score = max(score, _jaccard(boosted_tri, self._node_name_trigrams.get(n.id, set())) * 0.95)
                score = max(score, _jaccard(boosted_tri, self._node_trigrams.get(n.id, set())) * 0.80)
            if score > best_score:
                best_score = score
                best_node = n

        best_core_overlap = lexical_core_overlap(product, best_node.name) if best_node else False
        best_quality = lexical_quality(product, best_node.name, best_score) if best_node else ""

        if best_node and best_score >= 0.18 and best_core_overlap and best_quality != "noisy":
            return {
                "product": product,
                "node_id": best_node.id,
                "name": best_node.name,
                "path": best_node.path_str,
                "confidence": round(best_score, 3),
                "reason": "trigram 全量匹配（无 LLM）",
                "source": "pageindex_trigram",
                "latency_ms": round((time.time() - t0) * 1000, 1),
                "trace": [],
                "lexical_quality": best_quality,
                "core_overlap": best_core_overlap,
            }

        if best_node and best_score >= 0.08:
            reason = "Route B 弱匹配候选（无 LLM，需人工复核）"
            if not best_core_overlap or best_quality == "noisy":
                reason = "Route B 字面相似但核心词不一致，降为弱匹配候选"
            return {
                "product": product,
                "node_id": best_node.id,
                "name": best_node.name,
                "path": best_node.path_str,
                "confidence": round(best_score, 3),
                "reason": reason,
                "source": "pageindex_trigram_weak",
                "latency_ms": round((time.time() - t0) * 1000, 1),
                "trace": [],
                "weak_match": True,
                "lexical_quality": best_quality,
                "core_overlap": best_core_overlap,
            }

        return {
            "product": product,
            "node_id": None,
            "name": None,
            "path": None,
            "confidence": 0.0,
            "reason": "trigram 也无合适匹配",
            "source": "pageindex_empty",
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "trace": [],
        }


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / len(a | b) if inter else 0.0


def _fallback_alias(product: str) -> str:
    """Map common long product suffixes to broader taxonomy-friendly terms."""
    p = product.strip()
    if any(x in p for x in ("测定试剂盒", "检测试剂盒", "诊断试剂盒", "试剂盒")):
        return "生物试剂盒"
    if any(x in p for x in ("水质检测仪", "水质测试仪", "检测仪", "测试仪", "监测仪")):
        return "测试仪器"
    if any(x in p for x in ("比色计", "分析仪")):
        return "分析仪器"
    return p


# ── 模块入口 ───────────────────────────────────────────────────────
if __name__ == "__main__":
    mapper = PageIndexMapper()
    print(f"PageIndex 映射器就绪：{len(mapper.nodes)} 节点，"
          f"LLM={'已启用' if mapper._llm_available() else '未启用(trigram降级)'}")

    test_products = ["苞米", "独头蒜", "Vigna radiata", "红富士苹果", "笔记本电脑",
                     "工业机器人", "电动汽车电池", "太阳能光伏组件"]

    for p in test_products:
        result = mapper.map(p, verbose=True)
        source_tag = {
            "pageindex_exact": "[精确匹配]",
            "pageindex": "[PageIndex 树搜索]",
            "pageindex_trigram": "[trigram 降级]",
            "pageindex_empty": "[无结果]",
        }.get(result["source"], result["source"])

        print(f"\n产品：{p}  （{result['latency_ms']}ms，来源={source_tag}）")
        if result["node_id"]:
            print(f"  → 命中节点 id={result['node_id']}｜{result['path']}")
            print(f"     置信度={result['confidence']}｜理由：{result['reason']}")
        else:
            print(f"  → 未找到合适节点")
        print()
