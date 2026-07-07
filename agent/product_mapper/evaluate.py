"""评测：Route A（RAG）vs Route B（PageIndex）对比。

- Recall@K：正确节点是否落入召回候选（衡量召回层）。
- Top-1 准确率：经精排/树搜索后是否等于正确节点。
- 延迟 & 成本对比。
"""
import time

from . import config
from .taxonomy import load_nodes
from .recall import get_recall
from .rerank import rerank
from .evalset import load as load_eval
from .pageindex_mapper import PageIndexMapper


def evaluate(n_eval: int = 200, n_llm_limit: int = 100):
    """在评测集上对 Route A 和 Route B 进行对比评测。

    Args:
        n_eval: 用于召回评测的样本数
        n_llm_limit: 用于 LLM 端到端评测的样本数（控制 API 费用）
    """
    nodes = load_nodes()
    recaller = get_recall(nodes)
    pairs = load_eval()
    pi_mapper = PageIndexMapper(nodes)

    print(f"{'='*60}")
    print(f"评测集总量: {len(pairs)} 条")
    print(f"召回评测: {min(n_eval, len(pairs))} 条 | 端到端评测: {min(n_llm_limit, len(pairs))} 条")
    print(f"LLM: {'DeepSeek 已启用' if config.has_llm() else '未启用(降级)'}")
    print(f"{'='*60}")

    # ═══════════════════════════════════════════════════════════════
    # 1. 召回评测（共用同一 recaller）
    # ═══════════════════════════════════════════════════════════════
    sub = pairs[:n_eval]
    t0 = time.time()
    route_a_hit1 = route_a_hitk = 0

    for p in sub:
        cands = recaller.recall(p["product"])
        ids = [c.node.id for c in sorted(
            cands, key=lambda c: (c.trgm + c.vec), reverse=True)]
        if ids and ids[0] == p["node_id"]:
            route_a_hit1 += 1
        if p["node_id"] in ids:
            route_a_hitk += 1

    dt_a_recall = time.time() - t0

    print(f"\n{'─'*40}")
    print(f"【Route A: RAG 召回】")
    print(f"  Recall@1（融合分）: {route_a_hit1/len(sub):.3f}")
    print(f"  Recall@K（K≈{config.K_TRGM}+{config.K_VEC}）: {route_a_hitk/len(sub):.3f}")
    print(f"  平均延迟: {dt_a_recall/len(sub)*1000:.1f} ms/条")
    print(f"  总耗时: {dt_a_recall:.1f}s")

    # ═══════════════════════════════════════════════════════════════
    # 2. 端到端对比（Route A LLM 精排 vs Route B PageIndex 树搜索）
    # ═══════════════════════════════════════════════════════════════
    if config.has_llm():
        n_llm = min(n_llm_limit, len(pairs))
        sub_llm = pairs[:n_llm]

        # ── Route A: RAG + LLM 精排 ──
        route_a_correct = 0
        route_a_latencies = []
        t0 = time.time()
        for p in sub_llm:
            t1 = time.time()
            cands = recaller.recall(p["product"])
            res = rerank(p["product"], cands)
            if res["node_id"] == p["node_id"]:
                route_a_correct += 1
            route_a_latencies.append((time.time() - t1) * 1000)
        dt_a = time.time() - t0

        # ── Route B: PageIndex 树搜索 ──
        route_b_correct = 0
        route_b_latencies = []
        route_b_layers = []
        t0 = time.time()
        for p in sub_llm:
            t1 = time.time()
            res = pi_mapper.map(p["product"])
            if res["node_id"] == p["node_id"]:
                route_b_correct += 1
            route_b_latencies.append((time.time() - t1) * 1000)
            route_b_layers.append(res.get("n_layers_visited", 0))
        dt_b = time.time() - t0

        # ── 打印对比报告 ──
        print(f"\n{'─'*40}")
        print(f"【端到端 Top-1 准确率对比（{n_llm} 条）】")
        print()
        print(f"{'指标':<25} {'Route A (RAG+LLM)':<22} {'Route B (PageIndex)':<22}")
        print(f"{'-'*25} {'-'*22} {'-'*22}")
        print(f"{'Top-1 准确率':<25} {route_a_correct/n_llm:<22.3f} {route_b_correct/n_llm:<22.3f}")
        print(f"{'正确数':<25} {route_a_correct:<22} {route_b_correct:<22}")
        print(f"{'总耗时':<25} {dt_a:<21.1f}s {dt_b:<21.1f}s")
        print(f"{'平均延迟/条':<25} {sum(route_a_latencies)/len(route_a_latencies):<21.0f}ms {sum(route_b_latencies)/len(route_b_latencies):<21.0f}ms")
        print(f"{'延迟中位数':<25} {sorted(route_a_latencies)[len(route_a_latencies)//2]:<21.0f}ms {sorted(route_b_latencies)[len(route_b_latencies)//2]:<21.0f}ms")
        print(f"{'平均搜索层数':<25} {'N/A (2-stage)':<22} {sum(route_b_layers)/len(route_b_layers):<21.1f}")
        print()

        # Route A 来源分布
        sources_a = {}
        # We'd need to track sources more carefully...

        # Route B 来源分布（exact vs tree search）
        sources_b = {"exact_match": 0, "tree_search": 0}
        # re-process briefly to count
        for p in sub_llm[:20]:  # just sample
            res = pi_mapper.map(p["product"])
            src = res.get("source", "")
            if "exact" in src:
                sources_b["exact_match"] += 1
            else:
                sources_b["tree_search"] += 1

        print(f"{'─'*40}")
        print(f"【Route B 来源分布（前 20 条采样）】")
        print(f"  精确匹配短路: {sources_b['exact_match']}")
        print(f"  树搜索: {sources_b['tree_search']}")

        # ── 错误分析 ──
        print(f"\n{'─'*40}")
        print(f"【错误样例对比（前 5 条）】")
        shown = 0
        for p in sub_llm:
            if shown >= 5:
                break
            res_a = rerank(p["product"], recaller.recall(p["product"]))
            res_b = pi_mapper.map(p["product"])
            a_ok = res_a["node_id"] == p["node_id"]
            b_ok = res_b["node_id"] == p["node_id"]
            if not a_ok or not b_ok:
                shown += 1
                print(f"\n  产品: {p['product']}")
                print(f"  正确答案: [{p['node_id']}] {p['true_path']}")
                if a_ok:
                    print(f"  Route A: [OK] 正确")
                else:
                    print(f"  Route A: [XX] [{res_a['node_id']}] {res_a.get('path', 'N/A')}")
                if b_ok:
                    print(f"  Route B: [OK] 正确")
                else:
                    print(f"  Route B: [XX] [{res_b['node_id']}] {res_b.get('path', 'N/A')}")

    else:
        print("\n[提示] 未配置 DeepSeek key，跳过 LLM 端到端评测。")

    # ═══════════════════════════════════════════════════════════════
    # 3. 精确匹配覆盖率（Route B 特色）
    # ═══════════════════════════════════════════════════════════════
    exact_hits = 0
    for p in pairs[:n_eval]:
        res = pi_mapper.map(p["product"])
        if "exact" in res.get("source", ""):
            exact_hits += 1
    print(f"\n{'─'*40}")
    print(f"【Route B 精确匹配覆盖率（{min(n_eval, len(pairs))} 条）】")
    print(f"  精确命中: {exact_hits} ({exact_hits/min(n_eval, len(pairs)):.1%})")
    print(f"  需树搜索: {min(n_eval, len(pairs)) - exact_hits} ({(min(n_eval, len(pairs)) - exact_hits)/min(n_eval, len(pairs)):.1%})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Route A vs Route B 评测对比")
    parser.add_argument("--eval", type=int, default=200, help="召回评测样本数")
    parser.add_argument("--llm", type=int, default=100, help="LLM 端到端评测样本数")
    args = parser.parse_args()
    evaluate(n_eval=args.eval, n_llm_limit=args.llm)
