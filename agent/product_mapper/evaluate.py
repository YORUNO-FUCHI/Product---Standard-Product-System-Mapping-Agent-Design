"""评测：在自动构造的评测集上计算召回与端到端准确率。

- Recall@K：正确节点是否落入召回候选（衡量召回层，不花 LLM 费用，可跑全量）。
- Top-1 准确率：经精排后选中的节点是否等于正确节点（走 LLM，控量以省成本）。
"""
import time

from . import config
from .taxonomy import load_nodes
from .recall import get_recall
from .rerank import rerank
from .evalset import load as load_eval


def evaluate(n_recall: int = 500, n_llm: int = 50):
    nodes = load_nodes()
    recaller = get_recall(nodes)
    pairs = load_eval()

    # ── 召回评测 ──
    sub = pairs[:n_recall]
    hit1 = hitk = 0
    t0 = time.time()
    for p in sub:
        cands = recaller.recall(p["product"])
        ids = [c.node.id for c in sorted(
            cands, key=lambda c: (c.trgm + c.vec), reverse=True)]
        if ids and ids[0] == p["node_id"]:
            hit1 += 1
        if p["node_id"] in ids:
            hitk += 1
    dt = time.time() - t0
    print(f"\n=== 召回评测（{len(sub)} 条）===")
    print(f"Recall@1（融合分）: {hit1/len(sub):.3f}")
    print(f"Recall@K（K≈{config.K_TRGM}+{config.K_VEC}）: {hitk/len(sub):.3f}")
    print(f"平均召回延迟: {dt/len(sub)*1000:.1f} ms/条")

    # ── 端到端精排评测（LLM）──
    if config.has_llm():
        subl = pairs[:n_llm]
        correct = 0
        t0 = time.time()
        for p in subl:
            cands = recaller.recall(p["product"])
            res = rerank(p["product"], cands)
            if res["node_id"] == p["node_id"]:
                correct += 1
        dt = time.time() - t0
        print(f"\n=== 端到端 Top-1 准确率（LLM 精排，{len(subl)} 条）===")
        print(f"Top-1 Accuracy: {correct/len(subl):.3f}")
        print(f"平均端到端延迟: {dt/len(subl)*1000:.1f} ms/条")
    else:
        print("\n[提示] 未配置 DeepSeek key，跳过 LLM 精排评测。"
              "配置后可得端到端 Top-1 准确率。")


if __name__ == "__main__":
    evaluate()
