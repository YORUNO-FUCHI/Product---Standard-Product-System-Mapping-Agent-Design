"""用同义词自动构造评测集：每个同义词 = 一个"输入产品"，正确答案 = 其所属节点。

这是数据集里 6 万+ 条同义词的免费复用：无需人工标注即可得到大规模
（产品 → 正确 node_id）评测对。可控制每个节点最多取几条、总量上限。
"""
import json
import random

from . import config
from .taxonomy import load_nodes


def build(max_per_node: int = 2, limit: int = 2000, seed: int = 42):
    nodes = load_nodes()
    rng = random.Random(seed)
    pairs = []
    for n in nodes:
        if not n.synonyms:
            continue
        syns = list(dict.fromkeys(n.synonyms))  # 去重保序
        rng.shuffle(syns)
        for s in syns[:max_per_node]:
            if s and s != n.name:
                pairs.append({"product": s, "node_id": n.id,
                              "true_path": n.path_str})
    rng.shuffle(pairs)
    if limit:
        pairs = pairs[:limit]

    config.CACHE_DIR.mkdir(exist_ok=True)
    config.EVALSET_PATH.write_text(
        json.dumps(pairs, ensure_ascii=False, indent=1), encoding="utf-8")
    return pairs


def load():
    if not config.EVALSET_PATH.exists():
        return build()
    return json.loads(config.EVALSET_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    pairs = build()
    print(f"评测集已生成：{len(pairs)} 条 → {config.EVALSET_PATH}")
    for p in pairs[:5]:
        print("  ", p["product"], "→", p["true_path"])
