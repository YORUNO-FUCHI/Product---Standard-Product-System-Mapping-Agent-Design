"""标准产品体系树的解析与加载。

把 `产品标准体系.xlsx` 解析成节点列表 + 父子关系，并缓存到 JSON。
Excel 字段：category_id / category_name / category_group_id /
category_pids / category_group_name / syn_list
"""
import json
import re
from dataclasses import dataclass, asdict, field

from . import config

_BRACKET = re.compile(r"\[(-?\d+)\]")


@dataclass
class Node:
    id: int
    name: str
    parent_id: int              # -1 表示挂在虚拟根下
    depth: int                  # 含自身的层级数（根下第一层为 1）
    path_ids: list              # 从顶层到自身的 id 链（不含根 -1）
    path_names: list            # 对应的名称链（含自身）
    synonyms: list              # 同义词
    is_leaf: bool = True

    @property
    def path_str(self) -> str:
        return " > ".join(self.path_names)

    def search_text(self) -> str:
        """用于字面匹配 / trigram / 语义门槛的综合文本（含路径上下文）。"""
        parts = [self.name] + self.synonyms + self.path_names[:-1]
        return " ".join(parts)

    def embed_text(self) -> str:
        """用于向量召回的编码文本：名称 + 同义词，**不含路径**。

        路径上下文会稀释核心产品词的向量（如「笔记本电脑 产业链 新型显示」中
        '产业链/新型显示' 拉低与「笔记本电脑」类查询的相似度）。去路径后节点向量
        聚焦产品本体，与"产品名"查询形态对齐；标注集实测中位名次 13→7、top10 47%→53%。
        """
        parts = [self.name] + self.synonyms
        return " ".join(p for p in parts if p)


def _parse_synonyms(raw) -> list:
    if not raw:
        return []
    try:
        arr = json.loads(raw)
        return [str(x).strip() for x in arr if str(x).strip()]
    except Exception:
        return []


def _build_from_excel() -> list:
    import openpyxl

    wb = openpyxl.load_workbook(config.EXCEL_PATH, read_only=True, data_only=True)
    ws = wb["Sheet1"]
    rows = ws.iter_rows(values_only=True)
    next(rows)  # 跳过表头

    raw = []          # (id, name, ancestors_excl_root, parent_id, synonyms)
    id2name = {}
    for r in rows:
        cid, cname, _cgid, pids, _gname, syn = r
        if cid is None:
            continue
        cid = int(cid)
        cname = (cname or "").strip()
        ids = [int(x) for x in _BRACKET.findall(pids or "")]  # 例: [-1, 2, 3, 4]
        ancestors = [i for i in ids if i != -1]
        parent_id = ids[-1] if ids else -1
        raw.append((cid, cname, ancestors, parent_id, _parse_synonyms(syn)))
        id2name[cid] = cname

    children_count = {}
    for cid, _n, _anc, parent_id, _syn in raw:
        children_count[parent_id] = children_count.get(parent_id, 0) + 1

    nodes = []
    for cid, cname, ancestors, parent_id, syn in raw:
        path_ids = ancestors + [cid]
        path_names = [id2name.get(i, str(i)) for i in ancestors] + [cname]
        nodes.append(Node(
            id=cid, name=cname, parent_id=parent_id,
            depth=len(path_ids), path_ids=path_ids, path_names=path_names,
            synonyms=syn, is_leaf=(children_count.get(cid, 0) == 0),
        ))
    return nodes


def load_nodes(use_cache: bool = True) -> list:
    """加载节点列表（优先读缓存，缺失则从 Excel 解析并写缓存）。"""
    if use_cache and config.NODES_CACHE.exists():
        data = json.loads(config.NODES_CACHE.read_text(encoding="utf-8"))
        return [Node(**d) for d in data]

    nodes = _build_from_excel()
    config.CACHE_DIR.mkdir(exist_ok=True)
    config.NODES_CACHE.write_text(
        json.dumps([asdict(n) for n in nodes], ensure_ascii=False),
        encoding="utf-8",
    )
    return nodes


def build_json_tree(nodes: list) -> list:
    """构造 PageIndex 同构的 JSON 树（供 Route B 复用），返回顶层节点列表。"""
    by_id = {n.id: {"node_id": n.id, "title": n.name,
                    "synonyms": n.synonyms, "nodes": []} for n in nodes}
    roots = []
    for n in nodes:
        item = by_id[n.id]
        if n.parent_id in by_id:
            by_id[n.parent_id]["nodes"].append(item)
        else:
            roots.append(item)
    return roots


if __name__ == "__main__":
    ns = load_nodes(use_cache=False)
    leaves = sum(1 for n in ns if n.is_leaf)
    print(f"节点总数: {len(ns)} | 叶子: {leaves} | 内部: {len(ns) - leaves}")
    print("示例:", ns[3].path_str, "| 同义词:", ns[3].synonyms)
