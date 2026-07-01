"""字面匹配工具：字符三元组（trigram）相似度，语义对齐 PostgreSQL 的 pg_trgm。

pg_trgm 的做法：字符串前补 2 个空格、后补 1 个空格，切成长度 3 的滑窗集合，
相似度 = |A∩B| / |A∪B|（Jaccard）。这里在纯 Python 中复刻同一逻辑，
因此内存后端与未来的 pg 后端行为一致、可对照。
"""


def _to_halfwidth(s: str) -> str:
    res = []
    for ch in s:
        code = ord(ch)
        if code == 0x3000:          # 全角空格
            code = 0x20
        elif 0xFF01 <= code <= 0xFF5E:  # 全角 ASCII
            code -= 0xFEE0
        res.append(chr(code))
    return "".join(res)


def norm(s: str) -> str:
    """归一化：全角转半角 + 去首尾空白 + 小写。"""
    if not s:
        return ""
    return _to_halfwidth(s).strip().lower()


def trigrams(s: str) -> set:
    """返回字符三元组集合（pg_trgm 式补白）。"""
    s = norm(s)
    if not s:
        return set()
    padded = "  " + s + " "
    return {padded[i:i + 3] for i in range(len(padded) - 2)}


def similarity(a: str, b: str) -> float:
    """两字符串的 trigram Jaccard 相似度，范围 [0, 1]。"""
    ta, tb = trigrams(a), trigrams(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    if inter == 0:
        return 0.0
    return inter / len(ta | tb)
