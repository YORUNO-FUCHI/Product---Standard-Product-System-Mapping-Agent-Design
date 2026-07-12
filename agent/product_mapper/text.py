"""字面匹配工具：字符三元组（trigram）相似度，语义对齐 PostgreSQL 的 pg_trgm。

pg_trgm 的做法：字符串前补 2 个空格、后补 1 个空格，切成长度 3 的滑窗集合，
相似度 = |A∩B| / |A∪B|（Jaccard）。这里在纯 Python 中复刻同一逻辑，
因此内存后端与未来的 pg 后端行为一致、可对照。
"""
import re


NOISE_TERMS = {
    "智能", "系统", "设备", "产品", "解决方案", "方案", "高性能", "联合开发",
    "新型", "其他", "专用", "通用", "多功能", "全能量段", "型", "款",
    "5g", "4g", "3g", "2g",
}

PRODUCT_SUFFIXES = [
    "基因突变检测试剂盒", "测定试剂盒", "检测试剂盒", "试剂盒",
    "生产线设备", "生产线", "安全帽", "机器人", "注射液", "激光器",
    "发动机", "稳定剂", "光伏组件", "组件", "储能电芯", "电芯",
    "芯片", "薄膜", "锅炉", "均质机", "开关设备", "潜水系统",
    "安全帽", "终端", "电池", "原料药", "细胞", "基板",
]

PRODUCT_SUFFIXES = sorted(set(PRODUCT_SUFFIXES), key=len, reverse=True)


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


def normalize_product_name(s: str) -> str:
    """面向产品名分析的轻量归一化。"""
    s = norm(s)
    s = re.sub(r"[\s·,，。；;:：()（）\[\]【】]+", "", s)
    return s


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


def extract_product_type(s: str) -> str:
    """识别产品核心后缀，识别不到返回空字符串。"""
    text = normalize_product_name(s)
    for suffix in PRODUCT_SUFFIXES:
        if suffix.lower() in text:
            return suffix.lower()
    return ""


def extract_core_terms(s: str) -> set:
    """提取用于校验字面召回的核心词。

    目标是排除“智能/系统/设备/5G”等修饰词，保留安全帽、芯片、试剂盒等产品对象。
    """
    text = normalize_product_name(s)
    if not text:
        return set()

    terms = set()
    ptype = extract_product_type(text)
    if ptype:
        terms.add(ptype)

    work = text
    work = re.sub(r"[a-z]*\d+(?:\.\d+)?[a-zμ/+\-]*", "", work)
    for noise in sorted(NOISE_TERMS, key=len, reverse=True):
        work = work.replace(noise.lower(), "")
    for suffix in PRODUCT_SUFFIXES:
        if suffix.lower() in work:
            terms.add(suffix.lower())
            work = work.replace(suffix.lower(), "")

    # 保留较长中文片段，避免单字和型号残片制造假重叠。
    for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", work):
        if chunk not in NOISE_TERMS:
            terms.add(chunk)
    return terms


def core_overlap(product: str, candidate_name: str) -> bool:
    """输入产品与候选节点是否有核心词重叠。"""
    product_terms = extract_core_terms(product)
    candidate_terms = extract_core_terms(candidate_name)
    if not product_terms or not candidate_terms:
        return False
    if product_terms & candidate_terms:
        return True
    for a in product_terms:
        for b in candidate_terms:
            if a in b or b in a:
                return True
    return False


def lexical_quality(product: str, candidate_name: str, trgm_score: float = 0.0) -> str:
    """评估字面候选质量：strong / medium / weak / noisy。"""
    overlap = core_overlap(product, candidate_name)
    product_type = extract_product_type(product)
    candidate_type = extract_product_type(candidate_name)
    type_mismatch = bool(
        product_type and candidate_type
        and product_type != candidate_type
        and product_type not in candidate_type
        and candidate_type not in product_type
    )

    if not overlap or type_mismatch:
        return "noisy" if trgm_score >= 0.18 else "weak"
    if trgm_score >= 0.9:
        return "strong"
    if trgm_score >= 0.45:
        return "medium"
    return "weak"
