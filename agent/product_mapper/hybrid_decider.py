"""Hybrid Route A / Route B 冲突仲裁。

当两条路线都可靠命中但 node_id 不一致时，交给 LLM 在两个候选中裁决。
"""
from __future__ import annotations

from . import config
from .extension import route_a_reliable, route_b_reliable, semantic_sim, suggest_extension
from .llm import chat_json


ADJUDICATION_SYSTEM = """你是产品标准体系映射仲裁专家。
给定一个待映射产品，以及 Route A 和 Route B 给出的两个可靠候选节点。
请判断哪个候选更准确。只能选择 A、B 或 neither。
如果两个候选都不适合，选择 neither。
必须优先判断产品核心对象，而不是通信属性、智能属性、型号或规格。
例如“5G/4G智能安全帽”的核心对象是安全帽，不是5G智能终端或智慧智能设备；若 A/B 都偏离核心对象，应选择 neither。
严格输出 JSON：
{"choice":"A|B|neither","confidence":<0到1小数>,"reason":"<简短理由>"}"""


def has_conflict(result_a: dict | None, result_b: dict | None,
                 a_ok: bool | None = None, b_ok: bool | None = None) -> bool:
    """A/B 都可靠命中但 node_id 不同时，认为存在可靠冲突。"""
    if a_ok is None:
        a_ok = route_a_reliable(result_a)
    if b_ok is None:
        b_ok = route_b_reliable(result_b)
    if not (a_ok and b_ok):
        return False
    return result_a.get("node_id") != result_b.get("node_id")


def adjudicate_conflict(product: str, result_a: dict, result_b: dict,
                        use_llm: bool = True) -> dict:
    """对 A/B 可靠冲突做 LLM 仲裁；不可用时降级采用 A。"""
    base = {
        "triggered": True,
        "status": "unavailable",
        "choice": "fallback_A",
        "confidence": 0.0,
        "reason": "LLM 未启用，降级沿用旧规则采用 Route A",
        "llm_used": False,
    }
    if not use_llm or not config.has_llm():
        return base

    user = f"""待映射产品：{product}

Route A 候选：
- node_id: {result_a.get("node_id")}
- 节点名: {result_a.get("name")}
- 路径: {result_a.get("path")}
- 置信度: {result_a.get("confidence")}
- 来源: {result_a.get("source")}
- 字面质量: {result_a.get("lexical_quality", "")}
- 核心词一致: {result_a.get("core_overlap", "")}
- 理由: {result_a.get("reason", "")}

Route B 候选：
- node_id: {result_b.get("node_id")}
- 节点名: {result_b.get("name")}
- 路径: {result_b.get("path")}
- 置信度: {result_b.get("confidence")}
- 来源: {result_b.get("source")}
- 字面质量: {result_b.get("lexical_quality", "")}
- 核心词一致: {result_b.get("core_overlap", "")}
- 理由: {result_b.get("reason", "")}
"""
    out = chat_json(ADJUDICATION_SYSTEM, user, timeout=60)
    if not out:
        return {
            **base,
            "status": "failed",
            "reason": "LLM 仲裁调用失败，降级采用 Route A",
            "llm_used": True,
        }

    choice = str(out.get("choice", "")).strip()
    if choice not in {"A", "B", "neither"}:
        return {
            **base,
            "status": "failed",
            "reason": f"LLM 仲裁返回非法 choice={choice!r}，降级采用 Route A",
            "llm_used": True,
        }

    return {
        "triggered": True,
        "status": "done",
        "choice": choice,
        "confidence": float(out.get("confidence", 0.0) or 0.0),
        "reason": out.get("reason", ""),
        "llm_used": True,
    }


def _final_from(route: str, result: dict) -> dict:
    return {
        "route": route,
        "node_id": result.get("node_id"),
        "name": result.get("name"),
        "path": result.get("path"),
        "confidence": result.get("confidence", 0),
        "source": result.get("source", ""),
    }


def choose_hybrid_final(product: str, result_a: dict, result_b: dict,
                        mapper, use_llm: bool = True) -> dict:
    """返回 Hybrid 最终决策、冲突仲裁信息和必要的体系扩展建议。"""
    # 预存 Route B 的 bge 语义分，供 route_b_reliable 以"语义为主"判定可靠性
    if result_b is not None and "semantic_sim" not in result_b:
        result_b["semantic_sim"] = semantic_sim(mapper, product, result_b)
    a_ok = route_a_reliable(result_a)
    b_ok = route_b_reliable(result_b)
    conflict = has_conflict(result_a, result_b, a_ok, b_ok)
    adjudication = {
        "triggered": False,
        "status": "not_needed",
        "choice": "",
        "confidence": 0.0,
        "reason": "",
        "llm_used": False,
    }
    extension = None

    if conflict:
        adjudication = adjudicate_conflict(product, result_a, result_b, use_llm=use_llm)
        choice = adjudication.get("choice")
        if choice == "A":
            final = _final_from("Hybrid仲裁-Route A", result_a)
        elif choice == "fallback_A":
            final = _final_from("仲裁失败/降级采用Route A", result_a)
        elif choice == "B":
            final = _final_from("Hybrid仲裁-Route B", result_b)
        else:
            extension = suggest_extension(product, mapper, result_a, result_b, use_llm=use_llm)
            final = {
                "route": "体系扩展",
                "node_id": None,
                "name": None,
                "path": None,
                "confidence": 0.0,
                "source": "extension",
            }
        return {
            "final": final,
            "extension": extension,
            "conflict": True,
            "adjudication": adjudication,
            "a_ok": a_ok,
            "b_ok": b_ok,
        }

    if a_ok:
        final = _final_from("Route A", result_a)
    elif b_ok:
        final = _final_from("Route B", result_b)
    else:
        extension = suggest_extension(product, mapper, result_a, result_b, use_llm=use_llm)
        final = {
            "route": "体系扩展",
            "node_id": None,
            "name": None,
            "path": None,
            "confidence": 0.0,
            "source": "extension",
        }

    return {
        "final": final,
        "extension": extension,
        "conflict": False,
        "adjudication": adjudication,
        "a_ok": a_ok,
        "b_ok": b_ok,
    }
