"""DeepSeek 客户端（OpenAI 兼容 chat 接口，仅用 requests，无需 openai 包）。

未配置 DEEPSEEK_API_KEY 时，chat_json 返回 None，调用方自动降级。
"""
import json

import requests

from . import config


class LLMUnavailable(Exception):
    pass


_LLM_STATS = {
    "requests": 0,
    "failures": 0,
    "prompt_chars": 0,
    "completion_chars": 0,
    "total_tokens": 0,
}


def reset_llm_stats():
    for key in _LLM_STATS:
        _LLM_STATS[key] = 0


def get_llm_stats():
    return dict(_LLM_STATS)


def chat_json(system: str, user: str, temperature: float = 0.0, timeout: int = 60):
    """调用 DeepSeek，返回解析后的 JSON dict；不可用或失败时返回 None。"""
    if not config.has_llm():
        return None

    url = config.DEEPSEEK_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    _LLM_STATS["requests"] += 1
    _LLM_STATS["prompt_chars"] += len(system or "") + len(user or "")
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout,
                             proxies={"http": None, "https": None})
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage") or {}
        _LLM_STATS["total_tokens"] += int(usage.get("total_tokens") or 0)
        content = data["choices"][0]["message"]["content"]
        _LLM_STATS["completion_chars"] += len(content or "")
        return _safe_json(content)
    except Exception as e:
        _LLM_STATS["failures"] += 1
        print(f"[LLM] 调用失败，降级处理：{e}")
        return None


def _safe_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                return None
    return None
