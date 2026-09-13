#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm.py — 多 Provider LLM 调用封装（默认 DeepSeek，OpenAI 兼容接口），
rag_tool.py / query_intent.py / memory.py 共用。

通过 LLMProvider 工厂 + llm_providers.yaml 支持 DeepSeek / DashScope / 硅基流动 /
OpenAI 兼容端点四类 Provider 配置化切换（环境变量 RAG_LLM_PROVIDER 选择）。

把手写的 LLM 通信协议层做完整，覆盖生产级常见需求：
  - 错误分类：auth / rate_limit / bad_request / server / network / parse
  - 自动重试 + 指数退避（限流、服务端 5xx、网络/超时；鉴权与 4xx 不重试）
  - 空回答兜底：推理模型把 max_tokens 全花在“思考”上导致正文为空时，追加提示并放大 max_tokens 重试一次
  - 可配置超时、结构化输出（response_format=json_object）与 SSE 流式输出
  - token 用量统计（可选 usage 累加器）
  - chat_json()：优先走 JSON mode，失败自动降级为「纯文本 + 正则解析 + 修正重试」

接口（与旧版向后兼容）：
  - call_llm(messages, api_key, ...)           返回原始文本
  - chat_json(messages, api_key, ...)          要求 JSON 输出并解析为 dict
  - call_llm_stream(messages, api_key, ...)    SSE 流式生成器，逐段 yield 增量文本
"""

import http.client
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

try:
    import yaml
except ImportError:  # 部署环境未装 pyyaml 时回退内置默认配置，不影响主链路
    yaml = None

from rag_metrics import rag_metrics

# --------------------------------------------------------------------------
# 可插拔 LLM Provider（Factory + YAML）
# 环境变量 RAG_LLM_PROVIDER 选择（默认 deepseek），配置见 llm_providers.yaml。
# 覆盖优先级：环境变量 > auth.json > yaml 默认。
# --------------------------------------------------------------------------
LLM_PROVIDER_NAME = os.environ.get("RAG_LLM_PROVIDER", "deepseek").strip().lower()
LLM_API_KEY_ENV = os.environ.get("RAG_LLM_API_KEY", "").strip()      # 通用 Key 覆盖
LLM_BASE_URL_ENV = os.environ.get("RAG_LLM_BASE_URL", "").strip()    # 通用 base_url 覆盖

CONFIG_PATH = Path(__file__).resolve().parent / "llm_providers.yaml"

# 内置默认（yaml 缺失/损坏/未装 yaml 时的兜底）
_DEFAULT_PROVIDERS = {
    "deepseek": {"base_url": "https://api.deepseek.com", "model": "deepseek-v4-flash",
                 "api_key_env": "DEEPSEEK_API_KEY", "auth_key": "deepseek.key"},
    "dashscope": {"base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                  "model": "qwen-plus", "api_key_env": "DASHSCOPE_API_KEY",
                  "auth_key": "dashscope.key"},
    "siliconflow": {"base_url": "https://api.siliconflow.cn/v1", "model": "Qwen/Qwen3-8B",
                    "api_key_env": "SILICONFLOW_API_KEY", "auth_key": "siliconflow.key"},
    "openai": {"base_url": "", "model": "gpt-4o-mini",
               "api_key_env": "OPENAI_API_KEY", "auth_key": "openai.key"},
}


@dataclass
class LLMProvider:
    name: str
    base_url: str
    model: str
    api_key_env: str
    auth_key: str
    api_key: str = ""


def _read_auth_json() -> dict:
    auth_path = Path.home() / ".pi" / "agent" / "auth.json"
    if auth_path.exists():
        try:
            return json.loads(auth_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _load_providers_yaml() -> dict:
    """读 llm_providers.yaml；缺失/损坏/未装 yaml 时回退内置默认。"""
    if yaml is not None and CONFIG_PATH.exists():
        try:
            data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
            providers = data.get("providers") if isinstance(data, dict) else None
            if isinstance(providers, dict) and providers:
                return {k: dict(v) for k, v in providers.items() if isinstance(v, dict)}
        except Exception as e:
            print(f"[LLM] 读取 {CONFIG_PATH.name} 失败（{e}），回退内置默认配置。")
    return {k: dict(v) for k, v in _DEFAULT_PROVIDERS.items()}


_provider: LLMProvider | None = None


def get_llm_provider() -> LLMProvider:
    """工厂：按 RAG_LLM_PROVIDER 装配 LLMProvider，解析 base_url / model / api_key。

    优先级（高 → 低）：
      base_url : RAG_LLM_BASE_URL > auth.json 的 {provider}.base_url > yaml 默认
      model    : RAG_LLM_MODEL > yaml 默认
      api_key  : RAG_LLM_API_KEY > {provider}.api_key_env 环境变量 > auth.json 的 {provider}.key
    """
    global _provider
    if _provider is not None:
        return _provider

    providers = _load_providers_yaml()
    cfg = providers.get(LLM_PROVIDER_NAME)
    if cfg is None:
        print(f"[LLM] 未知 Provider '{LLM_PROVIDER_NAME}'，回退 deepseek。"
              f"可选：{', '.join(sorted(providers))}")
        cfg = _DEFAULT_PROVIDERS["deepseek"]
        name = "deepseek"
    else:
        name = LLM_PROVIDER_NAME

    auth = _read_auth_json()
    auth_node = auth.get(name) if isinstance(auth, dict) else None
    auth_base = auth_node.get("base_url", "") if isinstance(auth_node, dict) else ""
    auth_key = auth_node.get("key", "") if isinstance(auth_node, dict) else ""

    base_url = (LLM_BASE_URL_ENV or auth_base or cfg.get("base_url", "") or "").rstrip("/")
    model = os.environ.get("RAG_LLM_MODEL", "").strip() or cfg.get("model", "")
    env_key = os.environ.get(cfg.get("api_key_env") or "", "") or ""
    api_key = LLM_API_KEY_ENV or env_key or auth_key or ""

    _provider = LLMProvider(name=name, base_url=base_url, model=model,
                            api_key_env=cfg.get("api_key_env", ""),
                            auth_key=cfg.get("auth_key", ""), api_key=api_key)
    return _provider


# 默认模型 / 接口地址由激活的 Provider 派生（保持向后兼容：外部仍可 import LLM_MODEL）
LLM_MODEL = get_llm_provider().model or "deepseek-v4-flash"
DEEPSEEK_BASE_URL = get_llm_provider().base_url or "https://api.deepseek.com"

# 可重试的错误类别（鉴权 / 客户端参数错误不重试）
RETRYABLE_KINDS = {"rate_limit", "server", "network"}


class LLMError(RuntimeError):
    """LLM 调用/解析失败。kind 用于区分错误类别，调用方可据此决定降级或重试。"""

    def __init__(self, message: str, kind: str = "unknown", status_code: int | None = None):
        super().__init__(message)
        self.kind = kind            # auth | rate_limit | bad_request | server | network | parse
        self.status_code = status_code


def get_api_key() -> str:
    """返回激活 LLM Provider 的 API Key（由 get_llm_provider() 工厂解析）。"""
    return get_llm_provider().api_key


def _classify_http_error(status: int) -> str:
    if status in (401, 403):
        return "auth"
    if status == 429:
        return "rate_limit"
    if status >= 500:
        return "server"
    return "bad_request"   # 400 / 404 / 422 等


def _build_request(payload: dict, api_key: str) -> urllib.request.Request:
    base = get_llm_provider().base_url
    if not base:
        raise LLMError("当前 LLM Provider 未配置 base_url，请设置 RAG_LLM_BASE_URL "
                       "或在 llm_providers.yaml 中补齐。", kind="bad_request")
    return urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )


def _accumulate_usage(usage: dict | None, u: dict) -> None:
    if usage is None or not isinstance(u, dict):
        return
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        try:
            usage[k] = usage.get(k, 0) + int(u.get(k, 0))
        except (TypeError, ValueError):
            pass


def _post_chat(payload: dict, api_key: str, timeout: float,
               usage: dict | None) -> str:
    """发送一次 chat/completions 请求，返回正文，并累加 token 用量。"""
    with urllib.request.urlopen(_build_request(payload, api_key),
                                timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    _accumulate_usage(usage, data.get("usage") or {})
    return content


def _empty_answer_retry_payload(messages: list[dict], payload: dict) -> dict:
    """空回答兜底：追加「直接给答案、别思考」提示，并放大 max_tokens 再试。"""
    msgs = [dict(m) for m in messages]
    msgs.append({
        "role": "user",
        "content": "（请直接给出最终答案，不要输出任何推理、思考过程或解释。）",
    })
    p = dict(payload)
    p["messages"] = msgs
    old_max = int(p.get("max_tokens") or 4096)
    p["max_tokens"] = min(16384, max(8192, old_max * 2))
    return p


def call_llm(messages: list[dict], api_key: str,
             temperature: float = 0.1, max_tokens: int = 4096,
             model: str | None = None,
             timeout: float = 120.0, retries: int = 3, backoff: float = 1.0,
             response_format: dict | None = None,
             usage: dict | None = None) -> str:
    """一次非流式调用，返回正文。

    - 限流/5xx/网络错误自动指数退避重试；
    - 推理模型可能把 max_tokens 全花在“思考”上导致正文为空，此时追加提示并
      放大 max_tokens 兜底重试一次。
    """
    payload = {
        "model": model or LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if response_format:
        payload["response_format"] = response_format

    last_err: LLMError | None = None
    empty_retried = False
    for attempt in range(retries + 1):
        try:
            content = _post_chat(payload, api_key, timeout, usage)
            if not content.strip() and not empty_retried:
                empty_retried = True
                payload = _empty_answer_retry_payload(messages, payload)
                continue
            return content
        except urllib.error.HTTPError as e:
            kind = _classify_http_error(e.code)
            body = e.read().decode("utf-8", "ignore")[:400]
            last_err = LLMError(f"LLM 调用失败（HTTP {e.code}）：{body}",
                                kind=kind, status_code=e.code)
            if kind not in RETRYABLE_KINDS:
                raise last_err
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError,
                http.client.HTTPException) as e:
            last_err = LLMError(f"LLM 网络错误：{e}", kind="network")

        if attempt < retries:
            time.sleep(backoff * (2 ** attempt))

    raise last_err  # type: ignore[misc]


def call_llm_stream(messages: list[dict], api_key: str,
                    temperature: float = 0.1, max_tokens: int = 4096,
                    model: str | None = None, timeout: float = 120.0,
                    usage: dict | None = None):
    """SSE 流式调用：逐段 yield 增量文本（生成器）。

    流式响应按行返回 `data: {chunk}` 直到 `data: [DONE]`；
    末块若带 usage 则累加进 usage。
    """
    payload = {
        "model": model or LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    t_start = time.perf_counter()
    ttft_sent = False
    model_name = model or LLM_MODEL
    # 把 HTTP/网络错误统一包装成 LLMError，这样调用方 `except LLMError` 的
    # 「流式失败 → 降级一次性调用」兑底逻辑才能真正生效（否则抛的是 URLError）。
    try:
        with urllib.request.urlopen(_build_request(payload, api_key), timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                _accumulate_usage(usage, chunk.get("usage") or {})
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = (choices[0].get("delta") or {}).get("content")
                if delta:
                    if not ttft_sent:
                        ttft_sent = True
                        rag_metrics().llm_ttft.record(
                            (time.perf_counter() - t_start) * 1000, {"model": model_name})
                    yield delta
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")[:400]
        raise LLMError(f"LLM 流式调用失败（HTTP {e.code}）：{body}",
                       kind=_classify_http_error(e.code), status_code=e.code) from e
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError,
            http.client.HTTPException) as e:
        raise LLMError(f"LLM 流式网络错误：{e}", kind="network") from e


# --------------------------------------------------------------------------
# JSON 结构化输出
# --------------------------------------------------------------------------
def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_json_text(text: str) -> dict | None:
    """从文本里定位第一个 {…} 并解析；失败返回 None。"""
    if not isinstance(text, str):
        return None
    cleaned = _strip_code_fences(text)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(cleaned[start:end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _ensure_json_hint(messages: list[dict]) -> list[dict]:
    """json_object 模式要求 prompt 里出现 "json" 字样；没有则补一条提示。"""
    if any("json" in (m.get("content", "") or "").lower() for m in messages):
        return messages
    msgs = [dict(m) for m in messages]
    if msgs and msgs[0].get("role") == "system":
        msgs[0] = dict(msgs[0])
        msgs[0]["content"] = (msgs[0].get("content", "") or "") + "\n请以 JSON 对象输出。"
    else:
        msgs.insert(0, {"role": "system", "content": "请以 JSON 对象输出。"})
    return msgs


def chat_json(messages: list[dict], api_key: str,
              temperature: float = 0.0, max_tokens: int = 2048,
              model: str | None = None, retries: int = 2) -> dict:
    """要求 LLM 输出 JSON 并解析为 dict。

    优先走结构化输出（response_format=json_object）；模型不支持或解析失败时，
    自动降级为「纯文本 + 正则定位 + 修正重试」。
    """
    msgs = _ensure_json_hint(messages)

    # 1) 优先 JSON mode（一次调用，尽量别浪费重试）
    try:
        text = call_llm(msgs, api_key, temperature=temperature,
                        max_tokens=max_tokens, model=model,
                        response_format={"type": "json_object"}, retries=1)
        obj = _parse_json_text(text)
        if obj is not None:
            return obj
        rag_metrics().json_parse_fail.add(1)   # JSON mode 输出无法解析 → 记一次失败
    except LLMError:
        pass

    # 2) 兜底：纯文本 + 正则解析，解析失败把原文回传要求修正
    last_text = ""
    for attempt in range(retries + 1):
        m = msgs if attempt == 0 else msgs + [
            {"role": "assistant", "content": last_text},
            {"role": "user",
             "content": "上面的输出不是合法 JSON。请只输出一个合法 JSON 对象，"
                        "不要任何多余文字、解释或代码块标记。"},
        ]
        last_text = call_llm(m, api_key, temperature=temperature,
                             max_tokens=max_tokens, model=model)
        obj = _parse_json_text(last_text)
        if obj is not None:
            return obj
    raise LLMError("LLM 多次输出都无法解析为合法 JSON", kind="parse")
