#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reranker.py — Qwen3-Reranker 精排（云端优先，本地可选）

对第一阶段召回（向量 + BM25 → RRF）的候选做深度重排。默认走云端托管的
Qwen3-Reranker API，无需下载权重、无需 torch/GPU；也保留本地推理（Reranker 类）
作为可选后端。

云端后端（provider）：
    dashscope    阿里云百炼（Model Studio），OpenAI 兼容 /compatible-mode/v1/reranks，
                 模型 qwen3-rerank；工作空间地址写在 auth.json 的 dashscope.base_url
    siliconflow  硅基流动，OpenAI 兼容 /v1/rerank，模型 Qwen/Qwen3-Reranker-0.6B/4B/8B
    openai       任意 OpenAI 兼容 rerank 端点（自建 vLLM / 其他聚合平台），自定义 base_url
    local        本地推理（需 torch + transformers + 下载权重，首次使用会拉模型）

用法：
    from reranker import create_reranker
    r = create_reranker()               # 按环境变量选后端；无 key 时返回 None（禁用精排）
    scores = r.score("怎么退货", ["7天内可无理由退货", "客服时间 09:00-22:00"])
    hits = r.rerank(query, hits, top_k=6)

环境变量：
    RAG_RERANK_PROVIDER     后端：auto(默认自动探测) / dashscope / siliconflow / openai / local
    RAG_RERANK_API_KEY      精排 API Key（通用；也可用 provider 专属 key，见下）
    DASHSCOPE_API_KEY       阿里云百炼 key（等价 RAG_RERANK_API_KEY）
    SILICONFLOW_API_KEY     硅基流动 key（等价 RAG_RERANK_API_KEY）
    RAG_RERANK_MODEL        模型名（默认按 provider 自动选）
    RAG_RERANK_BASE_URL     自定义 base_url（覆盖默认；工作空间地址也可写在 auth.json 的 base_url 字段）
    RAG_RERANK_BATCH_SIZE   本地后端批量大小（默认 8）
    RAG_RERANK_MAX_LENGTH   本地后端最大长度（默认 8192）

Key 也会尝试从 ~/.pi/agent/auth.json 读取（字段 dashscope.key / siliconflow.key / openai.key），
base_url 会尝试读 dashscope.base_url / siliconflow.base_url / openai.base_url。
"""

import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger("reranker")

RERANK_PROVIDER = os.environ.get("RAG_RERANK_PROVIDER", "auto").strip().lower()
RERANK_MODEL = os.environ.get("RAG_RERANK_MODEL", "").strip()
RERANK_BASE_URL = os.environ.get("RAG_RERANK_BASE_URL", "").strip()
RERANK_API_KEY = os.environ.get("RAG_RERANK_API_KEY", "").strip()

# 本地后端参数
BATCH_SIZE = int(os.environ.get("RAG_RERANK_BATCH_SIZE", "8"))
MAX_LENGTH = int(os.environ.get("RAG_RERANK_MAX_LENGTH", "8192"))
LOCAL_MODEL = os.environ.get("RAG_RERANK_LOCAL_MODEL", "Qwen/Qwen3-Reranker-0.6B")
LOCAL_DEVICE = os.environ.get("RAG_RERANK_DEVICE", "").strip()

DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

# 各云端的端点与默认模型
CLOUD_PROVIDERS = {
    "dashscope": {
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "path": "/reranks",
        "model": "qwen3-rerank",
        "key_envs": ["RAG_RERANK_API_KEY", "DASHSCOPE_API_KEY"],
        "auth_key": "dashscope.key",
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "path": "/rerank",
        "model": "Qwen/Qwen3-Reranker-0.6B",
        "key_envs": ["RAG_RERANK_API_KEY", "SILICONFLOW_API_KEY"],
        "auth_key": "siliconflow.key",
    },
    "openai": {
        "base_url": "",            # 必须通过 RAG_RERANK_BASE_URL 指定
        "path": "/rerank",
        "model": "Qwen/Qwen3-Reranker-0.6B",
        "key_envs": ["RAG_RERANK_API_KEY", "OPENAI_API_KEY"],
        "auth_key": "openai.key",
    },
}


class RerankError(RuntimeError):
    """rerank 调用失败。"""


def _read_auth_json() -> dict:
    p = Path.home() / ".pi" / "agent" / "auth.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _resolve_key(cfg: dict) -> str:
    for env in cfg["key_envs"]:
        v = os.environ.get(env)
        if v:
            return v
    node = _read_auth_json()
    for part in cfg["auth_key"].split("."):
        node = node.get(part) if isinstance(node, dict) else None
        if node is None:
            return ""
    return node if isinstance(node, str) else ""


# --------------------------------------------------------------------------
# 云端后端
# --------------------------------------------------------------------------
class CloudReranker:
    """通过 HTTP 调用云端托管的 Qwen3-Reranker。"""

    def __init__(self, provider: str | None = None, model: str | None = None,
                 api_key: str | None = None, base_url: str | None = None,
                 timeout: float = 30.0):
        self.provider = (provider or RERANK_PROVIDER).lower()
        if self.provider not in CLOUD_PROVIDERS:
            raise RerankError(f"未知 rerank provider：{self.provider}（可选 "
                              f"{', '.join(CLOUD_PROVIDERS)}）")
        cfg = CLOUD_PROVIDERS[self.provider]
        # base_url 优先级：显式参数 > 环境变量 > auth.json 里的 base_url > 默认
        auth_node = _read_auth_json().get(self.provider)
        auth_base = auth_node.get("base_url", "") if isinstance(auth_node, dict) else ""
        self.base_url = (base_url or RERANK_BASE_URL or auth_base or cfg["base_url"]).rstrip("/")
        if not self.base_url:
            raise RerankError(f"{self.provider} 后端需要设置 RAG_RERANK_BASE_URL")
        self.model = model or RERANK_MODEL or cfg["model"]
        self.timeout = timeout
        self.api_key = api_key or _resolve_key(cfg)

    # ---------------- 请求 / 响应（OpenAI 兼容 rerank，三种后端统一） ----------------
    def _build_payload(self, query: str, documents: list[str]) -> dict:
        return {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
            "return_documents": False,
        }

    def _parse(self, data: dict, n: int) -> list[float]:
        results = data.get("results") or []
        by_idx: dict[int, float] = {}
        for r in results:
            if isinstance(r, dict) and "index" in r:
                by_idx[int(r["index"])] = float(r.get("relevance_score", 0.0))
        return [by_idx.get(i, 0.0) for i in range(n)]

    def _post(self, payload: dict) -> dict:
        url = self.base_url + CLOUD_PROVIDERS[self.provider]["path"]
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")[:400]
            raise RerankError(f"rerank API 失败（HTTP {e.code}）：{body}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            raise RerankError(f"rerank API 网络错误：{e}") from e

    # ---------------- 统一接口（与本地 Reranker 一致） ----------------
    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        if not self.api_key:
            raise RerankError(
                f"缺少 rerank API Key：请设置环境变量 RAG_RERANK_API_KEY 或 "
                f"{CLOUD_PROVIDERS[self.provider]['key_envs'][1]}（也可写入 "
                f"~/.pi/agent/auth.json 的 {CLOUD_PROVIDERS[self.provider]['auth_key']}）。")
        return self._parse(self._post(self._build_payload(query, documents)), len(documents))

    def rerank(self, query: str, hits: list[dict], top_k: int | None = None) -> list[dict]:
        if not hits:
            return []
        documents = []
        for h in hits:
            c = h["chunk"]
            documents.append(c["text"] if isinstance(c, dict) else c.text)
        raw = self.score(query, documents)
        out = []
        for h, s in zip(hits, raw):
            h = dict(h)
            h["rerank_score"] = float(s)
            out.append(h)
        out.sort(key=lambda h: -h["rerank_score"])
        return out[:top_k] if top_k else out


# --------------------------------------------------------------------------
# 本地后端（torch + transformers，懒加载；默认不启用，避免下载）
# --------------------------------------------------------------------------
class Reranker:
    """本地 Qwen3-Reranker 推理（生成式交叉编码器，yes/no logit 打分）。"""

    _PREFIX = ('<|im_start|>system\n'
               'Judge whether the Document meets the requirements based on the Query and the '
               'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
               '<|im_start|>user\n')
    _SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

    def __init__(self, model_name: str | None = None, device: str | None = None,
                 batch_size: int | None = None, max_length: int | None = None,
                 instruction: str | None = None):
        self.model_name = model_name or LOCAL_MODEL
        self.device = device or LOCAL_DEVICE or None
        self.batch_size = batch_size or BATCH_SIZE
        self.max_length = max_length or MAX_LENGTH
        self.instruction = instruction or DEFAULT_INSTRUCTION
        self._model = None
        self._tokenizer = None
        self._true_id = None
        self._false_id = None

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("加载本地 reranker：%s（device=%s）…", self.model_name, device)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        self._true_id = self._tokenizer.convert_tokens_to_ids("yes")
        self._false_id = self._tokenizer.convert_tokens_to_ids("no")

        kwargs: dict = {"trust_remote_code": True}
        if device == "cuda":
            kwargs["torch_dtype"] = torch.float16
        try:
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        except Exception:
            kwargs.pop("trust_remote_code", None)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)

        self._model.to(device).eval()
        self._device = device
        logger.info("本地 reranker 加载完成")

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._load()

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        self._ensure_loaded()
        import torch
        import torch.nn.functional as F

        body = f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: "
        texts = [self._PREFIX + body + d + self._SUFFIX for d in documents]
        scores: list[float] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            inputs = self._tokenizer(batch, return_tensors="pt", padding=True,
                                     truncation=True, max_length=self.max_length).to(self._device)
            with torch.no_grad():
                logits = self._model(**inputs).logits[:, -1, :]
                true_v = logits[:, self._true_id]
                false_v = logits[:, self._false_id]
                prob = F.log_softmax(torch.stack([false_v, true_v], dim=1), dim=1)[:, 1].exp()
            scores.extend(prob.float().cpu().tolist())
        return scores

    def rerank(self, query: str, hits: list[dict], top_k: int | None = None) -> list[dict]:
        if not hits:
            return []
        documents = []
        for h in hits:
            c = h["chunk"]
            documents.append(c["text"] if isinstance(c, dict) else c.text)
        raw = self.score(query, documents)
        out = []
        for h, s in zip(hits, raw):
            h = dict(h)
            h["rerank_score"] = float(s)
            out.append(h)
        out.sort(key=lambda h: -h["rerank_score"])
        return out[:top_k] if top_k else out


# --------------------------------------------------------------------------
# 工厂：按环境变量选后端（默认 auto 自动探测）；无 key 时返回 None（禁用精排）
# --------------------------------------------------------------------------
def _detect_provider() -> str | None:
    for p in ("dashscope", "siliconflow", "openai"):
        if _resolve_key(CLOUD_PROVIDERS[p]):
            return p
    return None


def create_reranker():
    provider = RERANK_PROVIDER
    if provider == "local":
        return Reranker()
    if provider == "auto":
        provider = _detect_provider()
        if provider is None:
            print("  [精排] 未找到任何云端 rerank API Key，已禁用精排（仍用 RRF 排序）。\n"
                  "         可设置 RAG_RERANK_API_KEY / DASHSCOPE_API_KEY / SILICONFLOW_API_KEY，"
                  "或写入 ~/.pi/agent/auth.json 的 dashscope.key / siliconflow.key。")
            return None
    if provider not in CLOUD_PROVIDERS:
        print(f"  [精排] 未知 provider '{provider}'，已禁用精排。"
              f"可选：{', '.join(CLOUD_PROVIDERS)} / local")
        return None
    try:
        r = CloudReranker(provider=provider)
    except RerankError as e:
        print(f"  [精排] {e}，已禁用精排。")
        return None
    if not r.api_key:
        cfg = CLOUD_PROVIDERS[provider]
        print(f"  [精排] 未找到 {provider} 的 API Key，已禁用精排（仍用 RRF 排序）。\n"
              f"         请设置环境变量 {cfg['key_envs'][1]} 或 RAG_RERANK_API_KEY，"
              f"或写入 ~/.pi/agent/auth.json 的 {cfg['auth_key']}。")
        return None
    return r


# --------------------------------------------------------------------------
# 独立自检
# --------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    r = create_reranker()
    if r is None:
        raise SystemExit("无可用 reranker（请先配置云端 API Key）。")
    query = "普通商品退货期限是多久？"
    docs = [
        "普通商品退货期限为7天，支持无理由退货，运费由平台承担。",
        "在线客服：每日09:00-22:00。",
        "生鲜商品退货期限为24小时，不支持无理由退货。",
        "普通商品 7天 是 平台",
    ]
    scores = r.score(query, docs)
    order = sorted(range(len(docs)), key=lambda i: -scores[i])
    print("provider:", getattr(r, "provider", "local"), "| query:", query)
    for i in order:
        print(f"  {scores[i]:.4f}  {docs[i]}")


if __name__ == "__main__":
    main()
