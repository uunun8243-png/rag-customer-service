#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
answer_cache.py — 答案缓存（Redis 优先，进程内内存兜底）

只缓存「自包含问题」的完整答案：命中后跳过「意图识别 → 检索 → 精排 → 生成」全链路，
直接返回上次的答案与出处（降延迟、降 LLM 成本）。

key = rag:cache:answer:{index_fingerprint[:12]}:{sha1(normalize(query))}
  - index_fingerprint：知识库内容指纹（rag_index.compute_fingerprint）。
    文档一变指纹就变 → 旧缓存自然失效，无需手动清缓存。
  - normalize(query)：去首尾空白 / 压缩空白 / 去尾部标点 / 小写化，
    让「退货期限多久？」与「退货期限多久」命中同一缓存。

依赖上下文指代的问题（「那北京呢」）不应按字面命中缓存——是否查缓存由调用方用
query_intent.has_reference() 决策，本模块只负责 get/set。

环境变量：
  RAG_ANSWER_CACHE_TTL  答案缓存 TTL 秒数（默认 3600；<=0 表示禁用缓存）
  REDIS_URL             Redis 连接串（与 session_store 共用；默认 redis://127.0.0.1:6379/0）
"""

import hashlib
import json
import os
import re
import threading
import time

ANSWER_CACHE_TTL = int(os.environ.get("RAG_ANSWER_CACHE_TTL", "3600"))
REDIS_URL = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
KEY_PREFIX = "rag:cache:answer:"


def normalize_query(q: str) -> str:
    """归一化查询：去空白、去尾部标点、小写化，让等价问法命中同一缓存。"""
    q = (q or "").strip().lower()
    q = re.sub(r"\s+", " ", q)
    return q.rstrip("？?。！!，,、；;：:")


def _cache_key(fingerprint: str, query: str) -> str:
    fp = (fingerprint or "")[:12]
    digest = hashlib.sha1(normalize_query(query).encode("utf-8")).hexdigest()
    return f"{KEY_PREFIX}{fp}:{digest}"


class MemoryAnswerCache:
    """进程内兜底实现（仅单进程部署可用）。"""

    def __init__(self, ttl: int, fingerprint: str):
        self._ttl = ttl
        self._fp = fingerprint
        self._data: dict[str, tuple[float, dict]] = {}
        self._lock = threading.Lock()

    def _key(self, query: str) -> str:
        return _cache_key(self._fp, query)

    def get(self, query: str) -> dict | None:
        key = self._key(query)
        with self._lock:
            rec = self._data.get(key)
            if rec is None:
                return None
            ts, value = rec
            if time.time() - ts > self._ttl:
                self._data.pop(key, None)
                return None
            return value

    def set(self, query: str, answer: str, hits: list[dict]) -> None:
        key = self._key(query)
        with self._lock:
            self._data[key] = (time.time(), {"answer": answer, "hits": hits})
            # 惰性清理过期项，防止无界膨胀
            if len(self._data) > 10000:
                now = time.time()
                expired = [k for k, (ts, _) in self._data.items()
                           if now - ts > self._ttl]
                for k in expired:
                    self._data.pop(k, None)


class RedisAnswerCache:
    """Redis 实现：多 Worker 共享同一份答案缓存。"""

    def __init__(self, url: str, ttl: int, fingerprint: str):
        import redis
        self._r = redis.Redis.from_url(url, decode_responses=True)
        self._r.ping()          # 提前失败，便于上层回退到内存实现
        self._ttl = ttl
        self._fp = fingerprint

    def _key(self, query: str) -> str:
        return _cache_key(self._fp, query)

    def get(self, query: str) -> dict | None:
        raw = self._r.get(self._key(query))
        return json.loads(raw) if raw else None

    def set(self, query: str, answer: str, hits: list[dict]) -> None:
        self._r.set(self._key(query),
                    json.dumps({"answer": answer, "hits": hits}, ensure_ascii=False),
                    ex=self._ttl)


def create_answer_cache(fingerprint: str = ""):
    """工厂：Redis 优先，失败回退内存；RAG_ANSWER_CACHE_TTL<=0 返回 None（禁用）。"""
    if ANSWER_CACHE_TTL <= 0:
        return None
    try:
        c = RedisAnswerCache(REDIS_URL, ANSWER_CACHE_TTL, fingerprint)
        print(f"[缓存] 答案缓存：Redis {REDIS_URL}"
              f"（TTL {ANSWER_CACHE_TTL}s，知识库指纹 {fingerprint[:12]}）")
        return c
    except Exception as e:
        print(f"[缓存] Redis 不可用（{e}），答案缓存回退到进程内内存（仅单进程可用）。")
        return MemoryAnswerCache(ANSWER_CACHE_TTL, fingerprint)
