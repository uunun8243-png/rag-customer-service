#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
session_store.py — 会话状态存储（Redis 优先，进程内内存兜底）

每个 session 保存：
  conversation  近期对话（role/content 列表；memory.py 会裁剪，早期历史折叠进 summary）
  summary       早期对话滚动摘要（memory.py 维护，长度有界）
  folded        累计已折叠进摘要的消息条数（观测用）
  usage         累计 token 用量

多 Worker 通过 Redis 共享会话，任意 Worker 都能接管同一 session 的后续请求；
Redis 不可用时回退到进程内内存（仅单进程部署可用，多 Worker 请务必起 Redis）。

环境变量：
  REDIS_URL        Redis 连接串（默认 redis://127.0.0.1:6379/0）
  RAG_SESSION_TTL  会话过期秒数（默认 24 小时，每次读写滑动续期）
"""

import copy
import json
import os
import secrets
import threading
import time

SESSION_TTL = int(os.environ.get("RAG_SESSION_TTL", str(24 * 3600)))
REDIS_URL = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
KEY_PREFIX = "rag:session:"


def new_session_id() -> str:
    return secrets.token_urlsafe(16)


def empty_session() -> dict:
    return {"conversation": [], "summary": "", "folded": 0, "usage": {}}


class MemorySessionStore:
    """进程内兜底实现（仅单进程可用）。"""

    def __init__(self, ttl: int = SESSION_TTL):
        self._ttl = ttl
        self._data: dict[str, dict] = {}
        self._lock = threading.Lock()

    def load(self, sid: str) -> dict | None:
        with self._lock:
            rec = self._data.get(sid)
            if rec is None:
                return None
            rec["ts"] = time.time()
            return copy.deepcopy(rec["state"])

    def save(self, sid: str, state: dict) -> None:
        with self._lock:
            self._data[sid] = {"state": copy.deepcopy(state), "ts": time.time()}
            self._prune_locked()

    def reset(self, sid: str) -> None:
        with self._lock:
            self._data.pop(sid, None)

    def _prune_locked(self) -> None:
        now = time.time()
        expired = [s for s, r in self._data.items() if now - r["ts"] > self._ttl]
        for s in expired:
            self._data.pop(s, None)


class RedisSessionStore:
    def __init__(self, url: str = REDIS_URL, ttl: int = SESSION_TTL):
        import redis
        self._r = redis.Redis.from_url(url, decode_responses=True)
        self._r.ping()          # 提前失败，便于上层回退到内存实现
        self._ttl = ttl

    def _key(self, sid: str) -> str:
        return KEY_PREFIX + sid

    def load(self, sid: str) -> dict | None:
        raw = self._r.get(self._key(sid))
        if raw is None:
            return None
        self._r.expire(self._key(sid), self._ttl)   # 滑动续期
        return json.loads(raw)

    def save(self, sid: str, state: dict) -> None:
        self._r.set(self._key(sid), json.dumps(state, ensure_ascii=False), ex=self._ttl)

    def reset(self, sid: str) -> None:
        self._r.delete(self._key(sid))


def create_session_store():
    try:
        store = RedisSessionStore()
        print(f"[会话] 使用 Redis：{REDIS_URL}")
        return store
    except Exception as e:
        print(f"[会话] Redis 不可用（{e}），回退到进程内内存会话"
              f"（仅单进程部署可用，多 Worker 请先启动 Redis）")
        return MemorySessionStore()
