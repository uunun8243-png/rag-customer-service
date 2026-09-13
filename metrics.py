#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metrics.py — 轻量埋点聚合器（零依赖、线程安全）

记录每个请求各阶段的耗时，实时聚合出：
  - 每个阶段的 count / avg / min / max / p50 / p90 / p99（毫秒）
  - QPS：累计 QPS（服务启动以来）+ 滑动窗口 QPS（最近 WINDOW_SECONDS 秒）
  - token 吞吐（completion_tokens / 生成耗时，tokens/s）

阶段命名约定（由调用方 record）：
  intent_llm  意图识别 + 上下文补全 + 查询改写/扩展/HyDE 等「检索前」的 LLM 调用
  retrieval   向量 + BM25 + RRF 融合 + 合并 + 槽位过滤（本地计算，不含 LLM）
  rerank      精排（Qwen3-Reranker API / 本地推理）
  generation  最终答案 LLM 生成（含首 token 延迟 + 总时长）
  e2e         端到端（从收到请求到答案写回）

用法：
    from metrics import METRICS
    METRICS.record("rerank", 0.12)
    print(METRICS.snapshot())
"""

import threading
import time
from collections import deque

WINDOW_SECONDS = 60          # 滑动窗口长度（用于最近 QPS）
MAX_SAMPLES = 100000         # 每阶段保留的分位数样本上限


class _Stage:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.min = float("inf")
        self.max = 0.0
        self.samples = deque()          # 最近样本，用于分位数
        self.tokens = 0                 # 仅 generation 阶段累计（completion tokens）

    def record(self, seconds: float, tokens: int = 0) -> None:
        self.count += 1
        self.total += seconds
        if seconds < self.min:
            self.min = seconds
        if seconds > self.max:
            self.max = seconds
        self.tokens += tokens
        self.samples.append(seconds)
        if len(self.samples) > MAX_SAMPLES:
            self.samples.popleft()


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self._stages: dict[str, _Stage] = {}
        self._started = time.time()
        self._completions = deque()     # e2e 完成时间戳，用于窗口 QPS

    # ---------------- 记录 ----------------
    def record(self, stage: str, seconds: float, tokens: int = 0) -> None:
        if seconds < 0:
            seconds = 0.0
        with self._lock:
            s = self._stages.setdefault(stage, _Stage())
            s.record(seconds, tokens)
            if stage == "e2e":
                self._completions.append(time.time())
                cutoff = time.time() - WINDOW_SECONDS
                while self._completions and self._completions[0] < cutoff:
                    self._completions.popleft()

    def reset(self) -> None:
        with self._lock:
            self._stages.clear()
            self._completions.clear()
            self._started = time.time()

    # ---------------- 分位数 ----------------
    @staticmethod
    def _percentile(sorted_samples: list, p: float) -> float:
        if not sorted_samples:
            return 0.0
        k = (len(sorted_samples) - 1) * p
        lo = int(k)
        hi = min(lo + 1, len(sorted_samples) - 1)
        frac = k - lo
        return sorted_samples[lo] * (1 - frac) + sorted_samples[hi] * frac

    # ---------------- 快照 ----------------
    def snapshot(self) -> dict:
        with self._lock:
            uptime = time.time() - self._started
            out = {"uptime_seconds": round(uptime, 2), "stages": {}}
            for name, s in self._stages.items():
                avg = s.total / s.count if s.count else 0.0
                ordered = sorted(s.samples)
                stage = {
                    "count": s.count,
                    "avg_ms": round(avg * 1000, 1),
                    "min_ms": round(s.min * 1000, 1) if s.count else 0.0,
                    "max_ms": round(s.max * 1000, 1),
                    "p50_ms": round(self._percentile(ordered, 0.50) * 1000, 1),
                    "p90_ms": round(self._percentile(ordered, 0.90) * 1000, 1),
                    "p99_ms": round(self._percentile(ordered, 0.99) * 1000, 1),
                }
                if name == "generation" and s.tokens:
                    gen_sec = s.total if s.total > 0 else 0.0
                    stage["tokens_per_sec"] = round(s.tokens / gen_sec, 1) \
                        if gen_sec > 0 else 0.0
                    stage["total_tokens"] = s.tokens
                out["stages"][name] = stage

            n_total = out["stages"].get("e2e", {}).get("count", 0)
            qps_cum = n_total / uptime if uptime > 0 else 0.0
            now = time.time()
            cutoff = now - WINDOW_SECONDS
            window = [t for t in self._completions if t >= cutoff]
            out["qps"] = {
                "total_requests": n_total,
                "cumulative": round(qps_cum, 3),
                f"window_{WINDOW_SECONDS}s": round(len(window) / WINDOW_SECONDS, 3),
            }
            return out

    def pretty(self) -> str:
        """人类可读的单行汇总（用于打印到日志）。"""
        snap = self.snapshot()
        lines = [f"uptime={snap['uptime_seconds']}s"]
        for name, s in snap["stages"].items():
            lines.append(
                f"{name}: n={s['count']} avg={s['avg_ms']}ms "
                f"p50={s['p50_ms']}ms p90={s['p90_ms']}ms p99={s['p99_ms']}ms"
            )
        q = snap["qps"]
        lines.append(f"QPS: cum={q['cumulative']} win{q['window_%ds' % WINDOW_SECONDS]}")
        return "  ".join(lines)


# 全局单例（webapp / bench 共用）
METRICS = Metrics()
