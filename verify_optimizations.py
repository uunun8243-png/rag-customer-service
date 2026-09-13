#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_optimizations.py — 验证三项成本/延迟优化的真实收益（真实调用 LLM/rerank）

  1. 规则优先意图：简单事实查询跳过 LLM 意图分析
  2. 答案缓存：相同问题二次命中直接返回
  3. 多轮合并：指代问题补全+意图合并为一次 LLM 调用

用法：
    .venv/bin/python verify_optimizations.py
"""

import contextlib
import io
import time

import query_intent
from llm import get_api_key
from rag_index import INDEX_FINGERPRINT, TOP_K, load_index, retrieve
from rag_tool import ask
from reranker import create_reranker
from answer_cache import create_answer_cache

# ---------------- 统计真实 LLM 调用次数 ----------------
_orig_chat_json = query_intent.chat_json
_orig_call_llm = query_intent.call_llm
counts = {"chat_json": 0, "call_llm": 0}


def counting_chat_json(*a, **k):
    counts["chat_json"] += 1
    return _orig_chat_json(*a, **k)


def counting_call_llm(*a, **k):
    counts["call_llm"] += 1
    return _orig_call_llm(*a, **k)


query_intent.chat_json = counting_chat_json
query_intent.call_llm = counting_call_llm


def reset_counts():
    counts["chat_json"] = counts["call_llm"] = 0


def run(q: str, state: dict) -> tuple[dict, float]:
    buf = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(buf):
        r = ask(q, state)
    return r, (time.time() - t0) * 1000


def main() -> None:
    api_key = get_api_key()
    if not api_key:
        raise SystemExit("找不到 DeepSeek API Key")
    print("加载索引…")
    index = load_index()
    state = {
        "chunks": index.chunks,
        "api_key": api_key,
        "retriever": lambda q, k=TOP_K: retrieve(index, q, top_k=k),
        "reranker": create_reranker(),
        "conversation": [],
        "summary": "",
        "folded": 0,
        "usage": {},
        "answer_cache": create_answer_cache(INDEX_FINGERPRINT),
    }

    print("=" * 70)

    # 场景 1：规则优先意图（单轮冷启动，简单事实查询）
    reset_counts()
    r1, ms1 = run("普通商品的退货期限是多久？", state)
    rule1 = r1.get("intent") and r1["intent"].rule_based
    print(f"[1] 规则意图    : {ms1:7.0f}ms  意图LLM调用={counts['chat_json']} "
          f"规则命中={rule1}")

    # 记录第一轮对话（模拟交互循环）
    state["conversation"].append({"role": "user", "content": "普通商品的退货期限是多久？"})
    state["conversation"].append({"role": "assistant", "content": r1.get("answer", "")})

    # 场景 2：答案缓存命中（相同问题二次提问）
    reset_counts()
    r2, ms2 = run("普通商品的退货期限是多久？", state)
    print(f"[2] 缓存命中    : {ms2:7.0f}ms  意图LLM调用={counts['chat_json']} "
          f"cached={r2.get('cached', False)}")

    # 场景 3：多轮合并（指代问题 → 补全+意图 1 次调用）
    reset_counts()
    r3, ms3 = run("那生鲜商品呢？", state)
    comp3 = r3.get("completed_query", "")
    print(f"[3] 多轮合并    : {ms3:7.0f}ms  意图LLM调用={counts['chat_json']} "
          f"补全后={comp3!r}")

    print("=" * 70)
    print("说明：chat_json 为意图阶段的 LLM 调用（补全/意图分析/改写扩展），")
    print("      call_llm 为 HyDE 等辅助调用（默认关闭）。生成阶段另有 1 次流式调用。")
    print("      基线（优化前）意图阶段平均 5.7s、每次至少 1 次 chat_json。")


if __name__ == "__main__":
    main()
