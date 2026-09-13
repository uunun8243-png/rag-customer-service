#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench.py — 批量压测：逐请求埋点，输出各阶段延迟分布 + QPS + 生成吞吐

用 golden_test_set.json 的查询串行跑完整链路（意图 → 检索 → 精排 → 生成），
每次请求把 intent_llm / retrieval / rerank / generation / e2e 的耗时喂给
metrics.METRICS，跑完后打印分位数汇总与 QPS。

用法：
    .venv/bin/python bench.py                    # 20 条各跑 1 遍
    .venv/bin/python bench.py --limit 5          # 只跑前 5 条
    .venv/bin/python bench.py --repeat 3         # 每条跑 3 遍（放大样本）
    .venv/bin/python bench.py --quiet            # 不逐条打印回答

口径说明：
    - 串行跑，因此「累计 QPS」≈ 1 / 平均端到端延迟，即单并发下的 QPS 上限。
    - 要测并发 QPS，需要多进程/去掉 webapp 的串行锁（见 webapp.py 的 _state_lock）。
"""

import argparse
import json
import time
from pathlib import Path

from llm import LLMError, call_llm, call_llm_stream, get_api_key
from metrics import METRICS
from query_intent import retrieve_with_intent
from rag_index import TOP_K, load_index, retrieve
from rag_tool import build_messages
from reranker import create_reranker

GOLDEN_FILE = Path(__file__).resolve().parent / "golden_test_set.json"


def build_state() -> dict:
    api_key = get_api_key()
    if not api_key:
        raise SystemExit("找不到 DeepSeek API Key（DEEPSEEK_API_KEY 或 ~/.pi/agent/auth.json）")
    print("加载索引…")
    index = load_index()
    return {
        "chunks": index.chunks,
        "api_key": api_key,
        "retriever": lambda q, k=TOP_K: retrieve(index, q, top_k=k),
        "reranker": create_reranker(),
        "conversation": [],
        "summary": "",
        "folded": 0,
        "usage": {},
    }


def run_one(question: str, state: dict) -> dict:
    """跑一次完整链路并埋点，返回 {question, action, answer, e2e}。"""
    t0 = time.time()
    result = retrieve_with_intent(question, state, retriever=state["retriever"],
                                  base_top_k=TOP_K, metrics=METRICS)
    if result["action"] != "answer":
        e2e = time.time() - t0
        METRICS.record("e2e", e2e)
        return {"question": question, "action": result["action"], "answer": "", "e2e": e2e}

    hits = result["hits"]
    usage = state.setdefault("usage", {})
    messages = build_messages(question, hits)
    prev = usage.get("completion_tokens", 0)
    t_gen = time.time()
    parts: list[str] = []
    try:
        for delta in call_llm_stream(messages, state["api_key"], usage=usage):
            parts.append(delta)
    except LLMError:
        if not parts:
            parts.append(call_llm(messages, state["api_key"], usage=usage))
    METRICS.record("generation", time.time() - t_gen,
                   tokens=usage.get("completion_tokens", 0) - prev)

    answer = "".join(parts)
    e2e = time.time() - t0
    METRICS.record("e2e", e2e)
    return {"question": question, "action": "answer", "answer": answer, "e2e": e2e}


def main() -> None:
    ap = argparse.ArgumentParser(description="批量压测：延迟分布 + QPS + 生成吞吐")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条查询（0=全部）")
    ap.add_argument("--repeat", type=int, default=1, help="每条查询重复跑 R 遍")
    ap.add_argument("--quiet", action="store_true", help="不逐条打印回答")
    args = ap.parse_args()

    state = build_state()
    queries = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))["queries"]
    if args.limit:
        queries = queries[:args.limit]
    total = len(queries) * args.repeat
    print(f"查询数：{len(queries)} × {args.repeat} 轮 = {total} 个请求")
    print("=" * 90)

    METRICS.reset()
    t_wall0 = time.time()
    for rnd in range(args.repeat):
        for q in queries:
            row = run_one(q["query"], state)
            if not args.quiet:
                print(f"[{q['id']}] e2e={row['e2e'] * 1000:6.0f}ms  {q['query']}")
                if row["action"] == "answer":
                    print(f"        {row['answer'][:80]}")
                else:
                    print(f"        （{row['action']}）")
    wall = time.time() - t_wall0

    print("=" * 90)
    print(METRICS.pretty())
    print(f"墙钟总耗时：{wall:.2f}s，墙钟 QPS = {total / wall:.3f}")


if __name__ == "__main__":
    main()
