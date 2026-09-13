#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_generation.py — 生成侧评估：回答是否忠于原文（有无编造）、是否正确

把选定的切块策略检索出的 top-k 块喂给 DeepSeek 生成回答，再用 DeepSeek 当
「裁判」逐条判断两个维度：
  - faithful ：AI 回答的每个信息点是否都能在检索到的原文中找到依据（false = 有编造/幻觉）
  - correct  ：AI 回答是否与 golden_test_set.json 里的标准答案一致

用法：
    .venv/bin/python eval_generation.py                     # 默认 merge_400_o20 + top3
    .venv/bin/python eval_generation.py --strategy merge_500_o10 --topk 5
    .venv/bin/python eval_generation.py --llm deepseek-v4-pro

与 eval_retrieval.py（Hit-Rate / MRR）的区别：
    检索指标只看「答案出处有没有被捞回来」，本脚本看「LLM 拿到这些块后，
    生成的回答是不是照着原文说、有没有自己编、对不对」。
"""

import argparse
import json
import re
import sys
from pathlib import Path

import httpx
import numpy as np

from eval_retrieval import DOC_FILE, CHUNKERS, load_embedder, embed

GOLDEN_FILE = Path(__file__).resolve().parent / "golden_test_set.json"
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-flash"


# --------------------------------------------------------------------------
# API Key / LLM 调用
# --------------------------------------------------------------------------
def get_api_key() -> str:
    import os
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key
    p = Path.home() / ".pi" / "agent" / "auth.json"
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        key = (data.get("deepseek") or {}).get("key")
        if key:
            return key
    sys.exit("找不到 DeepSeek API Key。")


def llm_chat(messages: list[dict], model: str = MODEL, temperature: float = 0.0) -> str:
    r = httpx.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {get_api_key()}",
                 "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": temperature},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------
# 检索 + 生成 + 裁判
# --------------------------------------------------------------------------
def retrieve(chunks, query: str, model, topk: int):
    cv = embed(model, [c.text for c in chunks])
    qv = embed(model, [query])
    sims = (cv @ qv.T).ravel()
    order = np.argsort(-sims)[:topk]
    return [(chunks[i], float(sims[i])) for i in order]


GEN_SYSTEM = (
    "你是一个售后客服知识库助手。请只根据下面提供的「参考资料」回答问题。\n"
    "要求：\n"
    "1. 只依据参考资料，资料里没有的信息就明确说「资料中没有提到」，不要编造；\n"
    "2. 直接给出结论，简洁，用中文。"
)


def generate(query: str, context: str, model_name: str) -> str:
    user = f"问题：{query}\n\n参考资料：\n{context}\n\n请回答："
    return llm_chat(
        [{"role": "system", "content": GEN_SYSTEM},
         {"role": "user", "content": user}],
        model=model_name,
    )


JUDGE_SYSTEM = (
    "你是一个严格的评估裁判。请判断「AI 回答」是否忠实于「参考资料」，以及是否正确。\n"
    "只输出一个 JSON 对象，不要输出任何其他文字。格式：\n"
    '{"faithful": true/false, "hallucination": "编造的具体内容，无则空字符串", '
    '"correct": true/false, "reason": "一句话理由"}\n\n'
    "判断标准：\n"
    "- faithful=true：AI 回答的每个信息点都能在参考资料中找到依据，没有编造原文没有的内容；\n"
    "  false：存在编造/幻觉（说了原文里没有的、或与原文矛盾的内容）。\n"
    "- correct=true：AI 回答与标准答案在关键信息上一致；false：不一致（答错、答非所问、遗漏关键信息等）。"
)


def judge(query: str, context: str, answer: str, golden: str, model_name: str) -> dict:
    user = (
        f"问题：{query}\n\n"
        f"参考资料：\n{context}\n\n"
        f"AI 回答：{answer}\n\n"
        f"标准答案：{golden}\n\n"
        "请输出 JSON："
    )
    raw = llm_chat(
        [{"role": "system", "content": JUDGE_SYSTEM},
         {"role": "user", "content": user}],
        model=model_name,
    )
    return parse_json(raw)


def parse_json(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {"faithful": None, "hallucination": raw.strip()[:200],
                "correct": None, "reason": "JSON 解析失败"}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {"faithful": None, "hallucination": raw.strip()[:200],
                "correct": None, "reason": "JSON 解析失败"}


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="生成侧评估（忠实性 / 正确性）")
    ap.add_argument("--strategy", default="window_500_o10", help="切块策略名（见 eval_retrieval.CHUNKERS）")
    ap.add_argument("--topk", type=int, default=3, help="检索块数（默认 3）")
    ap.add_argument("--llm", default=MODEL, help="LLM 模型名")
    ap.add_argument("--limit", type=int, default=0, help="只测前 N 条（0=全部）")
    args = ap.parse_args()

    if args.strategy not in CHUNKERS:
        sys.exit(f"未知切块策略：{args.strategy}。可选：{', '.join(CHUNKERS)}")

    text = DOC_FILE.read_text(encoding="utf-8")
    golden = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))
    queries = golden["queries"]
    if args.limit:
        queries = queries[:args.limit]

    print(f"切块策略：{args.strategy}｜Top-K：{args.topk}｜模型：{args.llm}")
    print(f"查询数：{len(queries)}")
    print("加载嵌入模型…")
    model = load_embedder()

    chunks = CHUNKERS[args.strategy](text)
    print(f"切块数：{len(chunks)}，开始逐条生成 + 裁判…\n")

    n_faithful = n_correct = 0
    results = []
    for q in queries:
        hits = retrieve(chunks, q["query"], model, args.topk)
        context = "\n\n".join(f"[块{i}] {c.text}" for i, (c, _) in enumerate(hits, 1))
        answer = generate(q["query"], context, args.llm)
        verdict = judge(q["query"], context, answer, q["answer"], args.llm)

        faithful = verdict.get("faithful")
        correct = verdict.get("correct")
        n_faithful += 1 if faithful is True else 0
        n_correct += 1 if correct is True else 0
        results.append((q, answer, verdict))

        print("=" * 72)
        print(f"[{q['id']}] {q['query']}")
        print(f"  回答：{answer}")
        print(f"  忠实原文：{faithful}   正确：{correct}")
        if verdict.get("hallucination"):
            print(f"  编造内容：{verdict['hallucination']}")
        if verdict.get("reason"):
            print(f"  理由：{verdict['reason']}")

    n = len(queries)
    print("\n" + "=" * 72)
    print("汇总")
    print(f"  忠实原文率（不编造）：{n_faithful}/{n} = {n_faithful/n:.3f}")
    print(f"  正确率：{n_correct}/{n} = {n_correct/n:.3f}")
    print(f"  存在编造/幻觉：{n - n_faithful} 条")
    print(f"  答错/不一致：{n - n_correct} 条")


if __name__ == "__main__":
    main()
