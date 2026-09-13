#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_ragas.py — 用 Ragas 端到端评测这套 RAG（检索 + 生成）

与 eval_retrieval.py（Hit/MRR/Recall/Precision，只看检索）和 eval_generation.py
（自研 LLM 裁判判「忠实/正确」）不同，本脚本用业界标准的 Ragas 框架，把你**线上真正
在用的检索 + 精排 + 生成链路**跑一遍，再用 Ragas 的五项指标打分：

  - context_recall        上下文召回：标准答案所需的原文块，检索有没有捞回来
  - context_precision     上下文精度：检索回来的块里，有多少真的和问题相关（带参考答案版）
  - faithfulness          忠实性：生成的回答是否都来自检索到的原文（有无编造）
  - answer_relevancy      答案相关性：回答是否切题、不啰嗦
  - answer_correctness    答案正确性：回答与标准答案是否一致（语义 + 事实）

复用生产代码（保证「评测的就是上线的」）：
  - 检索/索引   rag_index.load_index() + rag_index.retrieve()（向量+BM25→RRF）
  - 精排        reranker.create_reranker()（Qwen3-Reranker，云端）
  - 生成        rag_tool.build_messages()（生产 SYSTEM_PROMPT + 上下文格式）+ llm.call_llm()
  - 切块        rag_index 内 structure_merge(500, 0.20)（即 merge_500_o20）

Ragas 的两个依赖组件：
  - 裁判 LLM     DeepSeek（OpenAI 兼容，走 instructor 的 JSON 结构化输出）
  - 嵌入模型     本地 fastembed 的 bge-small-zh-v1.5（离线，与线上检索同一个模型）

用法：
    .venv/bin/python eval_ragas.py                    # 全量 20 条 + 全部 5 项指标
    .venv/bin/python eval_ragas.py --limit 3          # 先跑 3 条冒烟
    .venv/bin/python eval_ragas.py --no-rerank        # 关精排，只测 RRF
    .venv/bin/python eval_ragas.py --topk 3           # 改最终上下文块数（默认 6）
    .venv/bin/python eval_ragas.py --metrics faithfulness answer_correctness
    .venv/bin/python eval_ragas.py --force            # 强制重跑生成（清缓存）
    .venv/bin/python eval_ragas.py --golden ragas_testset_golden.json   # 用 ragas 生成的评测集

产物：
    eval_ragas_run_{golden}.json     中间结果缓存（问题/检索块/回答/标准答案，重跑指标不重跑生成）
    eval_ragas_metrics_{golden}.csv  每个查询 × 每项指标的分数表
"""

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
GOLDEN_FILE = BASE_DIR / "golden_test_set.json"

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"


# --------------------------------------------------------------------------
# Ragas 嵌入封装：用本地 fastembed（与线上检索同一个 bge-small-zh-v1.5，离线、免 torch）
# --------------------------------------------------------------------------
def FastEmbedEmbedding(model_name: str = EMBED_MODEL):
    """把 fastembed 适配成 Ragas 的 BaseRagasEmbedding 实例（embed_text / aembed_text）。"""
    from ragas.embeddings.base import BaseRagasEmbedding
    from fastembed import TextEmbedding

    class _Impl(BaseRagasEmbedding):
        def __init__(self):
            super().__init__(cache=None)
            self._model = TextEmbedding(model_name)

        def embed_text(self, text: str, **kwargs):
            return next(iter(self._model.embed([text]))).tolist()

        async def aembed_text(self, text: str, **kwargs):
            return self.embed_text(text, **kwargs)

        def embed_texts(self, texts: list, **kwargs):
            return [list(v) for v in self._model.embed(list(texts))]

        # 兼容 legacy 指标（AnswerRelevancy 走 embed_query / embed_documents）
        def embed_query(self, text: str):
            return self.embed_text(text)

        def embed_documents(self, texts: list):
            return self.embed_texts(list(texts))

    return _Impl()


def build_llm(model: str):
    """DeepSeek（OpenAI 兼容）→ Ragas 裁判 LLM（langchain ChatOpenAI + 强制 JSON 输出）。

    注意：ragas 0.4.3 里 evaluate() 只认 legacy Metric（ragas.metrics._*），
    它们要的是 LangchainLLMWrapper（BaseRagasLLM），不是新的 collections 指标。
    """
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    from llm import get_api_key

    key = get_api_key()
    if not key:
        sys.exit("找不到 DeepSeek API Key。请设置 DEEPSEEK_API_KEY，"
                 "或把 key 放到 ~/.pi/agent/auth.json。")
    chat = ChatOpenAI(
        model=model,
        base_url=DEEPSEEK_BASE_URL,
        api_key=key,
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    # bypass_n/bypass_temperature：DeepSeek 推理模型不一定支持 n/温度参数，
    # 交给 ChatOpenAI 自身的默认值，避免 wrapper 覆盖。
    return LangchainLLMWrapper(chat, bypass_n=True, bypass_temperature=True)


def build_metrics(llm, embeddings, names: list[str]):
    """按名字构造 Ragas legacy 指标对象（都绑定同一个裁判 LLM / 嵌入模型）。"""
    from ragas.metrics._answer_correctness import AnswerCorrectness
    from ragas.metrics._answer_relevance import AnswerRelevancy
    from ragas.metrics._context_precision import ContextPrecision
    from ragas.metrics._context_recall import ContextRecall
    from ragas.metrics._faithfulness import Faithfulness

    registry = {
        "context_recall": lambda: ContextRecall(llm=llm),
        "context_precision": lambda: ContextPrecision(llm=llm),
        "faithfulness": lambda: Faithfulness(llm=llm),
        "answer_relevancy": lambda: AnswerRelevancy(llm=llm, embeddings=embeddings),
        "answer_correctness": lambda: AnswerCorrectness(llm=llm, embeddings=embeddings),
    }
    return [registry[n]() for n in names]


# --------------------------------------------------------------------------
# 阶段一：跑线上链路（检索 + 精排 + 生成）
# --------------------------------------------------------------------------
def run_pipeline(queries: list[dict], index, reranker, api_key: str,
                 top_k: int, temperature: float) -> list[dict]:
    """逐条查询跑真实链路，返回 [{query, retrieved_contexts, response, reference, reference_contexts}]。

    检索路径与 rag_tool.answer_direct 一致：先取宽窗口（top_k*2 与 top_k+4 的较大者）
    再精排截断到 top_k；生成用生产 SYSTEM_PROMPT + 上下文格式（rag_tool.build_messages）。
    """
    from llm import call_llm
    from rag_index import retrieve
    from rag_tool import build_messages

    fetch_k = max(top_k * 2, top_k + 4)
    records = []
    for q in queries:
        hits = retrieve(index, q["query"], top_k=fetch_k)
        if reranker is not None and len(hits) > top_k:
            try:
                hits = reranker.rerank(q["query"], hits, top_k=top_k)
            except Exception as e:
                print(f"  [精排失败，回退 RRF 排序] {q['id']}: {e}")
                hits = hits[:top_k]
        else:
            hits = hits[:top_k]

        contexts = [h["chunk"]["text"] for h in hits]
        answer = call_llm(build_messages(q["query"], hits), api_key,
                          temperature=temperature)
        records.append({
            "id": q["id"],
            "query": q["query"],
            "retrieved_contexts": contexts,
            "response": answer,
            "reference": q["answer"],
            "reference_contexts": q["golden_passages"],
        })
        print(f"  [{q['id']}] 完成（{len(contexts)} 个上下文块，回答 {len(answer)} 字）")
    return records


# --------------------------------------------------------------------------
# 阶段二：Ragas 打分
# --------------------------------------------------------------------------
def build_dataset(records: list[dict]):
    from ragas import EvaluationDataset, SingleTurnSample

    samples = [
        SingleTurnSample(
            user_input=r["query"],
            retrieved_contexts=r["retrieved_contexts"],
            reference_contexts=r["reference_contexts"],
            response=r["response"],
            reference=r["reference"],
        )
        for r in records
    ]
    return EvaluationDataset(samples=samples)


def print_report(result, records: list[dict], csv_path: Path) -> None:
    import numpy as np
    import pandas as pd

    df = result.to_pandas()
    score_cols = [c for c in df.columns
                  if c not in ("user_input", "retrieved_contexts", "reference",
                               "reference_contexts", "response")]
    print("\n" + "=" * 78)
    print("Ragas 汇总（0~1，越大越好）")
    print("=" * 78)
    for c in score_cols:
        vals = pd.to_numeric(df[c], errors="coerce")
        n_valid = int(vals.notna().sum())
        mean = float(vals.mean()) if n_valid else float("nan")
        print(f"  {c:<24} mean={mean:.3f}  （有效 {n_valid}/{len(df)}）")

    print(f"\n  token 消耗：未统计（需给 evaluate() 传 token_usage_parser）")

    # 逐条明细表（加回查询 ID 便于对照）
    ids = [r["id"] for r in records]
    queries = [r["query"] for r in records]
    out = df.copy()
    out.insert(0, "id", ids)
    out.insert(1, "query", queries)
    out.to_csv(csv_path, index=False)
    print(f"\n逐条明细已写入 {csv_path}")

    # 打印最差的几条，方便定位
    if score_cols:
        worst_col = score_cols[0]
        ranked = out.sort_values(worst_col, na_position="last")
        print(f"\n按 {worst_col} 从低到高（前 5 条）：")
        for _, row in ranked.head(5).iterrows():
            scores = " ".join(f"{c}={row[c]:.2f}" if pd.notna(row[c]) else f"{c}=NaN"
                              for c in score_cols)
            print(f"  {row['id']} {row['query'][:34]:<36} | {scores}")


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="用 Ragas 端到端评测 RAG（检索+精排+生成）")
    ap.add_argument("--topk", type=int, default=6, help="最终交给生成的上下文块数（默认 6）")
    ap.add_argument("--limit", type=int, default=0, help="只测前 N 条（0=全部）")
    ap.add_argument("--no-rerank", action="store_true", help="关闭精排，只用 RRF 混合检索")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="生成回答的温度（默认 0.0 保证可复现）")
    ap.add_argument("--llm", default="", help="裁判/生成 LLM 模型（默认取 RAG_LLM_MODEL 或 deepseek-v4-flash）")
    ap.add_argument("--metrics", nargs="+", default=[
        "context_recall", "context_precision", "faithfulness",
        "answer_relevancy", "answer_correctness"],
        help="要计算的指标（默认全部 5 项）")
    ap.add_argument("--golden", default=str(GOLDEN_FILE),
                    help="评测集 JSON 路径（默认 golden_test_set.json，可用 ragas_testset_golden.json）")
    ap.add_argument("--force", action="store_true", help="忽略缓存，强制重跑生成链路")
    args = ap.parse_args()

    from llm import LLM_MODEL, get_api_key

    model = args.llm or LLM_MODEL
    api_key = get_api_key()
    if not api_key:
        sys.exit("找不到 DeepSeek API Key。请设置 DEEPSEEK_API_KEY，"
                 "或把 key 放到 ~/.pi/agent/auth.json。")

    golden_file = Path(args.golden)
    if not golden_file.is_file():
        sys.exit(f"评测集文件不存在：{golden_file}")
    run_cache = BASE_DIR / f"eval_ragas_run_{golden_file.stem}.json"
    metrics_csv = BASE_DIR / f"eval_ragas_metrics_{golden_file.stem}.csv"

    golden = json.loads(golden_file.read_text(encoding="utf-8"))
    queries = golden["queries"]
    if args.limit:
        queries = queries[:args.limit]

    print("=" * 78)
    print(f"Ragas 评测 · {golden['name']}")
    print(f"查询：{len(queries)} 条｜Top-K：{args.topk}｜精排：{'关' if args.no_rerank else '开'}")
    print(f"LLM：{model}｜指标：{'、'.join(args.metrics)}")
    print("=" * 78)

    # ---- 阶段一：跑链路（带缓存）----
    records = None
    if run_cache.exists() and not args.force and args.limit == 0:
        try:
            records = json.loads(run_cache.read_text(encoding="utf-8"))
            if len(records) == len(queries):
                print(f"命中缓存 {run_cache.name}（{len(records)} 条），跳过生成链路。")
            else:
                records = None
        except Exception:
            records = None

    if records is None:
        from rag_index import load_index
        from reranker import create_reranker

        print("加载索引…")
        index = load_index()
        print(f"索引后端：{'Qdrant 服务端' if type(index).__name__ == 'QdrantIndex' else '内存 numpy'}，"
              f"共 {len(index.chunks)} 块")

        reranker = None if args.no_rerank else create_reranker()
        if reranker is None and not args.no_rerank:
            print("  [提示] 未配置精排 API Key，本次只走 RRF 混合检索。")

        print("开始逐条跑检索 + 精排 + 生成…")
        records = run_pipeline(queries, index, reranker, api_key, args.topk, args.temperature)
        if args.limit == 0:
            run_cache.write_text(json.dumps(records, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
            print(f"中间结果已缓存到 {run_cache.name}（改 --topk/--force 可重跑）")

    # ---- 阶段二：Ragas 打分 ----
    print("\n初始化 Ragas 裁判 LLM 与嵌入模型…")
    llm = build_llm(model)
    embeddings = FastEmbedEmbedding(EMBED_MODEL)
    metrics = build_metrics(llm, embeddings, args.metrics)
    dataset = build_dataset(records)

    print(f"开始 Ragas 打分（{len(records)} 条 × {len(metrics)} 项指标，LLM 调用较多，耐心等待）…")
    from ragas import evaluate

    result = evaluate(dataset, metrics=metrics)
    print_report(result, records, metrics_csv)


if __name__ == "__main__":
    main()
