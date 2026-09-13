#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_retrieval.py — 用 Golden Test Set 评估切块策略的检索质量（向量 vs BM25 vs 混合）

在实现真正的切块策略之前，先用 golden_test_set.json 里的真实查询 + 标注答案出处，
把「切块 → 检索 → 命中」的评估闭环跑通，之后任何新的切块策略只需注册成一个函数
放进 CHUNKERS 即可横向对比。

指标（@K 表示只看 Top-K 检索结果）：
  - Hit-Rate@K ：Top-K 结果里至少命中 1 个正确答案块（golden chunk）的查询占比
  - MRR@K      ：首个正确答案块排名倒数的平均值（1/rank），命中越靠前越接近 1
  - Recall@K   ：Top-K 命中的相关块数 / 该查询全部相关块数，按查询平均
  - Precision@K：Top-K 里相关块所占比例（相关块数 / K），按查询平均

用法：
    .venv/bin/python eval_retrieval.py                 # 向量 / BM25 / 混合(RRF) 一起对比
    .venv/bin/python eval_retrieval.py --mode rrf      # 只看混合检索
    .venv/bin/python eval_retrieval.py --mode bm25     # 只看 BM25
    .venv/bin/python eval_retrieval.py --rerank        # 对比 RRF 与 RRF+Qwen3-Reranker 精排
    .venv/bin/python eval_retrieval.py --topk 1 3 5    # 自定义 Top-K
    .venv/bin/python eval_retrieval.py --verbose       # 打印每条查询的命中详情

原理：
    1. 读 clean.md 全文，记录每个 golden passage 的字符区间 [start, end)
    2. 用不同策略把全文切成若干 chunk（保留每个 chunk 的字符区间）
    3. chunk 与某个 golden passage 的重叠比 ≥ min_overlap_ratio 即视为"正确答案块"
    4. 三种检索方式：向量相似度（bge-small-zh-v1.5 余弦）、BM25（jieba 分词）、
       以及两者 RRF（Reciprocal Rank Fusion）混合
    5. 统计 Hit-Rate、MRR、Recall、Precision，输出对比表
"""

import argparse
import json
import re
import zlib
from pathlib import Path

import numpy as np

from chunker import Chunk, full_doc, structure_merge

BASE_DIR = Path(__file__).resolve().parent
DOC_FILE = BASE_DIR / "notes" / "clean.md"
GOLDEN_FILE = BASE_DIR / "golden_test_set.json"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"


# 参与对比的策略（名字 -> 切块函数）—— 切块实现统一在 chunker.py（单一数据源）
CHUNKERS = {
    "full_doc": full_doc,                                          # 天花板对照（1 块）
    "merge_400_o20": lambda t: structure_merge(t, 400, 0.20),     # ★ 推荐：实测最优
    "merge_400_o10": lambda t: structure_merge(t, 400, 0.10),
    "merge_300_o20": lambda t: structure_merge(t, 300, 0.20),
    "merge_500_o20": lambda t: structure_merge(t, 500, 0.20),
    "merge_400": lambda t: structure_merge(t, 400, 0.0),          # 无 overlap 对照
}


# --------------------------------------------------------------------------
# Golden passage 区间定位
# --------------------------------------------------------------------------
def locate_golden_spans(text: str, queries: list[dict]) -> dict[str, list[tuple[int, int]]]:
    """把每条查询的 golden passages 定位成全文中的字符区间（可能多处出现）。"""
    spans_by_id = {}
    for q in queries:
        spans = []
        for p in q["golden_passages"]:
            idx = text.find(p)
            while idx != -1:
                spans.append((idx, idx + len(p)))
                idx = text.find(p, idx + 1)
        if not spans:
            print(f"  [警告] {q['id']} {q['query']} 的出处未在文档中找到")
        spans_by_id[q["id"]] = spans
    return spans_by_id


def overlap_ratio(cs: int, ce: int, ps: int, pe: int) -> float:
    """chunk 区间 [cs,ce) 与 golden 区间 [ps,pe) 的重叠占 golden 长度的比例。"""
    o = max(0, min(ce, pe) - max(cs, ps))
    return o / max(1, pe - ps)


# --------------------------------------------------------------------------
# 检索 + 指标
# --------------------------------------------------------------------------
class HashingEmbedder:
    """无网络降级：字符 n-gram 特征哈希向量（确定性、本地、零下载）。

    仅用于离线环境下跑通评估流程；有网络时应优先用 bge-small-zh-v1.5 语义向量。
    """

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _stable_hash(self, s: str, seed: int) -> int:
        return zlib.crc32(f"{seed}:{s}".encode("utf-8"))

    def _features(self, text: str):
        feats = []
        for i, ch in enumerate(text):
            if not ch.isspace():
                feats.append(ch)          # 单字
            if i + 1 < len(text) and not text[i].isspace() and not text[i + 1].isspace():
                feats.append(text[i:i + 2])  # 相邻字 bigram
        return feats

    def embed(self, texts: list[str]):
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            for f in self._features(t):
                idx = self._stable_hash(f, 0) % self.dim
                sign = 1.0 if (self._stable_hash(f, 1) & 1) else -1.0
                v[idx] += sign
            yield v


def load_embedder():
    try:
        from fastembed import TextEmbedding
        print(f"  加载嵌入模型 {EMBED_MODEL}（首次会自动下载）…")
        return TextEmbedding(EMBED_MODEL)
    except Exception as e:  # 网络不可达 / 模型未缓存
        print(f"  [降级] 无法加载 bge 模型（{type(e).__name__}），"
              f"改用本地字符 n-gram 哈希向量。")
        return HashingEmbedder()


def embed(model, texts: list[str]) -> np.ndarray:
    vecs = np.array(list(model.embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def tokenize(text: str) -> list[str]:
    """中文分词（jieba）；不可用时退化为「单字 + 相邻字 bigram」。"""
    try:
        import jieba
        return [w for w in jieba.lcut(text) if w.strip()]
    except ImportError:
        t = re.sub(r"\s+", "", text)
        return list(t) + [t[i:i + 2] for i in range(len(t) - 1)]


def vector_ranker(model, chunk_vecs: np.ndarray):
    """返回 rank(query_text) -> 按余弦相似度降序的 chunk 下标。"""
    def rank(qtext: str) -> np.ndarray:
        qvec = embed(model, [qtext])
        sims = (chunk_vecs @ qvec.T).ravel()
        return np.argsort(-sims)
    return rank


def bm25_ranker(chunks: list[Chunk]):
    """返回 rank(query_text) -> 按 BM25 分数降序的 chunk 下标（jieba 分词）。"""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise SystemExit("缺少 rank-bm25：请运行 uv pip install --python .venv/bin/python rank-bm25")
    bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def rank(qtext: str) -> np.ndarray:
        scores = np.asarray(bm25.get_scores(tokenize(qtext)), dtype=np.float32)
        return np.argsort(-scores)
    return rank


def rrf_ranker(vec_rank_fn, bm_rank_fn, n_chunks: int, k_rrf: float = 60.0):
    """混合检索：RRF（Reciprocal Rank Fusion）融合向量与 BM25 的排序。

    score(chunk) = Σ 1/(k + rank)，rank 为该 chunk 在各检索器中的名次（1 起）。
    融合后能同时继承向量的语义排序与 BM25 的精确词召回。
    """
    def rank(qtext: str) -> np.ndarray:
        vr = vec_rank_fn(qtext)
        br = bm_rank_fn(qtext)
        v_pos = np.empty(n_chunks, dtype=np.float32)
        b_pos = np.empty(n_chunks, dtype=np.float32)
        v_pos[vr] = np.arange(1, n_chunks + 1, dtype=np.float32)
        b_pos[br] = np.arange(1, n_chunks + 1, dtype=np.float32)
        scores = 1.0 / (k_rrf + v_pos) + 1.0 / (k_rrf + b_pos)
        return np.argsort(-scores)
    return rank


def rerank_ranker(rrf_rank_fn, chunks: list[Chunk], reranker, n_candidates: int = 20):
    """在 RRF 排序之上加 Qwen3-Reranker 精排：取 RRF 前 n_candidates 个候选做
    深度重排，重排后的候选排前，其余候选按原 RRF 顺序拼在后面。"""
    def rank(qtext: str) -> np.ndarray:
        order = rrf_rank_fn(qtext)
        top = [int(i) for i in order[:n_candidates]]
        hits = [{"idx": i, "chunk": chunks[i]} for i in top]
        scored = reranker.rerank(qtext, hits)
        reranked = [h["idx"] for h in scored]
        seen = set(reranked)
        tail = [int(i) for i in order if int(i) not in seen]
        return np.array(reranked + tail, dtype=np.int64)
    return rank


def compute_metrics(chunks: list[Chunk], queries: list[dict],
                    golden_spans: dict, rank_fn, top_k_values: list[int],
                    min_overlap: float, verbose: bool = False, label: str = ""):
    """给定 rank_fn(query_text)->ranked chunk 下标，计算四类指标。"""
    # 每个 chunk 是否与某条查询的 golden 区间重叠（按查询分别算）
    q_rel = {}
    for q in queries:
        gs = golden_spans[q["id"]]
        q_rel[q["id"]] = [
            any(overlap_ratio(c.start, c.end, ps, pe) >= min_overlap for ps, pe in gs)
            for c in chunks
        ]

    metrics = {k: {"hit": 0, "mrr": 0.0, "recall": 0.0, "precision": 0.0}
               for k in top_k_values}
    n_rel = {k: 0 for k in top_k_values}   # 有相关块的查询数（Recall 平均的分母）
    details = []

    for q in queries:
        order = rank_fn(q["query"])
        rel = q_rel[q["id"]]

        first_rank = None
        for rank, ci in enumerate(order, start=1):
            if rel[ci]:
                first_rank = rank
                break

        total_rel = int(sum(rel))
        for k in top_k_values:
            rel_retrieved = int(sum(rel[ci] for ci in order[:k]))
            metrics[k]["hit"] += int(rel_retrieved > 0)
            mrr = (1.0 / first_rank) if (first_rank is not None and first_rank <= k) else 0.0
            metrics[k]["mrr"] += mrr
            metrics[k]["precision"] += rel_retrieved / k
            if total_rel > 0:
                metrics[k]["recall"] += rel_retrieved / total_rel
                n_rel[k] += 1

        details.append({"id": q["id"], "query": q["query"], "first_rank": first_rank})

    n = len(queries)
    for k in top_k_values:
        metrics[k]["hit"] = metrics[k]["hit"] / n
        metrics[k]["mrr"] = metrics[k]["mrr"] / n
        metrics[k]["precision"] = metrics[k]["precision"] / n
        metrics[k]["recall"] = metrics[k]["recall"] / (n_rel[k] or 1)

    if verbose:
        print(f"\n  [{label}] 逐条命中详情：")
        for d in details:
            fr = d["first_rank"]
            fr_txt = f"首个答案块排第 {fr}" if fr is not None else "未命中"
            print(f"    {d['id']} {d['query']:<28} -> {fr_txt}")

    return metrics, len(chunks)


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="切块策略检索评估（向量 vs BM25：Hit/MRR/Recall/Precision）")
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 3, 5],
                        help="评估的 Top-K 值（默认 1 3 5）")
    parser.add_argument("--min-overlap", type=float, default=0.5,
                        help="chunk 与 golden passage 的最小重叠比（默认 0.5）")
    parser.add_argument("--mode", choices=["vector", "bm25", "rrf", "both", "all"], default="all",
                        help="检索方式：vector / bm25 / rrf(混合) / both / all（默认 all）")
    parser.add_argument("--rrf-k", type=float, default=60.0,
                        help="RRF 融合常数 k（默认 60）")
    parser.add_argument("--rerank", action="store_true",
                        help="额外对比 RRF+Qwen3-Reranker 精排（云端或本地，见 reranker.py）")
    parser.add_argument("--rerank-n", type=int, default=20,
                        help="精排时对 RRF 前 N 个候选做重排（默认 20）")
    parser.add_argument("--verbose", action="store_true", help="打印逐条查询命中详情")
    args = parser.parse_args()

    text = DOC_FILE.read_text(encoding="utf-8")
    golden = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))
    queries = golden["queries"]
    top_k = sorted(args.topk)

    print("=" * 72)
    print(f"Golden Test Set：{golden['name']}")
    print(f"文档：{DOC_FILE}（{len(text)} 字符）｜ 查询：{len(queries)} 条")
    print(f"最小重叠比：{args.min_overlap}｜ Top-K：{top_k}｜ 检索：{args.mode}")
    print("=" * 72)

    golden_spans = locate_golden_spans(text, queries)
    model = load_embedder()

    print("\n正在评估各切块策略 × 检索方式…\n")

    header = f"{'检索':<8}{'策略':<18}{'块数':>5}{'平均块长':>9}"
    for k in top_k:
        header += (f"{'Hit@'+str(k):>8}{'MRR@'+str(k):>8}"
                   f"{'Recall@'+str(k):>8}{'Prec@'+str(k):>8}")
    print(header)
    print("-" * len(header))

    for name, fn in CHUNKERS.items():
        chunks = fn(text)
        avg_len = int(np.mean([c.end - c.start for c in chunks])) if chunks else 0

        need_vec = args.mode in ("vector", "both", "all", "rrf")
        need_bm25 = args.mode in ("bm25", "both", "all", "rrf")
        chunk_vecs = embed(model, [c.text for c in chunks]) if need_vec else None
        vec_rf = vector_ranker(model, chunk_vecs) if need_vec else None
        bm_rf = bm25_ranker(chunks) if need_bm25 else None

        rankers = []
        if args.mode in ("vector", "both", "all"):
            rankers.append(("vector", vec_rf))
        if args.mode in ("bm25", "both", "all"):
            rankers.append(("bm25", bm_rf))
        if args.mode in ("rrf", "all"):
            rrf_rf = rrf_ranker(vec_rf, bm_rf, len(chunks), args.rrf_k)
            rankers.append(("rrf", rrf_rf))
            if args.rerank:
                from reranker import create_reranker
                rr = create_reranker()
                if rr is not None:
                    rankers.append(("rrf+rerank",
                                    rerank_ranker(rrf_rf, chunks, rr, args.rerank_n)))

        for rlabel, rank_fn in rankers:
            metrics, nchunks = compute_metrics(
                chunks, queries, golden_spans, rank_fn, top_k,
                args.min_overlap, args.verbose, label=f"{name}/{rlabel}"
            )
            row = f"{rlabel:<8}{name:<18}{nchunks:>5}{avg_len:>9}"
            for k in top_k:
                row += (f"{metrics[k]['hit']:>8.3f}{metrics[k]['mrr']:>8.3f}"
                        f"{metrics[k]['recall']:>8.3f}{metrics[k]['precision']:>8.3f}")
            print(row)
            if args.verbose:
                print()

    print("-" * 72)
    print("注：Hit@K=命中率，MRR@K=平均倒数排名，Recall@K=召回率，Prec@K=精确率（均越大越好）。")
    print("    vector=向量相似度(bge-small-zh-v1.5)，bm25=词法检索(jieba 分词)，rrf=两者 RRF 融合，")
    print("    rrf+rerank=RRF 召回后再用 Qwen3-Reranker 精排（--rerank 时显示）。")
    print("    切块策略在 CHUNKERS 字典中注册，实现新策略后加进去即可横向对比。")


if __name__ == "__main__":
    main()
