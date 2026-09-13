#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 问答工具（查询意图识别 + 混合检索：向量 + BM25 → RRF 融合，向量存 Qdrant 服务端）

把你目录下的 Markdown 笔记做成一个能回答问题的工具。

用法：
    python rag_tool.py                  # 进入交互式问答
    python rag_tool.py "普通商品退货期限多久？"   # 单次提问
    python rag_tool.py --rebuild        # 强制重建索引（写入 Qdrant 服务端）
    python rag_tool.py --intent "2024年上海人才补贴怎么申请？"   # 只看意图分析

原理：
    1. 查询意图识别（query_intent.py）：判断问题类型（定义/事实/流程/比较/总结/政策/排查），
       抽取实体与槽位（地区/时间/政策类型等），据此决定检索策略
       （直接 / 改写 / 扩展 / HyDE / 分解 / 澄清）
    2. 读取知识库目录下所有 .md，用结构切块（500 字符 + 20% overlap）
    3. 向量检索：本地嵌入 bge-small-zh-v1.5 余弦相似度，向量存入 Qdrant 服务端
    4. BM25 检索：jieba 分词
    5. RRF 融合两者排序，按策略把多路结果合并，召回一个宽窗口
    6. 精排（rerank）：用 Qwen3-Reranker（默认云端 API，见 reranker.py）重排，取 top-K
    7. 把 top-K 原文拼成上下文，交给 DeepSeek 大模型综合回答（带出处）
    8. 多轮对话：上下文补全+意图识别（query_intent.complete_and_analyze 合并一次调用）+ 近期对话窗口 +
       早期对话滚动摘要（memory.py），生成回答时带入历史
    9. 答案缓存（answer_cache.py）：自包含问题命中直接返回，跳过全链路（降成本/延迟）

索引与检索统一在 rag_index.py（Qdrant 服务端为主，内存 numpy 兜底）：
    - 设置 RAG_QDRANT_URL=http://127.0.0.1:6333 时走 Qdrant 服务端（docker 部署）
    - 未设置时回退到进程内内存向量（单进程开发/CI 用）

检索策略依据：eval_retrieval.py 在 golden test set 上实测，
merge_500_o20 + RRF 取得 Hit@1=0.750 / MRR@3=0.875 / Recall@5=1.000（全表最优）。

依赖（API Key）：
    - 大模型：DeepSeek（OpenAI 兼容接口）
    - 嵌入：本地模型 bge-small-zh-v1.5，首次运行会自动下载一次权重
    - 向量库：Qdrant 服务端（docker，见 deploy/docker-compose.yml；未配置则用内存兜底）
    - 精排：Qwen3-Reranker 云端 API（默认硅基流动 siliconflow，可切 dashscope/local）
"""

import argparse
import os
import sys

from llm import LLMError, call_llm, call_llm_stream, get_api_key
from memory import format_dialogue, maintain_memory, recent_window
from query_intent import (analyze_intent, describe_intent, has_reference,
                          retrieve_with_intent)
from rag_index import (INDEX_FINGERPRINT, TOP_K, InMemoryIndex, QdrantIndex,
                       build_index, load_index, retrieve)
from reranker import CloudReranker, create_reranker
from answer_cache import create_answer_cache

# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------
# 索引/切块/检索参数统一在 rag_index.py（Qdrant 服务端为主，内存兜底），此处只保留生成侧配置。

# 精排（rerank）开关：环境变量 RAG_ENABLE_RERANK=0 或 --no-rerank 可关闭
RERANK_ENABLED = os.environ.get("RAG_ENABLE_RERANK", "1").strip().lower() not in ("0", "false", "no", "off")

SYSTEM_PROMPT = (
    "你是一位专业又有耐心的客服助手，正在用中文回答客户的问题。\n"
    "要求：\n"
    "1. 语气自然、口语、有温度，像真人客服一样直接给出答案或建议，先结论后简要说明，"
    "不要像念条款一样逐条罗列；\n"
    "2. 严格只依据提供的资料回答，绝不编造资料里没有的内容；资料没有明确答案时，"
    "用自然的话告知并给出建议（例如「这个情况暂时没有查到明确规定，建议以订单页面和商家说明为准」）；\n"
    "3. 回答中不要出现「笔记」「资料」「根据上下文」「[编号]」这类内部标记或元语言，直接自然地陈述内容；\n"
    "4. 若有对话历史，结合它理解客户的指代和省略，但只回答当前问题。"
)


# --------------------------------------------------------------------------
# 大模型生成
# --------------------------------------------------------------------------
def build_context(hits: list[dict]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        fname = h["chunk"].get("file_name", "")
        parts.append(f"[{i}]（{fname}）\n{h['chunk']['text']}")
    return "\n\n".join(parts)


def build_messages(question: str, hits: list[dict],
                   conversation: list[dict] | None = None,
                   summary: str = "") -> list[dict]:
    """组装最后一步「生成回答」的 messages（system + user，含对话上下文）。"""
    ctx_parts = []
    if summary:
        ctx_parts.append(f"【之前对话摘要】\n{summary}")
    recent = recent_window(conversation or [])
    if recent:
        ctx_parts.append("【近期对话】\n" + format_dialogue(recent))

    user = (f"【已知信息】\n{build_context(hits)}\n\n"
            f"【客户问题】\n{question}")
    if ctx_parts:
        user = "【对话上下文】\n" + "\n\n".join(ctx_parts) + "\n\n" + user
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def generate(question: str, hits: list[dict], api_key: str,
             conversation: list[dict] | None = None, summary: str = "",
             usage: dict | None = None, on_delta=None) -> str:
    messages = build_messages(question, hits, conversation, summary)
    if on_delta is not None:
        parts = []
        try:
            for delta in call_llm_stream(messages, api_key, usage=usage):
                parts.append(delta)
                on_delta(delta)
            if not "".join(parts).strip():
                # 推理模型可能把额度全花在“思考”上、正文为空：降级一次性调用
                # （call_llm 内部有同样的空回答兜底重试）
                return call_llm(messages, api_key, usage=usage)
            return "".join(parts)
        except LLMError:
            if parts:
                return "".join(parts)   # 已流式输出一部分，保留已生成内容
            return call_llm(messages, api_key, usage=usage)  # 流式失败 → 降级一次性
    return call_llm(messages, api_key, usage=usage)


# --------------------------------------------------------------------------
# 问答
# --------------------------------------------------------------------------
def _print_sources(hits: list[dict]) -> None:
    """打印检索出处（present 与缓存命中共用）。"""
    has_rerank = any(h.get("rerank_score") is not None for h in hits)
    label = ("精排分 / RRF / 向量 / BM25 / 来源" if has_rerank
             else "RRF 融合分 / 向量相似度 / BM25 分 / 策略来源")
    print(f"\n参考出处（{label}）：")
    for i, h in enumerate(hits, 1):
        fname = h["chunk"].get("file_name", "未知文件")
        snippet = h["chunk"]["text"].strip().replace("\n", " ")[:80]
        origins = ",".join(h.get("origins", [])) or "-"
        rr = h.get("rerank_score")
        rr_s = f"{rr:.3f}" if rr is not None else "-"
        print(f"  [{i}] {fname}（{rr_s} / {h.get('rrf', 0):.3f} / "
              f"{h.get('vec_sim', 0):.3f} / {h.get('bm25', 0):.3f} / {origins}）\n"
              f"      {snippet}…")


def present(question: str, hits: list[dict], state: dict, stream: bool = True) -> str:
    """把检索结果交给大模型生成回答并打印出处，返回回答文本。"""
    if not hits:
        print("没有检索到相关内容。")
        return ""
    usage = state.setdefault("usage", {})
    print("\n" + "=" * 60)
    print("回答：")
    try:
        if stream:
            answer = generate(question, hits, state["api_key"],
                              conversation=state.get("conversation"),
                              summary=state.get("summary", ""),
                              usage=usage,
                              on_delta=lambda d: print(d, end="", flush=True))
            print()
        else:
            answer = generate(question, hits, state["api_key"],
                              conversation=state.get("conversation"),
                              summary=state.get("summary", ""),
                              usage=usage)
            print(answer)
    except LLMError as e:
        raise SystemExit(str(e))
    print("=" * 60)

    _print_sources(hits)
    if usage.get("total_tokens"):
        print(f"[累计 tokens] prompt={usage.get('prompt_tokens', 0)} "
              f"completion={usage.get('completion_tokens', 0)} "
              f"total={usage.get('total_tokens', 0)}")
    print()
    return answer


def ask(question: str, state: dict, base_top_k: int = TOP_K) -> dict:
    """意图识别 →（澄清 | 检索+回答）。返回结果 dict，便于交互循环处理澄清。"""
    cache = state.get("answer_cache")

    # 答案缓存：自包含问题命中则直接返回，跳过意图/检索/精排/生成
    if cache is not None and not has_reference(question):
        cached = cache.get(question)
        if cached is not None:
            print("⚡ 缓存命中（跳过检索与生成）")
            print("\n" + "=" * 60)
            print("回答：")
            print(cached.get("answer", ""))
            print("=" * 60)
            _print_sources(cached.get("hits", []))
            return {"action": "answer", "answer": cached.get("answer", ""),
                    "cached": True}

    print("思考中…")
    conversation = state.get("conversation", [])
    result = retrieve_with_intent(
        question, state,
        retriever=state["retriever"],
        base_top_k=base_top_k,
        conversation=conversation,
    )
    intent = result["intent"]
    print("\n【意图】" + describe_intent(intent).replace("\n", "｜"))
    completed = result.get("completed_query", "")
    if completed and completed != question:
        print("【上下文补全】" + completed)

    if result["action"] == "clarify":
        print("\n" + "=" * 60)
        print("这个问题缺少关键信息，先向你确认：")
        for i, q in enumerate(result["questions"], 1):
            print(f"  {i}. {q}")
        print("=" * 60 + "\n")
        return result

    answer = present(question, result["hits"], state)
    result["answer"] = answer
    # 写回答案缓存（key 用补全后的完整查询，便于后续直接命中）
    if cache is not None and answer:
        cache.set(completed or question, answer, result["hits"])
    return result


def answer_direct(question: str, state: dict, top_k: int = TOP_K) -> str:
    """用户不愿补充澄清信息时，绕过意图识别直接检索。返回回答文本。

    先召回一个宽窗口，再精排截断到 top_k（与意图路径一致）。
    """
    fetch_k = max(top_k * 2, top_k + 4)
    hits = state["retriever"](question, fetch_k)
    reranker = state.get("reranker")
    if reranker is not None and len(hits) > top_k:
        try:
            hits = reranker.rerank(question, hits, top_k=top_k)
        except Exception as e:
            print(f"  [精排失败，回退 RRF 排序] {e}")
            hits = hits[:top_k]
    else:
        hits = hits[:top_k]
    return present(question, hits, state)


def interactive(state: dict, base_top_k: int = TOP_K) -> None:
    rr = state.get("reranker")
    backend = "云端" if isinstance(rr, CloudReranker) else ("本地" if rr is not None else "-")
    rerank_on = "开" if rr is not None else "关"
    print("\n欢迎使用笔记问答工具（意图识别 + 上下文补全 + 混合检索 + 精排）！")
    print(f"索引：{state.get('index_backend', '-')}；"
          f"精排（Qwen3-Reranker · {backend}）：{rerank_on}；"
          f"输入问题即可，输入 exit / quit / q 退出。\n")
    conversation = state.setdefault("conversation", [])
    while True:
        try:
            q = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            print("再见！")
            break

        result = ask(q, state, base_top_k)
        if result.get("action") == "clarify":
            detail = input("补充说明（直接回车则按原问题直接检索）：").strip()
            if detail:
                q = f"{q}（补充：{detail}）"
                result = ask(q, state, base_top_k)
            else:
                ans = answer_direct(q, state)
                result = {"action": "answer", "answer": ans}
        if result.get("action") == "answer":
            conversation.append({"role": "user", "content": q})
            conversation.append({"role": "assistant", "content": result.get("answer", "")})
            maintain_memory(state, state["api_key"])


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="意图识别 + 混合检索（向量 + BM25 → RRF，向量存 Qdrant 服务端）笔记问答工具")
    parser.add_argument("question", nargs="?", help="直接提问（不加则进入交互模式）")
    parser.add_argument("--rebuild", action="store_true", help="强制重建索引（写入 Qdrant 服务端）")
    parser.add_argument("--no-rerank", action="store_true", help="关闭精排（只用 RRF 混合检索）")
    parser.add_argument("--intent", action="store_true",
                        help="只做意图分析（类型/实体槽位/策略），不检索不回答")
    args = parser.parse_args()

    api_key = get_api_key()
    if not api_key:
        sys.exit("找不到 DeepSeek API Key。请设置环境变量 DEEPSEEK_API_KEY，"
                 "或把 key 放到 ~/.pi/agent/auth.json。")

    if args.intent:
        if not args.question:
            sys.exit("--intent 需要同时给出一个问题，例如："
                     'python rag_tool.py --intent "2024年上海人才补贴怎么申请？"')
        intent = analyze_intent(args.question, api_key)
        print(describe_intent(intent))
        if intent.missing_info:
            print("待澄清：" + "；".join(intent.missing_info))
        return

    index = build_index() if args.rebuild else load_index()
    backend = "Qdrant 服务端" if isinstance(index, QdrantIndex) else "内存 numpy"
    print(f"索引后端：{backend}（{len(index.chunks)} 块）")

    use_rerank = RERANK_ENABLED and not args.no_rerank
    state = {
        "chunks": index.chunks,
        "index": index,
        "index_backend": backend,
        "api_key": api_key,
        "retriever": lambda q, k=TOP_K: retrieve(index, q, top_k=k),
        "reranker": create_reranker() if use_rerank else None,   # 精排（云端优先，懒加载）
        "conversation": [],   # 完整对话（role/content），内存保留
        "summary": "",        # 早期对话滚动摘要（memory.py 维护）
        "folded": 0,          # 已折叠进摘要的消息条数
        "usage": {},          # 累计 token 用量（prompt/completion/total）
        "answer_cache": create_answer_cache(INDEX_FINGERPRINT),   # 答案缓存（Redis/内存）
    }

    if args.question:
        result = ask(args.question, state)
        if result["action"] == "clarify":
            print("（单次模式无法交互澄清；进入交互模式后可继续补充。）")
    else:
        interactive(state)


if __name__ == "__main__":
    main()
