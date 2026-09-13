#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
memory.py — 长对话记忆管理：滚动窗口 + 早期对话滚动摘要。

对话状态存在 state["conversation"]（role/content 列表，只保留最近若干条）。
喂给 LLM 时只用两样：
  - 近期对话窗口 recent_window()（最近若干条消息 + 字符预算上限）
  - 早期对话滚动摘要 state["summary"]（由 maintain_memory() 增量维护）

maintain_memory() 采用增量折叠 + 物理裁剪：当对话超过阈值时，把「滑出窗口」的
消息并入摘要，然后真正从 state["conversation"] 里删掉这些消息，只保留最近 keep 条。
这样不仅喂给 LLM 的上下文有界，session 落盘（Redis/内存）的体积也有界，
长对话不丢关键实体与约束，也不会随对话无限膨胀。
"""

from llm import LLMError, call_llm

# 生成上下文里保留的最近消息条数与字符预算
DEFAULT_MAX_MESSAGES = 8
DEFAULT_MAX_CHARS = 2400
# 对话超过该条数时，把超出窗口的部分折叠进摘要
DEFAULT_SUMMARY_THRESHOLD = 12


def recent_window(conversation: list[dict],
                  max_messages: int = DEFAULT_MAX_MESSAGES,
                  max_chars: int = DEFAULT_MAX_CHARS) -> list[dict]:
    """取最近若干条消息，总字符数不超预算（从最旧的开始丢弃）。"""
    if not conversation:
        return []
    msgs = list(conversation[-max_messages:])
    total = sum(len(m.get("content", "")) for m in msgs)
    dropped = 0
    while total > max_chars and len(msgs) > 1:
        total -= len(msgs[0].get("content", ""))
        msgs = msgs[1:]
        dropped += 1
    if dropped:
        # 近期窗口因字符预算被裁剪 → 「上下文超长被截断」的触发点。
        # 懒加载 rag_metrics：CLI（rag_tool.py）未初始化 MeterProvider 时是无副作用 no-op。
        from rag_metrics import rag_metrics
        rag_metrics().context_truncated.add(dropped)
    return msgs


def format_dialogue(messages: list[dict]) -> str:
    """把 role/content 消息列表格式化成可读对话文本。"""
    lines = []
    for m in messages:
        role = "用户" if m.get("role") == "user" else "助手"
        lines.append(f"{role}：{m.get('content', '')}")
    return "\n".join(lines)


def summarize_turns(messages: list[dict], api_key: str) -> str:
    """把一段对话压缩成简短摘要。失败时退化为截断原文。"""
    if not messages:
        return ""
    text = format_dialogue(messages)
    try:
        return call_llm([
            {"role": "system",
             "content": "把下面这段对话压缩成一段简短的中文摘要，保留关键实体、"
                        "重要结论和尚未解决的约束条件。只输出摘要，不要任何前缀。"},
            {"role": "user", "content": text},
        ], api_key, temperature=0.2, max_tokens=400).strip()
    except LLMError:
        return text[-600:]


def maintain_memory(state: dict, api_key: str,
                    keep: int = DEFAULT_MAX_MESSAGES,
                    threshold: int = DEFAULT_SUMMARY_THRESHOLD) -> None:
    """增量维护滚动摘要，并物理裁剪 state["conversation"]。

    当对话超过 threshold 条时，把「滑出 keep 窗口」的旧消息折叠进 state["summary"]，
    然后从 state["conversation"] 里真正删除这些消息，只保留最近 keep 条。
    新摘要 = 摘要(旧摘要 + 新溢出消息)，信息不丢、摘要长度有界；
    conversation 也随之上界，落盘体积不再随对话无限增长。
    """
    conv = state.get("conversation", [])
    if len(conv) <= threshold:
        return
    drop_until = len(conv) - keep
    if drop_until <= 0:
        return
    overflow = conv[:drop_until]

    prev = state.get("summary", "")
    combined = (f"【之前对话摘要】\n{prev}\n\n" if prev else "") \
        + "【新增对话】\n" + format_dialogue(overflow)
    try:
        summary = call_llm([
            {"role": "system",
             "content": "把下面的新增对话合并进已有摘要，输出一段更完整但仍简短的中文摘要，"
                        "保留关键实体、重要结论和尚未解决的约束。只输出摘要。"},
            {"role": "user", "content": combined},
        ], api_key, temperature=0.2, max_tokens=400).strip()
        state["summary"] = summary
    except LLMError as e:
        # 摘要失败也不阻塞裁剪：用截断原文兜底，保证 conversation 始终有界。
        fallback = (prev + "\n" + format_dialogue(overflow)) if prev else format_dialogue(overflow)
        state["summary"] = fallback[-1200:]
        print(f"  [记忆压缩失败，已按截断兜底并裁剪] {e}")
    else:
        print(f"  [记忆压缩] 已折叠 {drop_until} 条消息 → 摘要 {len(state['summary'])} 字，"
              f"保留最近 {keep} 条")

    # 物理裁剪：删除已折叠进摘要的前缀，只保留最近 keep 条。
    state["conversation"] = conv[drop_until:]
    state["folded"] = int(state.get("folded", 0) or 0) + drop_until
