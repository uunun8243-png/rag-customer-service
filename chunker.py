#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chunker.py — 结构切块器（单一数据源）

选定策略：结构切块 + 目标 400 字符 + 20% 结构块级 overlap（structure_merge）。
在 golden test set 上实测最优（Hit@1/MRR@1 = 0.650，Hit@3 = 1.000，MRR@3 = 0.808）。

eval_retrieval.py（评估）与 rag_tool.py（建索引）共用本模块，保证
「评估的切块」与「上线的切块」是同一份代码。

用法：
    from chunker import structure_merge, Chunk
    chunks = structure_merge(text)                       # 默认 400 / 0.20
    chunks = structure_merge(text, target=500, overlap_ratio=0.15)
    # -> list[Chunk]，每个 Chunk 有 .text / .start / .end
"""

import re
from dataclasses import dataclass

# 选定策略的默认参数（对应 golden test set 里实测最优的 merge_400_o20）
DEFAULT_TARGET = 400
DEFAULT_OVERLAP_RATIO = 0.20


@dataclass
class Chunk:
    text: str
    start: int   # 在全文中的起始字符下标
    end: int     # 在全文中的结束字符下标（不含）


# --------------------------------------------------------------------------
# 基础工具：行区间 / 块构造
# --------------------------------------------------------------------------
HEADING_RE = re.compile(r"^(?:#{1,6}\s|[一二三四五六七八九十百]+、)")


def line_spans(text: str) -> list[tuple[int, int]]:
    """把全文按行切分，返回每行（不含换行符）的字符区间。"""
    spans = []
    i, n = 0, len(text)
    while i < n:
        j = text.find("\n", i)
        if j == -1:
            spans.append((i, n))
            break
        spans.append((i, j))
        i = j + 1
    return spans


def make_chunk(text: str, spans: list[tuple[int, int]]) -> Chunk:
    """把若干连续行的区间合并成一个 chunk（用换行符连接）。"""
    s = spans[0][0]
    e = spans[-1][1]
    return Chunk(text[s:e], s, e)


def split_long(chunk: Chunk, text: str, max_chars: int) -> list[Chunk]:
    """把超过 max_chars 的块按固定窗口切开（尽量在句末/换行处断）。"""
    if chunk.end - chunk.start <= max_chars:
        return [chunk]
    out = []
    start = chunk.start
    end = chunk.end
    while start < end:
        stop = min(start + max_chars, end)
        if stop < end:
            window = text[start:stop]
            for m in re.finditer(r"[。！？；\n]", window):
                stop = start + m.end()
        out.append(Chunk(text[start:stop], start, stop))
        start = stop
    return out


# --------------------------------------------------------------------------
# 切块策略（每个函数：text -> list[Chunk]）
# --------------------------------------------------------------------------
def structure(text: str, max_chars: int) -> list[Chunk]:
    """结构切块：按标题行/空行分块（标题与紧随其后的内容归入同一块），
    超过 max_chars 的块再切一刀（尽量在句末/换行处断）。"""
    chunks, cur = [], []
    for s, e in line_spans(text):
        line = text[s:e]
        is_blank = not line.strip()
        is_heading = bool(HEADING_RE.match(line.strip()))
        if (is_heading or is_blank) and cur:
            chunks.append(make_chunk(text, cur))
            cur = []
        if not is_blank:
            cur.append((s, e))
    if cur:
        chunks.append(make_chunk(text, cur))

    flat = []
    for c in chunks:
        flat.extend(split_long(c, text, max_chars))
    return flat


def structure_merge(text: str, target: int = DEFAULT_TARGET,
                    overlap_ratio: float = DEFAULT_OVERLAP_RATIO) -> list[Chunk]:
    """★ 选定策略：结构切块（合并版）。

    先按结构边界（标题/空行）切原子块，再把相邻小块贪心合并到接近 target 字符
    （不跨过 target）。overlap_ratio 是相邻块的重叠比例：下一块的起点从当前块
    末尾回退 overlap 字符、并对齐到最近的原子块起点，使相邻块内容重叠，
    避免答案恰好被切在块边界上而丢失。
    """
    atoms, cur = [], []
    for s, e in line_spans(text):
        line = text[s:e]
        is_blank = not line.strip()
        is_heading = bool(HEADING_RE.match(line.strip()))
        if (is_heading or is_blank) and cur:
            atoms.append(make_chunk(text, cur))
            cur = []
        if not is_blank:
            cur.append((s, e))
    if cur:
        atoms.append(make_chunk(text, cur))

    overlap = int(target * overlap_ratio)
    chunks = []
    i, n = 0, len(atoms)
    while i < n:
        cur_start = atoms[i].start
        j = i
        while j < n and atoms[j].end - cur_start <= target:
            j += 1
        if j == i:            # 单个原子块就超 target：整块单独成 chunk
            j = i + 1
        chunk = make_chunk(text, [(a.start, a.end) for a in atoms[i:j]])
        chunks.append(chunk)

        if j >= n:
            break
        if overlap == 0:
            i = j
        else:
            back_to = chunk.end - overlap
            k = j - 1
            while k > i and atoms[k].start > back_to:
                k -= 1
            i = max(k, i + 1)   # 至少前进一个原子块，避免死循环
    return chunks


def structure_window(text: str, target: int, overlap_ratio: float) -> list[Chunk]:
    """对照：结构感知字符窗口 + 精确 overlap。
    窗口终点尽量对齐句末/换行，下一块起点精确回退 overlap 字符（允许切在句子中间）。"""
    overlap = int(target * overlap_ratio)
    chunks = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + target, n)
        if end < n:
            best = None
            for m in re.finditer(r"[。！？；\n]", text[start:end]):
                best = m.end()
            if best and best >= target // 2:
                end = start + best
        chunks.append(Chunk(text[start:end], start, end))
        if end >= n:
            break
        start = end - overlap
    return chunks


def full_doc(text: str) -> list[Chunk]:
    """对照：不切块，整篇文档作为一个 chunk（评估时的天花板参照）。"""
    return [Chunk(text, 0, len(text))]
