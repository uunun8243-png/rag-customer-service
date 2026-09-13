#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
规则清洗器 clean_md.py
====================
把 MarkItDown 从 PDF 转出的"脏" Markdown，用可配置的规则清洗成适合
Chunking / Embedding 的干净 Markdown。

所有规则集中在文件顶部的「规则配置区」，按需增删即可。

用法：
    .venv/bin/python clean_md.py                       # 清洗默认的脏 PDF
    .venv/bin/python clean_md.py 数据清洗/xxx.pdf       # 先转 PDF 再清洗
    .venv/bin/python clean_md.py 数据清洗/xxx.md        # 直接清洗某个 .md
    .venv/bin/python clean_md.py --semantic            # 追加语义级近似去重（嵌入向量）

输出：
    数据清洗/dirty_raw.md    —— 清洗前（markitdown 原始输出）
    数据清洗/clean.md        —— 清洗后
    并打印一份清洗报告（改了什么、删了什么、修了什么）。
"""

import os
import re
import sys
import unicodedata
from pathlib import Path

# ==========================================================================
# 规则配置区（每一条都对应"规则清洗清单"里的一类问题）
# ==========================================================================

# ① 版式噪声：页眉 / 页脚 / 页码 / URL —— 整行匹配即删除
LAYOUT_NOISE_PATTERNS = [
    # 页眉：XX商城售后服务中心 | 客户服务部内部资料
    r"^XX商城售后服务中心\s*[|｜]\s*客户服务部内部资料\s*$",
    # 页脚：www.example-shop.com | 资料编号：AF-2026-08 第 1 页 / 共 5 页
    r"^www\.example-shop\.com\s*[|｜]\s*资料编号\s*[：:]\s*AF-2026-08\s+"
    r"第\s*\d+\s*页\s*/\s*共\s*\d+\s*页\s*$",
    # 裸页眉片段（第十一节里故意展示的噪声）
    r"^XX商城售后服务中心\s*$",
    # 重复的"客户服务部内部资料 客户服务部内部资料"
    r"^客户服务部内部资料(?:\s*客户服务部内部资料)+\s*$",
    # 无意义分隔线
    r"^#{4,}\s*$",
    r"^-{2,}\s*分隔线\s*-{2,}\s*$",
]

# 演示标注（本测试 PDF 自带的"答案提示"，真实资料里不会有，清洗时删除）
REMOVE_DEMO_ANNOTATIONS = True
DEMO_ANNOTATION_PATTERN = re.compile(r"^【(?:异常数据\s*\d+|清洗注意)】")

# ② 错别字 / 近似字符 —— 确定性词典替换（只放"确定性可修"的错误）
TYPO_DICT = {
    "7夭": "7天",        # 夭 → 天
    "退挨货": "退换货",   # 挨 → 换
    "联纱": "联系",       # 纱 → 系
    "中习会": "系统会",   # 中习 → 系统
    "进渡": "进度",       # 渡 → 度
}

# ④ 一句话结尾的标点：不以此结尾的行才考虑与下一行合并（跨页/断行修复）
TERMINAL_PUNCT = set("。！？；：，、）)]】》」』…\"'")

# 中文标题行（一、二、三、… 或 # 开头），不做跨行合并
HEADING_RE = re.compile(r"^(?:#{1,6}\s|[一二三四五六七八九十百]+、)")

# 元信息字段行（备注 / 更新时间 / 免责声明 等），独立成行，不与后文合并
META_FIELD_RE = re.compile(r"^(备注|更新时间|免责声明|日期|版本|作者|编号|文档名称)\s*[：:]")

# ⑨ 语义级近似去重（嵌入向量 + 余弦相似度）
SEMANTIC_MODEL = os.environ.get("RAG_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
SEMANTIC_AUTO_THRESHOLD = 0.90    # ≥ 此值自动删除（高置信度重复）
SEMANTIC_REVIEW_THRESHOLD = 0.85  # 此值 ~ 自动阈值之间：仅报告供复核# ==========================================================================
# 基础工具
# ==========================================================================


def is_table_row(line: str) -> bool:
    return line.strip().startswith("|")


def is_heading(line: str) -> bool:
    return bool(HEADING_RE.match(line.strip()))


def is_list_item(line: str) -> bool:
    """清洗后的列表项统一以 '- ' 开头。"""
    return line.strip().startswith("- ")


def is_special(line: str) -> bool:
    """空行 / 标题 / 表格行 / 列表项 / 元信息字段 —— 这些行不参与跨行合并。"""
    s = line.strip()
    return (
        not s
        or is_heading(s)
        or is_table_row(s)
        or is_list_item(s)
        or bool(META_FIELD_RE.match(s))
    )


def ends_terminal(s: str) -> bool:
    return bool(s) and s[-1] in TERMINAL_PUNCT


# ==========================================================================
# 预处理：去控制字符、删版式噪声
# ==========================================================================


def preprocess(lines, report):
    out = []
    for line in lines:
        # 去 NUL / BOM / 行首行尾空白
        s = line.replace("\x00", "").replace("\ufeff", "").strip()
        if not s:
            out.append("")
            continue
        # 演示标注
        if REMOVE_DEMO_ANNOTATIONS and DEMO_ANNOTATION_PATTERN.match(s):
            report["demo_annotations"].append(s)
            continue
        # 版式噪声
        if any(re.match(p, s) for p in LAYOUT_NOISE_PATTERNS):
            report["layout_noise"].append(s)
            continue
        out.append(s)
    return out


# ==========================================================================
# ④ 列表格式统一
# ==========================================================================

_LIST_MARKER = r"[、.．)）]"


def normalize_list_line(line, known_items):
    """把各种列表写法统一成 '- '。可能返回多行（行内编号列表会拆开）。"""
    s = line.strip()

    # (cid:127) 这类 PDF 解析失败的符号 → 还原为 '- '
    m = re.match(r"^\(cid:\d+\)\s*(.*)$", s)
    if m:
        c = m.group(1).strip()
        known_items.add(c)
        return ["- " + c]

    # 小写 L 假 bullet：l 商品未使用
    m = re.match(r"^l\s+(.*)$", s)
    if m:
        c = m.group(1).strip()
        known_items.add(c)
        return ["- " + c]

    # 标准 markdown bullet
    m = re.match(r"^[-*•]\s+(.*)$", s)
    if m:
        c = m.group(1).strip()
        known_items.add(c)
        return ["- " + c]

    # 编号列表：1、 2. 3） 4．
    if re.match(r"^\d+" + _LIST_MARKER, s):
        parts = re.split(r"(?=\d+" + _LIST_MARKER + r")", s)
        result = []
        for p in parts:
            p = p.strip()
            m = re.match(r"^\d+" + _LIST_MARKER + r"\s*(.*)$", p)
            if m:
                c = m.group(1).strip().rstrip("。；;，,")
                known_items.add(c)
                result.append("- " + c)
        return result if result else [line]

    # 裸列表项（前面没有任何标记，但内容和已知列表项一致）→ 补上 '- '
    if s in known_items:
        return ["- " + s]

    return [line]


# ==========================================================================
# ④ 跨页 / 断行合并
# ==========================================================================


def join_broken_lines(lines, report):
    out = []
    i = 0
    n = len(lines)
    first_content = True
    while i < n:
        line = lines[i]
        if is_special(line):
            out.append(line)
            if line.strip():
                first_content = False
            i += 1
            continue
        if first_content:
            # 文档标题行独立成行，不与副标题/正文合并
            out.append(line)
            first_content = False
            i += 1
            continue
        buf = line
        while not ends_terminal(buf):
            if i + 1 >= n:
                break
            nxt = lines[i + 1]
            if is_special(nxt):
                break
            # 中文 + 英文/数字交界处补一个空格，避免粘连（转换为Markdown → 转换为 Markdown）
            if (
                buf
                and nxt
                and "\u4e00" <= buf[-1] <= "\u9fff"
                and nxt[0].isascii()
                and nxt[0].isalnum()
            ):
                buf += " " + nxt
            else:
                buf += nxt
            i += 1
            report["joined_lines"] += 1
        out.append(buf)
        i += 1
    return out


# ==========================================================================
# ④ 异常空格修复（中文之间、数字与中文之间）
# ==========================================================================


def normalize_spaces(line: str) -> str:
    s = line
    s = re.sub(r"(?<=[\u4e00-\u9fff])[ \t]+(?=[\u4e00-\u9fff])", "", s)  # 退 货 → 退货
    s = re.sub(r"(?<=[\u4e00-\u9fff])[ \t]+(?=\d)", "", s)               # 签收后 7 → 签收后7
    s = re.sub(r"(?<=\d)[ \t]+(?=[\u4e00-\u9fff])", "", s)               # 7 天 → 7天
    return s


# ==========================================================================
# ② 错别字修复
# ==========================================================================


def fix_typos(line: str, report):
    for wrong, right in TYPO_DICT.items():
        if wrong in line:
            line = line.replace(wrong, right)
            report["typos"][wrong] = report["typos"].get(wrong, 0) + line.count(right) - line.count(wrong)
    return line


# ==========================================================================
# ③ 表格清洗
# ==========================================================================


def _norm_cell(cell: str) -> str:
    c = cell.strip()
    c = re.sub(r"(\d+)\s*日", r"\1天", c)   # 7日 → 7天
    c = re.sub(r"(\d+)\s+天", r"\1天", c)   # 7 天 → 7天
    c = re.sub(r"\s+", " ", c)
    return c


def clean_table_block(block, seen_keys, report):
    """清洗一个连续表格块。返回清洗后的行列表（可能为空 = 整个表被删除）。"""
    rows = []
    for line in block:
        parts = line.split("|")
        cells = [p.strip() for p in parts[1:-1]]  # 去首尾空段
        rows.append(cells)

    if len(rows) < 2:
        return block  # 不完整，原样保留

    header = rows[0]
    # 找分隔行
    sep_idx = None
    for idx, r in enumerate(rows[1:], start=1):
        if all(re.match(r"^:?-{3,}:?$", c or "---") for c in r):
            sep_idx = idx
            break
    if sep_idx is None:
        return block

    data = rows[sep_idx + 1:]

    # 1) 删除空列：表头为空 且 所有数据行该列也为空
    ncol = max(len(r) for r in rows)
    keep = []
    for col in range(ncol):
        h = header[col] if col < len(header) else ""
        all_data_empty = all(
            (col >= len(r)) or (not r[col].strip()) for r in data
        )
        if not (not h.strip() and all_data_empty):
            keep.append(col)
    if not keep:
        return []

    def strip_cols(r):
        return [r[c] if c < len(r) else "" for c in keep]

    new_header = strip_cols(header)
    new_data = [strip_cols(r) for r in data]

    # 2) 规范化单元格 + 3) 行内去重 + 全局按首列去重（跨页/跨节重复行）
    header_key = tuple(_norm_cell(c) for c in new_header)
    cleaned_data = []
    seen_in_block = set()
    for r in new_data:
        nr = tuple(_norm_cell(c) for c in r)
        if nr in seen_in_block:
            report["table_dup_rows"] += 1
            continue
        # 跨表重复：首列（商品类型）之前出现过 → 视为跨页重复表头/重复行
        key = nr[0] if nr else ""
        if key and key in seen_keys:
            report["table_cross_rows"] += 1
            continue
        seen_in_block.add(nr)
        if key:
            seen_keys.add(key)
        cleaned_data.append(nr)

    if not cleaned_data:
        return []  # 全是重复行，整表删除（跨页重复表头场景）

    # 跨页重复表头：与上一个表头完全相同 → 只留数据，不再重复输出表头
    if getattr(clean_table_block, "_last_header", None) == header_key:
        report["table_dup_headers"] += 1
        header_lines = []
    else:
        header_lines = [
            "| " + " | ".join(header_key) + " |",
            "| " + " | ".join("---" for _ in header_key) + " |",
        ]
    clean_table_block._last_header = header_key

    out = header_lines + [
        "| " + " | ".join(r) + " |" for r in cleaned_data
    ]
    return out


def clean_tables(lines, report):
    out = []
    i = 0
    seen_keys = set()
    clean_table_block._last_header = None
    while i < len(lines):
        line = lines[i]
        if is_table_row(line):
            block = []
            while i < len(lines) and is_table_row(lines[i]):
                block.append(lines[i])
                i += 1
            out.extend(clean_table_block(block, seen_keys, report))
        else:
            out.append(line)
            i += 1
    return out


# ==========================================================================
# ② 重复行去重（内容级：'- X' 与裸 'X' 视为同一内容）
# ==========================================================================


def dedup_lines(lines, report):
    seen = set()
    out = []
    for line in lines:
        s = line.strip()
        if not s:
            if out and out[-1].strip():
                out.append("")
            continue
        key = re.sub(r"^[-*•]\s+", "", s)
        if key in seen:
            report["dup_lines"].append(s)
            continue
        seen.add(key)
        out.append(line)
    # 收尾去空行
    while out and not out[-1].strip():
        out.pop()
    while out and not out[0].strip():
        out.pop(0)
    return out


def sentence_dedup(lines, report):
    """句子级去重：把段落按句切分，跨全文去除完全重复的句子（保留首次出现）。

    解决"同一句话既嵌在长段落里、又单独成段"这类行级去重抓不到的重复。
    标题 / 表格 / 列表 / 元信息行不参与（它们已是独立单元）。
    """
    seen = set()
    out = []
    for line in lines:
        s = line.strip()
        if not s:
            if out and out[-1].strip():
                out.append("")
            continue
        # 标题 / 表格 / 列表 / 元信息 行不参与句子级去重
        if (
            is_heading(s)
            or is_table_row(s)
            or is_list_item(s)
            or META_FIELD_RE.match(s)
        ):
            out.append(line)
            continue

        # 按句切分（保留句末标点）
        segments = re.split(r"(?<=[。！？；])", s)
        kept = []
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            # 过短的碎片不去重（避免误伤）
            if len(seg) < 8:
                kept.append(seg)
                continue
            if seg in seen:
                report["dup_sentences"].append(seg)
                continue
            seen.add(seg)
            kept.append(seg)

        if kept:
            out.append("".join(kept))
        else:
            # 整段都是重复句子 → 丢弃该段
            report["emptied_paragraphs"] += 1
    return out


# ==========================================================================
# ⑧ 近似重复检测
# ==========================================================================

_PUNCT_TABLE = str.maketrans("", "", "，。！？；：、·…—“”‘’（）()【】[]{}《》〈〉「」『』\"'`~～,;:!?./|\\-_*•")


def normalize_unit(s: str) -> str:
    """归一化：全角→半角、去标点、去空白、转小写。用于判断"准重复"。"""
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_PUNCT_TABLE)
    s = re.sub(r"\s+", "", s)
    return s.lower()


def char_bigrams(s: str) -> set:
    if len(s) < 2:
        return {s} if s else set()
    return {s[i:i + 2] for i in range(len(s) - 1)}


def jaccard(a: str, b: str) -> float:
    A, B = char_bigrams(a), char_bigrams(b)
    if not A or not B:
        return 1.0 if a == b else 0.0
    return len(A & B) / len(A | B)


def near_dup_detect(lines, report, fuzzy_threshold=0.8, min_len=6):
    """近似重复检测：

    1) 归一化后完全相同 → 准重复，自动删除（保留首次出现）；
    2) Jaccard 相似度 ≥ 阈值 → 疑似近似，只报告不删除（供人工/模型复核）。
    """
    # 解析每行 → (kind, segments)
    #   kind: skip=原样保留；list=列表项(单段)；para=段落(按句切)
    parsed = []
    for line in lines:
        s = line.strip()
        if not s:
            parsed.append(["skip", [line]])
            continue
        if is_heading(s) or is_table_row(s) or META_FIELD_RE.match(s):
            parsed.append(["skip", [line]])
            continue
        if is_list_item(s):
            content = re.sub(r"^[-*•]\s+", "", s)
            parsed.append(["list", [content]])
        else:
            segs = [seg.strip() for seg in re.split(r"(?<=[。！？；])", s) if seg.strip()]
            parsed.append(["para", segs])

    # 1) 归一化精确去重（准重复）
    seen = {}  # norm -> 首次出现的原文
    for item in parsed:
        kind, segs = item
        if kind == "skip":
            continue
        kept = []
        for seg in segs:
            norm = normalize_unit(seg)
            if len(norm) < min_len:
                kept.append(seg)
                continue
            if norm in seen:
                report["near_dup_exact"].append((seg, seen[norm]))
                continue
            seen[norm] = seg
            kept.append(seg)
        item[1] = kept

    # 2) 模糊近似：Jaccard 扫描（只报告，不删除）
    units = []
    for kind, segs in parsed:
        if kind == "skip":
            continue
        for seg in segs:
            n = normalize_unit(seg)
            if len(n) >= min_len:
                units.append(seg)
    weak_pairs = []
    for i in range(len(units)):
        for j in range(i + 1, len(units)):
            a, b = units[i], units[j]
            na, nb = normalize_unit(a), normalize_unit(b)
            if na == nb:
                continue
            sim = jaccard(na, nb)
            if sim >= fuzzy_threshold:
                report["near_dup_fuzzy"].append((a, b, round(sim, 3)))
            elif sim >= 0.4:
                weak_pairs.append((a, b, round(sim, 3)))
    weak_pairs.sort(key=lambda x: -x[2])
    report["near_dup_weak"] = weak_pairs[:3]

    # 重建行
    out = []
    for kind, segs in parsed:
        if kind == "skip":
            out.extend(segs)
        elif kind == "list":
            if segs:
                out.append("- " + segs[0])
            else:
                report["near_dup_emptied"] += 1
        elif kind == "para":
            if segs:
                out.append("".join(segs))
            else:
                report["near_dup_emptied"] += 1

    # 压缩连续空行、去首尾空行
    result = []
    for line in out:
        if not line.strip():
            if result and result[-1].strip():
                result.append(line)
            continue
        result.append(line)
    while result and not result[-1].strip():
        result.pop()
    while result and not result[0].strip():
        result.pop(0)
    return result


# ==========================================================================
# 主流程
# ==========================================================================


def semantic_dedup(lines, report, auto_threshold=SEMANTIC_AUTO_THRESHOLD, review_threshold=SEMANTIC_REVIEW_THRESHOLD):
    """⑨ 语义级近似去重：用嵌入向量 + 余弦相似度抓"文本不同但语义相同"的重复。

    默认只自动删除相似度 ≥ auto_threshold 的高置信度重复；
    review_threshold ~ auto_threshold 之间的进入"复核"清单（只报告不删除）。
    """
    try:
        from fastembed import TextEmbedding
        import numpy as np
    except ImportError:
        print("  [跳过] 未安装 fastembed，无法做语义去重。", file=sys.stderr)
        return lines

    # 提取单元（与 near_dup 相同的拆分：段落按句切，列表项整条算）
    parsed = []
    for line in lines:
        s = line.strip()
        if not s:
            parsed.append(["skip", [line]])
            continue
        if is_heading(s) or is_table_row(s) or META_FIELD_RE.match(s):
            parsed.append(["skip", [line]])
            continue
        if is_list_item(s):
            parsed.append(["list", [re.sub(r"^[-*•]\s+", "", s)]])
        else:
            segs = [seg.strip() for seg in re.split(r"(?<=[。！？；])", s) if seg.strip()]
            parsed.append(["para", segs])

    unit_texts, unit_locs = [], []
    for pi, (kind, segs) in enumerate(parsed):
        if kind == "skip":
            continue
        for si, seg in enumerate(segs):
            if len(seg.strip()) >= 6:
                unit_texts.append(seg.strip())
                unit_locs.append((pi, si))

    if len(unit_texts) < 2:
        return lines

    print(f"  [语义去重] 加载嵌入模型 {SEMANTIC_MODEL}，对 {len(unit_texts)} 个单元向量化…")
    model = TextEmbedding(SEMANTIC_MODEL)
    vecs = np.array(list(model.embed(unit_texts)), dtype=np.float32)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    sim = vecs @ vecs.T

    kept_idx = []
    remove_flags = [False] * len(unit_texts)
    for i in range(len(unit_texts)):
        best_k, best_s = -1, 0.0
        for k in kept_idx:
            s = float(sim[i, k])
            if s > best_s:
                best_k, best_s = k, s
        if best_k >= 0 and best_s >= auto_threshold:
            remove_flags[i] = True
            report["semantic_removed"].append((unit_texts[i], unit_texts[best_k], round(best_s, 3)))
        else:
            if best_k >= 0 and best_s >= review_threshold:
                report["semantic_review"].append((unit_texts[i], unit_texts[best_k], round(best_s, 3)))
            kept_idx.append(i)

    report["semantic_review"].sort(key=lambda x: -x[2])

    # 应用删除
    for ui in range(len(unit_texts)):
        if remove_flags[ui]:
            pi, si = unit_locs[ui]
            parsed[pi][1][si] = None

    out = []
    for kind, segs in parsed:
        if kind == "skip":
            out.extend(segs)
        elif kind == "list":
            kept = [seg for seg in segs if seg is not None]
            if kept:
                out.append("- " + kept[0])
            else:
                report["semantic_emptied"] += 1
        elif kind == "para":
            kept = [seg for seg in segs if seg is not None]
            if kept:
                out.append("".join(kept))
            else:
                report["semantic_emptied"] += 1

    # 压缩连续空行、去首尾空行
    result = []
    for line in out:
        if not line.strip():
            if result and result[-1].strip():
                result.append(line)
            continue
        result.append(line)
    while result and not result[-1].strip():
        result.pop()
    while result and not result[0].strip():
        result.pop(0)
    return result


def clean_text(text: str, semantic: bool = False) -> tuple[list[str], dict]:
    report = {
        "layout_noise": [],
        "demo_annotations": [],
        "joined_lines": 0,
        "typos": {},
        "table_dup_rows": 0,
        "table_cross_rows": 0,
        "table_dup_headers": 0,
        "dup_lines": [],
        "dup_sentences": [],
        "emptied_paragraphs": 0,
        "near_dup_exact": [],
        "near_dup_fuzzy": [],
        "near_dup_weak": [],
        "near_dup_emptied": 0,
        "semantic_removed": [],
        "semantic_review": [],
        "semantic_emptied": 0,
    }

    lines = text.splitlines()

    # 0. 预处理：去控制字符 + 删版式噪声
    lines = preprocess(lines, report)

    # 1. 列表格式统一（顺带建立"已知列表项"集合，供裸列表项识别）
    known_items = set()
    normalized = []
    for line in lines:
        if not line.strip():
            normalized.append(line)
            continue
        normalized.extend(normalize_list_line(line, known_items))
    lines = normalized

    # 2. 跨页 / 断行合并
    lines = join_broken_lines(lines, report)

    # 3. 异常空格修复
    lines = [normalize_spaces(l) for l in lines]

    # 4. 错别字修复
    lines = [fix_typos(l, report) for l in lines]

    # 5. 表格清洗（空列 / 重复行 / 期限格式 / 跨页重复表头）
    lines = clean_tables(lines, report)

    # 6. 重复行去重 + 压缩多余空行
    lines = dedup_lines(lines, report)

    # 7. 句子级去重（嵌在段落里的重复句子）
    lines = sentence_dedup(lines, report)

    # 8. 近似重复检测（归一化精确匹配 + Jaccard 相似度扫描）
    lines = near_dup_detect(lines, report)

    # 9. 语义级近似去重（嵌入向量 + 余弦相似度，可选）
    if semantic:
        lines = semantic_dedup(lines, report)

    return lines, report


def convert_pdf_to_md(pdf_path: Path) -> str:
    try:
        from markitdown import MarkItDown
    except ImportError:
        sys.exit("未安装 markitdown，请先执行：uv pip install --python .venv/bin/python 'markitdown[pdf]'")
    md = MarkItDown()
    result = md.convert(str(pdf_path))
    if not result or not getattr(result, "text_content", "").strip():
        sys.exit("PDF 转换失败或结果为空。")
    return result.text_content


def print_report(report, raw_lines, clean_lines, semantic: bool = False):
    def p(title):
        print(f"\n--- {title} ---")

    print("\n" + "=" * 62)
    print("清洗报告")
    print("=" * 62)
    print(f"清洗前行数：{len(raw_lines)}")
    print(f"清洗后行数：{len(clean_lines)}")

    p("① 版式噪声（页眉/页脚/页码/URL/分隔线）已删除")
    for l in report["layout_noise"]:
        print(f"  [删] {l}")

    p("演示标注（【异常数据】/【清洗注意】）已删除")
    for l in report["demo_annotations"]:
        print(f"  [删] {l}")

    p("④ 断行合并")
    print(f"  合并了 {report['joined_lines']} 处断行")

    p("② 错别字修复")
    if report["typos"]:
        for w, cnt in report["typos"].items():
            print(f"  [改] {w} → {TYPO_DICT[w]} （{cnt} 处）")
    else:
        print("  （无）")

    p("③ 表格清洗")
    print(f"  删除表内重复行：{report['table_dup_rows']} 行")
    print(f"  删除跨页/跨节重复行：{report['table_cross_rows']} 行")
    print(f"  删除跨页重复表头：{report['table_dup_headers']} 个")

    p("② 重复行去重")
    print(f"  删除完全重复内容：{len(report['dup_lines'])} 行")
    for l in report["dup_lines"]:
        print(f"  [删] {l[:60]}")

    p("② 句子级去重（嵌在段落里的重复句）")
    print(f"  删除重复句子：{len(report['dup_sentences'])} 句")
    for l in report["dup_sentences"]:
        print(f"  [删] {l[:60]}")
    if report["emptied_paragraphs"]:
        print(f"  因整段重复而丢弃的段落：{report['emptied_paragraphs']} 段")

    p("⑧ 近似重复检测（归一化精确匹配）")
    print(f"  删除准重复：{len(report['near_dup_exact'])} 处")
    for removed, kept in report["near_dup_exact"]:
        print(f"  [删] {removed[:40]}")
        print(f"        ↳ 与已保留的「{kept[:40]}」归一化后相同")
    if report["near_dup_emptied"]:
        print(f"  因准重复整行清空：{report['near_dup_emptied']} 行")

    p("⑧ 近似重复检测（Jaccard 相似度，仅报告不删除）")
    if report["near_dup_fuzzy"]:
        print("  [强疑似，≥0.8]")
        for a, b, sim in report["near_dup_fuzzy"]:
            print(f"    {sim:.3f}")
            print(f"      A: {a[:40]}")
            print(f"      B: {b[:40]}")
    else:
        print("  未发现相似度 ≥0.8 的强近似重复。")
    if report["near_dup_weak"]:
        print("  相似度最高的几对（未达 0.8，供人工/模型复核）：")
        for a, b, sim in report["near_dup_weak"]:
            print(f"    {sim:.3f}  {a[:30]}  <=>  {b[:30]}")

    if semantic:
        p("⑨ 语义级近似去重（嵌入向量 + 余弦相似度）")
        if report["semantic_removed"]:
            print(f"  自动删除高置信度重复（≥{SEMANTIC_AUTO_THRESHOLD}）：{len(report['semantic_removed'])} 处")
            for removed, kept, s in report["semantic_removed"]:
                print(f"  [删] {removed[:40]}")
                print(f"        ↳ 与「{kept[:40]}」相似度 {s:.3f}")
        else:
            print("  未发现需要自动删除的高置信度重复。")
        if report["semantic_review"]:
            print(f"  需人工复核（{SEMANTIC_REVIEW_THRESHOLD}~{SEMANTIC_AUTO_THRESHOLD}）：")
            for a, b, s in report["semantic_review"]:
                print(f"    {s:.3f}  {a[:30]}  <=>  {b[:30]}")
        if report["semantic_emptied"]:
            print(f"  因语义重复整行清空：{report['semantic_emptied']} 行")

    print("\n" + "=" * 62)


def main():
    argv = sys.argv[1:]
    semantic = "--semantic" in argv
    positional = [a for a in argv if not a.startswith("--")]
    arg = positional[0] if positional else "数据清洗/dirty_rag_demo.pdf"
    src = Path(arg)
    if not src.exists():
        sys.exit(f"文件不存在：{src}")

    data_dir = Path(__file__).resolve().parent / "数据清洗"
    data_dir.mkdir(exist_ok=True)
    raw_path = data_dir / "dirty_raw.md"
    clean_path = data_dir / "clean.md"

    # 1. 转 PDF（如果输入是 PDF）
    if src.suffix.lower() == ".pdf":
        print(f"正在用 MarkItDown 转换：{src}")
        text = convert_pdf_to_md(src)
        raw_path.write_text(text, encoding="utf-8")
        print(f"清洗前已保存：{raw_path}")
    else:
        text = src.read_text(encoding="utf-8")
        raw_path.write_text(text, encoding="utf-8")

    raw_lines = text.splitlines()

    # 2. 规则清洗（+ 可选语义去重）
    clean_lines, report = clean_text(text, semantic=semantic)

    clean_path.write_text("\n".join(clean_lines) + "\n", encoding="utf-8")
    print(f"清洗后已保存：{clean_path}")

    print_report(report, raw_lines, clean_lines, semantic=semantic)


if __name__ == "__main__":
    main()
