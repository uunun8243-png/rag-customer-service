#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
query_intent.py — 查询阶段意图识别（问题类型 / 实体槽位 / 检索策略）

三个阶段：
  1. 判断问题类型：definition（定义）/ fact（事实）/ process（流程）/
     comparison（比较）/ summary（总结）/ policy（政策）/ troubleshooting（排查）
     不同类型适合不同检索方式：
       事实/定义 → 精确检索
       比较      → 拆多个子问题（decompose）
       总结      → 扩大召回范围（recall_scale 调大 top-k）
       政策      → 结合地区/时间/政策类型等槽位做过滤
  2. 抽取实体与槽位：实体=用户真正关心的对象；槽位=检索约束
     （region / time / policy_type / product_category …），可直接用于过滤。
  3. 决定检索策略：
        direct    直接检索（问题已明确）
        rewrite   查询改写（口语化 / 指代不明 / 依赖上下文）
        expand    查询扩展（用户用词与文档术语可能不一致）
        hyde      生成假设性文档辅助召回（问题短而抽象）
        decompose 查询分解（多个子目标 / 比较多个对象）
        clarify   澄清（缺少关键信息，先问用户，而不是强行检索）

设计：三步用一次 LLM 结构化输出完成（analyze_intent），策略执行由代码确定性完成
（execute_strategies）。本模块不 import rag_tool，通过外部传入的 retriever 回调执行检索，
与索引/检索实现解耦；LLM 失败时降级为「fact + direct」直接检索。

多轮对话：complete_and_analyze() 把「上下文补全 + 意图识别」合并为一次 LLM 调用
（无历史时直接 analyze_intent）；retrieve_with_intent 接受 conversation（role/content 列表）
理解指代。has_reference() 规则判断问题是否依赖上文，供答案缓存层决策。
独立工具 complete_query() 保留（仅补全、不分析意图）。

独立用法（只做意图分析、不检索，便于调试）：
    .venv/bin/python query_intent.py "2024年上海人才补贴怎么申请？"
    .venv/bin/python query_intent.py --demo
"""

import argparse
import logging
import os
import time
from dataclasses import dataclass, field

from llm import LLMError, call_llm, chat_json, get_api_key
from rag_spans import stage
from rag_metrics import rag_metrics

# 默认 top-k（运行时由调用方传 base_top_k 覆盖，与 rag_tool 的 TOP_K 保持一致）
DEFAULT_TOP_K = 6

# 查询扩展开关：rewrite 的额外 LLM 调用、expand、hyde 默认关闭（省成本省延迟）。
# 设 RAG_ENABLE_EXPANSION=1 恢复完整扩展能力。
ENABLE_EXPANSION = os.environ.get("RAG_ENABLE_EXPANSION", "0").strip().lower() in (
    "1", "true", "yes", "on")

# 上下文指代词表（供 has_reference 规则判断，答案缓存层用它决定是否查缓存）
_REFERENCE_TOKENS = (
    "这个", "那个", "这些", "那些", "这里", "那里", "这儿", "那儿",
    "它", "他", "她", "他们", "她们", "它们", "其",
    "上面", "前面", "之前", "刚才", "上述", "以上", "后者", "前者",
    "另外", "其他", "其它",
)

# 规则优先意图：高置信度模式命中后直接判定、跳过 LLM（省成本省延迟）。
# 只覆盖「直接检索就能答」的简单查询，复杂/多目标问题一律走 LLM（见 _RULE_COMPLEX）。
_RULE_DEFINITION = ("是什么", "什么是", "定义", "什么意思", "指的是")
_RULE_FAST_FACT = (
    "多久", "几天", "多长时间", "几点", "什么时候", "什么时间", "期限", "有效期",
    "在哪里", "哪里", "地址", "入口", "电话", "渠道", "怎么联系", "联系方式",
    "是否", "能不能", "可以吗", "需要吗", "要不要", "哪些",
    "要求", "条件", "需要提供", "需要满足", "需要提交",
)
_RULE_TROUBLESHOOTING = ("怎么办",)
# 必须走 LLM 的复杂/多目标信号（比较/总结/政策/复合问句）
_RULE_COMPLEX = (
    "区别", "不同", "对比", "哪个", "哪种", "比较", "一样", "相同",
    "总结", "概括", "概述", "归纳", "都讲", "全部",
    "补贴", "政策",
    "并且", "以及", "同时", "还有",
)

# 问题类型 → 默认策略（LLM 未给策略时的兜底）
DEFAULT_STRATEGY_BY_TYPE = {
    "definition": ["direct"],
    "fact": ["direct"],
    "process": ["direct"],
    "comparison": ["decompose"],
    "summary": ["direct"],
    "policy": ["direct"],
    "troubleshooting": ["rewrite", "expand"],
    "other": ["direct"],
}

# 问题类型 → 默认召回规模倍数（总结类需要更大上下文，政策类需要更宽候选再过滤）
RECALL_SCALE_BY_TYPE = {
    "definition": 1.0,
    "fact": 1.0,
    "process": 1.2,
    "comparison": 1.0,
    "summary": 2.5,
    "policy": 1.5,
    "troubleshooting": 1.2,
    "other": 1.0,
}

# 政策类过滤时视为「必须命中」的槽位类型
CRITICAL_SLOT_TYPES = {"region", "time", "policy_type", "product_category"}

STRATEGIES = {"direct", "rewrite", "expand", "hyde", "decompose", "clarify"}
QUESTION_TYPES = {
    "definition", "fact", "process", "comparison",
    "summary", "policy", "troubleshooting", "other",
}

TYPE_ZH = {
    "definition": "定义", "fact": "事实", "process": "流程",
    "comparison": "比较", "summary": "总结", "policy": "政策",
    "troubleshooting": "排查", "other": "其它",
}
STRATEGY_ZH = {
    "direct": "直接检索", "rewrite": "查询改写", "expand": "查询扩展",
    "hyde": "HyDE", "decompose": "查询分解", "clarify": "澄清",
}


@dataclass
class Intent:
    question_type: str = "fact"
    entities: list = field(default_factory=list)      # [{type, value}]
    slots: list = field(default_factory=list)         # [{type, value}]
    needs_clarification: bool = False
    missing_info: list = field(default_factory=list)
    strategies: list = field(default_factory=list)
    rewritten_query: str = ""
    sub_queries: list = field(default_factory=list)
    recall_scale: float = 1.0
    rule_based: bool = False      # True=规则判定，未调 LLM
    raw: dict = field(default_factory=dict)


INTENT_SYSTEM_PROMPT = (
    "你是检索意图分析器。分析用户问题，只输出一个 JSON 对象，"
    "不要输出任何解释、Markdown 代码块或多余文字。\n"
    "\n"
    "背景：本系统在已有知识库上做检索问答，知识库内容已就绪，用户的问题默认针对知识库内容。"
    "不要因为用户没有粘贴文档、没有注明地区/平台就要求澄清。\n"
    "\n"
    "JSON 字段（键名固定，不得增删）：\n"
    "{\n"
    '  "question_type": "definition|fact|process|comparison|summary|policy|troubleshooting|other",\n'
    '  "entities": [{"type": "topic|product|organization|person|policy_name|...", "value": "实体"}],\n'
    '  "slots": [{"type": "region|time|policy_type|product_category|quantity|...", "value": "约束值"}],\n'
    '  "needs_clarification": true 或 false,\n'
    '  "missing_info": ["缺少的关键信息，以问句形式写给用户"],\n'
    '  "strategies": ["direct|rewrite|expand|hyde|decompose|clarify 中的一个或多个"],\n'
    '  "rewritten_query": "改写后更适合检索的查询语句，没有就填空字符串",\n'
    '  "sub_queries": ["分解出的子问题，比较/多目标时填写"],\n'
    '  "recall_scale": 召回规模倍数（数字，总结类建议 2.0~3.0，其余 1.0 左右）\n'
    "}\n"
    "\n"
    "判定规则：\n"
    "1. question_type：问定义→definition；查具体事实/数值/条款→fact；"
    "问步骤/怎么操作/如何申请→process；比较两个及以上对象→comparison；"
    "要求概括或总结→summary；涉及政策/补贴/规定且带地区时间等约束→policy；"
    "描述故障/异常/报错并想排查原因→troubleshooting；其余→other。\n"
    "2. entities 是用户真正关心的对象（话题/产品/机构/人物等）；slots 是检索必须满足的限制条件。"
    "地区（region）、时间（time）一律放 slots，不要放 entities。有则填，无则空数组。\n"
    "3. strategies 选择：\n"
    "   - 问题清晰明确→direct；\n"
    "   - 口语化、指代不明、依赖上下文→rewrite；\n"
    "   - 用户用词与文档术语可能不一致（俗称/缩写/症状）→expand；\n"
    "   - 问题短而抽象（如「LangGraph 适合做什么」）→hyde；\n"
    "   - 包含多个子目标或比较多个对象→decompose；\n"
    "   - 问题本身不完整、指代不明、无法确定用户想要什么"
    "（如「这个怎么退」不知道「这个」指什么、「那后来呢」没有上下文）→clarify。\n"
    "4. needs_clarification 仅在问题本身不完整或指代不明、无法据此检索时才为 true"
    "（如「这个怎么退」）。只要问题已是一个完整、可检索的提问，即使缺少地区/平台/时间等范围信息，"
    "也不要 clarify，照常检索即可；地区/时间等信息放进 slots 用于过滤，而不是必须澄清的前置条件。"
)


# --------------------------------------------------------------------------
# 意图分析
# --------------------------------------------------------------------------
def _norm_list(v):
    return v if isinstance(v, list) else []


def _norm_str(v):
    return v.strip() if isinstance(v, str) else ""


def _pick(raw, key, default):
    return raw.get(key, default) if isinstance(raw, dict) else default


def _normalize(raw: dict) -> Intent:
    raw = raw if isinstance(raw, dict) else {}
    qtype = _norm_str(_pick(raw, "question_type", "fact")).lower()
    if qtype not in QUESTION_TYPES:
        qtype = "fact"

    entities = [e for e in _norm_list(_pick(raw, "entities", []))
                if isinstance(e, dict) and _norm_str(e.get("value"))]
    slots = [s for s in _norm_list(_pick(raw, "slots", []))
             if isinstance(s, dict) and _norm_str(s.get("value"))]

    strategies = [s for s in _norm_list(_pick(raw, "strategies", [])) if s in STRATEGIES]
    if not strategies:
        strategies = list(DEFAULT_STRATEGY_BY_TYPE.get(qtype, ["direct"]))
    needs_clar = bool(_pick(raw, "needs_clarification", False))
    if needs_clar or "clarify" in strategies:
        strategies = ["clarify"]

    try:
        recall = float(_pick(raw, "recall_scale", 0))
    except (TypeError, ValueError):
        recall = 0.0
    if recall <= 0:
        recall = RECALL_SCALE_BY_TYPE.get(qtype, 1.0)

    return Intent(
        question_type=qtype,
        entities=entities,
        slots=slots,
        needs_clarification=needs_clar,
        missing_info=[m for m in _norm_list(_pick(raw, "missing_info", []))
                      if isinstance(m, str) and m.strip()],
        strategies=strategies,
        rewritten_query=_norm_str(_pick(raw, "rewritten_query", "")),
        sub_queries=[q for q in _norm_list(_pick(raw, "sub_queries", []))
                     if isinstance(q, str) and q.strip()],
        recall_scale=recall,
        raw=raw,
    )


def has_reference(question: str) -> bool:
    """规则判断问题是否依赖上下文指代（本地零成本，供答案缓存层决策）。

    保守策略：宁多判「有指代」也不漏判——漏判只会让缓存按字面 key 查（几乎不命中），
    不影响正确性；多判最多损失一次缓存命中，也不影响正确性。
    注意：本函数只决定「查不查答案缓存」，不影响「是否补全」——补全由
    complete_and_analyze() 独立判断（有历史时合并补全）。
    """
    q = (question or "").strip()
    if not q:
        return False
    if any(t in q for t in _REFERENCE_TOKENS):
        return True
    # 「那北京呢」「这怎么退」这类短省略式指代
    if len(q) <= 8 and q[0] in ("那", "这"):
        return True
    return False


def _conversation_text(conversation, max_turns: int = 6) -> str:
    """把对话历史格式化成文本（兼容 str 列表与 role/content 列表）。"""
    if not conversation:
        return ""
    lines = []
    for m in list(conversation)[-max_turns:]:
        if isinstance(m, str):
            lines.append(f"用户：{m.strip()}")
        elif isinstance(m, dict):
            role = "用户" if m.get("role") == "user" else "助手"
            lines.append(f"{role}：{m.get('content', '')}")
    return "\n".join(lines)


def complete_query(question: str, api_key: str, conversation=None) -> str:
    """上下文补全：把依赖上文的省略/指代补全成可独立检索的完整查询。

    例：上文问过「2024年上海人才补贴怎么申请」，当前问「那北京呢」
      → 「2024年北京人才补贴怎么申请」（继承时间、替换地区）。
    无历史或补全失败时返回原问题。
    """
    if not conversation:
        return question
    hist = _conversation_text(conversation)
    if not hist.strip():
        return question
    try:
        out = chat_json([
            {"role": "system",
             "content": "你是查询补全器。结合对话历史，把用户的当前问题补全成一个不依赖上下文、"
                        "可独立用于检索的完整查询：补全指代（它/这个/上面/第二个等）、继承上文约束"
                        "（地区/时间/对象等）。只输出 JSON：{\"query\": \"补全后的完整查询\"}。"
                        "若当前问题已完整独立，query 返回原问题即可。"},
            {"role": "user",
             "content": f"【对话历史】\n{hist}\n\n【当前问题】\n{question}"},
        ], api_key)
        q = _norm_str(out.get("query"))
        return q or question
    except LLMError:
        return question


def rule_based_intent(question: str) -> Intent | None:
    """规则优先意图：对高置信度的简单事实/定义/排查查询直接判定，返回 Intent；
    否则返回 None（走 LLM）。

    只处理「直接检索原问题就能答」的查询，绝不碰需要改写/扩展/分解/澄清的复杂问题：
    - 指代不明、比较、总结、政策、复合多目标 → 返回 None（走 LLM）。
    - 误判的代价仅是「用原问题直接检索」，与 LLM 判成 fact+direct 等价，不会答错。
    """
    q = (question or "").strip()
    if not q:
        return None
    # 依赖上文指代 / 复杂多目标 → 走 LLM
    if has_reference(q) or any(p in q for p in _RULE_COMPLEX):
        return None
    if any(p in q for p in _RULE_DEFINITION):
        return Intent(question_type="definition", strategies=["direct"],
                      rewritten_query=q, recall_scale=1.0, rule_based=True)
    if any(p in q for p in _RULE_FAST_FACT):
        return Intent(question_type="fact", strategies=["direct"],
                      rewritten_query=q, recall_scale=1.0, rule_based=True)
    if any(p in q for p in _RULE_TROUBLESHOOTING):
        return Intent(question_type="troubleshooting", strategies=["direct"],
                      rewritten_query=q, recall_scale=1.0, rule_based=True)
    # 「…吗？」这类 yes/no 疑问直接判定为 fact
    if q.endswith("吗") or q.endswith("吗？"):
        return Intent(question_type="fact", strategies=["direct"],
                      rewritten_query=q, recall_scale=1.0, rule_based=True)
    return None


def analyze_intent(question: str, api_key: str, conversation=None) -> Intent:
    """一次 LLM 调用完成「类型判断 + 实体槽位抽取 + 策略决策」。

    规则优先：简单事实/定义/排查查询先由 rule_based_intent 判定，命中则跳过 LLM。
    """
    rule = rule_based_intent(question)
    if rule is not None:
        return rule
    hist = _conversation_text(conversation)
    user_msg = (f"【对话历史（仅用于理解指代，不参与问题类型判断）】\n{hist}\n\n【当前问题】\n{question}"
                if hist else f"【当前问题】\n{question}")
    try:
        raw = chat_json(
            [{"role": "system", "content": INTENT_SYSTEM_PROMPT},
             {"role": "user", "content": user_msg}],
            api_key,
        )
    except LLMError as e:
        print(f"  [意图分析失败，降级为直接检索] {e}")
        return Intent(question_type="fact", strategies=["direct"], rewritten_query=question)
    return _normalize(raw)


COMPLETE_AND_ANALYZE_SYSTEM = (
    "你是检索意图分析器。先结合对话历史，把用户当前问题补全成一个不依赖上下文、可独立检索的"
    "完整查询，再分析补全后查询的意图。只输出一个 JSON 对象，不要输出任何解释、Markdown 代码块"
    "或多余文字。\n"
    "\n"
    "背景：本系统在已有知识库上做检索问答，知识库内容已就绪，用户的问题默认针对知识库内容。"
    "不要因为用户没有粘贴文档、没有注明地区/平台就要求澄清。\n"
    "\n"
    "JSON 字段（键名固定，不得增删）：\n"
    "{\n"
    '  "completed_query": "补全后的完整查询（不依赖上下文；若已完整独立，返回原问题）",\n'
    '  "question_type": "definition|fact|process|comparison|summary|policy|troubleshooting|other",\n'
    '  "entities": [{"type": "topic|product|organization|person|policy_name|...", "value": "实体"}],\n'
    '  "slots": [{"type": "region|time|policy_type|product_category|quantity|...", "value": "约束值"}],\n'
    '  "needs_clarification": true 或 false,\n'
    '  "missing_info": ["缺少的关键信息，以问句形式写给用户"],\n'
    '  "strategies": ["direct|rewrite|expand|hyde|decompose|clarify 中的一个或多个"],\n'
    '  "rewritten_query": "改写后更适合检索的查询语句，没有就填空字符串",\n'
    '  "sub_queries": ["分解出的子问题，比较/多目标时填写"],\n'
    '  "recall_scale": 召回规模倍数（数字，总结类建议 2.0~3.0，其余 1.0 左右）\n'
    "}\n"
    "\n"
    "判定规则：\n"
    "1. 先补全：把「它/这个/上面/那北京呢」等依赖上文的省略与指代，结合历史补成完整查询"
    "（继承上文的时间/地区/对象等约束）。当前问题已完整独立时，completed_query 返回原问题。\n"
    "2. question_type：问定义→definition；查具体事实/数值/条款→fact；"
    "问步骤/怎么操作/如何申请→process；比较两个及以上对象→comparison；"
    "要求概括或总结→summary；涉及政策/补贴/规定且带地区时间等约束→policy；"
    "描述故障/异常/报错并想排查原因→troubleshooting；其余→other。\n"
    "3. entities 是用户真正关心的对象（话题/产品/机构/人物等）；slots 是检索必须满足的限制条件。"
    "地区（region）、时间（time）一律放 slots，不要放 entities。有则填，无则空数组。\n"
    "4. strategies 选择：问题清晰明确→direct；口语化、指代不明、依赖上下文→rewrite；"
    "用户用词与文档术语可能不一致（俗称/缩写/症状）→expand；问题短而抽象→hyde；"
    "包含多个子目标或比较多个对象→decompose；问题本身不完整、指代不明、无法确定用户想要什么→clarify。\n"
    "5. needs_clarification 仅在问题本身不完整或指代不明、无法据此检索时才为 true。"
    "只要问题已是一个完整、可检索的提问，即使缺少地区/平台/时间等范围信息，也不要 clarify。"
)


def complete_and_analyze(question: str, api_key: str, conversation=None) -> tuple[str, Intent]:
    """合并「上下文补全 + 意图识别」为一次 LLM 调用（多轮场景省掉一次调用）。

    - 规则优先：自包含的简单问题直接 rule_based_intent 判定，跳过 LLM（无论有无历史）。
    - 无历史：直接 analyze_intent（completed = 原问题）。
    - 有历史且非规则命中：一次调用同时输出 completed_query 与意图字段。
    返回 (completed_query, Intent)。
    """
    rule = rule_based_intent(question)
    if rule is not None:
        return question, rule

    if not conversation:
        intent = analyze_intent(question, api_key)
        return question, intent

    hist = _conversation_text(conversation)
    if not hist.strip():
        intent = analyze_intent(question, api_key)
        return question, intent

    try:
        raw = chat_json(
            [{"role": "system", "content": COMPLETE_AND_ANALYZE_SYSTEM},
             {"role": "user",
              "content": f"【对话历史】\n{hist}\n\n【当前问题】\n{question}"}],
            api_key,
        )
    except LLMError as e:
        print(f"  [意图分析失败，降级为直接检索] {e}")
        return question, Intent(question_type="fact", strategies=["direct"],
                                rewritten_query=question)

    completed = _norm_str(raw.get("completed_query")) or question
    intent = _normalize(raw)
    return completed, intent


# --------------------------------------------------------------------------
# 策略辅助：改写 / 扩展 / HyDE（各自一次小调用）
# --------------------------------------------------------------------------
def _rewrite(question: str, api_key: str) -> str:
    try:
        out = chat_json([
            {"role": "system",
             "content": "把用户口语化、含指代的问题改写成适合信息检索的简洁查询语句。"
                        "只输出 JSON：{\"query\": \"改写后的检索语句\"}"},
            {"role": "user", "content": question},
        ], api_key)
        return _norm_str(out.get("query")) or question
    except LLMError:
        return question


def _expand(question: str, api_key: str) -> list[str]:
    try:
        out = chat_json([
            {"role": "system",
             "content": "为信息检索生成查询扩展，覆盖用户用词与文档用词可能不一致的情况"
                        "（同义词、近义词、简称/全称、症状词等）。"
                        "只输出 JSON：{\"queries\": [\"2~4 个检索短语\"]}"},
            {"role": "user", "content": question},
        ], api_key)
        qs = _norm_list(out.get("queries"))
        return [q.strip() for q in qs if isinstance(q, str) and q.strip()]
    except LLMError:
        return []


def _hyde(question: str, api_key: str) -> str:
    try:
        text = call_llm([
            {"role": "system",
             "content": "针对用户问题，写一段可能出现在知识库中的假设性回答段落（HyDE）。"
                        "只输出段落本身，不要任何前缀说明。"},
            {"role": "user", "content": question},
        ], api_key, temperature=0.3)
        return text.strip()
    except LLMError:
        return question


# --------------------------------------------------------------------------
# 策略执行：生成检索变体 → 多路检索 → 合并 → 槽位过滤
# --------------------------------------------------------------------------
def collect_query_variants(intent: Intent, question: str, api_key: str) -> list[tuple[str, str]]:
    """按策略生成 (检索文本, 来源) 列表，去重。

    rewrite/expand/hyde 可能额外触发 LLM 调用，默认（RAG_ENABLE_EXPANSION=0）关闭：
    - rewrite 仅复用意图阶段已产出的 rewritten_query，不再单独调 LLM；
    - expand/hyde 不执行。
    设 RAG_ENABLE_EXPANSION=1 恢复完整扩展能力。
    """
    variants: list[tuple[str, str]] = []
    seen: set[str] = set()
    strategies = set(intent.strategies)
    primary = intent.rewritten_query or question

    def add(q: str, origin: str) -> None:
        q = (q or "").strip()
        if q and q not in seen:
            seen.add(q)
            variants.append((q, origin))

    if "decompose" in strategies:
        for sq in intent.sub_queries:
            add(sq, "decompose")
    if "direct" in strategies:
        add(primary, "direct")
    if not variants:
        add(primary, "direct")
    if "rewrite" in strategies:
        add(intent.rewritten_query or (question if not ENABLE_EXPANSION
                                       else _rewrite(question, api_key)), "rewrite")
    if "expand" in strategies and ENABLE_EXPANSION:
        for q in _expand(question, api_key):
            add(q, "expand")
    if "hyde" in strategies and ENABLE_EXPANSION:
        add(_hyde(question, api_key), "hyde")
    return variants


def scaled_top_k(intent: Intent, base_top_k: int) -> int:
    scale = intent.recall_scale if intent.recall_scale > 0 else 1.0
    return max(1, int(round(base_top_k * scale)))


def merge_hits(hits: list[dict]) -> list[dict]:
    """把多个检索变体的命中按 chunk idx 归并，累加 RRF 分（相关度跨变体累加）。"""
    by_idx: dict[int, dict] = {}
    for h in hits:
        idx = h.get("idx")
        if idx is None:
            continue
        e = by_idx.get(idx)
        if e is None:
            e = {"idx": idx, "chunk": h.get("chunk"), "score": 0.0,
                 "origins": [], "vec_sim": 0.0, "bm25": 0.0, "best": h}
            by_idx[idx] = e
        e["score"] += float(h.get("rrf", 0.0))
        e["vec_sim"] = max(e["vec_sim"], float(h.get("vec_sim", 0.0)))
        e["bm25"] = max(e["bm25"], float(h.get("bm25", 0.0)))
        if h.get("origin"):
            e["origins"].append(h["origin"])
        if float(h.get("rrf", -1.0)) > float(e["best"].get("rrf", -2.0)):
            e["best"] = h

    merged = sorted(by_idx.values(), key=lambda e: -e["score"])
    out = []
    for e in merged:
        h = dict(e["best"])
        h["score"] = e["score"]
        h["origins"] = sorted(set(e["origins"]))
        h["vec_sim"] = e["vec_sim"]
        h["bm25"] = e["bm25"]
        out.append(h)
    return out


def apply_slot_filter(hits: list[dict], slots: list[dict], chunks: list[dict]) -> list[dict]:
    """政策类软过滤：离线阶段没有逐块元数据，退化为「槽位值是否出现在块文本」的软降权。

    有真实元数据（地区/时间/政策类型）时，这里应替换为硬过滤：chunk.meta["region"] == "上海"。
    """
    if not slots or not chunks:
        return hits
    critical = [str(s.get("value")).strip() for s in slots
                if s.get("type") in CRITICAL_SLOT_TYPES and _norm_str(s.get("value"))]
    if not critical:
        return hits
    for h in hits:
        idx = h.get("idx")
        text = chunks[idx]["text"] if (idx is not None and 0 <= idx < len(chunks)) else ""
        matched = [v for v in critical if v in text]
        h["slot_matched"] = matched
        h["slot_score"] = float(h.get("score", 0.0)) * (1.0 if matched else 0.35)
    hits.sort(key=lambda h: -h.get("slot_score", 0.0))
    return hits


def retrieve_with_intent(question: str, state: dict, retriever,
                         base_top_k: int = DEFAULT_TOP_K,
                         conversation=None, metrics=None) -> dict:
    """主入口：上下文补全 → 分析意图 → 澄清 or 执行策略 → 返回合并后的命中。

    返回 dict：{"action": "clarify", "intent":…, "questions":[…], "completed_query":…}
            或 {"action": "answer",  "intent":…, "hits":[…], "variants":[…], "completed_query":…}
    """
    api_key = state.get("api_key", "")
    t_pre = time.time()
    # 第一步：上下文补全 + 意图识别（合并为一次 LLM 调用）
    with stage("intent_llm", user_query=question):
        completed, intent = complete_and_analyze(question, api_key, conversation=conversation)
    rag_metrics().intent_type.add(1, {"type": intent.question_type})

    if intent.needs_clarification or "clarify" in intent.strategies:
        if metrics is not None:
            metrics.record("intent_llm", time.time() - t_pre)
        rag_metrics().search_duration.record((time.time() - t_pre) * 1000, {"stage": "intent_llm"})
        questions = [q for q in intent.missing_info if isinstance(q, str) and q.strip()]
        if not questions:
            questions = ["请补充问题的具体背景或限制条件（例如地区、时间、具体对象）。"]
        return {"action": "clarify", "intent": intent, "questions": questions,
                "completed_query": completed}

    chunks = state["chunks"]
    variants = collect_query_variants(intent, completed, api_key)
    top_k = scaled_top_k(intent, base_top_k)
    fetch_k = max(top_k * 2, top_k + 4)   # 先取宽窗口，给槽位过滤留余地
    if metrics is not None:
        metrics.record("intent_llm", time.time() - t_pre)
    rag_metrics().search_duration.record((time.time() - t_pre) * 1000, {"stage": "intent_llm"})

    t_ret = time.time()
    all_hits: list[dict] = []
    with stage("retrieval", variants=len(variants), user_query=completed):
        for qtext, origin in variants:
            try:
                hits = retriever(qtext, fetch_k)
            except Exception as e:
                logging.getLogger("rag.retrieval").warning(
                    "检索失败，跳过该变体",
                    extra={"stage": "retrieval", "error_code": type(e).__name__})
                continue
            for h in hits:
                h["origin"] = origin
                h["query"] = qtext
            all_hits.extend(hits)

    merged = merge_hits(all_hits)
    merged = apply_slot_filter(merged, intent.slots, chunks)
    if metrics is not None:
        metrics.record("retrieval", time.time() - t_ret)
    m = rag_metrics()
    m.search_duration.record((time.time() - t_ret) * 1000, {"stage": "retrieval"})
    m.search_hits.record(len(merged))

    # 精排（rerank）：用 Qwen3-Reranker 对候选做深度重排，再截到 top_k。
    # 只精排 RRF 排名靠前的候选（top_k 的 3 倍），省算力且不丢尾部召回。
    reranker = state.get("reranker")
    t_rr = time.time()
    if reranker is not None and len(merged) > 1:
        pool = min(len(merged), max(top_k * 3, top_k + 8))
        head, tail = merged[:pool], merged[pool:]
        with stage("rerank", input_docs=pool, user_query=completed):
            try:
                merged = reranker.rerank(completed, head) + tail
            except Exception as e:
                logging.getLogger("rag.rerank").warning(
                    "精排失败，回退 RRF 排序",
                    extra={"stage": "rerank", "error_code": type(e).__name__})
        if metrics is not None:
            metrics.record("rerank", time.time() - t_rr)
        rag_metrics().search_duration.record((time.time() - t_rr) * 1000, {"stage": "rerank"})

    merged = merged[:top_k]
    return {"action": "answer", "intent": intent, "hits": merged,
            "variants": variants, "completed_query": completed}


# --------------------------------------------------------------------------
# 展示
# --------------------------------------------------------------------------
def describe_intent(intent: Intent) -> str:
    suffix = "（规则判定，未调 LLM）" if intent.rule_based else ""
    lines = [f"类型：{TYPE_ZH.get(intent.question_type, intent.question_type)}（{intent.question_type}）{suffix}"]
    if intent.entities:
        lines.append("实体：" + "、".join(f"{e.get('type')}={e.get('value')}" for e in intent.entities))
    if intent.slots:
        lines.append("槽位：" + "、".join(f"{s.get('type')}={s.get('value')}" for s in intent.slots))
    lines.append("策略：" + "、".join(STRATEGY_ZH.get(s, s) for s in intent.strategies))
    if intent.sub_queries:
        lines.append("子问题：" + "；".join(intent.sub_queries))
    if intent.rewritten_query:
        lines.append("改写查询：" + intent.rewritten_query)
    lines.append(f"召回倍数：{intent.recall_scale:.1f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 独立调试入口（只分析意图，不检索）
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="查询意图分析（问题类型 / 实体槽位 / 检索策略）")
    parser.add_argument("question", nargs="?", help="要分析的问题")
    parser.add_argument("--demo", action="store_true", help="用内置示例演示")
    args = parser.parse_args()

    api_key = get_api_key()
    if not api_key:
        raise SystemExit("找不到 DeepSeek API Key（DEEPSEEK_API_KEY 或 ~/.pi/agent/auth.json）")

    samples = [
        "普通商品的退货期限是多久？",
        "LangGraph 和 LangChain 有什么区别？",
        "总结一下这套售后规则都讲了什么。",
        "2024年上海人才补贴怎么申请？",
        "订单提交后一直显示处理中，是什么问题？",
        "这个怎么退？",
    ]
    questions = samples if (args.demo or not args.question) else [args.question]

    for q in questions:
        print("=" * 68)
        print("问题：", q)
        intent = analyze_intent(q, api_key)
        print(describe_intent(intent))
        if intent.missing_info:
            print("待澄清：", "；".join(intent.missing_info))
        if intent.needs_clarification:
            print("→ 结论：缺少关键信息，应先向用户澄清，而不是强行检索。")
    print("=" * 68)


if __name__ == "__main__":
    main()
