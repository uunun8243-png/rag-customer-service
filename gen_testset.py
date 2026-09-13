#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_testset.py — 用 Ragas 从知识库文档自动生成评测集（问题 + 标准答案 + 出处）

输入：知识库 Markdown（默认 notes/clean.md）
输出：合成评测集，每条包含
  - query              用户问题
  - reference          标准答案（Ragas 生成）
  - reference_contexts 回答该问题所需的原文出处（golden passages，来自文档切块）
  - synthesizer_name   生成该条用的策略（single-hop / multi-hop 等）

原理（ragas 0.4.3 testset 流程）：
  1. 读文档，按 token 数选策略（>500 token → 抽标题 + 按标题切块 + 摘要 + 主题 + 实体）
  2. 构建知识图谱（文档/切块节点 + 相似关系）
  3. 用 query synthesizer 按分布生成问题：单跳具体 / 多跳抽象 / 多跳具体
  4. 每条同时产出标准答案（reference）与出处（reference_contexts）

依赖：DeepSeek（生成用 LLM）+ 本地 bge-small-zh-v1.5（主题聚类/相似度）

用法：
    .venv/bin/python gen_testset.py                    # 默认 20 条，读 notes/clean.md
    .venv/bin/python gen_testset.py --size 15          # 生成 15 条
    .venv/bin/python gen_testset.py --file notes/clean.md
    .venv/bin/python gen_testset.py --llm deepseek-v4-pro

产物：
    ragas_testset.json         完整评测集（含 synthesizer 元信息）
    ragas_testset_golden.json  与 golden_test_set.json 同构（id/query/answer/golden_passages），
                               可直接喂给 eval_retrieval.py / eval_generation.py / eval_ragas.py
"""

import argparse
import json
import sys
from pathlib import Path

from langchain_core.documents import Document
from ragas.testset.persona import Persona

BASE_DIR = Path(__file__).resolve().parent
DOC_FILE = BASE_DIR / "notes" / "clean.md"
OUT_FULL = BASE_DIR / "ragas_testset.json"
OUT_GOLDEN = BASE_DIR / "ragas_testset_golden.json"

# 固定 persona：让生成的问题统一从“消费者咨询售后”的视角出发，
# 避免 Ragas 自动从文档内容生成 persona 时把“数据清洗”等无关视角混进问题。
PERSONAS = [
    Persona(
        name="普通消费者",
        role_description="在电商平台购物的普通用户，正在咨询售后政策（退货/换货期限、运费、无理由退货、申请流程、服务渠道等）",
    ),
]

# 让生成的问题/答案用简体中文，并贴合业务场景
LLM_CONTEXT = (
    "这是某电商平台的售后规则文档。请围绕退货期限、无理由退货、运费承担、"
    "售后申请流程、服务渠道、常见问题等业务点生成测试问题，全部用简体中文。"
)


def load_documents(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        sys.exit(f"文档为空：{path}")
    return [Document(page_content=text, metadata={"source": str(path)})]


def generate_testset(documents: list[Document], size: int, model: str):
    from eval_ragas import FastEmbedEmbedding, build_llm  # 复用同一套 LLM/嵌入
    from ragas.testset import TestsetGenerator

    llm = build_llm(model)
    embeddings = FastEmbedEmbedding()

    print("初始化 TestsetGenerator（LLM=DeepSeek，嵌入=bge-small-zh-v1.5）…")
    generator = TestsetGenerator(
        llm=llm,
        embedding_model=embeddings,
        persona_list=PERSONAS,
        llm_context=LLM_CONTEXT,
    )

    print(f"开始生成评测集（目标 {size} 条，先建知识图谱再生成问题，LLM 调用较多）…")
    testset = generator.generate_with_langchain_docs(
        documents=documents,
        testset_size=size,
        raise_exceptions=True,
    )
    return testset


def save(testset, source_name: str) -> None:
    rows = testset.to_list()

    # 1) 完整版（含 synthesizer）
    full = {
        "name": f"{source_name} · Ragas 合成评测集",
        "source_file": source_name,
        "generator": "ragas",
        "count": len(rows),
        "samples": rows,
    }
    OUT_FULL.write_text(json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完整评测集已写入 {OUT_FULL.name}（{len(rows)} 条）")

    # 2) golden_test_set.json 同构版，可直接复用现有 eval 脚本
    queries = []
    for i, r in enumerate(rows, 1):
        ref_ctxs = r.get("reference_contexts") or []
        queries.append({
            "id": f"q{i:02d}",
            "query": r.get("user_input", ""),
            "answer": r.get("reference", ""),
            "synthesizer": r.get("synthesizer_name", ""),
            "golden_passages": list(ref_ctxs),
        })
    golden = {
        "name": f"{source_name} · Ragas 合成评测集",
        "version": "1.0",
        "description": "由 ragas 从文档自动合成（含 single-hop / multi-hop 问题与出处）。",
        "doc_file": source_name,
        "queries": queries,
    }
    OUT_GOLDEN.write_text(json.dumps(golden, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(f"golden 同构版已写入 {OUT_GOLDEN.name}（可直接用于 eval_retrieval.py 等）")

    # 控制台预览
    print("\n" + "=" * 72)
    print("生成样例预览：")
    for q in queries[:5]:
        print(f"  [{q['id']}] {q['query']}")
        print(f"       答：{q['answer'][:60]}{'…' if len(q['answer']) > 60 else ''}")
        print(f"       出处块数：{len(q['golden_passages'])}｜策略：{q['synthesizer']}")


def main():
    ap = argparse.ArgumentParser(description="用 Ragas 从文档生成评测集")
    ap.add_argument("--file", default=str(DOC_FILE), help="知识库 Markdown 路径")
    ap.add_argument("--size", type=int, default=20, help="生成的评测条目数（默认 20）")
    ap.add_argument("--llm", default="", help="生成用 LLM（默认取 RAG_LLM_MODEL 或 deepseek-v4-flash）")
    args = ap.parse_args()

    from llm import LLM_MODEL

    path = Path(args.file)
    if not path.is_file():
        sys.exit(f"文件不存在：{path}")
    model = args.llm or LLM_MODEL

    print("=" * 72)
    print(f"Ragas 评测集生成 · {path}")
    print(f"目标条数：{args.size}｜LLM：{model}")
    print("=" * 72)

    docs = load_documents(path)
    print(f"已加载文档：{path}（{len(docs[0].page_content)} 字符）")

    testset = generate_testset(docs, args.size, model)
    save(testset, path.name)


if __name__ == "__main__":
    main()
