#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag_index.py — 索引与混合检索（Qdrant Server 为主，内存 numpy 兜底）

多 Worker 部署：
  - 向量存 Qdrant 服务端（docker 部署，见 deploy/docker-compose.yml），所有 Worker 共享；
  - BM25 在每个 Worker 本地计算（语料 tokenize 后常驻内存，无锁、只读）；
  - 会话仍走 Redis（见 session_store.py）。

后端选择（环境变量）：
  RAG_QDRANT_URL         Qdrant 服务端地址，例如 http://127.0.0.1:6333（REST 端口）
                        未设置时回退到「内存 numpy 全量余弦」（仅单进程开发/CI 用）
  RAG_QDRANT_COLLECTION  collection 名（默认 notes）

离线构建索引（写入 Qdrant 服务端）：
    .venv/bin/python rag_index.py --rebuild
Worker 启动时 load_index()：若 collection 不存在则自动构建，否则直接 scroll 加载。

检索：向量（Qdrant）+ BM25（本地）→ RRF 融合，返回结构与 rag_tool.retrieve 一致。
"""

import hashlib
import logging
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import jieba
import numpy as np
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from chunker import structure_merge

jieba.setLogLevel(logging.WARNING)

BASE_DIR = Path(__file__).resolve().parent
NOTES_DIR = Path(os.environ.get("RAG_NOTES_DIR", str(BASE_DIR / "notes")))

EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
CHUNK_TARGET = int(os.environ.get("RAG_CHUNK_SIZE", "500"))
CHUNK_OVERLAP_RATIO = float(os.environ.get("RAG_CHUNK_OVERLAP_RATIO", "0.20"))
TOP_K = 6
RRF_K = float(os.environ.get("RAG_RRF_K", "60"))

QDRANT_URL = os.environ.get("RAG_QDRANT_URL", "").strip()
COLLECTION = os.environ.get("RAG_QDRANT_COLLECTION", "notes")

# 不入库的文件名关键词（数据清洗的中间产物，如清洗前的脏数据）
EXCLUDE_FILENAME_KEYWORDS = ["dirty_raw"]

# 不入库的目录名（虚拟环境、缓存、索引自身等；点号开头的隐藏目录一并兜底排除）
EXCLUDE_DIR_NAMES = {".venv", ".git", ".rag_index", "__pycache__",
                     ".pytest_cache", ".mypy_cache", ".ruff_cache",
                     "node_modules", ".idea", ".vscode"}


# --------------------------------------------------------------------------
# 嵌入 / 分词
# --------------------------------------------------------------------------
_embed_lock = threading.Lock()   # fastembed/onnxruntime 潜在非线程安全，encode 时串行


def load_embedder() -> TextEmbedding:
    return TextEmbedding(EMBED_MODEL)


def encode(embedder: TextEmbedding, texts: list[str]) -> np.ndarray:
    """向量化并做 L2 归一化（即余弦相似度）。内部加锁保护 fastembed 并发。

    只锁 fastembed 推理这一小段，BM25 打分与 Qdrant 查询都在锁外，
    避免旧实现里「整段检索串行化」导致的并发瓶颈。
    """
    with _embed_lock:
        vecs = np.array(list(embedder.embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def tokenize(text: str) -> list[str]:
    return [w for w in jieba.lcut(text) if w.strip()]


# --------------------------------------------------------------------------
# 读取文档
# --------------------------------------------------------------------------
def list_md_files() -> list[Path]:
    if not NOTES_DIR.is_dir():
        raise SystemExit(f"笔记目录不存在：{NOTES_DIR}。请用环境变量 RAG_NOTES_DIR 指定正确的目录。")
    files = sorted(NOTES_DIR.rglob("*.md"))
    files = [f for f in files
             if not any(k in f.name for k in EXCLUDE_FILENAME_KEYWORDS)
             and not any(p in EXCLUDE_DIR_NAMES or p.startswith(".")
                         for p in f.parts)]
    if not files:
        raise SystemExit(f"在 {NOTES_DIR} 下没有找到任何 .md 笔记文件，"
                         f"请把知识库 .md 放进该目录（或用 RAG_NOTES_DIR 指定）。")
    return files


# 知识库内容指纹：文档一变指纹就变，答案缓存（answer_cache.py）据此自动失效
INDEX_FINGERPRINT = ""


def compute_fingerprint() -> str:
    h = hashlib.sha1()
    for f in list_md_files():
        text = f.read_text(encoding="utf-8", errors="ignore")
        h.update(f.name.encode("utf-8"))
        h.update(b"\0")
        h.update(text.encode("utf-8"))
    return h.hexdigest()


def read_docs() -> list[dict]:
    files = list_md_files()
    print(f"[索引] 读取笔记目录：{NOTES_DIR}（{len(files)} 篇 .md）")
    chunks = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        for c in structure_merge(text, CHUNK_TARGET, CHUNK_OVERLAP_RATIO):
            chunks.append({"text": c.text, "file_name": f.name})
    return chunks


# --------------------------------------------------------------------------
# 两种索引后端
# --------------------------------------------------------------------------
@dataclass
class InMemoryIndex:
    chunks: list[dict]        # [{"text", "file_name"}]
    chunk_vecs: np.ndarray    # (n, d) L2 归一化
    bm25: BM25Okapi
    embedder: TextEmbedding


@dataclass
class QdrantIndex:
    client: QdrantClient      # Qdrant 服务端客户端（url 模式）
    collection: str
    chunks: list[dict]        # scroll 出来的全部 chunk（与 point id 对齐）
    bm25: BM25Okapi           # 本地 BM25
    embedder: TextEmbedding


def get_server_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, timeout=30)


def _build_inmemory(embedder: TextEmbedding, chunks: list[dict]) -> InMemoryIndex:
    vecs = encode(embedder, [c["text"] for c in chunks])
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    return InMemoryIndex(chunks=chunks, chunk_vecs=vecs, bm25=bm25, embedder=embedder)


def _upsert_qdrant(client: QdrantClient, embedder: TextEmbedding, chunks: list[dict]) -> None:
    vecs = encode(embedder, [c["text"] for c in chunks])
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION,
        vectors_config=VectorParams(size=vecs.shape[1], distance=Distance.COSINE),
    )
    client.upsert(COLLECTION, [
        PointStruct(id=i, vector=vecs[i].tolist(),
                    payload={"text": c["text"], "file_name": c["file_name"]})
        for i, c in enumerate(chunks)
    ])
    print(f"[索引] 已写入 Qdrant 服务端 {QDRANT_URL}（collection={COLLECTION}，{len(chunks)} 条）")


def _scroll_chunks(client: QdrantClient) -> list[dict]:
    points, offset = [], None
    while True:
        batch, offset = client.scroll(COLLECTION, limit=1000, with_payload=True, offset=offset)
        points.extend(batch)
        if offset is None:
            break
    chunks = [None] * len(points)
    for p in points:
        chunks[p.id] = {"text": p.payload["text"], "file_name": p.payload.get("file_name", "")}
    return chunks


# --------------------------------------------------------------------------
# 构建 / 加载
# --------------------------------------------------------------------------
def build_index(embedder: TextEmbedding | None = None):
    """离线构建/重建索引并写入后端（Qdrant 服务端或内存）。CLI：python rag_index.py --rebuild"""
    global INDEX_FINGERPRINT
    embedder = embedder or load_embedder()
    INDEX_FINGERPRINT = compute_fingerprint()
    chunks = read_docs()
    if QDRANT_URL:
        client = get_server_client()
        _upsert_qdrant(client, embedder, chunks)
        bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
        return QdrantIndex(client=client, collection=COLLECTION, chunks=chunks,
                           bm25=bm25, embedder=embedder)
    return _build_inmemory(embedder, chunks)


def load_index(embedder: TextEmbedding | None = None):
    """Worker 启动时加载索引：Qdrant 服务端不存在则自动构建，再 scroll + 本地 BM25。"""
    global INDEX_FINGERPRINT
    embedder = embedder or load_embedder()
    INDEX_FINGERPRINT = compute_fingerprint()
    if not QDRANT_URL:
        return _build_inmemory(embedder, read_docs())
    client = get_server_client()
    if not client.collection_exists(COLLECTION):
        print(f"[索引] Qdrant 服务端尚无 collection，自动构建…")
        _upsert_qdrant(client, embedder, read_docs())
    chunks = _scroll_chunks(client)
    if not chunks:
        raise SystemExit(f"[索引] Qdrant collection={COLLECTION} 为空，"
                         f"请先离线构建：.venv/bin/python rag_index.py --rebuild")
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    print(f"[索引] 从 Qdrant 服务端加载：{QDRANT_URL}（collection={COLLECTION}，{len(chunks)} 条）")
    return QdrantIndex(client=client, collection=COLLECTION, chunks=chunks,
                       bm25=bm25, embedder=embedder)


# --------------------------------------------------------------------------
# RRF 混合检索
# --------------------------------------------------------------------------
def retrieve(index, query: str, top_k: int = TOP_K) -> list[dict]:
    """向量（Qdrant 服务端 / 内存）+ 本地 BM25 → RRF 融合，返回 top_k。

    返回每项含 idx / chunk / rrf / vec_sim / bm25。
    注：当前为「全量排序」版 RRF；大语料时应改为「向量 top-M + BM25 top-M 候选再融合」，
    避免把整库排名都算一遍。
    """
    qvec = encode(index.embedder, [query])[0]
    n = len(index.chunks)

    if isinstance(index, QdrantIndex):
        qres = index.client.query_points(index.collection, query=qvec.tolist(), limit=n)
        vec_order = np.array([p.id for p in qres.points], dtype=np.int64)
        vec_sim = np.zeros(n, dtype=np.float32)
        for p in qres.points:
            vec_sim[p.id] = p.score
    else:
        sims = index.chunk_vecs @ qvec
        vec_order = np.argsort(-sims)
        vec_sim = sims

    bm_scores = np.asarray(index.bm25.get_scores(tokenize(query)), dtype=np.float32)
    bm_order = np.argsort(-bm_scores)

    v_pos = np.empty(n, dtype=np.float32)
    b_pos = np.empty(n, dtype=np.float32)
    v_pos[vec_order] = np.arange(1, n + 1, dtype=np.float32)
    b_pos[bm_order] = np.arange(1, n + 1, dtype=np.float32)
    rrf = 1.0 / (RRF_K + v_pos) + 1.0 / (RRF_K + b_pos)

    order = np.argsort(-rrf)[:top_k]
    return [
        {"idx": int(i), "chunk": index.chunks[int(i)], "rrf": float(rrf[i]),
         "vec_sim": float(vec_sim[i]), "bm25": float(bm_scores[i])}
        for i in order
    ]


# --------------------------------------------------------------------------
# 离线构建入口
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="构建/重建 RAG 索引（写入 Qdrant 服务端或内存）")
    ap.add_argument("--rebuild", action="store_true", help="强制重建（删除旧 collection 重新入库）")
    args = ap.parse_args()

    if args.rebuild or not QDRANT_URL:
        idx = build_index()
    else:
        idx = load_index()
    backend = "Qdrant 服务端" if isinstance(idx, QdrantIndex) else "内存 numpy"
    print(f"索引就绪：backend={backend}，chunks={len(idx.chunks)}")
    sys.exit(0)
