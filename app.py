#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py — RAG 问答 Web 服务（Starlette ASGI，多 Worker + Redis 会话）

架构：
    Nginx → Uvicorn / Gunicorn → 多个 Worker（各自持有内存索引，互不加锁）
                                    ↓
                                  Redis（session / 记忆）

- 向量检索走 Qdrant 服务端（docker 部署，所有 Worker 共享），BM25 在每个 Worker 本地
  计算（见 rag_index.py），实现真正的并发（旧 webapp.py 是单进程 + 全请求一把全局锁）。
- 会话状态（conversation / summary / folded / usage）存 Redis，任意 Worker 都能
  接管同一 session 的后续请求；Redis 不可用时回退到进程内内存（仅单进程可用）。
- SSE 流式输出，与旧 webapp.py 的事件协议一致（status/intent/sources/delta/usage/done）。

启动：
    单进程开发：  .venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
    多 Worker：   .venv/bin/gunicorn -c deploy/gunicorn.conf.py app:app
    前面再加 Nginx（见 deploy/nginx.conf，负责 TLS / 限流 / SSE 不缓冲 / 反向代理）。

环境变量：
    REDIS_URL             Redis 连接串（默认 redis://127.0.0.1:6379/0）
    RAG_SESSION_TTL       会话过期秒数（默认 24 小时）
    RAG_NOTES_DIR         知识库目录（默认 notes/，见 rag_index.py）
    RAG_AUTH_TOKEN        访问令牌；设置后 /api/* 需带 `Authorization: Bearer <token>`
                          或 `X-Auth-Token: <token>`，否则 401（不设置则不鉴权）
    RAG_MAX_BODY_BYTES    请求体大小上限（默认 1MB，超限 413）
    RAG_MAX_QUESTION_CHARS 问题长度上限（默认 2000 字，超限 400）
    RAG_WEB_TLS_CERT / RAG_WEB_TLS_KEY   HTTPS 证书与私钥路径（uvicorn 单进程模式）
    RAG_COOKIE_SECURE     置 1 时会话 Cookie 加 Secure（HTTPS 部署时建议开启）
    其余：RAG_EMBED_MODEL / RAG_CHUNK_SIZE / RAG_ENABLE_RERANK / RAG_LLM_MODEL 等
"""

import copy
import hmac
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from otel_setup import setup_telemetry
setup_telemetry()   # 必须在 import rag_spans / rag_metrics（会 get_tracer/get_meter）之前

from rag_spans import end_stage_span, start_stage_span  # noqa: E402
from rag_metrics import rag_metrics  # noqa: E402

import anyio
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.routing import Route

from llm import LLMError, call_llm, call_llm_stream, get_api_key
from memory import maintain_memory
from metrics import METRICS
from query_intent import describe_intent, has_reference, retrieve_with_intent
from rag_index import INDEX_FINGERPRINT, TOP_K, load_index, retrieve
from rag_tool import build_messages
from reranker import create_reranker
from session_store import create_session_store, empty_session, new_session_id
from answer_cache import create_answer_cache

COOKIE_NAME = "rag_sid"

log = logging.getLogger("rag.api")
OTEL_MODEL = os.environ.get("RAG_LLM_MODEL", "deepseek-v4-flash")

# 成本单价（USD / 1K tokens）。默认按 DeepSeek 常见价位估算，上线前按实际价格设：
#   RAG_COST_PER_1K_PROMPT / RAG_COST_PER_1K_COMPLETION
COST_PER_1K_PROMPT = float(os.environ.get("RAG_COST_PER_1K_PROMPT", "0.00027"))
COST_PER_1K_COMPLETION = float(os.environ.get("RAG_COST_PER_1K_COMPLETION", "0.00110"))

# ---- 安全相关配置（环境变量）----
AUTH_TOKEN = os.environ.get("RAG_AUTH_TOKEN", "").strip()                 # 设置后 /api/* 需鉴权
MAX_BODY_BYTES = int(os.environ.get("RAG_MAX_BODY_BYTES", str(1024 * 1024)))    # 请求体上限（1MB）
MAX_QUESTION_CHARS = int(os.environ.get("RAG_MAX_QUESTION_CHARS", "2000"))      # 问题长度上限
TLS_CERT = os.environ.get("RAG_WEB_TLS_CERT", "").strip()                 # HTTPS 证书路径
TLS_KEY = os.environ.get("RAG_WEB_TLS_KEY", "").strip()                   # HTTPS 私钥路径
COOKIE_SECURE = os.environ.get("RAG_COOKIE_SECURE", "").strip().lower() in ("1", "true", "yes")

HTML_PAGE = (Path(__file__).resolve().parent / "static" / "index.html").read_text(encoding="utf-8")

# 每个 Worker 一份的只读资源（lifespan 里初始化）
APP = {
    "index": None,
    "retriever": None,
    "reranker": None,
    "api_key": "",
    "store": None,
}


# --------------------------------------------------------------------------
# 会话 / 状态组装
# --------------------------------------------------------------------------
def set_cookie(sid: str) -> dict:
    cookie = f"{COOKIE_NAME}={sid}; HttpOnly; Path=/; SameSite=Lax"
    if COOKIE_SECURE:
        cookie += "; Secure"          # HTTPS 下才下发 Secure Cookie，防明文网络泄露
    return {"Set-Cookie": cookie}


def unauthorized() -> JSONResponse:
    return JSONResponse({"error": "unauthorized"}, status_code=401)


def is_authorized(request: Request) -> bool:
    """RAG_AUTH_TOKEN 未设置时不鉴权；设置后 /api/* 需带 Bearer 或 X-Auth-Token。"""
    if not AUTH_TOKEN:
        return True
    auth = request.headers.get("Authorization", "")
    token = auth[7:].strip() if auth.startswith("Bearer ") else \
        request.headers.get("X-Auth-Token", "")
    return hmac.compare_digest(token, AUTH_TOKEN)


async def read_json_body(request: Request) -> tuple[dict | None, int]:
    """带大小上限地读取 JSON 请求体，返回 (dict, 200) 或 (None, 错误码)。

    - 依据 Content-Length 与流式读取双重限制，防止超大 body 打爆内存；
    - 校验 JSON 必须是对象（修复此前 list/str 导致 500 的问题）。
    """
    cl = request.headers.get("Content-Length", "")
    if cl:
        try:
            if int(cl) > MAX_BODY_BYTES:
                return None, 413
        except ValueError:
            pass

    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_BODY_BYTES:
            return None, 413
        chunks.append(chunk)

    raw = b"".join(chunks)
    if not raw:
        return {}, 200
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, 400
    if not isinstance(data, dict):
        return None, 400
    return data, 200


def make_state(session: dict) -> dict:
    """把「只读索引资源 + 会话态」合成 retrieve_with_intent 需要的 state。"""
    return {
        "chunks": APP["index"].chunks,
        "retriever": APP["retriever"],
        "reranker": APP["reranker"],
        "api_key": APP["api_key"],
        "conversation": copy.deepcopy(session.get("conversation", [])),
        "summary": session.get("summary", ""),
        "folded": int(session.get("folded", 0) or 0),
        "usage": copy.deepcopy(session.get("usage", {}) or {}),
    }


def write_back_session(session: dict, state: dict) -> None:
    session["conversation"] = state["conversation"]
    session["summary"] = state["summary"]
    session["folded"] = state["folded"]
    session["usage"] = state["usage"]


def format_hit(h: dict) -> dict:
    c = h.get("chunk") or {}
    return {
        "idx": h.get("idx"),
        "file": c.get("file_name", ""),
        "text": c.get("text", ""),
        "snippet": (c.get("text", "") or "").strip().replace("\n", " ")[:120],
        "rerank": h.get("rerank_score"),
        "rrf": h.get("rrf"),
        "vec_sim": h.get("vec_sim"),
        "bm25": h.get("bm25"),
        "origins": h.get("origins", []),
    }


# --------------------------------------------------------------------------
# 核心流程（同步生成器，StreamingResponse 会在线程池里迭代它 → 阻塞 LLM 调用不卡事件循环）
# --------------------------------------------------------------------------
def ask_events(question: str, session: dict, sid: str = ""):
    t_start = time.time()
    m = rag_metrics()
    m.query_len.record(len(question))
    log.info("收到问答请求：%s", question,
             extra={"stage": "user_request", "session_id": sid})
    state = make_state(session)
    cache = APP.get("answer_cache")

    # 答案缓存：自包含问题命中则直接返回，跳过意图/检索/精排/生成
    if cache is not None and not has_reference(question):
        cached = cache.get(question)
        if cached is not None:
            m.cache_total.add(1, {"hit": "yes"})
            log.info("答案缓存命中", extra={"stage": "cache", "session_id": sid})
            yield {"type": "intent", "text": "⚡ 缓存命中", "completed_query": question}
            yield {"type": "sources",
                   "hits": [format_hit(h) for h in cached.get("hits", [])]}
            answer = cached.get("answer", "")
            yield {"type": "delta", "text": answer}
            state["conversation"].append({"role": "user", "content": question})
            state["conversation"].append({"role": "assistant", "content": answer})
            maintain_memory(state, state["api_key"])
            write_back_session(session, state)
            yield {"type": "usage",
                   **{k: int(v) for k, v in state["usage"].items() if k.endswith("tokens")}}
            m.requests_total.add(1, {"status": "ok"})
            METRICS.record("e2e", time.time() - t_start)
            yield {"type": "done", "answer": answer}
            return
        m.cache_total.add(1, {"hit": "no"})

    yield {"type": "status", "text": "意图分析中…"}
    conversation = state["conversation"]

    try:
        result = retrieve_with_intent(
            question, state, retriever=state["retriever"],
            base_top_k=TOP_K, conversation=conversation, metrics=METRICS)
    except Exception as e:
        m.requests_total.add(1, {"status": "error"})
        yield {"type": "error", "text": f"检索阶段失败：{e}"}
        METRICS.record("e2e", time.time() - t_start)
        return

    intent = result["intent"]
    yield {"type": "intent",
           "text": describe_intent(intent).replace("\n", "｜"),
           "completed_query": result.get("completed_query", "")}
    log.info("意图：%s", describe_intent(intent).replace("\n", "｜"),
             extra={"stage": "intent_llm", "session_id": sid})

    if result["action"] == "clarify":
        m.clarify.add(1)
        log.info("澄清反问", extra={"stage": "clarify", "session_id": sid})
        yield {"type": "clarify", "questions": result["questions"]}
        m.requests_total.add(1, {"status": "ok"})
        METRICS.record("e2e", time.time() - t_start)
        yield {"type": "done"}
        return

    hits = result["hits"]
    if not hits:
        m.empty_recall.add(1)
        m.requests_total.add(1, {"status": "error"})
        log.warning("空召回", extra={"stage": "retrieval", "hits": 0, "session_id": sid})
        yield {"type": "error", "text": "没有检索到相关内容。"}
        METRICS.record("e2e", time.time() - t_start)
        yield {"type": "done"}
        return

    m.top_score.record(max(float(h.get("vec_sim") or 0.0) for h in hits))
    top_files = "、".join((h.get("chunk") or {}).get("file_name", "") for h in hits[:3])
    log.info("检索命中 %d 篇：%s", len(hits), top_files,
             extra={"stage": "retrieval", "session_id": sid, "hits": len(hits)})
    yield {"type": "sources", "hits": [format_hit(h) for h in hits]}
    yield {"type": "status", "text": "生成中…"}

    usage = state.setdefault("usage", {})
    messages = build_messages(question, hits, conversation, state.get("summary", ""))
    m.context_chars.record(sum(len(mm.get("content", "")) for mm in messages))
    prev_comp = usage.get("completion_tokens", 0)
    prev_prompt = usage.get("prompt_tokens", 0)
    t_gen = time.time()
    gen_span = start_stage_span("llm_generate", model=OTEL_MODEL,
                                user_query=question, session_id=sid)
    parts: list[str] = []
    gen_error: Exception | None = None
    try:
        try:
            for delta in call_llm_stream(messages, state["api_key"], usage=usage):
                parts.append(delta)
                yield {"type": "delta", "text": delta}
        except LLMError as e:
            gen_error = e
            if parts:
                # 已向客户端下发过部分增量：不再整段重发（否则前端会重复显示），
                # 直接把已生成内容当作最终回答。
                yield {"type": "status", "text": "流式输出中断，已保留已生成的内容。"}
            else:
                # 尚未产出任何 token 就失败 → 降级一次性调用；再失败则显式报错，不中断 SSE。
                try:
                    text = call_llm(messages, state["api_key"], usage=usage)
                except LLMError as e2:
                    gen_error = e2
                    m.requests_total.add(1, {"status": "error"})
                    yield {"type": "error", "text": f"LLM 生成失败：{e2}"}
                    METRICS.record("e2e", time.time() - t_start)
                    yield {"type": "done"}
                    return
                parts.append(text)
                yield {"type": "delta", "text": text}
    finally:
        comp_tokens = usage.get("completion_tokens", 0) - prev_comp
        prompt_tokens = usage.get("prompt_tokens", 0) - prev_prompt
        gen_span.set_attribute("prompt_tokens", prompt_tokens)
        gen_span.set_attribute("completion_tokens", comp_tokens)
        gen_span.set_attribute("context_chars", sum(len(mm.get("content", "")) for mm in messages))
        METRICS.record("generation", time.time() - t_gen, tokens=comp_tokens)
        m.tokens_total.add(prompt_tokens, {"model": OTEL_MODEL, "type": "prompt"})
        m.tokens_total.add(comp_tokens, {"model": OTEL_MODEL, "type": "completion"})
        m.llm_duration.record((time.time() - t_gen) * 1000, {"model": OTEL_MODEL})
        cost_usd = round((prompt_tokens * COST_PER_1K_PROMPT
                          + comp_tokens * COST_PER_1K_COMPLETION) / 1000.0, 6)
        gen_span.set_attribute("cost_usd", cost_usd)
        m.cost.add(cost_usd, {"model": OTEL_MODEL})
        end_stage_span(gen_span, gen_error)
        if gen_error is not None:
            log.error("LLM 生成异常：%s", gen_error,
                      extra={"stage": "llm_generate", "error_code": type(gen_error).__name__,
                             "session_id": sid, "model": OTEL_MODEL})

    answer = "".join(parts)
    state["conversation"].append({"role": "user", "content": question})
    state["conversation"].append({"role": "assistant", "content": answer})
    maintain_memory(state, state["api_key"])
    write_back_session(session, state)

    # 写回答案缓存（key 用补全后的完整查询，便于后续直接命中）
    if cache is not None and answer:
        cache.set(result.get("completed_query") or question, answer, hits)

    yield {"type": "usage", **{k: int(v) for k, v in usage.items() if k.endswith("tokens")}}
    m.requests_total.add(1, {"status": "ok"})
    METRICS.record("e2e", time.time() - t_start)
    latency_ms = round((time.time() - t_start) * 1000, 1)
    log.info("问答完成（%d tokens / %s）：%s",
             comp_tokens, f"{latency_ms}ms", answer[:60],
             extra={"stage": "e2e", "session_id": sid, "latency_ms": latency_ms,
                    "prompt_tokens": prompt_tokens, "completion_tokens": comp_tokens})
    yield {"type": "done", "answer": answer}


def sse_stream(question: str, sid: str, session: dict):
    """SSE 帧序列；无论正常结束、异常还是客户端中途断开，都落盘会话。"""
    try:
        for ev in ask_events(question, session, sid):
            yield "data: " + json.dumps(ev, ensure_ascii=False) + "\n\n"
    except Exception as e:
        # 兜底：把未预期异常也包装成 error 事件，避免流直接断掉。
        yield "data: " + json.dumps({"type": "error", "text": f"服务端异常：{e}"},
                                    ensure_ascii=False) + "\n\n"
    finally:
        try:
            APP["store"].save(sid, session)
        except Exception as e:
            print(f"[会话] 保存失败：{e}")


# --------------------------------------------------------------------------
# 路由
# --------------------------------------------------------------------------
async def index_page(request: Request):
    sid = request.cookies.get(COOKIE_NAME) or new_session_id()
    return HTMLResponse(HTML_PAGE, headers=set_cookie(sid))


async def healthz(request: Request):
    return JSONResponse({"ok": True, "pid": os.getpid(),
                         "chunks": len(APP["index"].chunks) if APP["index"] else 0})


async def api_metrics(request: Request):
    if not is_authorized(request):
        return unauthorized()
    snap = METRICS.snapshot()
    snap["pid"] = os.getpid()          # 便于确认由哪个 worker 应答
    return JSONResponse(snap)


async def api_reset(request: Request):
    if not is_authorized(request):
        return unauthorized()
    sid = request.cookies.get(COOKIE_NAME)
    if sid:
        await anyio.to_thread.run_sync(APP["store"].reset, sid)
    return JSONResponse({"ok": True})


async def api_ask(request: Request):
    if not is_authorized(request):
        return unauthorized()

    body, status = await read_json_body(request)
    if body is None:
        msg = "payload too large" if status == 413 else "bad request"
        return JSONResponse({"error": msg}, status_code=status)

    question = (body.get("question") or "").strip()
    if not question:
        return JSONResponse({"error": "empty question"}, status_code=400)
    if len(question) > MAX_QUESTION_CHARS:
        return JSONResponse({"error": "question too long"}, status_code=400)

    sid = request.cookies.get(COOKIE_NAME) or new_session_id()
    session = await anyio.to_thread.run_sync(APP["store"].load, sid)
    if session is None:
        session = empty_session()

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",     # 通知 Nginx 不要缓冲 SSE
        **set_cookie(sid),
    }
    return StreamingResponse(sse_stream(question, sid, session),
                             media_type="text/event-stream", headers=headers)


@asynccontextmanager
async def lifespan(app):
    api_key = get_api_key()
    if not api_key:
        raise SystemExit("找不到 DeepSeek API Key（DEEPSEEK_API_KEY 或 ~/.pi/agent/auth.json）")
    print(f"[启动 pid={os.getpid()}] 加载索引…")
    index = load_index()

    def safe_retrieve(q, k=TOP_K):
        # fastembed 并发保护已在 rag_index.encode 内部加锁，此处无需再锁
        return retrieve(index, q, top_k=k)

    APP["index"] = index
    APP["retriever"] = safe_retrieve
    APP["reranker"] = create_reranker()
    APP["api_key"] = api_key
    APP["store"] = create_session_store()
    APP["answer_cache"] = create_answer_cache(INDEX_FINGERPRINT)
    print(f"[启动 pid={os.getpid()}] 就绪：chunks={len(index.chunks)}")
    yield


app = Starlette(
    debug=False,
    lifespan=lifespan,
    routes=[
        Route("/", index_page, methods=["GET"]),
        Route("/healthz", healthz, methods=["GET"]),
        Route("/api/metrics", api_metrics, methods=["GET"]),
        Route("/api/reset", api_reset, methods=["POST"]),
        Route("/api/ask", api_ask, methods=["POST"]),
    ],
)

from otel_setup import instrument_app  # noqa: E402
instrument_app(app)   # 自动埋点：给每个接口建根 span + 自动 http 指标


if __name__ == "__main__":
    import uvicorn

    kwargs: dict = {}
    if TLS_CERT and TLS_KEY:
        kwargs["ssl_certfile"] = TLS_CERT
        kwargs["ssl_keyfile"] = TLS_KEY
    elif TLS_CERT or TLS_KEY:
        print("警告：HTTPS 需同时设置 RAG_WEB_TLS_CERT 与 RAG_WEB_TLS_KEY，当前按 HTTP 启动。")

    uvicorn.run("app:app", host="127.0.0.1",
                port=int(os.environ.get("RAG_WEB_PORT", "8000")), reload=False, **kwargs)
