#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
webapp.py — RAG 问答 Web 前端（零依赖，标准库 http.server + SSE 流式输出）

复用 rag_tool.py / query_intent.py / memory.py / reranker.py 的既有能力，
浏览器里输入问题 → 意图分析 → 混合检索 → 精排 → DeepSeek 流式回答 + 出处展示。

启动：
    .venv/bin/python webapp.py            # 默认 http://127.0.0.1:8000
    RAG_WEB_PORT=9000 .venv/bin/python webapp.py

可选环境变量：
    RAG_WEB_HOST      监听地址（默认 127.0.0.1；对外部署设 0.0.0.0，前面加 nginx）
    RAG_WEB_PORT      监听端口（默认 8000）
    RAG_WEB_REBUILD   置 1 启动时强制重建索引（默认用已有索引）
    RAG_AUTH_TOKEN    访问令牌；设置后 /api/* 需带 `Authorization: Bearer <token>`
                      或 `X-Auth-Token: <token>` 头，否则 401（不设置则不鉴权，仅限本机开发）
    RAG_WEB_TLS_CERT / RAG_WEB_TLS_KEY   TLS 证书与私钥路径，同时设置则启用 HTTPS
    RAG_SESSION_TTL   会话空闲过期秒数（默认 3600；0 表示不过期）

会话与并发说明：
    - 每次请求按 `X-Session-Id` 头隔离会话（前端自动生成并存 localStorage），
      不同浏览器/用户互不串话；未带会话头时使用一次性临时会话（无历史）。
    - 检索资源（索引/嵌入/reranker）进程内只读共享，仅检索阶段短暂加锁；
      LLM 流式生成与精排不持锁，多个请求可并发。
"""

import hmac
import json
import os
import ssl
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from llm import LLMError, call_llm, call_llm_stream, get_api_key
from metrics import METRICS
from memory import maintain_memory
from query_intent import describe_intent, has_reference, retrieve_with_intent
from rag_index import INDEX_FINGERPRINT, TOP_K, build_index, load_index, retrieve
from rag_tool import build_messages
from reranker import CloudReranker, create_reranker
from answer_cache import create_answer_cache

HOST = os.environ.get("RAG_WEB_HOST", "127.0.0.1")
PORT = int(os.environ.get("RAG_WEB_PORT", "8000"))
FORCE_REBUILD = os.environ.get("RAG_WEB_REBUILD", "").strip().lower() in ("1", "true", "yes")
AUTH_TOKEN = os.environ.get("RAG_AUTH_TOKEN", "").strip()
TLS_CERT = os.environ.get("RAG_WEB_TLS_CERT", "").strip()
TLS_KEY = os.environ.get("RAG_WEB_TLS_KEY", "").strip()
SESSION_TTL = int(os.environ.get("RAG_SESSION_TTL", "3600") or "0")

MAX_BODY_BYTES = int(os.environ.get("RAG_MAX_BODY_BYTES", str(1024 * 1024)))   # 1MB
MAX_QUESTION_CHARS = int(os.environ.get("RAG_MAX_QUESTION_CHARS", "2000"))

# ---------------- 共享只读检索资源 + 每会话状态 ----------------
# SHARED：进程启动时加载一次，所有会话/请求共享（只读，检索阶段短暂加锁）。
# SESSIONS：session_id -> {conversation, summary, folded, usage}，按会话隔离。
SHARED: dict = {}
SESSIONS: dict = {}
_sessions_lock = threading.Lock()     # 保护 SESSIONS 的读写


def build_shared() -> dict:
    """加载索引、组装检索器（进程内只读共享，不含会话状态）。"""
    api_key = get_api_key()
    if not api_key:
        sys.exit("找不到 DeepSeek API Key。请设置 DEEPSEEK_API_KEY，"
                 "或把 key 放到 ~/.pi/agent/auth.json。")
    print("正在加载索引…")
    index = build_index() if FORCE_REBUILD else load_index()
    reranker = create_reranker()
    answer_cache = create_answer_cache(INDEX_FINGERPRINT)

    def _retriever(q, k=TOP_K):
        # 向量检索走 Qdrant 服务端/内存，BM25 本地计算。
        # fastembed 并发保护已在 rag_index.encode 内部加锁，此处无需再锁。
        return retrieve(index, q, top_k=k)

    return {
        "index": index,
        "chunks": index.chunks,
        "api_key": api_key,
        "retriever": _retriever,
        "reranker": reranker,
        "answer_cache": answer_cache,
    }


def new_session() -> dict:
    return {"conversation": [], "summary": "", "folded": 0, "usage": {},
            "_atime": time.time()}


def get_session(sid: str) -> dict:
    """按会话 id 取（或建）会话；带惰性过期清理；无 id 返回一次性临时会话。"""
    now = time.time()
    with _sessions_lock:
        if SESSION_TTL > 0:
            for k in list(SESSIONS):
                if now - SESSIONS[k].get("_atime", now) > SESSION_TTL:
                    del SESSIONS[k]
        if not sid:
            return new_session()
        s = SESSIONS.get(sid)
        if s is None:
            s = new_session()
            SESSIONS[sid] = s
        s["_atime"] = now
        return s


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


def ask_events(question: str, session: dict, shared: dict):
    """生成 SSE 事件序列：status → intent → sources → delta… → usage → done。

    只在成功生成回答后把本轮问答写入 session["conversation"] 并维护记忆。
    session 为当前请求专属（per-session），shared 为只读共享资源。
    """
    t_start = time.time()
    conversation = session.get("conversation", [])
    summary = session.get("summary", "")
    cache = shared.get("answer_cache")

    # 答案缓存：自包含问题命中则直接返回，跳过意图/检索/精排/生成
    if cache is not None and not has_reference(question):
        cached = cache.get(question)
        if cached is not None:
            yield {"type": "intent", "text": "⚡ 缓存命中", "completed_query": question}
            yield {"type": "sources",
                   "hits": [format_hit(h) for h in cached.get("hits", [])]}
            answer = cached.get("answer", "")
            yield {"type": "delta", "text": answer}
            with _sessions_lock:
                session["conversation"].append({"role": "user", "content": question})
                session["conversation"].append({"role": "assistant", "content": answer})
            maintain_memory(session, shared["api_key"])
            with _sessions_lock:
                totals = {k: int(v) for k, v in session["usage"].items()
                          if k.endswith("tokens")}
            yield {"type": "usage", **totals}
            METRICS.record("e2e", time.time() - t_start)
            yield {"type": "done", "answer": answer}
            return

    yield {"type": "status", "text": "意图分析中…"}

    state = dict(shared)
    state["conversation"] = conversation
    state["summary"] = summary

    try:
        result = retrieve_with_intent(
            question, state,
            retriever=shared["retriever"],
            base_top_k=TOP_K,
            conversation=conversation,
            metrics=METRICS,
        )
    except Exception as e:
        yield {"type": "error", "text": f"检索阶段失败：{e}"}
        METRICS.record("e2e", time.time() - t_start)
        return

    intent = result["intent"]
    yield {"type": "intent",
           "text": describe_intent(intent).replace("\n", "｜"),
           "completed_query": result.get("completed_query", "")}

    if result["action"] == "clarify":
        yield {"type": "clarify", "questions": result["questions"]}
        METRICS.record("e2e", time.time() - t_start)
        yield {"type": "done"}
        return

    hits = result["hits"]
    if not hits:
        yield {"type": "error", "text": "没有检索到相关内容。"}
        METRICS.record("e2e", time.time() - t_start)
        yield {"type": "done"}
        return

    yield {"type": "sources", "hits": [format_hit(h) for h in hits]}
    yield {"type": "status", "text": "生成中…"}

    # 局部 usage，流式结束后再合并回 session（避免并发写同一 session 的累计值）
    usage_local = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    messages = build_messages(question, hits, conversation, summary)
    prev_completion = usage_local.get("completion_tokens", 0)
    t_gen = time.time()
    answer_parts: list[str] = []
    try:
        for delta in call_llm_stream(messages, shared["api_key"], usage=usage_local):
            answer_parts.append(delta)
            yield {"type": "delta", "text": delta}
    except LLMError:
        if answer_parts:
            # 已下发过部分增量：不再整段重发（避免前端重复显示），以已生成内容为准。
            yield {"type": "status", "text": "流式输出中断，已保留已生成的内容。"}
        else:
            # 尚未产出任何 token 就失败 → 降级一次性调用；再失败则显式报错。
            try:
                text = call_llm(messages, shared["api_key"], usage=usage_local)
            except LLMError as e:
                yield {"type": "error", "text": f"LLM 生成失败：{e}"}
                METRICS.record("e2e", time.time() - t_start)
                yield {"type": "done"}
                return
            answer_parts.append(text)
            yield {"type": "delta", "text": text}
    gen_sec = time.time() - t_gen
    comp_tokens = usage_local.get("completion_tokens", 0) - prev_completion
    METRICS.record("generation", gen_sec, tokens=comp_tokens)

    answer = "".join(answer_parts)
    with _sessions_lock:
        session["conversation"].append({"role": "user", "content": question})
        session["conversation"].append({"role": "assistant", "content": answer})
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            session["usage"][k] = session["usage"].get(k, 0) + usage_local.get(k, 0)
    maintain_memory(session, shared["api_key"])

    # 写回答案缓存（key 用补全后的完整查询，便于后续直接命中）
    if cache is not None and answer:
        cache.set(result.get("completed_query") or question, answer, hits)

    with _sessions_lock:
        totals = {k: int(v) for k, v in session["usage"].items() if k.endswith("tokens")}
    yield {"type": "usage", **totals}
    METRICS.record("e2e", time.time() - t_start)
    yield {"type": "done", "answer": answer}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "RagWeb/1.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (self.address_string(), fmt % args))

    # ---------------- 鉴权 ----------------
    def _authorized(self) -> bool:
        if not AUTH_TOKEN:
            return True
        auth = self.headers.get("Authorization", "")
        token = auth[7:].strip() if auth.startswith("Bearer ") else \
            self.headers.get("X-Auth-Token", "")
        return hmac.compare_digest(token, AUTH_TOKEN)

    # ---------------- 基础响应 ----------------
    def _send(self, status: int, content_type: str, body: bytes, extra: dict | None = None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _sse_write(self, ev: dict) -> None:
        payload = json.dumps(ev, ensure_ascii=False).encode("utf-8")
        data = b"data: " + payload + b"\n\n"
        # 手动 chunked 帧，保证 SSE 逐条即时下发
        self.wfile.write(("%X\r\n" % len(data)).encode() + data + b"\r\n")
        self.wfile.flush()

    # ---------------- 路由 ----------------
    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", HTML_PAGE.encode("utf-8"))
        elif path == "/api/metrics":
            if not self._authorized():
                self._send(401, "application/json; charset=utf-8",
                           json.dumps({"error": "unauthorized"}).encode("utf-8"))
                return
            self._send(200, "application/json; charset=utf-8",
                       json.dumps(METRICS.snapshot(), ensure_ascii=False).encode("utf-8"))
        else:
            self._send(404, "text/plain; charset=utf-8", b"not found")

    def do_POST(self):
        path = urlparse(self.path).path

        if not self._authorized():
            self._send(401, "application/json; charset=utf-8",
                       json.dumps({"error": "unauthorized"}).encode("utf-8"))
            return

        if path == "/api/reset":
            sid = self.headers.get("X-Session-Id", "").strip()
            with _sessions_lock:
                SESSIONS.pop(sid, None)
            self._send(200, "application/json; charset=utf-8",
                       json.dumps({"ok": True}).encode("utf-8"))
            return

        if path != "/api/ask":
            self._send(404, "text/plain; charset=utf-8", b"not found")
            return

        # 读请求体（带大小上限，防止超大 body 打爆内存）
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
        except ValueError:
            length = 0
        if length > MAX_BODY_BYTES:
            self._send(413, "application/json; charset=utf-8",
                       json.dumps({"error": "payload too large"}).encode("utf-8"))
            return
        try:
            raw = self.rfile.read(length) if length > 0 else b""
        except Exception:
            self._send(400, "application/json; charset=utf-8",
                       json.dumps({"error": "bad request"}).encode("utf-8"))
            return
        try:
            req = json.loads(raw.decode("utf-8") or "{}")
            question = (req.get("question") or "").strip()
        except Exception:
            self._send(400, "application/json; charset=utf-8",
                       json.dumps({"error": "bad request"}).encode("utf-8"))
            return
        if not question:
            self._send(400, "application/json; charset=utf-8",
                       json.dumps({"error": "empty question"}).encode("utf-8"))
            return
        if len(question) > MAX_QUESTION_CHARS:
            self._send(400, "application/json; charset=utf-8",
                       json.dumps({"error": "question too long"}).encode("utf-8"))
            return

        sid = self.headers.get("X-Session-Id", "").strip()
        session = get_session(sid)

        # SSE 流式响应（chunked，逐事件下发）
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        try:
            for ev in ask_events(question, session, SHARED):
                self._sse_write(ev)
        except Exception as e:
            self._sse_write({"type": "error", "text": f"服务端异常：{e}"})
        finally:
            self.wfile.write(b"0\r\n\r\n")   # chunked 结束帧
            self.wfile.flush()


HTML_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>笔记问答</title>
<style>
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body {
    margin: 0; height: 100vh; display: flex; flex-direction: column;
    font: 15px/1.6 -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
    background: #f5f6f8; color: #1f2328;
  }
  header {
    padding: 12px 20px; background: #fff; border-bottom: 1px solid #e3e6ea;
    display: flex; align-items: center; gap: 12px;
  }
  header h1 { font-size: 17px; margin: 0; }
  header .spacer { flex: 1; }
  #status { font-size: 13px; color: #6b7280; }
  button {
    padding: 6px 14px; border: 1px solid #d0d4da; background: #fff;
    border-radius: 8px; cursor: pointer; font-size: 13px;
  }
  button:hover { background: #f0f1f3; }
  #chat {
    flex: 1; overflow-y: auto; padding: 20px; max-width: 860px;
    width: 100%; margin: 0 auto;
  }
  .msg { display: flex; margin-bottom: 16px; }
  .msg.user { justify-content: flex-end; }
  .bubble {
    max-width: 78%; padding: 10px 14px; border-radius: 14px;
    white-space: pre-wrap; word-break: break-word;
  }
  .msg.user .bubble { background: #1677ff; color: #fff; border-bottom-right-radius: 4px; }
  .msg.assistant .bubble { background: #fff; border: 1px solid #e3e6ea; border-bottom-left-radius: 4px; }
  .meta { font-size: 12px; color: #9aa1a9; margin: 2px 2px 6px; }
  .meta.intent { color: #6b7280; }
  .sources { margin-top: 10px; font-size: 13px; }
  .src {
    background: #f7f8fa; border: 1px solid #e6e9ed; border-radius: 8px;
    padding: 8px 10px; margin-bottom: 6px;
  }
  .src summary { cursor: pointer; color: #57606a; }
  .src .sc { color: #9aa1a9; font-size: 12px; }
  .src .full { color: #333; white-space: pre-wrap; margin-top: 6px; border-top: 1px dashed #e0e3e8; padding-top: 6px; }
  .clarify { color: #b45309; background: #fff7ed; border: 1px solid #fed7aa; padding: 10px 12px; border-radius: 8px; }
  footer {
    padding: 12px 20px; background: #fff; border-top: 1px solid #e3e6ea;
    display: flex; gap: 10px; align-items: flex-end;
    max-width: 900px; width: 100%; margin: 0 auto;
  }
  textarea {
    flex: 1; resize: none; height: 46px; padding: 12px 14px; font: inherit;
    border: 1px solid #d0d4da; border-radius: 10px; outline: none;
  }
  textarea:focus { border-color: #1677ff; }
  #send { height: 46px; padding: 0 22px; background: #1677ff; color: #fff; border: none; }
  #send:disabled { background: #a8c6f0; cursor: not-allowed; }
  .err { color: #c62828; }
  .usage { font-size: 12px; color: #9aa1a9; margin-top: 6px; }
  #metricsPanel {
    max-width: 860px; width: 100%; margin: 12px auto 0; background: #fff;
    border: 1px solid #e3e6ea; border-radius: 10px; padding: 14px 16px; font-size: 13px;
  }
  #metricsPanel table { border-collapse: collapse; width: 100%; margin-top: 8px; }
  #metricsPanel th, #metricsPanel td { text-align: right; padding: 4px 8px; border-bottom: 1px solid #f0f1f3; }
  #metricsPanel th:first-child, #metricsPanel td:first-child { text-align: left; }
  #metricsPanel .qps { margin-top: 8px; color: #57606a; }
</style>
</head>
<body>
<header>
  <h1>📝 笔记问答</h1>
  <span id="status"></span>
  <span class="spacer"></span>
  <button id="metrics" title="查看延迟与 QPS">指标</button>
  <button id="clear" title="清空对话与记忆">清空对话</button>
</header>
<div id="metricsPanel" hidden></div>
<main id="chat"></main>
<footer>
  <textarea id="q" placeholder="输入问题，Enter 发送，Shift+Enter 换行"></textarea>
  <button id="send">发送</button>
</footer>

<script>
const chat = document.getElementById('chat');
const input = document.getElementById('q');
const sendBtn = document.getElementById('send');
const statusEl = document.getElementById('status');
const clearBtn = document.getElementById('clear');
const metricsBtn = document.getElementById('metrics');
let busy = false;

// 会话隔离：每个浏览器生成一个随机 session id，随请求带上，互不串话
let sessionId = localStorage.getItem('rag_sid');
if (!sessionId) {
  sessionId = (typeof crypto !== 'undefined' && crypto.randomUUID)
    ? crypto.randomUUID()
    : String(Date.now()) + Math.random().toString(16).slice(2);
  localStorage.setItem('rag_sid', sessionId);
}

function addMsg(role) {
  const row = document.createElement('div');
  row.className = 'msg ' + role;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  row.appendChild(bubble);
  chat.appendChild(row);
  chat.scrollTop = chat.scrollHeight;
  return bubble;
}

function esc(s) {
  return String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}

function renderSources(hits) {
  const box = document.createElement('div');
  box.className = 'sources';
  hits.forEach((h, i) => {
    const d = document.createElement('details');
    d.className = 'src';
    const sc = [];
    if (h.rerank != null) sc.push('精排 ' + h.rerank.toFixed(3));
    sc.push('RRF ' + (h.rrf||0).toFixed(3));
    sc.push('vec ' + (h.vec_sim||0).toFixed(3));
    sc.push('BM25 ' + (h.bm25||0).toFixed(2));
    const origins = (h.origins||[]).join(',') || '-';
    d.innerHTML = '<summary>[' + (i+1) + '] ' + esc(h.file) +
      ' <span class="sc">' + esc(sc.join(' · ')) + ' · ' + esc(origins) + '</span></summary>' +
      '<div class="full">' + esc(h.snippet) + '…<br>（' + esc(h.text.length) + ' 字）</div>';
    box.appendChild(d);
  });
  return box;
}

function setStatus(t) { statusEl.textContent = t; }

async function ask(question) {
  busy = true; sendBtn.disabled = true;
  const userBubble = addMsg('user');
  userBubble.textContent = question;
  const asBubble = addMsg('assistant');
  let answerText = '';
  let sourcesBox = null;
  let metaBox = document.createElement('div');
  metaBox.className = 'meta';
  asBubble.appendChild(metaBox);

  try {
    const resp = await fetch('/api/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json', 'X-Session-Id': sessionId},
      body: JSON.stringify({question}),
    });
    if (!resp.ok || !resp.body) {
      asBubble.innerHTML = '<span class="err">请求失败：HTTP ' + resp.status + '</span>';
      return;
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    for (;;) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += decoder.decode(value, {stream: true});
      let idx;
      while ((idx = buf.indexOf('\n\n')) >= 0) {
        const raw = buf.slice(0, idx); buf = buf.slice(idx + 2);
        const line = raw.split('\n').find(l => l.startsWith('data: '));
        if (!line) continue;
        let ev;
        try { ev = JSON.parse(line.slice(6)); } catch { continue; }
        handleEvent(ev);
      }
    }
  } catch (e) {
    asBubble.innerHTML = '<span class="err">网络错误：' + esc(e.message) + '</span>';
  } finally {
    busy = false; sendBtn.disabled = false;
    setStatus('');
    input.focus();
    chat.scrollTop = chat.scrollHeight;
  }

  function handleEvent(ev) {
    switch (ev.type) {
      case 'status':
        setStatus(ev.text);
        metaBox.textContent = ev.text;
        break;
      case 'intent':
        metaBox.className = 'meta intent';
        metaBox.textContent = ev.text;
        break;
      case 'clarify':
        const c = document.createElement('div');
        c.className = 'clarify';
        c.textContent = '这个问题缺少关键信息，请补充：\n' +
          (ev.questions||[]).map((q,i) => (i+1) + '. ' + q).join('\n');
        asBubble.appendChild(c);
        metaBox.textContent = '';
        break;
      case 'sources':
        sourcesBox = renderSources(ev.hits);
        break;
      case 'delta':
        if (!answerText) { // 首个增量：先挂出处，再写正文
          if (sourcesBox) asBubble.appendChild(sourcesBox);
          answerText = document.createElement('span');
          asBubble.appendChild(answerText);
        }
        answerText.textContent += ev.text;
        chat.scrollTop = chat.scrollHeight;
        break;
      case 'usage':
        const u = document.createElement('div');
        u.className = 'usage';
        u.textContent = '本轮累计 tokens：prompt=' + (ev.prompt_tokens||0) +
          ' completion=' + (ev.completion_tokens||0) + ' total=' + (ev.total_tokens||0);
        asBubble.appendChild(u);
        break;
      case 'error':
        const e = document.createElement('div');
        e.className = 'err';
        e.textContent = ev.text;
        asBubble.appendChild(e);
        break;
    }
  }
}

function submit() {
  const q = input.value.trim();
  if (!q || busy) return;
  input.value = '';
  ask(q);
}

sendBtn.addEventListener('click', submit);
input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); }
});
clearBtn.addEventListener('click', async () => {
  await fetch('/api/reset', {method: 'POST', headers: {'X-Session-Id': sessionId}});
  chat.innerHTML = '';
  statusEl.textContent = '';
});

metricsBtn.addEventListener('click', showMetrics);
async function showMetrics() {
  const panel = document.getElementById('metricsPanel');
  if (!panel.hidden) { panel.hidden = true; return; }
  try {
    const r = await fetch('/api/metrics', {headers: {'X-Session-Id': sessionId}});
    if (!r.ok) { console.error('metrics 请求失败：HTTP ' + r.status); return; }
    const m = await r.json();
    let html = '<b>性能指标</b>（uptime ' + m.uptime_seconds + 's）';
    html += '<table><tr><th>阶段</th><th>count</th><th>avg(ms)</th><th>p50</th><th>p90</th><th>p99</th><th>max</th></tr>';
    for (const name of ['intent_llm', 'retrieval', 'rerank', 'generation', 'e2e']) {
      const s = m.stages[name];
      if (!s) continue;
      html += '<tr><td>' + name + '</td><td>' + s.count + '</td><td>' + s.avg_ms +
        '</td><td>' + s.p50_ms + '</td><td>' + s.p90_ms + '</td><td>' + s.p99_ms +
        '</td><td>' + s.max_ms + '</td></tr>';
    }
    html += '</table>';
    const q = m.qps;
    const winKey = Object.keys(q).find(k => k.startsWith('window_')) || '';
    html += '<div class="qps">QPS：累计 ' + q.cumulative +
      (winKey ? ' · ' + winKey + ' ' + q[winKey] : '') +
      ' · 总请求 ' + q.total_requests + '</div>';
    const g = m.stages.generation;
    if (g && g.tokens_per_sec) {
      html += '<div class="qps">生成吞吐：' + g.tokens_per_sec + ' tokens/s（累计 ' + g.total_tokens + ' tokens）</div>';
    }
    panel.innerHTML = html;
    panel.hidden = false;
  } catch (e) { console.error(e); }
}
</script>
</body>
</html>
"""


def main() -> None:
    global SHARED
    SHARED = build_shared()
    rr = SHARED.get("reranker")
    backend = "云端" if isinstance(rr, CloudReranker) else ("本地" if rr is not None else "-")

    scheme = "http"
    if TLS_CERT and TLS_KEY:
        scheme = "https"
    elif TLS_CERT or TLS_KEY:
        print("警告：TLS 需同时设置 RAG_WEB_TLS_CERT 与 RAG_WEB_TLS_KEY，当前按 HTTP 启动。")

    print("=" * 60)
    print(f"笔记问答 Web 前端已启动：{scheme}://{HOST}:{PORT}")
    print(f"精排（Qwen3-Reranker · {backend}）：{'开' if rr is not None else '关'}")
    print(f"鉴权：{'开启（Bearer token）' if AUTH_TOKEN else '关闭（仅限本机开发，生产请设 RAG_AUTH_TOKEN）'}")
    print(f"会话隔离：按 X-Session-Id 隔离，空闲 {SESSION_TTL}s 过期" if SESSION_TTL > 0
          else "会话隔离：按 X-Session-Id 隔离，不过期")
    print("按 Ctrl+C 退出。")
    print("=" * 60)

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    if scheme == "https":
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(TLS_CERT, TLS_KEY)
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")


if __name__ == "__main__":
    main()
