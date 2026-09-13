import os

# 监听地址（Nginx 反代到此；与 deploy/nginx.conf 的 upstream 一致）
bind = os.environ.get("RAG_BIND", "127.0.0.1:8000")

# HTTPS（可选）：同时设置 RAG_WEB_TLS_CERT 与 RAG_WEB_TLS_KEY 时，gunicorn 直接对外提供 TLS；
# 生产更推荐由 Nginx 统一终止 TLS（见 deploy/nginx.conf），此时无需设置这两项。
_cert = os.environ.get("RAG_WEB_TLS_CERT", "").strip()
_key = os.environ.get("RAG_WEB_TLS_KEY", "").strip()
if _cert and _key:
    certfile = _cert
    keyfile = _key

# Worker 数：每个 worker 独立加载一份内存索引 + bge 模型，
# 内存占用 ≈ workers × (索引向量 + 模型权重)。按机器内存调。
workers = int(os.environ.get("RAG_WORKERS", "3"))

# ASGI worker（支持 SSE 流式）
worker_class = "uvicorn.workers.UvicornWorker"

# SSE 长连接：生成回答期间不超时
timeout = int(os.environ.get("RAG_TIMEOUT", "300"))
graceful_timeout = 30
keepalive = 5

# 不 preload：每个 worker 各自 build_index()，
# 避免 fastembed/onnxruntime 模型在 fork 后共享引发的问题。
preload_app = False

accesslog = "-"
errorlog = "-"
loglevel = "info"
