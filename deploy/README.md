# 部署说明（Qdrant Server + 多 Worker + Redis）

架构：

    Nginx → Gunicorn(Uvicorn Worker) × N ─┬→ Qdrant Server（向量，docker）
                                          └→ Redis（session / memory）

## 1. 装依赖

    .venv/bin/pip install -r requirements.txt

## 2. 启动 Qdrant + Redis（docker）

    docker compose -f deploy/docker-compose.yml up -d

    Qdrant：REST http://127.0.0.1:6333（gRPC 6334），数据存 named volume qdrant_data
    Redis：127.0.0.1:6379，AOF 持久化，数据存 named volume redis_data

## 3. （可选，推荐）离线构建索引到 Qdrant

    export RAG_QDRANT_URL=http://127.0.0.1:6333
    .venv/bin/python rag_index.py --rebuild

    不跑这步也行：worker 启动时发现 collection 不存在会自动构建。

## 4. 启动多 Worker 服务

    export DEEPSEEK_API_KEY=...                      # 或写入 ~/.pi/agent/auth.json
    export RAG_QDRANT_URL=http://127.0.0.1:6333
    export REDIS_URL=redis://127.0.0.1:6379/0
    export RAG_AUTH_TOKEN=...                       # 生产必设：/api/* 鉴权令牌
    .venv/bin/gunicorn -c deploy/gunicorn.conf.py app:app

    验证：

    curl http://127.0.0.1:8000/healthz    # {"ok":true,"pid":12345,"chunks":5}
    curl -H "Authorization: Bearer $RAG_AUTH_TOKEN" http://127.0.0.1:8000/api/metrics

## 5. 配 Nginx

    把 deploy/nginx.conf 放到 /etc/nginx/conf.d/rag.conf，改 server_name 后 reload。

## 6. 上线 checklist

    - HTTPS：certbot 签证书，只暴露 443，80 强制跳转（见 nginx.conf 注释段）；
      也可直接设 RAG_WEB_TLS_CERT/RAG_WEB_TLS_KEY 让 gunicorn/uvicorn 自身提供 TLS
    - 鉴权：设置 RAG_AUTH_TOKEN（应用层，/api/* 401）；需要更细粒度再叠 Nginx basic auth / 网关
    - 请求体：应用层 RAG_MAX_BODY_BYTES（默认 1MB）+ Nginx client_max_body_size 64k 双保险；
      问题长度上限 RAG_MAX_QUESTION_CHARS（默认 2000）
    - Cookie：HTTPS 部署时设 RAG_COOKIE_SECURE=1，会话 Cookie 带 Secure 标记
    - Qdrant：默认无鉴权且仅监听本地，务必不要直接暴露 6333 到公网；生产建议开 API key
    - 会话：RAG_SESSION_TTL 默认 24h；Redis 已开 AOF，生产建议再设 maxmemory
    - 索引更新：改知识库后跑 rag_index.py --rebuild，再滚动重启 worker
    - 监控：/api/metrics 是「单 worker」口径（Nginx 轮询会随机落到某 worker）；
            多 worker 聚合需要上 Prometheus 导出

## 7. 成本与延迟优化相关环境变量（答案缓存 / 查询扩展）

    RAG_ANSWER_CACHE_TTL   答案缓存 TTL 秒数（默认 3600；<=0 禁用缓存）
                           命中后跳过「意图→检索→精排→生成」全链路，直接返回上次答案
                           仅对自包含问题生效（含上下文指代的问题不查缓存，由规则判断）
                           缓存 key 含知识库内容指纹，文档更新后旧缓存自动失效
    RAG_ENABLE_EXPANSION   查询扩展开关（默认 0 关闭）：rewrite 的额外 LLM 调用、
                           expand、HyDE 默认不再执行；设 1 恢复完整扩展能力

    REDIS_URL              答案缓存与会话共用同一 Redis（多 Worker 共享命中）

