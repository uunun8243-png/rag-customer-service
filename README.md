# RAG 智能客服问答系统（售后知识库）

面向企业售后场景的 RAG（检索增强生成）问答系统：从「脏 PDF 数据清洗 → 结构切块 → 混合检索 → 精排 → 流式问答」，到「多轮记忆、三层评估、可观测性」，覆盖完整工程链路。默认以 `XX商城售后规则` 为演示知识库。

## 核心能力

- **意图识别与检索策略**：一次结构化 LLM 调用完成 7 类问题类型（定义/事实/流程/比较/总结/政策/排查）、实体与槽位抽取、6 种检索策略（直接/改写/扩展/HyDE/分解/澄清）联合决策；高置信度简单查询走规则快速通道，跳过 LLM 以省成本延迟。
- **两阶段检索**：一阶段 BM25（jieba 分词）+ Dense 向量（bge-small-zh-v1.5，本地 fastembed 离线运行）RRF 融合召回；二阶段 Qwen3-Reranker Cross-Encoder 精排，仅精排 RRF 头部候选，失败自动回退 RRF 排序。
- **多轮记忆**：近期对话窗口 + 早期对话滚动摘要，喂给 LLM 的上下文与 session 落盘体积均有界，长对话不丢关键实体。
- **可插拔 LLM Provider**：`LLMProvider` 工厂 + `llm_providers.yaml`，支持 DeepSeek / DashScope / 硅基流动 / OpenAI 兼容端点 4 类 Provider 配置化切换。
- **多后端设计**：向量库 Qdrant 服务端 + 内存 numpy 兜底；会话与答案缓存 Redis + 进程内兜底；精排云端 API + 本地 torch/transformers 推理。
- **三层评估**：检索侧 Hit/MRR/Recall/Precision 横向对比多切块策略；生成侧 LLM-as-Judge 判忠实性与正确性；端到端 Ragas 打 5 项指标——三者复用生产链路，保证「评测的就是上线的」。
- **可观测性**：OpenTelemetry 统一 Trace / Metrics / Logs，按意图/检索/精排/生成/e2e 分阶段埋点，聚合 P50/P90/P99、QPS、Token 吞吐、TTFT 与成本。
- **成本优化**：规则优先意图、答案缓存（知识库指纹自动失效）、多轮「补全 + 意图」合并单次调用。

## 架构

```
用户问题
   │
   ▼
意图识别（7 类）+ 实体/槽位 + 检索策略（规则快速通道优先）
   │
   ├─ 澄清反问（缺关键信息）
   └─ 多路检索变体 ──► BM25 + Dense 向量 ──► RRF 融合
                                              │
                                              ▼
                                     槽位软过滤 ──► Qwen3-Reranker 精排（失败回退 RRF）
                                              │
                                              ▼
                                  DeepSeek 生成（带出处）── SSE 流式返回
                                              │
                                              ▼
                            多轮记忆（滚动窗口 + 滚动摘要）+ 答案缓存
```

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
export DEEPSEEK_API_KEY=...          # 主 LLM
# 或切换 Provider：
export RAG_LLM_PROVIDER=dashscope    # deepseek | dashscope | siliconflow | openai
# 精排（可选，未配置则自动禁用精排、只用 RRF）：
export RAG_RERANK_API_KEY=...        # 或 DASHSCOPE_API_KEY / SILICONFLOW_API_KEY
```

### 3. 命令行问答（单进程，内存索引）

```bash
.venv/bin/python rag_tool.py "普通商品的退货期限是多久？"   # 单次提问
.venv/bin/python rag_tool.py                                # 交互模式
.venv/bin/python rag_tool.py --rebuild                      # 重建索引
```

### 4. Web 服务（多 Worker + Redis + Qdrant，SSE 流式）

```bash
docker compose -f deploy/docker-compose.yml up -d    # 启动 Qdrant + Redis
export RAG_QDRANT_URL=http://127.0.0.1:6333
export REDIS_URL=redis://127.0.0.1:6379/0
.venv/bin/gunicorn -c deploy/gunicorn.conf.py app:app
# 前端：http://127.0.0.1:8000/
```

详见 [deploy/README.md](deploy/README.md)（Nginx、HTTPS、鉴权、上线 checklist）。

## 评估

```bash
.venv/bin/python eval_retrieval.py      # 检索：Hit/MRR/Recall/Precision，多切块策略 × 向量/BM25/RRF
.venv/bin/python eval_generation.py     # 生成：忠实性 / 正确性（LLM-as-Judge）
.venv/bin/python eval_ragas.py          # 端到端：Ragas 5 项指标
.venv/bin/python bench.py               # 压测：分阶段延迟分布 + QPS + 生成吞吐
```

评估集：`golden_test_set.json`（20 条真实查询 + 标注出处）。

## 主要环境变量

| 变量 | 说明 | 默认 |
|---|---|---|
| `RAG_LLM_PROVIDER` | LLM Provider（deepseek/dashscope/siliconflow/openai） | deepseek |
| `RAG_LLM_MODEL` / `RAG_LLM_BASE_URL` / `RAG_LLM_API_KEY` | Provider 覆盖项 | 见 `llm_providers.yaml` |
| `RAG_QDRANT_URL` | Qdrant 服务端地址（未设置则内存 numpy 兜底） | 空 |
| `REDIS_URL` | Redis（会话/缓存，未设置则进程内兜底） | redis://127.0.0.1:6379/0 |
| `RAG_RERANK_PROVIDER` | 精排后端 auto/dashscope/siliconflow/openai/local | auto |
| `RAG_NOTES_DIR` | 知识库目录 | notes/ |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP_RATIO` | 切块参数 | 500 / 0.20 |
| `RAG_ANSWER_CACHE_TTL` | 答案缓存 TTL 秒（<=0 禁用） | 3600 |
| `RAG_AUTH_TOKEN` | 设置后 /api/* 需鉴权 | 空 |

## 项目结构

```
app.py / webapp.py       Web 服务（Starlette SSE；webapp 为旧单进程版）
rag_tool.py              命令行问答入口
query_intent.py          意图识别 + 实体槽位 + 检索策略 + 上下文补全
rag_index.py             索引与混合检索（BM25 + Dense + RRF，Qdrant/内存）
reranker.py              Qwen3-Reranker 精排（云端/本地多后端）
chunker.py               结构切块器（评估与上线共用同一份）
memory.py                滚动窗口 + 滚动摘要记忆
llm.py                   多 Provider LLM 封装（重试/JSON/流式/用量）
llm_providers.yaml       LLM Provider 配置
answer_cache.py          答案缓存（Redis/内存）
session_store.py         会话存储（Redis/内存）
clean_md.py              脏 Markdown 规则清洗器
metrics.py               进程内延迟分位数 / QPS 聚合
otel_setup.py            OpenTelemetry 初始化（Trace/Metrics/Logs）
rag_spans.py / rag_metrics.py   链路 span 与业务指标
eval_retrieval.py / eval_generation.py / eval_ragas.py / gen_testset.py / bench.py   评估与压测
deploy/                  docker-compose / nginx / gunicorn 配置
notes/                   知识库（clean.md）
数据清洗/                演示用脏 PDF 与清洗产物
```
