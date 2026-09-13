"""RAG 项目专属业务指标。

设计原则：所有 label 必须是低基数的枚举值（模型名、场景名、状态）。
query 文本、doc_id、user_id、session_id 绝不作为 label，它们走 trace attribute 或日志字段。

在 otel_setup.setup_telemetry() 之后通过 rag_metrics() 取用。
"""
from opentelemetry import metrics

_instance = None


class RagMetrics:
    def __init__(self) -> None:
        meter = metrics.get_meter("rag.business")

        # 检索层
        self.search_duration = meter.create_histogram(
            name="rag_search_duration_ms",
            description="检索各阶段耗时，label: stage=embedding/vector_search/rerank",
            unit="ms",
        )
        self.search_hits = meter.create_histogram(
            name="rag_search_hits",
            description="单次检索召回的文档数",
            unit="{doc}",
        )
        self.empty_recall = meter.create_counter(
            name="rag_empty_recall_total",
            description="空召回次数。索引损坏或 embedding 模型被换时会先在这里炸",
        )
        self.top_score = meter.create_histogram(
            name="rag_top_score",
            description="召回文档的最高相似度，质量滑坡的第一预警信号",
        )

        # 生成层
        self.llm_duration = meter.create_histogram(
            name="rag_llm_duration_ms",
            description="模型生成总耗时，label: model",
            unit="ms",
        )
        self.llm_ttft = meter.create_histogram(
            name="rag_llm_ttft_ms",
            description="首 token 延迟，用户体感最直接的指标",
            unit="ms",
        )
        self.tokens_total = meter.create_counter(
            name="rag_tokens_total",
            description="token 消耗量，label: model, type=prompt/completion",
            unit="{token}",
        )
        self.context_truncated = meter.create_counter(
            name="rag_context_truncated_total",
            description="上下文超长被截断的次数。持续上升说明答案质量在无声下降",
        )
        self.context_chars = meter.create_histogram(
            name="rag_context_chars",
            description="送入模型的上下文字符数",
        )

        # 服务层
        self.requests_total = meter.create_counter(
            name="rag_requests_total",
            description="问答请求总数，label: status=ok/error",
        )
        self.active_requests = meter.create_up_down_counter(
            name="rag_active_requests",
            description="当前正在处理的请求数，即饱和度",
        )

        # 质量/行为层
        self.json_parse_fail = meter.create_counter(
            name="rag_json_parse_fail_total",
            description="意图/查询 JSON 解析失败次数（触发文本兜底）",
        )
        self.clarify = meter.create_counter(
            name="rag_clarify_total",
            description="澄清反问次数",
        )
        self.intent_type = meter.create_counter(
            name="rag_intent_type_total",
            description="意图类型分布，label: type=question_type",
        )
        self.query_len = meter.create_histogram(
            name="rag_query_len_chars",
            description="问题长度分布",
            unit="char",
        )

        # 缓存层
        self.cache_total = meter.create_counter(
            name="rag_cache_total",
            description="答案缓存查询次数，label: hit=yes/no",
        )

        # 成本
        self.cost = meter.create_counter(
            name="rag_cost_total",
            description="累计成本(USD)，label: model",
            unit="USD",
        )


def rag_metrics() -> RagMetrics:
    """懒加载单例，保证在 Provider 设置完成之后才创建指标。"""
    global _instance
    if _instance is None:
        _instance = RagMetrics()
    return _instance
