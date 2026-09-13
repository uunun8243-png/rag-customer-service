"""RAG 链路埋点：把一次问答拆成可观测的若干阶段。

用法：
    with stage("vector_search", top_k=5, collection="policy_docs") as span:
        docs = await retriever.search(question)
        span.set_attribute("hits", len(docs))

产出的一条链路长这样：
    user_request
     ├─ embedding
     ├─ vector_search   top_k=5, hits=5
     ├─ rerank
     └─ llm_generate    model=deepseek-v4-pro, prompt_tokens=2400
"""
import time
from contextlib import contextmanager

from opentelemetry import trace

tracer = trace.get_tracer("rag.pipeline")


@contextmanager
def stage(name: str, **attributes):
    with tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, value)

        started = time.perf_counter()
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
            raise
        finally:
            span.set_attribute("duration_ms", round((time.perf_counter() - started) * 1000, 2))


def start_stage_span(name: str, **attributes):
    """手动开启一个 span（用于跨 yield 的生成器场景，如 SSE 流式生成）。

    与 stage() 不同：不依赖 with 上下文（with 跨 yield 时，OpenTelemetry 的
    contextvar 在生成器每次 resume 于不同线程时会丢失当前 span）。父 span 取
    调用时的当前上下文，调用方负责在结束时调 end_stage_span()。
    """
    span = tracer.start_span(name)
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, value)
    span._started = time.perf_counter()
    return span


def end_stage_span(span, exc: Exception | None = None) -> None:
    """结束由 start_stage_span() 开启的 span，记录耗时与异常状态。"""
    if exc is not None:
        span.record_exception(exc)
        span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
    span.set_attribute("duration_ms", round((time.perf_counter() - span._started) * 1000, 2))
    span.end()


def add_span_attributes(**attributes) -> None:
    """给当前 span 补属性。注意：高基数的值只放这里，不要放 metric 的 label。"""
    span = trace.get_current_span()
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, value)


def current_trace_id() -> str:
    """取当前 trace_id，用于回写进业务日志或返回给前端排查。"""
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else ""
