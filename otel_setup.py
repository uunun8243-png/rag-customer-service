"""OpenTelemetry 初始化：日志 / 链路 / 指标 三件套统一走 OTLP 送到 Collector。

在你的项目入口最早的位置调用一次 setup_telemetry() 即可。
"""
import json
import logging
import os
import sys

from opentelemetry import _logs, metrics, trace
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "rag-api")
OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")

# 会从 log record 的 extra 里自动提取的业务字段
_EXTRA_FIELDS = (
    "stage",
    "session_id",
    "model",
    "latency_ms",
    "hits",
    "retrieved_doc_ids",
    "prompt_tokens",
    "completion_tokens",
    "error_code",
)

_initialized = False


class JsonFormatter(logging.Formatter):
    """单行 JSON 日志，自动带上 trace_id / span_id，方便和链路互跳。"""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname.lower(),
            "service": SERVICE_NAME,
            "logger": record.name,
            "message": record.getMessage(),
        }

        ctx = trace.get_current_span().get_span_context()
        if ctx.is_valid:
            payload["trace_id"] = format(ctx.trace_id, "032x")
            payload["span_id"] = format(ctx.span_id, "016x")

        for key in _EXTRA_FIELDS:
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value

        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def setup_telemetry() -> None:
    """初始化 Tracer / Meter / Logger 三个 Provider。重复调用安全。"""
    global _initialized
    if _initialized:
        return

    os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", OTLP_ENDPOINT)
    os.environ.setdefault("OTEL_SERVICE_NAME", SERVICE_NAME)

    resource = Resource.create(
        {
            "service.name": SERVICE_NAME,
            "service.version": os.getenv("SERVICE_VERSION", "0.1.0"),
            "deployment.environment": os.getenv("DEPLOY_ENV", "production"),
        }
    )

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(tracer_provider)

    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[
            PeriodicExportingMetricReader(OTLPMetricExporter(), export_interval_millis=15000)
        ],
    )
    metrics.set_meter_provider(meter_provider)

    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))
    _logs.set_logger_provider(logger_provider)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(JsonFormatter())
    root.addHandler(console)

    # LoggingHandler 会把日志同时以 OTLP 发出去，并自动附上当前 span 的 trace_id
    root.addHandler(LoggingHandler(level=logging.NOTSET, logger_provider=logger_provider))

    _initialized = True


def instrument_app(app) -> None:
    """给 Web 应用挂上自动埋点，拿到接口的 QPS / 延迟 / 错误率。

    本项目的生产入口 app.py 用的是 Starlette（不是 FastAPI），所以这里走
    StarletteInstrumentor；FastAPI 项目则用 FastAPIInstrumentor。
    """
    try:
        from opentelemetry.instrumentation.starlette import StarletteInstrumentor
        StarletteInstrumentor.instrument_app(app)
    except ImportError:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
