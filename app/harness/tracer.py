"""
TraceContext — request ID propagation across the entire call chain.

Ensures every log line, metric, and audit entry shares a common trace_id
that propagates from the HTTP request through LLM calls, tool invocations,
and external service calls (httpx, redis, mysql).
"""

import contextvars
import logging
import uuid

logger = logging.getLogger("superbizagent.harness.tracer")

# Context variable that propagates through asyncio tasks
_trace_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "trace_id", default=""
)


class TraceContext:
    """Thread-safe, async-safe trace ID propagation.

    Usage:
        with TraceContext("req_abc123") as ctx:
            logger.info("processing", extra={"trace_id": ctx.trace_id})
            # All nested calls within this context inherit the trace_id
    """

    def __init__(self, trace_id: str = "", request_id: str = ""):
        self.trace_id = trace_id or request_id or f"trace_{uuid.uuid4().hex[:12]}"
        self._token = None

    def __enter__(self):
        self._token = _trace_id_ctx.set(self.trace_id)
        return self

    def __exit__(self, *args):
        if self._token:
            _trace_id_ctx.reset(self._token)

    @staticmethod
    def current() -> str:
        return _trace_id_ctx.get() or ""

    @staticmethod
    def new() -> str:
        return f"trace_{uuid.uuid4().hex[:12]}"

    @staticmethod
    def format_log_extra(**kwargs) -> dict:
        """Return extra dict for structured logging with trace_id injected."""
        trace_id = TraceContext.current()
        extra = {"trace_id": trace_id} if trace_id else {}
        extra.update(kwargs)
        return extra
