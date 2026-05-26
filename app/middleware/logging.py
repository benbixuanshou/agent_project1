"""Request logging — pure ASGI middleware."""

import logging
import time
import uuid

from fastapi import Request

logger = logging.getLogger("superbizagent")


class LoggingMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        start = time.perf_counter()
        status_code = 500

        # Propagate trace_id via harness TraceContext
        from app.harness.tracer import TraceContext
        trace_ctx = TraceContext(trace_id=request_id)
        scope["_trace_ctx"] = trace_ctx

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 500)
            await send(message)

        try:
            with trace_ctx:
                await self.app(scope, receive, send_wrapper)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "request",
                extra={
                    "request_id": request_id,
                    "trace_id": trace_ctx.trace_id,
                    "method": request.method,
                    "path": str(request.url.path),
                    "status": status_code,
                    "duration_ms": round(elapsed_ms, 2),
                },
            )
