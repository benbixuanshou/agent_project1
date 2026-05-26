"""
HarnessPipeline — unified 5-step execution pipeline wrapping every agent request.

Pipeline flow:
    start()          → input guard + context assembly (returns PipelineContext)
    check_tool()     → pre-tool-call constraint enforcement (called per tool)
    check_output()   → output validation (called after agent completes)
    finish()         → audit log + incident recording (called at end)

Design: steps are split across the caller's agent loop because tool_guard
and output_guard depend on dynamic agent behavior. The PipelineContext
binds them all to a single request lifecycle.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from app.harness.guardrail_engine import GuardrailEngine, GuardResult
from app.harness.incident_learner import IncidentLearner
from app.self_monitor import agent_metrics

logger = logging.getLogger("superbizagent.harness")


@dataclass
class StepResult:
    """Result of a single pipeline step."""
    step_name: str
    allowed: bool = True
    action: str = "pass"  # pass | block | sanitize | circuit_break | require_approval
    message: str = ""
    data: Any = None
    latency_ms: float = 0.0


@dataclass
class PipelineContext:
    """Mutable context carried through all pipeline steps for one request."""
    query: str
    tenant_id: str = "default"
    session_id: str = ""
    target_agent: str = ""
    route_reason: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    guardrail_log: list[dict] = field(default_factory=list)
    final_output: str = ""
    error: str | None = None
    blocked: bool = False
    block_message: str = ""
    start_time: float = 0.0


class HarnessPipeline:
    """Unified 5-step execution pipeline.

    Usage in chat.py:
        pipeline = HarnessPipeline(guardrail_engine, incident_learner, context_assembler)
        ctx = pipeline.start(query, tenant_id, session_id)
        if ctx.blocked:
            return ctx.block_message

        for tool_name, params in agent_tool_calls:
            guard = pipeline.check_tool(ctx, tool_name, params)
            if guard.action in ("block", "circuit_break"):
                break
            # … execute tool …

        pipeline.check_output(ctx, agent_response)
        pipeline.finish(ctx)
    """

    def __init__(
        self,
        guardrail_engine: GuardrailEngine | None = None,
        incident_learner: IncidentLearner | None = None,
    ):
        self.guardrail_engine = guardrail_engine or GuardrailEngine()
        self.incident_learner = incident_learner or IncidentLearner()

    # ── Step 1 + 2: input guard + context assembly ──────────

    def start(self, query: str, tenant_id: str = "default",
              session_id: str = "") -> PipelineContext:
        """Create pipeline context and run input guard + context assembly.

        Returns the context. If ctx.blocked is True, the caller should
        short-circuit and return the block message to the user.
        """
        ctx = PipelineContext(
            query=query, tenant_id=tenant_id,
            session_id=session_id, start_time=time.time(),
        )
        self._input_guard(ctx)
        if not ctx.blocked:
            self._context_assembly(ctx)
        return ctx

    def _input_guard(self, ctx: PipelineContext):
        t0 = time.perf_counter()
        result = self.guardrail_engine.check("input", {
            "query": ctx.query,
            "tenant_id": ctx.tenant_id,
            "session_id": ctx.session_id,
        })

        ctx.guardrail_log.append({
            "step": "input_guard", "action": result.action,
            "message": str(result.message), "rule_match": result.rule_id,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        })

        if result.action == "block":
            ctx.blocked = True
            ctx.block_message = str(result.message) or "请求被拦截"

    def _context_assembly(self, ctx: PipelineContext):
        t0 = time.perf_counter()
        token_estimate = len(ctx.query) // 2  # rough: ~2 chars per token for CJK
        ctx.guardrail_log.append({
            "step": "context_assembly",
            "estimated_tokens": token_estimate,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        })

    # ── Step 3: per-tool-call guard ─────────────────────────

    def check_tool(self, ctx: PipelineContext, tool_name: str,
                   tool_params: dict | None = None,
                   extra_context: dict | None = None) -> StepResult:
        """Run pre-tool-call guardrail + circuit breaker check.

        Called by the agent execution loop before each tool invocation.

        extra_context can carry ReAct loop state (iteration_count,
        same_tool_consecutive_failures) that guardrail rules reference.
        """
        tool_params = tool_params or {}
        guardrail_ctx = {
            "tool": tool_name,
            "params": tool_params,
            "tenant_id": ctx.tenant_id,
        }
        if extra_context:
            guardrail_ctx.update(extra_context)
        t0 = time.perf_counter()
        result = self.guardrail_engine.check("tool_call", guardrail_ctx)

        ctx.guardrail_log.append({
            "step": "tool_guard", "tool": tool_name,
            "action": result.action, "message": str(result.message),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        })
        ctx.tool_calls.append({"name": tool_name, "params": tool_params,
                               "allowed": result.allowed, "action": result.action})

        if not result.allowed:
            return StepResult(
                step_name="tool_guard", allowed=False,
                action=result.action, message=str(result.message),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        return StepResult(
            step_name="tool_guard", allowed=True,
            action=result.action,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    # ── Step 4: output validation ───────────────────────────

    def check_output(self, ctx: PipelineContext, output: str) -> StepResult:
        """Run output guardrail after agent completes.

        Validates: format, safety, factual consistency, P0 escalation.
        """
        t0 = time.perf_counter()
        result = self.guardrail_engine.check("output", {
            "output": output,
            "query": ctx.query,
            "tool_results": ctx.tool_results,
            "target_agent": ctx.target_agent,
        })

        ctx.final_output = output
        ctx.guardrail_log.append({
            "step": "output_guard", "action": result.action,
            "message": str(result.message),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        })

        return StepResult(
            step_name="output_guard", allowed=result.allowed,
            action=result.action, message=str(result.message or ""),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    # ── Step 5: audit + incident ────────────────────────────

    def finish(self, ctx: PipelineContext):
        """Record audit trail and handle errors via incident learner.

        MUST be called at the end of every request lifecycle.
        """
        total_ms = (time.time() - ctx.start_time) * 1000
        ctx.guardrail_log.append({
            "step": "audit_log",
            "total_latency_ms": round(total_ms, 1),
            "guardrail_checks": len(ctx.guardrail_log),
            "tool_calls": len(ctx.tool_calls),
        })
        agent_metrics.record_llm_success(total_ms)

    async def record_incident(self, ctx: PipelineContext, error: Exception):
        """Record an incident for the self-healing feedback loop."""
        tool_calls = [t for t in ctx.tool_calls if not t.get("allowed", True)]
        await self.incident_learner.record_error(
            query=ctx.query,
            error_type=type(error).__name__,
            error_message=str(error),
            target_agent=ctx.target_agent,
            tool_calls=tool_calls,
            tenant_id=ctx.tenant_id,
        )

    def summary(self, ctx: PipelineContext) -> dict:
        """Return a compact summary of the full pipeline execution."""
        total_ms = (time.time() - ctx.start_time) * 1000 if ctx.start_time else 0
        return {
            "query": ctx.query[:200],
            "tenant_id": ctx.tenant_id,
            "target_agent": ctx.target_agent,
            "blocked": ctx.blocked,
            "block_message": ctx.block_message[:200] if ctx.block_message else "",
            "tool_calls": len(ctx.tool_calls),
            "guardrail_checks": len(ctx.guardrail_log),
            "output_length": len(ctx.final_output),
            "total_latency_ms": round(total_ms, 1),
            "steps": [
                {"step": g["step"], "action": g.get("action", ""), "latency_ms": g.get("latency_ms", 0)}
                for g in ctx.guardrail_log
            ],
        }
