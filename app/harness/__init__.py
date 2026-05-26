"""
Harness Engineering layer for SuperBizAgent.

Six dimensions of AI agent governance, implemented as explicit,
configurable, observable components:

1. Context Management — what the model can see (token budget, compression)
2. Tool System        — what the model can do (circuit breaker, result envelope)
3. Execution Orchestration — what the model does step by step (routing, audit)
4. State & Memory     — how the model maintains continuity (sessions, housekeeping)
5. Evaluation & Observability — how the model knows if it's right (metrics, tracing)
6. Constraints & Recovery — what happens when things go wrong (guardrails, incidents)

Active modules:
  guardrails/  — GuardrailEngine (13 rules) + IncidentLearner (self-healing)
  tools/       — CircuitBreaker + CircuitBreakerRegistry
  pipeline/    — HarnessPipeline (5-step execution lifecycle)
  observability/ — HarnessMetrics + TraceContext
  orchestration/ — DecisionLog (routing audit)
  memory/      — SessionHousekeeper + ScopedPendingActions
"""

from app.harness.pipeline import HarnessPipeline, StepResult, PipelineContext
from app.harness.guardrail_engine import GuardrailEngine, GuardResult
from app.harness.incident_learner import IncidentLearner, Incident
from app.harness.circuit_breaker import CircuitBreaker, CircuitState, CircuitBreakerRegistry

__all__ = [
    "HarnessPipeline", "StepResult", "PipelineContext",
    "GuardrailEngine", "GuardResult",
    "IncidentLearner", "Incident",
    "CircuitBreaker", "CircuitState", "CircuitBreakerRegistry",
]
