"""
HarnessMetrics — guardrail, circuit breaker, and incident counters.

Extends agent_metrics (LLM/tool counters in self_monitor.py) with
harness-specific observability: guardrail blocks, circuit trips, incidents.

Wired into:
  - pipeline.py: increments guardrail counters when rules fire
  - /metrics endpoint: exports counters as Prometheus metrics
"""

import logging

from app.self_monitor import agent_metrics

logger = logging.getLogger("superbizagent.harness.metrics")


class HarnessMetrics:
    """Harness-specific counters, complementing agent_metrics."""

    def __init__(self):
        self.guardrail_blocks: int = 0
        self.guardrail_passes: int = 0
        self.circuit_breaks: int = 0
        self.incidents_recorded: int = 0

    def record_guardrail_block(self, rule_id: str = ""):
        self.guardrail_blocks += 1

    def record_guardrail_pass(self):
        self.guardrail_passes += 1

    def record_circuit_break(self, tool_name: str = ""):
        self.circuit_breaks += 1

    def record_incident(self):
        self.incidents_recorded += 1

    def summary(self) -> dict:
        return {
            **agent_metrics.health_report(),
            "guardrail_blocks": self.guardrail_blocks,
            "guardrail_passes": self.guardrail_passes,
            "circuit_breaks": self.circuit_breaks,
            "incidents_recorded": self.incidents_recorded,
        }


harness_metrics = HarnessMetrics()
