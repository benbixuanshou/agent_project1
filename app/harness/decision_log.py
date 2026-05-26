"""
RouteDecision & DecisionLog — observable, auditable agent routing.

Every supervisor routing decision is logged with full context:
  - What was the query?
  - Which agent was chosen and why?
  - Was it a fast route or LLM route?
  - Did the route succeed or fail?

This makes agent routing decisions debuggable and auditable.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("superbizagent.harness.router")


@dataclass
class RouteDecision:
    """A single routing decision with full metadata."""
    query_snippet: str            # first 200 chars of the query
    target: str                   # rag | sre | platform | action | rag_then_sre | block
    reason: str                   # human-readable reason
    source: str = "unknown"       # "intent_gateway" | "llm_routing" | "default_fallback" | "override"
    confidence: float = 0.0
    latency_ms: float = 0.0
    success: bool = True
    error: str = ""


class DecisionLog:
    """Records and analyzes routing decisions over time.

    In-memory ring buffer for recent decisions, JSONL persistence for long-term.
    """

    MAX_RECENT = 1000

    def __init__(self):
        self._decisions: list[RouteDecision] = []
        self._stats = {
            "total": 0,
            "fast_route": 0,
            "llm_route": 0,
            "fallback": 0,
            "blocked": 0,
            "errors": 0,
            "by_target": {},
        }

    def record(self, decision: RouteDecision):
        self._decisions.append(decision)
        if len(self._decisions) > self.MAX_RECENT:
            self._decisions = self._decisions[-self.MAX_RECENT:]

        self._stats["total"] += 1
        if decision.source == "intent_gateway":
            self._stats["fast_route"] += 1
        elif decision.source == "llm_routing":
            self._stats["llm_route"] += 1
        elif decision.source == "default_fallback":
            self._stats["fallback"] += 1

        if decision.target == "block":
            self._stats["blocked"] += 1
        if not decision.success:
            self._stats["errors"] += 1

        target = decision.target
        if target not in self._stats["by_target"]:
            self._stats["by_target"][target] = 0
        self._stats["by_target"][target] += 1

    def recent(self, n: int = 20) -> list[RouteDecision]:
        return self._decisions[-n:]

    def stats(self) -> dict:
        fast_pct = (self._stats["fast_route"] / max(self._stats["total"], 1)) * 100
        return {
            **self._stats,
            "fast_route_pct": round(fast_pct, 1),
            "recent_decisions": len(self._decisions),
        }
