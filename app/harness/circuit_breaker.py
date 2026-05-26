"""
CircuitBreaker — programmatic failure protection for external tool calls.

Implements the standard circuit breaker pattern:
    CLOSED → (N consecutive failures) → OPEN → (cooldown) → HALF_OPEN → (success) → CLOSED

Integrates with GuardrailEngine: when a tool is circuit-broken, the guardrail
blocks the call and returns a cached/fallback result instead.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("superbizagent.harness.circuit")


class CircuitState(Enum):
    CLOSED = "closed"       # normal operation, calls pass through
    OPEN = "open"           # circuit tripped, calls are blocked
    HALF_OPEN = "half_open" # probe: allow one call to test recovery


@dataclass
class CircuitBreaker:
    """Per-tool circuit breaker.

    Usage:
        cb = CircuitBreaker("query_prometheus_alerts", failure_threshold=3, cooldown_seconds=30)
        if not cb.allow():
            return fallback_result
        try:
            result = await call_tool()
            cb.on_success()
            return result
        except Exception as e:
            cb.on_failure()
            raise
    """
    name: str
    failure_threshold: int = 3
    cooldown_seconds: int = 30
    half_open_timeout: int = 5

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    consecutive_failures: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    total_calls: int = 0
    total_failures: int = 0
    tripped_at: float = 0.0
    trip_count: int = 0
    _half_open_probe_sent: bool = False

    def allow(self) -> bool:
        """Check if the call should be allowed through."""
        self.total_calls += 1

        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            elapsed = time.time() - self.tripped_at
            if elapsed >= self.cooldown_seconds:
                self.state = CircuitState.HALF_OPEN
                self._half_open_probe_sent = False
                logger.info("circuit %s: OPEN → HALF_OPEN (cooldown elapsed)", self.name)
            else:
                return False

        if self.state == CircuitState.HALF_OPEN:
            if not self._half_open_probe_sent:
                self._half_open_probe_sent = True
                return True
            return False

        return True

    def on_success(self):
        """Record a successful call."""
        self.last_success_time = time.time()
        self.consecutive_failures = 0

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            logger.info("circuit %s: HALF_OPEN → CLOSED (probe succeeded)", self.name)

    def on_failure(self):
        """Record a failed call and potentially trip the circuit."""
        self.last_failure_time = time.time()
        self.consecutive_failures += 1
        self.failure_count += 1
        self.total_failures += 1

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.tripped_at = time.time()
            self.trip_count += 1
            logger.warning("circuit %s: HALF_OPEN → OPEN (probe failed)", self.name)
        elif (self.state == CircuitState.CLOSED
              and self.consecutive_failures >= self.failure_threshold):
            self.state = CircuitState.OPEN
            self.tripped_at = time.time()
            self.trip_count += 1
            logger.warning(
                "circuit %s: CLOSED → OPEN (%d consecutive failures, threshold=%d)",
                self.name, self.consecutive_failures, self.failure_threshold,
            )

    def reset(self):
        """Force-reset the circuit to CLOSED."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.consecutive_failures = 0
        self._half_open_probe_sent = False
        logger.info("circuit %s: force-reset → CLOSED", self.name)

    def stats(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "failure_rate": round(self.total_failures / max(self.total_calls, 1), 3),
            "consecutive_failures": self.consecutive_failures,
            "trip_count": self.trip_count,
            "last_failure": self.last_failure_time,
        }


class CircuitBreakerRegistry:
    """Manages per-tool circuit breakers."""

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(self, tool_name: str) -> CircuitBreaker:
        if tool_name not in self._breakers:
            self._breakers[tool_name] = CircuitBreaker(name=tool_name)
        return self._breakers[tool_name]

    def allow(self, tool_name: str) -> bool:
        return self.get(tool_name).allow()

    def on_success(self, tool_name: str):
        self.get(tool_name).on_success()

    def on_failure(self, tool_name: str):
        self.get(tool_name).on_failure()

    def stats(self) -> list[dict]:
        return [cb.stats() for cb in self._breakers.values()]

    def reset_all(self):
        for cb in self._breakers.values():
            cb.reset()
