"""Tests for CircuitBreaker — state transitions and recovery."""

from app.harness.circuit_breaker import CircuitBreaker, CircuitState, CircuitBreakerRegistry


class TestCircuitBreaker:
    def test_initial_state_is_closed(self):
        cb = CircuitBreaker("test")
        assert cb.state == CircuitState.CLOSED
        assert cb.allow() is True

    def test_trips_after_consecutive_failures(self):
        cb = CircuitBreaker("test", failure_threshold=2)
        cb.on_failure()
        assert cb.state == CircuitState.CLOSED
        assert cb.allow() is True

        cb.on_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow() is False

    def test_success_resets_consecutive_failures(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.on_failure()
        cb.on_failure()
        assert cb.consecutive_failures == 2
        cb.on_success()
        assert cb.consecutive_failures == 0

    def test_stays_open_during_cooldown(self):
        cb = CircuitBreaker("test", failure_threshold=1, cooldown_seconds=999)
        cb.on_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow() is False
        # Multiple checks all return False
        for _ in range(5):
            assert cb.allow() is False

    def test_half_open_probe_succeeds(self):
        cb = CircuitBreaker("test", failure_threshold=1, cooldown_seconds=0)
        cb.on_failure()
        # Cooldown expired immediately, first allow() transitions to HALF_OPEN + passes
        assert cb.allow() is True
        assert cb.state == CircuitState.HALF_OPEN

        cb.on_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_probe_fails(self):
        cb = CircuitBreaker("test", failure_threshold=1, cooldown_seconds=0)
        cb.on_failure()
        assert cb.allow() is True  # probe
        cb.on_failure()  # probe fails
        assert cb.state == CircuitState.OPEN

    def test_reset_returns_to_closed(self):
        cb = CircuitBreaker("test", failure_threshold=1)
        cb.on_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.allow() is True

    def test_only_one_probe_in_half_open(self):
        cb = CircuitBreaker("test", failure_threshold=1, cooldown_seconds=0)
        cb.on_failure()
        assert cb.allow() is True   # first call: probe allowed
        assert cb.allow() is False  # second call: blocked (probe already sent)

    def test_stats_reflect_state(self):
        cb = CircuitBreaker("test", failure_threshold=1)
        cb.on_failure()
        s = cb.stats()
        assert s["state"] == "open"
        assert s["trip_count"] == 1
        assert s["total_failures"] == 1


class TestCircuitBreakerRegistry:
    def test_creates_breakers_on_demand(self):
        reg = CircuitBreakerRegistry()
        cb = reg.get("prometheus")
        assert cb.name == "prometheus"
        assert cb.state == CircuitState.CLOSED

    def test_tracks_multiple_breakers(self):
        reg = CircuitBreakerRegistry()
        reg.get("prometheus").on_failure()
        reg.get("k8s")
        stats = reg.stats()
        assert len(stats) == 2

    def test_reset_all(self):
        reg = CircuitBreakerRegistry()
        reg.get("a").on_failure()
        reg.get("a").on_failure()
        reg.get("a").on_failure()
        assert reg.get("a").state == CircuitState.OPEN
        reg.reset_all()
        assert reg.get("a").state == CircuitState.CLOSED
