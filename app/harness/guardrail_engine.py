"""
GuardrailEngine — deterministic constraint enforcement for AI agents.

Loads YAML rules and evaluates them before/after tool calls and agent output.
Rules are programmatic (not prompt-based): if a rule matches, the action is
enforced by Python code, not by asking the LLM to comply.

Three scopes:
    input     — user query arrives → sanitize/reject
    tool_call — agent calls a tool → block/circuit_break/allow
    output    — agent produces answer → validate/retry/reject
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("superbizagent.harness.guardrails")


@dataclass
class GuardResult:
    allowed: bool = True
    action: str = "pass"  # pass | block | sanitize | circuit_break | validate_and_retry | require_approval
    message: str = ""
    rule_id: str = ""


@dataclass
class GuardRule:
    id: str
    description: str
    scope: str  # input | tool_call | output
    when: dict[str, Any]
    action: str
    message: str = ""
    cooldown_seconds: int = 0


class GuardrailEngine:
    """Loads and evaluates declarative YAML guardrail rules.

    Rules are loaded from rules.yaml at init time. Each rule has:
      - scope: when in the pipeline to check
      - when: conditions to match (field paths with values to compare)
      - action: what to do when matched

    When a circuit_break action fires and circuit_registry is wired,
    the corresponding CircuitBreaker is tripped immediately — subsequent
    tool calls will be blocked at the breaker level (not just the guardrail).
    """

    def __init__(self, rules_path: str | None = None,
                 circuit_registry=None):
        if rules_path is None:
            rules_path = str(Path(__file__).parent / "rules.yaml")
        self._rules: list[GuardRule] = []
        self._rules_by_scope: dict[str, list[GuardRule]] = {
            "input": [], "tool_call": [], "output": [],
        }
        self._circuit_registry = circuit_registry
        self._load_rules(rules_path)
        self._match_counter: dict[str, int] = {}

    def _load_rules(self, path: str):
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning("guardrail rules file not found: %s, using empty ruleset", path)
            return
        except yaml.YAMLError:
            logger.warning("guardrail rules file parse error: %s, using empty ruleset", path)
            return

        for rule_data in data.get("rules", data.get("guardrails", [])):
            rule = GuardRule(
                id=rule_data.get("id", ""),
                description=rule_data.get("description", ""),
                scope=rule_data.get("scope", ""),
                when=rule_data.get("when", {}),
                action=rule_data.get("action", "block"),
                message=rule_data.get("message", ""),
                cooldown_seconds=rule_data.get("cooldown_seconds", 0),
            )
            if rule.scope in self._rules_by_scope:
                self._rules_by_scope[rule.scope].append(rule)
                self._rules.append(rule)

        logger.info(
            "guardrail engine loaded %d rules (input=%d, tool_call=%d, output=%d)",
            len(self._rules),
            len(self._rules_by_scope["input"]),
            len(self._rules_by_scope["tool_call"]),
            len(self._rules_by_scope["output"]),
        )

    def reload(self, path: str | None = None):
        """Hot-reload rules without restarting."""
        self._rules.clear()
        for k in self._rules_by_scope:
            self._rules_by_scope[k].clear()
        self._load_rules(path or str(Path(__file__).parent / "rules.yaml"))

    def check(self, scope: str, context: dict) -> GuardResult:
        """Evaluate all rules for a given scope against the context.

        Returns the FIRST matching result. Rules are evaluated in order.
        """
        for rule in self._rules_by_scope.get(scope, []):
            if self._match(rule.when, context):
                return self._execute(rule, context)
        return GuardResult(allowed=True, action="pass", message="")

    def _match(self, conditions: dict, context: dict) -> bool:
        """Evaluate rule conditions against context.

        Supported conditions:
          - simple equality: {"tool": "query_k8s_events"} → context["tool"] == "query_k8s_events"
          - target_agent: {"target_agent": "sre"} → context["target_agent"] == "sre"
          - contains: {"input_contains": ["ignore previous", "DAN mode"]} → substring match
          - output_contains: {"output_contains": ["auto_remediate"]} → substring in output
          - regex: {"output_regex": "pattern"} → re.search on context["output"]
          - length checks: {"max_query_length": 50000} → len(context["query"]) > 50000
          - {"query_empty": true} → context["query"] is empty/whitespace
          - {"output_too_short": 100} → len(context["output"]) < 100
          - numeric threshold: {"same_tool_consecutive_failures": 3} → context[key] >= N
          - dot-path access: {"params.namespace": "production"} → context["params"]["namespace"] == "production"
        """
        if not conditions:
            return False

        for key, expected in conditions.items():
            # ── Input content checks ──
            if key == "input_contains":
                query = str(context.get("query", context.get("input", ""))).lower()
                patterns = expected if isinstance(expected, list) else [expected]
                if not any(p.lower() in query for p in patterns):
                    return False
                continue

            # ── Output content checks ──
            if key == "output_contains":
                output = str(context.get("output", "")).lower()
                patterns = expected if isinstance(expected, list) else [expected]
                if not any(p.lower() in output for p in patterns):
                    return False
                continue

            if key == "output_regex":
                output = str(context.get("output", ""))
                try:
                    # Guard against ReDoS: limit regex execution time
                    # Python's re module doesn't support native timeout,
                    # so we bound input length as a practical mitigation.
                    # For production, replace with the `re2` library or
                    # run regex in a subprocess with signal.alarm().
                    if len(output) > 100_000:
                        logger.warning("guardrail: output too long for regex check (%d chars)", len(output))
                        return False
                    if not re.search(expected, output[:100_000]):
                        return False
                except re.error:
                    logger.warning("guardrail: invalid regex in rule: %s", expected)
                    return False
                continue

            if key == "output_claims_entity":
                if not context.get("output"):
                    return False
                continue

            # ── Length / boundary checks ──
            if key == "max_query_length":
                query_len = len(str(context.get("query", context.get("input", ""))))
                if query_len <= expected:
                    return False
                continue

            if key == "query_empty":
                query = str(context.get("query", context.get("input", ""))).strip()
                if bool(query) != (not expected):  # expected=True means query SHOULD be empty
                    return False
                continue

            if key == "output_too_short":
                output_len = len(str(context.get("output", "")))
                if output_len >= expected:
                    return False
                continue

            # ── Target agent match ──
            if key == "target_agent":
                actual = str(context.get("target_agent", ""))
                if actual != str(expected):
                    return False
                continue

            # ── Numeric threshold: same_tool_consecutive_failures >= N ──
            if key in ("same_tool_consecutive_failures", "iteration_count"):
                actual = context.get(key, 0)
                if actual < expected:
                    return False
                continue

            # ── Alert severity: string equality ──
            if key == "alert_severity":
                actual = str(context.get(key, "")).lower()
                if actual != str(expected).lower():
                    return False
                continue

            # ── Dot-path access: params.namespace, params.action ──
            if "." in key:
                parts = key.split(".")
                value = context
                for part in parts:
                    if isinstance(value, dict):
                        value = value.get(part)
                    else:
                        value = None
                        break

                if isinstance(expected, list):
                    if value not in expected:
                        return False
                elif isinstance(expected, str) and isinstance(value, str):
                    if expected.lower() not in value.lower():
                        return False
                elif value != expected:
                    return False
                continue

            # ── Simple equality ──
            actual = context.get(key)
            if actual != expected:
                return False

        return True

    def _execute(self, rule: GuardRule, context: dict) -> GuardResult:
        """Execute the rule's action. Trips CircuitBreaker for circuit_break actions."""
        self._match_counter[rule.id] = self._match_counter.get(rule.id, 0) + 1

        # Wire circuit_break to the actual CircuitBreaker
        if rule.action == "circuit_break" and self._circuit_registry:
            tool_name = context.get("tool", "")
            if tool_name:
                cb = self._circuit_registry.get(tool_name)
                cb.cooldown_seconds = rule.cooldown_seconds or 30
                for _ in range(cb.failure_threshold):
                    cb.on_failure()
                logger.info(
                    "guardrail circuit_break: tripped %s via rule %s (cooldown=%ds)",
                    tool_name, rule.id, cb.cooldown_seconds,
                )

        return GuardResult(
            allowed=(rule.action not in ("block", "circuit_break")),
            action=rule.action,
            message=rule.message or f"Guardrail rule [{rule.id}] triggered",
            rule_id=rule.id,
        )

    def stats(self) -> dict:
        """Return rule match statistics for observability."""
        return {
            "total_rules": len(self._rules),
            "rules_by_scope": {k: len(v) for k, v in self._rules_by_scope.items()},
            "match_counts": dict(self._match_counter),
        }
