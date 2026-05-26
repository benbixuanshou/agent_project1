"""Tool registries for RAG Agent and SRE Agent with intent-based grouping.

Tools are wrapped with circuit breaker checks: before each tool call,
the circuit breaker for that tool is checked. If OPEN, the call is
blocked with a ToolResult error rather than hitting the failing backend.
"""

import asyncio
import functools
import logging
from typing import Callable

from app.tools.datetime_tool import get_current_datetime
from app.tools.prometheus_tool import query_prometheus_alerts
from app.tools.cls_logs_tool import query_logs, get_available_log_topics
from app.tools.change_tools import query_recent_deployments
from app.rag.rag_tool import search_knowledge_base
from app.config import settings

logger = logging.getLogger("superbizagent")

# Set during app startup by main.py lifespan
_circuit_registry = None


def set_circuit_registry(registry):
    global _circuit_registry
    _circuit_registry = registry


def _wrap_with_circuit_breaker(fn: Callable) -> Callable:
    """Wrap a tool function with circuit breaker pre-check.

    Before each call:
      1. Check CircuitBreaker.allow() — if OPEN, return error immediately
      2. On success → cb.on_success()
      3. On failure → cb.on_failure()

    The original function's return value is preserved.
    """
    tool_name = getattr(fn, "name", None) or fn.__name__

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if _circuit_registry is not None:
            cb = _circuit_registry.get(tool_name)
            if not cb.allow():
                logger.warning("circuit_breaker: %s is OPEN, blocking call", tool_name)
                return f"[circuit_breaker] {tool_name} 已熔断，跳过此次调用以避免雪崩。"
        try:
            result = fn(*args, **kwargs)
            if _circuit_registry is not None:
                _circuit_registry.get(tool_name).on_success()
            return result
        except Exception:
            if _circuit_registry is not None:
                _circuit_registry.get(tool_name).on_failure()
            raise

    @functools.wraps(fn)
    async def async_wrapper(*args, **kwargs):
        if _circuit_registry is not None:
            cb = _circuit_registry.get(tool_name)
            if not cb.allow():
                logger.warning("circuit_breaker: %s is OPEN, blocking async call", tool_name)
                return f"[circuit_breaker] {tool_name} 已熔断，跳过此次调用以避免雪崩。"
        try:
            result = await fn(*args, **kwargs)
            if _circuit_registry is not None:
                _circuit_registry.get(tool_name).on_success()
            return result
        except Exception:
            if _circuit_registry is not None:
                _circuit_registry.get(tool_name).on_failure()
            raise

    return async_wrapper if asyncio.iscoroutinefunction(fn) else wrapper


def gather_rag_tools() -> list:
    """Tools for RAG Agent: knowledge search + web search + datetime."""
    from app.tools.web_search_tool import web_search
    tools = [search_knowledge_base, get_current_datetime, web_search]
    return [_wrap_with_circuit_breaker(t) for t in tools]


def gather_sre_tools(include_cls: bool = None, intent: str = "") -> list:
    """Tools for SRE Agent with intent-based grouping.

    When intent is provided, returns only tools relevant to that intent category.
    When intent is empty (default), returns the full toolkit (15 tools + search_skill).

    Intent groups:
        troubleshooting  → monitoring + logging + k8s + skill + utility
        cost_analysis    → cost + monitoring
        compliance       → change
        technical_question → knowledge + utility
        general_question → knowledge + utility
    """
    from app.tools.k8s_tools import query_k8s_events, get_k8s_namespaces
    from app.tools.slo_tools import query_slo_status
    from app.tools.topology_tools import query_service_topology, query_blast_radius
    from app.tools.health_scorer import score_service_health
    from app.tools.compliance_tools import run_compliance_check
    from app.tools.cost_tools import check_cost_anomaly
    from app.tools.capacity_tools import predict_capacity
    from app.skills.loader import search_skill

    # Category → tool functions
    tool_groups = {
        "monitoring": [query_prometheus_alerts, query_slo_status, score_service_health],
        "logging": [query_logs, get_available_log_topics],
        "k8s": [query_k8s_events, get_k8s_namespaces, query_service_topology, query_blast_radius],
        "change": [query_recent_deployments, run_compliance_check],
        "cost": [check_cost_anomaly, predict_capacity],
        "knowledge": [search_knowledge_base],
        "skill": [search_skill],
        "utility": [get_current_datetime],
    }

    # Intent → relevant categories
    intent_categories = {
        "troubleshooting": ["monitoring", "logging", "k8s", "skill", "utility"],
        "technical_question": ["knowledge", "utility"],
        "configuration": ["knowledge"],
        "product_inquiry": ["knowledge"],
        "cost_analysis": ["cost", "monitoring"],
        "compliance": ["change"],
        "general_question": ["knowledge", "utility"],
    }

    if intent and intent in intent_categories:
        categories = intent_categories[intent]
        tools = []
        for cat in categories:
            tools.extend(tool_groups.get(cat, []))
        return [_wrap_with_circuit_breaker(t) for t in tools]

    # Full toolkit (backward-compatible default)
    tools = [
        get_current_datetime,
        search_knowledge_base,
        query_prometheus_alerts,
        query_k8s_events,
        get_k8s_namespaces,
        query_recent_deployments,
        query_slo_status,
        query_service_topology,
        query_blast_radius,
        score_service_health,
        run_compliance_check,
        check_cost_anomaly,
        predict_capacity,
        search_skill,
    ]
    if include_cls is None:
        include_cls = settings.cls_mock_enabled
    if include_cls:
        tools.extend([query_logs, get_available_log_topics])
    return [_wrap_with_circuit_breaker(t) for t in tools]
