"""Prometheus /metrics endpoint — export Agent health + Harness metrics."""

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse

from app.self_monitor import agent_metrics

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
async def metrics(request: Request):
    m = agent_metrics
    hm = getattr(request.app.state, "harness_metrics", None)
    ge = getattr(request.app.state, "guardrail_engine", None)
    il = getattr(request.app.state, "incident_learner", None)
    cb = getattr(request.app.state, "circuit_registry", None)
    dl = getattr(request.app.state.supervisor, "decision_log", None) if hasattr(request.app.state, "supervisor") else None

    gs = ge.stats() if ge else {}
    ils = il.stats() if il else {}
    cbs = cb.stats() if cb else []
    dls = dl.stats() if dl else {}

    lines = [
        "# HELP superbizagent_http_requests_total Total HTTP requests",
        "# TYPE superbizagent_http_requests_total counter",
        f"superbizagent_http_requests_total {getattr(hm, '_guardrail_passes', 0) + getattr(hm, '_guardrail_blocks', 0)}",
        "",
        "# HELP superbizagent_llm_calls_total Total LLM calls (wired via harness)",
        "# TYPE superbizagent_llm_calls_total counter",
        f"superbizagent_llm_calls_total {m.llm_calls}",
        "",
        "# HELP superbizagent_llm_failures_total Total LLM call failures",
        "# TYPE superbizagent_llm_failures_total counter",
        f"superbizagent_llm_failures_total {m.llm_failures}",
        "",
        "# HELP superbizagent_tool_calls_total Total tool calls",
        "# TYPE superbizagent_tool_calls_total counter",
        f"superbizagent_tool_calls_total {m.tool_calls}",
        "",
        "# HELP superbizagent_tool_failures_total Total tool call failures",
        "# TYPE superbizagent_tool_failures_total counter",
        f"superbizagent_tool_failures_total {m.tool_failures}",
        "",
        "# HELP superbizagent_llm_success_rate LLM call success rate (0-1)",
        "# TYPE superbizagent_llm_success_rate gauge",
        f"superbizagent_llm_success_rate {m.llm_success_rate}",
        "",
        "# HELP superbizagent_alert_storm 1 if alert storm detected",
        "# TYPE superbizagent_alert_storm gauge",
        f"superbizagent_alert_storm {1 if m.is_alert_storm else 0}",
        "",
        "# HELP superbizagent_guardrail_rules_total Guardrail rules loaded",
        "# TYPE superbizagent_guardrail_rules_total gauge",
        f"superbizagent_guardrail_rules_total {gs.get('total_rules', 0)}",
        "",
        "# HELP superbizagent_guardrail_blocks_total Guardrail blocks",
        "# TYPE superbizagent_guardrail_blocks_total counter",
        f"superbizagent_guardrail_blocks_total {getattr(hm, '_guardrail_blocks', 0)}",
        "",
        "# HELP superbizagent_circuit_breaks_total Circuit breaker trips",
        "# TYPE superbizagent_circuit_breaks_total counter",
        f"superbizagent_circuit_breaks_total {getattr(hm, '_circuit_breaks', 0)}",
        "",
        "# HELP superbizagent_incidents_total Total incidents recorded",
        "# TYPE superbizagent_incidents_total counter",
        f"superbizagent_incidents_total {ils.get('total_incidents', 0)}",
        "",
        "# HELP superbizagent_incident_clusters_total Incident clusters",
        "# TYPE superbizagent_incident_clusters_total gauge",
        f"superbizagent_incident_clusters_total {ils.get('total_clusters', 0)}",
        "",
        "# HELP superbizagent_routing_decisions_total Routing decisions",
        "# TYPE superbizagent_routing_decisions_total counter",
        f"superbizagent_routing_decisions_total {dls.get('total', 0)}",
        "",
        "# HELP superbizagent_fast_route_pct Fast route percentage",
        "# TYPE superbizagent_fast_route_pct gauge",
        f"superbizagent_fast_route_pct {dls.get('fast_route_pct', 0)}",
        "",
    ]

    # Per-circuit-breaker stats
    for cb_stat in cbs:
        name = cb_stat["name"].replace("-", "_")
        lines.extend([
            f"# HELP superbizagent_circuit_state_{name} Circuit breaker state (0=closed, 1=open, 2=half_open)",
            f"# TYPE superbizagent_circuit_state_{name} gauge",
        ])
        state_map = {"closed": 0, "open": 1, "half_open": 2}
        lines.append(f"superbizagent_circuit_state_{name} {state_map.get(cb_stat['state'], 0)}")
        lines.append("")

    return PlainTextResponse("\n".join(lines), media_type="text/plain")
