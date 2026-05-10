"""Tool registries for RAG Agent and SRE Agent."""

from app.tools.datetime_tool import get_current_datetime
from app.tools.prometheus_tool import query_prometheus_alerts
from app.tools.cls_logs_tool import query_logs, get_available_log_topics
from app.tools.change_tools import query_recent_deployments
from app.rag.rag_tool import search_knowledge_base
from app.config import settings


def gather_rag_tools() -> list:
    """Tools for RAG Agent: knowledge search + web search + datetime."""
    from app.tools.web_search_tool import web_search
    return [search_knowledge_base, get_current_datetime, web_search]


def gather_sre_tools(include_cls: bool = None) -> list:
    """Tools for SRE Agent: full toolkit (15 tools + search_skill)."""
    from app.tools.k8s_tools import query_k8s_events, get_k8s_namespaces
    from app.tools.slo_tools import query_slo_status
    from app.tools.topology_tools import query_service_topology, query_blast_radius
    from app.tools.health_scorer import score_service_health
    from app.tools.compliance_tools import run_compliance_check
    from app.tools.cost_tools import check_cost_anomaly
    from app.tools.capacity_tools import predict_capacity
    from app.skills.loader import search_skill

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
    return tools
