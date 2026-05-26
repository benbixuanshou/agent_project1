"""Admin API — login, global stats, tenant overview."""

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.middleware.auth import get_tenant_context
from app.self_monitor import agent_metrics
from app.tenant_store import tenant_registry, TenantContext, DEFAULT_TENANT_ID
from app.config import settings


def _require_admin(request: Request):
    """Reusable admin role check. Raises 403 if not admin."""
    ctx = get_tenant_context(request)
    if ctx is None or ctx.role != "admin":
        raise HTTPException(status_code=403, detail="admin role required")
    return ctx

logger = logging.getLogger("superbizagent")
router = APIRouter(tags=["admin"])


class LoginRequest(BaseModel):
    api_key: str = Field(..., alias="api_key")


@router.post("/login")
async def login(req: LoginRequest):
    """Validate API Key and return tenant info + role. No auth required."""
    key = req.api_key.strip()
    if not key:
        raise HTTPException(status_code=401, detail="api_key is required")

    ctx = tenant_registry.lookup(key)

    # Also check env API_KEYS fallback
    if not ctx:
        env_keys = {k.strip() for k in settings.api_keys.split(",") if k.strip()}
        if key in env_keys:
            ctx = TenantContext(
                tenant_id=DEFAULT_TENANT_ID,
                tenant_name="Default Tenant",
                role="admin",
            )

    if not ctx:
        raise HTTPException(status_code=401, detail="invalid api_key")

    return {
        "status": "ok",
        "tenant": {
            "id": ctx.tenant_id,
            "name": ctx.tenant_name,
            "role": ctx.role,
        },
    }


@router.get("/admin/stats")
async def admin_stats(request: Request):
    """Global dashboard stats — admin role only."""
    _require_admin(request)
    return {
        "tenants": {
            "count": tenant_registry.tenant_count,
            "keys": tenant_registry.key_count,
        },
        "agent": agent_metrics.health_report(),
        "server": "ok",
    }


@router.get("/admin/tenants")
async def admin_tenants(request: Request):
    """Current tenant info — any authenticated role."""
    ctx = get_tenant_context(request)
    return {
        "tenant": {
            "id": ctx.tenant_id,
            "name": ctx.tenant_name,
            "role": ctx.role,
        },
    }


# ═══════════════════════════════════════════════════════════
# Harness Engineering — management endpoints
# ═══════════════════════════════════════════════════════════

@router.get("/admin/harness/guardrails")
async def harness_guardrail_stats(request: Request):
    """Guardrail engine stats — admin role only."""
    ge = getattr(request.app.state, "guardrail_engine", None)
    if not ge:
        return {"enabled": False}
    return {"enabled": True, **ge.stats()}


@router.post("/admin/harness/guardrails/reload")
async def harness_guardrail_reload(request: Request):
    """Hot-reload guardrail rules from rules.yaml — admin only."""
    _require_admin(request)
    ge = getattr(request.app.state, "guardrail_engine", None)
    if not ge:
        return {"status": "guardrail engine not enabled"}
    ge.reload()
    return {"status": "reloaded", **ge.stats()}


@router.get("/admin/harness/incidents")
async def harness_incident_stats(request: Request):
    """Incident learner stats — admin role only."""
    il = getattr(request.app.state, "incident_learner", None)
    if not il:
        return {"enabled": False}
    return il.stats()


@router.get("/admin/harness/incidents/candidates")
async def harness_incident_candidates(request: Request):
    """Pending candidate rules awaiting human review."""
    il = getattr(request.app.state, "incident_learner", None)
    if not il:
        return {"candidates": []}
    return {"candidates": il.get_candidates()}


@router.get("/admin/harness/circuits")
async def harness_circuit_stats(request: Request):
    """Circuit breaker status for all tools."""
    cb = getattr(request.app.state, "circuit_registry", None)
    if not cb:
        return {"circuits": []}
    return {"circuits": cb.stats()}


@router.post("/admin/harness/circuits/reset")
async def harness_circuit_reset(request: Request):
    """Force-reset all circuit breakers."""
    _require_admin(request)
    cb = getattr(request.app.state, "circuit_registry", None)
    if not cb:
        return {"status": "circuit registry not available"}
    cb.reset_all()
    return {"status": "all circuits reset"}


@router.get("/admin/harness/routing")
async def harness_routing_stats(request: Request):
    """Supervisor routing decision stats."""
    sup = getattr(request.app.state, "supervisor", None)
    if not sup or not hasattr(sup, "decision_log"):
        return {"enabled": False}
    return sup.decision_log.stats()


@router.get("/admin/harness/pending_actions")
async def harness_pending_actions(request: Request):
    """Scoped pending actions overview."""
    spa = getattr(request.app.state, "scoped_pending_actions", None)
    if not spa:
        return {"actions": []}
    return {"stats": spa.stats(), "recent": spa.get("default")[-20:]}


@router.get("/admin/harness/overview")
async def harness_overview(request: Request):
    """Complete harness overview — all 6 dimensions in one view."""
    _require_admin(request)
    ge = getattr(request.app.state, "guardrail_engine", None)
    il = getattr(request.app.state, "incident_learner", None)
    cb = getattr(request.app.state, "circuit_registry", None)
    sup = getattr(request.app.state, "supervisor", None)

    return {
        "dimension_1_context": {
            "session_max_pairs": settings.session_max_pairs,
            "session_ttl_seconds": settings.session_ttl_seconds,
        },
        "dimension_2_tools": {
            "categories": ["monitoring", "logging", "k8s", "change", "cost", "knowledge", "skill", "utility"],
        },
        "dimension_3_orchestration": sup.decision_log.stats() if sup and hasattr(sup, "decision_log") else {},
        "dimension_4_memory": {
            "backend": settings.session_backend,
            "ttl_seconds": settings.session_ttl_seconds,
            "cleanup_interval": settings.session_cleanup_interval_seconds,
        },
        "dimension_5_observability": agent_metrics.health_report(),
        "dimension_6_guardrails": {
            "rules": ge.stats() if ge else {},
            "incidents": il.stats() if il else {},
            "circuits": cb.stats() if cb else [],
        },
    }
