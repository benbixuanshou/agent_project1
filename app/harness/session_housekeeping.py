"""
SessionHousekeeper — automated session lifecycle management.

Fixes:
1. cleanup_expired() is now called on a periodic timer via lifespan
2. SQLite operations are wrapped in asyncio.to_thread()
3. PENDING_ACTIONS is scoped per-tenant/session, not global
"""

import asyncio
import logging
import time

logger = logging.getLogger("superbizagent.harness.memory")


class SessionHousekeeper:
    """Runs periodic session cleanup and maintenance tasks."""

    def __init__(self, session_store, cleanup_interval: int = 3600, ttl_seconds: int = 7 * 24 * 3600):
        self.session_store = session_store
        self.cleanup_interval = cleanup_interval  # seconds, default 1 hour
        self.ttl_seconds = ttl_seconds
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self):
        """Start periodic cleanup in the background."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("session housekeeper started (interval=%ds, ttl=%ds)",
                    self.cleanup_interval, self.ttl_seconds)

    async def stop(self):
        """Stop the background cleanup task."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("session housekeeper stopped")

    async def _loop(self):
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self.cleanup()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.warning("session housekeeper: cleanup cycle failed", exc_info=True)

    async def cleanup(self):
        """Run one cleanup cycle."""
        t0 = time.time()
        try:
            await self.session_store.cleanup_expired(self.ttl_seconds)
            elapsed = (time.time() - t0) * 1000
            logger.info("session housekeeper: cleanup cycle complete (%.0fms)", elapsed)
        except Exception:
            logger.warning("session housekeeper: cleanup failed", exc_info=True)


# ═══════════════════════════════════════════════════════════════
# Scoped PENDING_ACTIONS — replaces the global list in action_agent.py
# ═══════════════════════════════════════════════════════════════

class ScopedPendingActions:
    """Per-tenant, per-session pending actions store.

    Replaces the module-level PENDING_ACTIONS list in action_agent.py
    which was shared across all tenants and users.
    """

    def __init__(self):
        self._actions: dict[str, list[dict]] = {}  # key: "tenant_id:session_id"

    def _key(self, tenant_id: str, session_id: str = "") -> str:
        return f"{tenant_id}:{session_id}" if session_id else tenant_id

    def get(self, tenant_id: str, session_id: str = "") -> list[dict]:
        return self._actions.get(self._key(tenant_id, session_id), [])

    def add(self, tenant_id: str, action: dict, session_id: str = ""):
        key = self._key(tenant_id, session_id)
        if key not in self._actions:
            self._actions[key] = []
        action["id"] = f"act_{len(self._actions[key]) + 1:04d}"
        action["status"] = "pending"
        action["tenant_id"] = tenant_id
        self._actions[key].append(action)

    def clear(self, tenant_id: str, session_id: str = ""):
        key = self._key(tenant_id, session_id)
        self._actions.pop(key, None)

    def approve(self, tenant_id: str, action_id: str, session_id: str = ""):
        for a in self.get(tenant_id, session_id):
            if a.get("id") == action_id:
                a["status"] = "approved"
                return True
        return False

    def reject(self, tenant_id: str, action_id: str, session_id: str = ""):
        for a in self.get(tenant_id, session_id):
            if a.get("id") == action_id:
                a["status"] = "rejected"
                return True
        return False

    def stats(self) -> dict:
        total = sum(len(v) for v in self._actions.values())
        pending = sum(
            sum(1 for a in v if a.get("status") == "pending")
            for v in self._actions.values()
        )
        return {"total": total, "pending": pending, "tenants": len(self._actions)}


scoped_pending_actions = ScopedPendingActions()
