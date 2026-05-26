"""Patrol Agent — scheduled health checks, proactive anomaly detection."""

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta

from app.config import settings
from app.notify.dingtalk import send_dingtalk_markdown

logger = logging.getLogger("superbizagent")

SHANGHAI_TZ = timezone(timedelta(hours=8))

CHECKLIST = [
    ("Prometheus 活跃告警", "query_prometheus_alerts"),
    ("K8s Pod 异常事件", "query_k8s_events(namespace='production')"),
    ("TLS 证书过期", "check_cert_expiry"),
]


class PatrolAgent:
    """Periodic patrol: checks defined health indicators and notifies on findings."""

    def __init__(self, tools: dict | None = None):
        self.tools = tools or {}
        self._task: asyncio.Task | None = None
        self._last_run: float = 0

    async def start(self):
        if settings.patrol_interval_minutes <= 0:
            logger.info("patrol: disabled (patrol_interval_minutes=0)")
            return

        logger.info("patrol: started (interval=%s min)", settings.patrol_interval_minutes)
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self):
        while True:
            await asyncio.sleep(settings.patrol_interval_minutes * 60)
            try:
                await self.patrol()
            except Exception:
                logger.error("patrol: run failed", exc_info=True)

    async def patrol(self) -> str | None:
        """Run one patrol cycle. Returns findings if anything notable, else None."""
        findings: list[str] = []
        now = datetime.now(SHANGHAI_TZ).strftime("%Y-%m-%d %H:%M")

        # 1. Check Prometheus alerts via tool
        prom_tool = self.tools.get("query_prometheus_alerts")
        if prom_tool:
            try:
                result = await asyncio.to_thread(prom_tool.invoke, {})
                if result and "no active" not in str(result).lower():
                    findings.append(f"### Prometheus 活跃告警\n\n```\n{str(result)[:2000]}\n```")
            except Exception:
                logger.warning("patrol: prometheus check failed", exc_info=True)

        # 2. Check K8s events via tool
        k8s_tool = self.tools.get("query_k8s_events")
        if k8s_tool:
            try:
                result = await asyncio.to_thread(k8s_tool.invoke, {"namespace": "production"})
                warning_keywords = ["OOM", "CrashLoop", "ImagePull", "Failed", "Unhealthy"]
                if any(kw in str(result) for kw in warning_keywords):
                    findings.append(f"### K8s 异常事件\n\n```\n{str(result)[:2000]}\n```")
            except Exception:
                logger.warning("patrol: k8s check failed", exc_info=True)

        # 3. Check TLS cert expiry (simple date-based)
        findings.append(self._cert_expiry_check())

        if not findings or all(not f.strip() for f in findings):
            self._last_run = time.time()
            return None

        report = f"# 定时巡检报告\n\n**巡检时间**: {now}\n\n" + "\n---\n".join(findings)
        self._last_run = time.time()

        if settings.notify_enabled:
            await send_dingtalk_markdown("定时巡检报告", report)

        return report

    @staticmethod
    def _cert_expiry_check() -> str:
        # Placeholder — in production this would query actual cert stores
        now_dt = datetime.now(SHANGHAI_TZ)
        mock_expiry = now_dt + timedelta(days=25)
        days_left = (mock_expiry - now_dt).days
        if days_left < 30:
            return f"### TLS 证书即将过期\n\n模拟证书 `*.example.com` 将在 **{days_left} 天后**过期，请及时续期。"
        return ""
