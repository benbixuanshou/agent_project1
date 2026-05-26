"""Notify Agent — manages notification delivery strategy and escalation chains."""

import asyncio
import logging

from app.config import settings
from app.notify.dingtalk import send_dingtalk_markdown
from app.notify.feishu import send_feishu_card
from app.notify.wecom import send_wecom_markdown

logger = logging.getLogger("superbizagent")

PRIORITY_CHANNELS = {
    "P0": ["dingtalk", "feishu", "wecom"],
    "P1": ["dingtalk", "feishu"],
    "P2": ["dingtalk"],
}


class NotifyAgent:
    """Routes notifications to the right channels based on severity and configuration."""

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False

    async def start(self):
        if self._running:
            return
        self._running = True
        asyncio.create_task(self._worker())

    async def stop(self):
        self._running = False

    async def send(self, title: str, content: str, severity: str = "P1"):
        await self._queue.put({"title": title, "content": content, "severity": severity})

    async def _worker(self):
        while self._running:
            try:
                msg = await asyncio.wait_for(self._queue.get(), timeout=30)
                await self._deliver(msg["title"], msg["content"], msg["severity"])
            except asyncio.TimeoutError:
                pass
            except Exception:
                logger.error("notify_agent: delivery failed", exc_info=True)

    async def _deliver(self, title: str, content: str, severity: str):
        channels = PRIORITY_CHANNELS.get(severity, ["dingtalk"])
        results = await asyncio.gather(
            *[self._try_send(ch, title, content) for ch in channels],
            return_exceptions=True,
        )
        success = sum(1 for r in results if r is True)
        logger.info("notify_delivered: %s [%s] → %d/%d channels", title, severity, success, len(channels))

    async def _try_send(self, channel: str, title: str, content: str) -> bool:
        if channel == "dingtalk" and settings.dingtalk_webhook_url:
            return await send_dingtalk_markdown(title, content)
        if channel == "feishu" and settings.feishu_webhook_url:
            return await send_feishu_card(title, content)
        if channel == "wecom" and settings.wecom_webhook_url:
            return await send_wecom_markdown(f"# {title}\n\n{content}")
        return False


notify_agent = NotifyAgent()
