"""
Action Agent — controlled auto-remediation with human confirmation.

All destructive actions require explicit human confirmation before execution.
The agent proposes, the human decides.

PENDING_ACTIONS is now scoped per-tenant via ScopedPendingActions (harness layer).
Replaces the former module-level global list that was shared across all users.
"""

import logging

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from app.harness.session_housekeeping import scoped_pending_actions

logger = logging.getLogger("superbizagent")

ACTION_SYSTEM_PROMPT = """你是受控的自动化止损 Agent。你可以提出操作建议，但所有破坏性操作都需要人工确认。

## 工作方式

1. 分析故障场景 → 提出止损建议
2. 每个建议标注：操作类型、影响范围、风险等级、回滚方案
3. 调用 propose_action 记录建议
4. 操作分为三类：
   - 安全操作（restart_pod, check_status）→ 可以直接执行
   - 需确认操作（scale_up, drain_node）→ 需要人工 approve
   - 紧急操作（rollback, toggle_feature）→ 需要双人确认

## 可用工具

- propose_action(action_type, target, params): 提出操作建议
- get_pending_actions(): 查看待确认的操作

## 操作类型

| 操作 | 类型 | 风险 | 需确认 |
|------|------|------|--------|
| restart_pod | 重启 Pod | 低 | 否 |
| scale_up | 扩容 | 低 | 是 |
| scale_down | 缩容 | 中 | 是 |
| drain_node | 摘除节点 | 中 | 是 |
| rollback | 回滚发布 | 高 | 是 |
| toggle_feature | 开关降级 | 中 | 是 |
| clear_cache | 清理缓存 | 低 | 否 |
"""

@tool
def propose_action(action_type: str, target: str, params: str = "") -> str:
    """提出一个止损操作建议，记录到待确认列表。

    Args:
        action_type: 操作类型 (restart_pod/scale_up/scale_down/drain_node/rollback/toggle_feature/clear_cache)
        target: 操作目标 (服务名/Pod名/Deployment名)
        params: 操作参数 (副本数/版本号等)
    """
    action = {
        "type": action_type,
        "target": target,
        "params": params,
    }
    scoped_pending_actions.add("default", action)
    action_id = f"act_{len(scoped_pending_actions.get('default')):04d}"
    return f"操作已记录 [{action_id}]: {action_type} → {target} (等待确认)"


@tool
def get_pending_actions() -> str:
    """查看所有待确认的操作建议。"""
    actions = scoped_pending_actions.get("default")
    if not actions:
        return "目前没有待确认的操作"
    lines = ["待确认的操作:"]
    for a in actions:
        lines.append(f"  [{a.get('id', '?')}] {a.get('type', '?')} → {a.get('target', '?')} ({a.get('status', 'pending')})")
    return "\n".join(lines)


def build_action_agent(llm, tools: list):
    return create_react_agent(
        model=llm,
        tools=tools + [propose_action, get_pending_actions],
        prompt=SystemMessage(content=ACTION_SYSTEM_PROMPT),
    )
