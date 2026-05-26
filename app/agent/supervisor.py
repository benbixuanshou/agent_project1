"""
Supervisor Agent — routes user queries to RAG Agent or SRE Agent.
Two-tier routing: IntentGateway rules (fast) + LLM fallback (edge cases).
Injects matched Skills into worker Agent context.

Harness Engineering integration:
  - DecisionLog: audit every routing decision
  - GuardrailEngine: input/tool/output constraint enforcement
  - IncidentLearner: error recording and rule generation
"""

import json
import logging
import re
import time

from langchain_core.messages import SystemMessage, HumanMessage

from app.rag.intent import IntentGateway, IntentType
from app.skills.loader import SkillLoader
from app.self_monitor import agent_metrics

logger = logging.getLogger("superbizagent")

SUPERVISOR_PROMPT = """你是路由 Supervisor。根据用户问题，决定分配给哪个 Agent:

## 决策规则

1. **技术问答** — 概念解释、配置、原理 → rag_agent
2. **故障排查** — 告警、日志、性能诊断 → sre_agent
3. **基础设施诊断** — Pod 异常、K8s 事件、网络问题 → platform_agent
4. **止损操作建议** — 重启、扩容、降级 → action_agent (只建议，不执行)
5. **不确定/混合** → rag_then_sre

## 输出格式

只输出一个 JSON:
{"target": "rag"|"sre"|"platform"|"action"|"rag_then_sre", "reason": "一句话理由"}

不要输出其他内容，不要调用工具。
"""


class SupervisorHarnessDeps:
    """Harness components injected into Supervisor."""
    def __init__(self, guardrail_engine=None, incident_learner=None):
        self.guardrail_engine = guardrail_engine
        self.incident_learner = incident_learner


class Supervisor:
    """Route queries to the right Agent with harness integration.

    Pipeline per request:
    1. IntentGateway: classify → decide worker agent
    2. GuardrailEngine: input check
    3. SkillLoader: match skills → inject into query context
    4. Worker Agent: invoke with skill-enhanced context
    5. GuardrailEngine: output check
    6. IncidentLearner: record errors

    Design rationale:
    - 2-tier routing: fast rules for clear cases, LLM fallback for edge cases.
    - T=0.01 and 200 max_tokens to minimize cost on routing decisions.
    - Skills are injected per-request rather than at agent construction time.
    """

    def __init__(self, llm, rag_agent, sre_agent,
                 platform_agent=None, action_agent=None,
                 harness: SupervisorHarnessDeps | None = None):
        self.llm = llm
        self.rag_agent = rag_agent
        self.sre_agent = sre_agent
        self.platform_agent = platform_agent
        self.action_agent = action_agent
        self.gateway = IntentGateway()
        self.skill_loader = SkillLoader()
        from app.skills.loader import set_skill_loader
        set_skill_loader(self.skill_loader)

        h = harness or SupervisorHarnessDeps()
        self.guardrail_engine = h.guardrail_engine
        self.incident_learner = h.incident_learner

        from app.harness.decision_log import DecisionLog, RouteDecision
        self.decision_log = DecisionLog()

    def _fast_route(self, query: str) -> str | None:
        config = self.gateway.route(query)
        if config.block:
            return "block"

        result = self.gateway.recognizer.recognize(query)
        if result.confidence > 0.15:
            if result.intent in (IntentType.TROUBLESHOOTING,):
                return "sre"
            if result.intent in (IntentType.TECHNICAL_QUESTION, IntentType.CONFIGURATION,
                                 IntentType.PRODUCT_INQUIRY):
                return "rag"
        return None

    def _resolve_intent(self, query: str) -> str:
        """Resolve the intent type string for tool guidance injection."""
        result = self.gateway.recognizer.recognize(query)
        return result.intent.value if result.intent else ""

    async def route(self, query: str) -> dict:
        t0 = time.perf_counter()
        fast = self._fast_route(query)

        if fast == "block":
            self.decision_log.record(RouteDecision(
                query_snippet=query[:200], target="block",
                reason="intent gateway rejected", source="intent_gateway",
                confidence=0.0, latency_ms=(time.perf_counter() - t0) * 1000,
            ))
            return {"target": "block", "reason": "intent gateway rejected",
                    "block_reply": "我是运维助手，只能回答运维和技术相关的问题。"}

        if fast:
            self.decision_log.record(RouteDecision(
                query_snippet=query[:200], target=fast,
                reason="intent gateway (confidence > 0.15)", source="intent_gateway",
                confidence=0.15, latency_ms=(time.perf_counter() - t0) * 1000,
            ))
            return {"target": fast, "reason": f"intent gateway (confidence > 0.15)"}

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=SUPERVISOR_PROMPT),
                HumanMessage(content=query),
            ])
            agent_metrics.record_llm_success((time.perf_counter() - t0) * 1000)
            content = response.content or ""
            match = re.search(r"\{[^}]+\}", content)
            if match:
                decision_data = json.loads(match.group())
                target = decision_data.get("target", "rag")
                if target in ("rag", "sre", "rag_then_sre", "platform", "action"):
                    decision = RouteDecision(
                        query_snippet=query[:200], target=target,
                        reason=decision_data.get("reason", "LLM routing"),
                        source="llm_routing",
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )
                    self.decision_log.record(decision)
                    return {"target": target, "reason": decision_data.get("reason", "LLM routing")}
        except Exception as e:
            agent_metrics.record_llm_failure()
            logger.warning("supervisor LLM routing failed: %s, falling back to RAG agent", e)
            # Record incident for self-healing
            if self.incident_learner:
                await self.incident_learner.record_error(
                    query=query, error_type=type(e).__name__,
                    error_message=str(e), target_agent="supervisor",
                )

        # Default fallback — explicitly logged
        decision = RouteDecision(
            query_snippet=query[:200], target="rag",
            reason="default fallback (LLM routing failed or no match)",
            source="default_fallback",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
        self.decision_log.record(decision)
        return {"target": "rag", "reason": "default fallback"}

    def _inject_context(self, query: str, target: str,
                         intent_type: str = "") -> str:
        """Build skill-enhanced query context with intent-based tool guidance.

        For SRE/Platform Agent:
          - inject skill catalog for on-demand loading via search_skill
          - inject intent-based tool guidance to focus LLM on relevant tools
          This reduces token waste from listing all 15 tools when only 3-5 are relevant.
        """
        parts = []

        # IntentGateway prompt extension
        config = self.gateway.route(query)
        if config.prompt_extension:
            parts.append(f"[场景指引]\n{config.prompt_extension}")

        # Intent-based tool guidance for SRE/Platform agents
        if target in ("sre", "platform"):
            tool_guidance = self._tool_guidance_for_intent(intent_type)
            if tool_guidance:
                parts.append(tool_guidance)

            catalog = self.skill_loader.get_catalog()
            guide = self.skill_loader.get_skill("skill-selector")
            if guide:
                parts.append(
                    f"[技能选择指引]\n"
                    f"请根据问题选择合适的排查技能。可用技能目录:\n\n"
                    f"{catalog}\n\n"
                    f"用法: 调用 search_skill(name) 加载选中技能的完整排查流程和输出格式。"
                    f"不确定时先加载最接近的一个试试。"
                )

        if parts:
            return "\n\n".join(parts) + f"\n\n用户问题: {query}"
        return query

    @staticmethod
    def _tool_guidance_for_intent(intent_type: str) -> str:
        """Return tool focus guidance based on intent category.

        Tells the LLM which tool categories to prioritize, reducing cognitive
        load from seeing all 15 tools in every request.
        """
        guidance = {
            "troubleshooting": (
                "[工具优先级]\n"
                "核心工具: query_prometheus_alerts → query_logs → query_k8s_events\n"
                "辅助工具: query_recent_deployments, search_knowledge_base, search_skill\n"
                "按此顺序排查：告警 → 日志 → 事件 → 变更。不要跳过步骤。"
            ),
            "cost_analysis": (
                "[工具优先级]\n"
                "核心工具: check_cost_anomaly → predict_capacity → query_prometheus_alerts\n"
                "先看成本异常，再预测容量趋势，最后交叉验证告警。"
            ),
            "compliance": (
                "[工具优先级]\n"
                "核心工具: run_compliance_check → query_recent_deployments\n"
                "专注于合规检查，不需要查告警或日志。"
            ),
            "technical_question": (
                "[工具优先级]\n"
                "核心工具: search_knowledge_base（优先）→ web_search（兜底）\n"
                "这是知识查询，不需要调用监控或日志工具。"
            ),
        }
        return guidance.get(intent_type, "")

    async def invoke(self, query: str, messages: list = None):
        decision = await self.route(query)

        if decision["target"] == "block":
            return decision["block_reply"]

        intent_type = self._resolve_intent(query)
        enhanced_query = self._inject_context(query, decision["target"], intent_type)

        if decision["target"] == "rag":
            agent = self.rag_agent
        elif decision["target"] == "sre":
            agent = self.sre_agent
        elif decision["target"] == "platform" and self.platform_agent:
            agent = self.platform_agent
        elif decision["target"] == "action" and self.action_agent:
            agent = self.action_agent
        else:  # rag_then_sre
            result = await self.rag_agent.ainvoke({
                "messages": [{"role": "user", "content": query}]
            })
            rag_answer = result["messages"][-1].content
            enhanced_query = f"知识库参考: {rag_answer[:500]}\n\n用户问题: {query}"
            agent = self.sre_agent

        result = await agent.ainvoke({
            "messages": messages or [{"role": "user", "content": enhanced_query}]
        })
        return result["messages"][-1].content

    async def astream(self, query: str, messages: list = None):
        decision = await self.route(query)

        if decision["target"] == "block":
            yield {"type": "content", "data": decision["block_reply"]}
            return

        intent_type = self._resolve_intent(query)
        enhanced_query = self._inject_context(query, decision["target"], intent_type)

        # Fixed: handle ALL agent targets, not just rag/sre
        target = decision["target"]
        if target == "rag":
            agent = self.rag_agent
        elif target == "sre":
            agent = self.sre_agent
        elif target == "platform" and self.platform_agent:
            agent = self.platform_agent
        elif target == "action" and self.action_agent:
            agent = self.action_agent
        elif target == "rag_then_sre":
            result = await self.rag_agent.ainvoke({
                "messages": [{"role": "user", "content": query}]
            })
            rag_answer = result["messages"][-1].content
            enhanced_query = f"知识库参考: {rag_answer[:500]}\n\n用户问题: {query}"
            agent = self.sre_agent
        else:
            agent = self.sre_agent
            logger.warning("astream: unknown target '%s', defaulting to sre_agent", target)

        async for event in agent.astream(
            {"messages": messages or [{"role": "user", "content": enhanced_query}]},
            stream_mode="values",
        ):
            yield event
