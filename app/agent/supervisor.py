"""
Supervisor Agent — routes user queries to RAG Agent or SRE Agent.
Two-tier routing: IntentGateway rules (fast) + LLM fallback (edge cases).
Injects matched Skills into worker Agent context.
"""
from langchain_core.messages import SystemMessage, HumanMessage

from app.rag.intent import IntentGateway, IntentType
from app.skills.loader import SkillLoader

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


class Supervisor:
    """Route queries to the right Agent with skill injection.

    Pipeline per request:
    1. IntentGateway: classify → decide worker agent
    2. SkillLoader: match skills → inject into query context
    3. Worker Agent: invoke with skill-enhanced context

    Design rationale:
    - 2-tier routing: fast rules for clear cases, LLM fallback for edge cases.
      This saves LLM calls on 90%+ of requests (IntentGateway is deterministic).
    - Supervisor LLM uses T=0.01 (nearly deterministic) and 200 max_tokens
      to minimize cost on routing decisions.
    - Skills are injected per-request rather than at agent construction time.
      This allows the same agent instances to handle diverse queries without
      bloating the system prompt.
    """

    def __init__(self, llm, rag_agent, sre_agent, platform_agent=None, action_agent=None):
        self.llm = llm
        self.rag_agent = rag_agent
        self.sre_agent = sre_agent
        self.platform_agent = platform_agent
        self.action_agent = action_agent
        self.gateway = IntentGateway()
        self.skill_loader = SkillLoader()
        from app.skills.loader import set_skill_loader
        set_skill_loader(self.skill_loader)

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

    async def route(self, query: str) -> dict:
        fast = self._fast_route(query)
        if fast == "block":
            return {"target": "block", "reason": "intent gateway rejected",
                    "block_reply": "我是运维助手，只能回答运维和技术相关的问题。"}

        if fast:
            return {"target": fast, "reason": f"intent gateway (confidence > 0.15)"}

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=SUPERVISOR_PROMPT),
                HumanMessage(content=query),
            ])
            import json, re
            content = response.content or ""
            match = re.search(r"\{[^}]+\}", content)
            if match:
                decision = json.loads(match.group())
                target = decision.get("target", "rag")
                if target in ("rag", "sre", "rag_then_sre", "platform", "action"):
                    return {"target": target, "reason": decision.get("reason", "LLM routing")}
        except Exception:
            pass

        return {"target": "rag", "reason": "default fallback"}

    def _inject_context(self, query: str, target: str) -> str:
        """Build skill-enhanced query context.

        For SRE/Platform Agent: inject skill catalog (on-demand via search_skill)
        instead of full skill bodies. This saves tokens and lets the Agent decide
        which skills are actually relevant to the problem.
        """
        parts = []

        # IntentGateway prompt extension
        config = self.gateway.route(query)
        if config.prompt_extension:
            parts.append(f"[场景指引]\n{config.prompt_extension}")

        # Skill catalog for agents that do troubleshooting (SRE / Platform)
        if target in ("sre", "platform"):
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

    async def invoke(self, query: str, messages: list = None):
        decision = await self.route(query)

        if decision["target"] == "block":
            return decision["block_reply"]

        enhanced_query = self._inject_context(query, decision["target"])

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

        enhanced_query = self._inject_context(query, decision["target"])

        if decision["target"] in ("rag", "sre"):
            agent = self.rag_agent if decision["target"] == "rag" else self.sre_agent
        else:
            agent = self.sre_agent

        async for event in agent.astream(
            {"messages": messages or [{"role": "user", "content": enhanced_query}]},
            stream_mode="values",
        ):
            yield event
