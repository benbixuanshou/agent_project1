"""
Skill loader for .claude/skills/*/SKILL.md files.
Compatible with Claude Code's native skill format.
Three-level progressive disclosure:
  Level 1 — Metadata (name + description) always loaded
  Level 2 — SKILL.md body injected when matched
  Level 3 — references/* scripts/* loaded on demand
"""

from pathlib import Path

import yaml
from langchain_core.tools import tool

# Global reference for the search_skill tool to access
_global_skill_loader: "SkillLoader" = None


def set_skill_loader(loader: "SkillLoader"):
    global _global_skill_loader
    _global_skill_loader = loader


@tool
def search_skill(skill_name: str) -> str:
    """查询并加载指定排查技能的完整流程、调用步骤和输出格式。
    在需要详细的排查步骤、分析维度或输出格式模板时调用此工具。

    Args:
        skill_name: 技能名称，如 'log-analyzer', 'sql-tuning', 'alert-triage', 'capacity-planning', 'report-writer'

    Returns:
        该技能的完整排查流程和输出格式
    """
    if _global_skill_loader is None:
        return "技能加载器未初始化"
    skill = _global_skill_loader.get_skill(skill_name)
    if skill is None:
        available = _global_skill_loader.list_skill_names()
        return f"未找到技能 '{skill_name}'。可用技能: {', '.join(available)}"
    return skill["body"]


class SkillLoader:
    def __init__(self, skill_dir: str = ".claude/skills"):
        self.skills: dict[str, dict] = {}
        self._load_all(skill_dir)

    def _load_all(self, base_dir: str):
        base = Path(base_dir)
        if not base.exists():
            return
        for skill_dir in base.iterdir():
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue

            meta, body = self._parse_frontmatter(skill_md.read_text(encoding="utf-8"))
            name = meta.get("name", skill_dir.name)

            self.skills[name] = {
                "name": name,
                "description": meta.get("description", ""),
                "allowed_tools": meta.get("allowed-tools", []),
                "body": body,
            }

    def _parse_frontmatter(self, text: str) -> tuple[dict, str]:
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                meta = yaml.safe_load(parts[1]) or {}
                body = parts[2].strip()
                return meta, body
        return {}, text

    # ── On-demand skill retrieval ──

    def get_skill(self, name: str) -> dict | None:
        """Return a single skill by name, or None if not found."""
        return self.skills.get(name)

    def list_skill_names(self) -> list[str]:
        """Return all loaded skill names."""
        return sorted(self.skills.keys())

    def get_catalog(self) -> str:
        """Return a compact catalog of all skills (name + description) for Agent context."""
        lines = []
        for name, skill in sorted(self.skills.items()):
            desc = skill["description"].replace("\n", " ").strip()
            # Truncate description to one sentence
            if "。" in desc:
                desc = desc.split("。")[0] + "。"
            elif ". " in desc:
                desc = desc.split(". ")[0] + "."
            elif len(desc) > 120:
                desc = desc[:120] + "..."
            lines.append(f"  **{name}**: {desc}")
        return "\n".join(lines)

    # ── Keyword matching (legacy, still used for non-SRE agents) ──

    def match(self, query: str, top_k: int = 2) -> list[dict]:
        """Match skills by keyword overlap with query."""
        query_lower = query.lower()
        scored = []
        for name, skill in self.skills.items():
            score = 0
            desc = skill["description"].lower()
            for word in query_lower.split():
                if word in desc or word in name.lower():
                    score += 1
            body = skill["body"].lower()
            for word in query_lower.split():
                if len(word) > 2 and word in body:
                    score += 2
            if score > 0:
                scored.append((score, name))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [self.skills[name] for _, name in scored[:top_k]]
