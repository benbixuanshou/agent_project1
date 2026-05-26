"""
IncidentLearner — self-healing loop: record errors → cluster → generate guardrail rules.

Philosophy (Mitchell Hashimoto):
    "Every time the AI makes a mistake, engineer a solution so
     it never makes that mistake again."

This module:
1. Records each agent error as an Incident with full context
2. Fingerprints and clusters similar incidents
3. When a cluster reaches threshold (3), auto-generates a candidate GuardrailRule
4. Candidate rules go to candidates/ directory for human review
5. Approved rules are merged into rules.yaml
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("superbizagent.harness.incident")


@dataclass
class Incident:
    """A single agent error event with full context for analysis."""
    id: str
    query: str
    error_type: str
    error_message: str
    target_agent: str
    tool_calls: list[dict]
    tenant_id: str
    timestamp: float
    fingerprint: str  # deterministic hash for clustering
    cluster_id: str = ""


@dataclass
class IncidentCluster:
    """A group of similar incidents."""
    cluster_id: str
    fingerprint: str
    incidents: list[Incident] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    has_rule: bool = False
    candidate_rule: dict[str, Any] | None = None

    @property
    def count(self) -> int:
        return len(self.incidents)

    def add(self, incident: Incident):
        if not self.first_seen:
            self.first_seen = incident.timestamp
        self.last_seen = incident.timestamp
        self.incidents.append(incident)
        incident.cluster_id = self.cluster_id


class IncidentLearner:
    """Records, clusters, and learns from agent errors.

    Data flow:
        agent error → Incident.record() → fingerprint → cluster
        cluster.count >= 3 → generate candidate GuardrailRule
        candidate → human review → merge into rules.yaml

    Persistence: writes are buffered. Call flush() periodically or
    on shutdown to persist. Buffer auto-flushes every N records.
    """

    FLUSH_INTERVAL = 10  # auto-flush every N incidents

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "candidates").mkdir(parents=True, exist_ok=True)

        self._incidents: list[Incident] = []
        self._clusters: dict[str, IncidentCluster] = {}
        self._pending_writes: int = 0  # buffer counter, flush when >= FLUSH_INTERVAL
        self._load_state()

    def _load_state(self):
        """Load persisted incidents and clusters from disk."""
        incidents_file = self.data_dir / "incidents.jsonl"
        clusters_file = self.data_dir / "clusters.json"

        if incidents_file.exists():
            try:
                with open(incidents_file, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            incident = Incident(**data)
                            self._incidents.append(incident)
                            fp = incident.fingerprint
                            if fp not in self._clusters:
                                self._clusters[fp] = IncidentCluster(
                                    cluster_id=f"cluster_{hashlib.md5(fp.encode()).hexdigest()[:8]}",
                                    fingerprint=fp,
                                )
                            self._clusters[fp].add(incident)
            except Exception:
                logger.warning("failed to load incident state")

        if clusters_file.exists():
            try:
                with open(clusters_file, encoding="utf-8") as f:
                    for cls_data in json.load(f):
                        fp = cls_data["fingerprint"]
                        if fp in self._clusters:
                            self._clusters[fp].has_rule = cls_data.get("has_rule", False)
                            self._clusters[fp].candidate_rule = cls_data.get("candidate_rule")
            except Exception:
                logger.warning("failed to load cluster state")

    def flush(self):
        """Persist buffered incidents and clusters to disk.

        Called automatically every FLUSH_INTERVAL records or when a rule
        is generated. Also call on application shutdown.
        """
        if self._pending_writes == 0:
            return
        self._save_state()
        self._pending_writes = 0

    def _save_state(self):
        """Persist incidents and clusters to disk."""
        incidents_file = self.data_dir / "incidents.jsonl"
        clusters_file = self.data_dir / "clusters.json"

        try:
            with open(incidents_file, "a", encoding="utf-8") as f:
                for incident in self._incidents[-self._pending_writes:]:
                    if self._pending_writes <= 0:
                        break
                    f.write(json.dumps(self._serialize_incident(incident), ensure_ascii=False) + "\n")
            self._rotate_if_needed(incidents_file, max_lines=10000)

            with open(clusters_file, "w", encoding="utf-8") as f:
                json.dump([
                    {
                        "cluster_id": c.cluster_id,
                        "fingerprint": c.fingerprint,
                        "count": c.count,
                        "has_rule": c.has_rule,
                        "candidate_rule": c.candidate_rule,
                        "first_seen": c.first_seen,
                        "last_seen": c.last_seen,
                    }
                    for c in self._clusters.values()
                ], f, ensure_ascii=False, indent=2)
        except Exception:
            logger.warning("failed to persist incident state")

    @staticmethod
    def _rotate_if_needed(filepath: Path, max_lines: int):
        """Keep only the last max_lines lines of the incident file."""
        if not filepath.exists():
            return
        lines = filepath.read_text(encoding="utf-8").splitlines()
        if len(lines) > max_lines:
            filepath.write_text("\n".join(lines[-max_lines:]), encoding="utf-8")

    @staticmethod
    def _serialize_incident(inc: Incident) -> dict:
        return {
            "id": inc.id,
            "query": inc.query[:500],
            "error_type": inc.error_type,
            "error_message": inc.error_message[:1000],
            "target_agent": inc.target_agent,
            "tool_calls": inc.tool_calls,
            "tenant_id": inc.tenant_id,
            "timestamp": inc.timestamp,
            "fingerprint": inc.fingerprint,
            "cluster_id": inc.cluster_id,
        }

    def _fingerprint(self, error_type: str, error_message: str, tool_name: str = "") -> str:
        """Create a deterministic fingerprint for clustering similar errors.

        Strategy: normalize aggressively — strip timestamps, ID numbers,
        attempt counters, durations, offsets — then hash. Only the error
        structure (type + tool + message shape) determines the fingerprint.
        """
        import re
        normalized = f"{error_type}|{tool_name}|{error_message}"

        # Normalize variable parts to placeholders
        normalized = re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?', '<TS>', normalized)
        normalized = re.sub(r'0x[0-9a-fA-F]+', '<HEX>', normalized)
        normalized = re.sub(r'\b\d{10,}\b', '<ID>', normalized)
        normalized = re.sub(r'/[a-zA-Z0-9/_\-.]{20,}', '<PATH>', normalized)
        normalized = re.sub(r'\b\d+\.\d+\.\d+\.\d+(?::\d+)?\b', '<IP>', normalized)
        normalized = re.sub(r'attempt \d+', 'attempt <N>', normalized)
        normalized = re.sub(r'#\d+', '#<N>', normalized)
        normalized = re.sub(r'at index \d+', 'at index <N>', normalized)
        normalized = re.sub(r'\btimed out after \d+\.\d+s\b', 'timed out after <DURATION>', normalized)
        normalized = re.sub(r'index \d+', 'index <N>', normalized)
        normalized = re.sub(r'\b\d+ms\b', '<MS>', normalized)
        normalized = re.sub(r'\b\d+s\b', '<SECONDS>', normalized)
        normalized = re.sub(r'\b\d+\.\d+s\b', '<DURATION>', normalized)
        normalized = re.sub(r'\b\d{6,}\b', '<LONG_NUM>', normalized)
        return hashlib.md5(normalized.encode()).hexdigest()[:16]

    async def record_error(
        self,
        query: str,
        error_type: str,
        error_message: str,
        target_agent: str = "",
        tool_calls: list[dict] | None = None,
        tenant_id: str = "default",
        tool_name: str = "",
    ) -> Incident:
        """Record a new error incident and check if it triggers rule generation."""
        tool_calls = tool_calls or []
        fp = self._fingerprint(error_type, error_message, tool_name)

        incident = Incident(
            id=f"inc_{int(time.time() * 1000)}_{hashlib.md5(fp.encode()).hexdigest()[:6]}",
            query=query[:500],
            error_type=error_type,
            error_message=error_message[:1000],
            target_agent=target_agent,
            tool_calls=tool_calls,
            tenant_id=tenant_id,
            timestamp=time.time(),
            fingerprint=fp,
        )

        if fp not in self._clusters:
            self._clusters[fp] = IncidentCluster(
                cluster_id=f"cluster_{hashlib.md5(fp.encode()).hexdigest()[:8]}",
                fingerprint=fp,
            )
            self._clusters[fp].first_seen = incident.timestamp

        self._clusters[fp].add(incident)
        self._incidents.append(incident)
        self._pending_writes += 1
        if self._pending_writes >= self.FLUSH_INTERVAL:
            self.flush()

        cluster = self._clusters[fp]
        if cluster.count >= 3 and not cluster.has_rule:
            candidate = self._generate_candidate_rule(cluster)
            cluster.candidate_rule = candidate
            cluster.has_rule = True
            await self._save_candidate(candidate)
            self.flush()  # persist cluster state change immediately (rule generated)
            logger.info(
                "incident cluster %s reached threshold (%d incidents), "
                "auto-generated candidate rule: %s",
                cluster.cluster_id, cluster.count, candidate.get("id", ""),
            )

        return incident

    def _generate_candidate_rule(self, cluster: IncidentCluster) -> dict:
        """Generate a candidate guardrail rule from a cluster of similar errors.

        Uses simple heuristics based on error type:
        - tool failures → circuit_break rule
        - hallucination → output validation rule
        - auth errors → input sanitize rule
        """
        error_type = cluster.incidents[0].error_type.lower()
        tool_calls = [t for inc in cluster.incidents for t in inc.tool_calls]
        tool_names = list(set(t.get("name", "") for t in tool_calls if t.get("name")))

        rule = {
            "id": f"auto_{cluster.cluster_id}",
            "description": f"Auto-generated from {cluster.count} incidents: {error_type}",
            "scope": "tool_call" if tool_names else "output",
            "action": "circuit_break" if "timeout" in error_type or "connection" in error_type else "block",
            "cooldown_seconds": 30,
            "message": f"[自动生成] 从 {cluster.count} 次同类错误中学习。类型: {error_type}。"
                       f"请人工审核后确认或修改。",
            "auto_generated": True,
            "source_cluster_id": cluster.cluster_id,
        }

        if tool_names:
            rule["when"] = {"tool": tool_names[0], "same_tool_consecutive_failures": 2}
        else:
            rule["when"] = {"output_regex": ""}
            rule["scope"] = "output"

        return rule

    async def _save_candidate(self, candidate: dict):
        """Write candidate rule to candidates/ directory for human review."""
        candidate_file = self.data_dir / "candidates" / f"{candidate['id']}.yaml"
        candidate_yaml = (
            f"# CANDIDATE RULE — auto-generated, needs human review\n"
            f"# Source: {candidate.get('source_cluster_id', 'unknown')}\n"
            f"# Auto-generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"# Review this rule, then move to app/harness/guardrails/rules.yaml\n\n"
        )
        candidate_yaml += f"- id: \"{candidate['id']}\"\n"
        candidate_yaml += f"  description: \"{candidate['description']}\"\n"
        candidate_yaml += f"  scope: \"{candidate['scope']}\"\n"
        candidate_yaml += f"  when: {candidate['when']}\n"
        candidate_yaml += f"  action: \"{candidate['action']}\"\n"
        if candidate.get("cooldown_seconds"):
            candidate_yaml += f"  cooldown_seconds: {candidate['cooldown_seconds']}\n"
        candidate_yaml += f"  message: \"{candidate['message']}\"\n"

        candidate_file.write_text(candidate_yaml, encoding="utf-8")

    def stats(self) -> dict:
        """Return incident statistics for observability."""
        total = len(self._incidents)
        clusters_with_rules = sum(1 for c in self._clusters.values() if c.has_rule)
        top_clusters = sorted(
            self._clusters.values(), key=lambda c: c.count, reverse=True
        )[:5]

        return {
            "total_incidents": total,
            "total_clusters": len(self._clusters),
            "clusters_with_rules": clusters_with_rules,
            "error_rate_change": None,  # would need historical baseline
            "top_clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "error_type": c.incidents[0].error_type if c.incidents else "",
                    "count": c.count,
                    "has_rule": c.has_rule,
                    "candidate_rule_id": c.candidate_rule.get("id") if c.candidate_rule else None,
                    "first_seen": c.first_seen,
                    "last_seen": c.last_seen,
                }
                for c in top_clusters
            ],
        }

    def get_candidates(self) -> list[dict]:
        """Return all pending candidate rules awaiting human review."""
        candidates_dir = self.data_dir / "candidates"
        if not candidates_dir.exists():
            return []
        candidates = []
        for f in sorted(candidates_dir.glob("*.yaml")):
            candidates.append({
                "file": f.name,
                "rule_id": f.stem,
                "content": f.read_text(encoding="utf-8")[:2000],
            })
        return candidates
