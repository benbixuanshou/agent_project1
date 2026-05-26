"""Tests for IncidentLearner — error fingerprinting, clustering, rule generation."""

import asyncio
import tempfile
from pathlib import Path

import pytest
from app.harness.incident_learner import IncidentLearner


@pytest.fixture
def learner():
    with tempfile.TemporaryDirectory() as tmpdir:
        il = IncidentLearner(data_dir=tmpdir)
        yield il


class TestFingerprinting:
    def test_same_error_same_fingerprint(self, learner):
        fp1 = learner._fingerprint("TimeoutError", "Connection to k8s-api.example.com:6443 timed out after 10.0s (attempt 1)", "query_k8s_events")
        fp2 = learner._fingerprint("TimeoutError", "Connection to k8s-api.example.com:6443 timed out after 10.0s (attempt 2)", "query_k8s_events")
        assert fp1 == fp2

    def test_different_tool_different_fingerprint(self, learner):
        fp1 = learner._fingerprint("TimeoutError", "Connection timed out after 10.0s", "query_k8s_events")
        fp2 = learner._fingerprint("TimeoutError", "Connection timed out after 10.0s", "query_prometheus_alerts")
        assert fp1 != fp2

    def test_different_error_type_different_fingerprint(self, learner):
        fp1 = learner._fingerprint("TimeoutError", "timed out", "")
        fp2 = learner._fingerprint("HTTPError", "timed out", "")
        assert fp1 != fp2

    def test_fingerprint_normalizes_timestamps(self, learner):
        fp1 = learner._fingerprint("Error", "failed at 2026-05-26T23:07:22Z", "")
        fp2 = learner._fingerprint("Error", "failed at 2025-01-01T00:00:00Z", "")
        assert fp1 == fp2

    def test_fingerprint_normalizes_attempt_numbers(self, learner):
        fp1 = learner._fingerprint("Error", "failed (attempt 1)", "")
        fp2 = learner._fingerprint("Error", "failed (attempt 999)", "")
        assert fp1 == fp2


class TestClustering:
    @pytest.mark.asyncio
    async def test_single_incident_does_not_generate_rule(self, learner):
        await learner.record_error(
            query="test query", error_type="TestError",
            error_message="test error", target_agent="sre",
            tool_name="test_tool",
        )
        stats = learner.stats()
        assert stats["total_incidents"] == 1
        assert stats["total_clusters"] == 1
        assert stats["clusters_with_rules"] == 0

    @pytest.mark.asyncio
    async def test_three_similar_incidents_generate_candidate_rule(self, learner):
        for i in range(3):
            await learner.record_error(
                query=f"query {i}", error_type="TimeoutError",
                error_message=f"Connection timed out (attempt {i})",
                target_agent="sre", tool_name="query_k8s_events",
            )
        stats = learner.stats()
        assert stats["total_incidents"] == 3
        assert stats["total_clusters"] == 1
        assert stats["clusters_with_rules"] == 1
        candidates = learner.get_candidates()
        assert len(candidates) == 1
        assert "auto_cluster_" in candidates[0]["rule_id"]

    @pytest.mark.asyncio
    async def test_two_incidents_no_rule_generated(self, learner):
        for i in range(2):
            await learner.record_error(
                query=f"query {i}", error_type="TimeoutError",
                error_message=f"Connection timed out (attempt {i})",
                target_agent="sre", tool_name="test_tool",
            )
        assert learner.stats()["clusters_with_rules"] == 0

    @pytest.mark.asyncio
    async def test_different_errors_different_clusters(self, learner):
        await learner.record_error(
            query="q1", error_type="TimeoutError",
            error_message="timed out", tool_name="tool_a",
        )
        await learner.record_error(
            query="q2", error_type="HTTPError",
            error_message="503 unavailable", tool_name="tool_b",
        )
        assert learner.stats()["total_clusters"] == 2


class TestBuffer:
    @pytest.mark.asyncio
    async def test_buffer_flushes_at_interval(self, learner):
        for i in range(learner.FLUSH_INTERVAL + 2):
            await learner.record_error(
                query=f"q{i}", error_type="TestError",
                error_message=f"error {i}", tool_name="test_tool",
            )
        # After FLUSH_INTERVAL incidents, writes should have been flushed
        incidents_file = Path(learner.data_dir) / "incidents.jsonl"
        assert incidents_file.exists()

    @pytest.mark.asyncio
    async def test_rule_generation_triggers_immediate_flush(self, learner):
        for i in range(3):
            await learner.record_error(
                query=f"q{i}", error_type="TimeoutError",
                error_message=f"timed out (attempt {i})",
                target_agent="sre", tool_name="test_tool",
            )
        # Rule generation should have triggered an immediate flush
        clusters_file = Path(learner.data_dir) / "clusters.json"
        assert clusters_file.exists()
