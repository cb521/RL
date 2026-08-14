# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for AI-search trajectory and Prometheus observability."""

import json

import pytest
from prometheus_client import generate_latest

from resources_servers.ai_search.observability import trace_span
from resources_servers.ai_search.prometheus import SearchPrometheusMetrics
from resources_servers.ai_search.retrieval.types import SearchTimings


def test_trace_span_writes_bounded_jsonl(monkeypatch, tmp_path) -> None:
    trace_path = tmp_path / "trajectory.jsonl"
    monkeypatch.setenv("AI_SEARCH_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("AI_SEARCH_TRACE_SAMPLE_RATE", "1")

    with trace_span(
        trace_id="session-7",
        component="resource_server",
        operation="search",
        attributes={"query": "x" * 400},
    ) as span:
        span.set_attributes(queue_ms=2.5)

    event = json.loads(trace_path.read_text(encoding="utf-8"))
    assert event["schema_version"] == 1
    assert event["trace_id"] == "session-7"
    assert event["component"] == "resource_server"
    assert event["operation"] == "search"
    assert event["status"] == "ok"
    assert event["duration_ms"] >= 0
    assert event["attributes"]["queue_ms"] == 2.5
    assert len(event["attributes"]["query"]) == 256


def test_trace_span_rejects_invalid_configuration(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AI_SEARCH_TRACE_PATH", str(tmp_path / "trace.jsonl"))
    monkeypatch.setenv("AI_SEARCH_TRACE_SAMPLE_RATE", "1.5")

    with pytest.raises(ValueError, match="must be in"):
        with trace_span(
            trace_id="session",
            component="agent",
            operation="model_generate",
        ):
            pass


def test_prometheus_metrics_have_bounded_labels_and_stage_breakdown() -> None:
    metrics = SearchPrometheusMetrics("http")
    initial_payload = generate_latest(metrics.registry).decode("utf-8")
    assert (
        'nemo_ai_search_requests_total{outcome="success",provider="http"} 0.0'
        in initial_payload
    )
    assert (
        'nemo_ai_search_stage_latency_ms_count{provider="http",stage="queue"} 0.0'
        in initial_payload
    )

    metrics.observe_success(
        SearchTimings(
            queue_ms=1.0,
            encode_ms=2.0,
            index_ms=3.0,
            fetch_ms=4.0,
            total_ms=10.0,
            cache_hits=0,
            cache_misses=4,
            batch_size=4,
        )
    )
    metrics.observe_rejection("budget_exhausted")

    payload = generate_latest(metrics.registry).decode("utf-8")
    assert (
        'nemo_ai_search_requests_total{outcome="success",provider="http"} 1.0'
        in payload
    )
    assert 'nemo_ai_search_rejections_total{reason="budget_exhausted"} 1.0' in payload
    assert 'provider="http",stage="queue"' in payload
    assert 'provider="http",stage="provider_total"' in payload
    assert "session-" not in payload
