# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for framework-neutral sampled trajectory spans."""

import json

import pytest

from framework_observability import trace_span


def test_trace_span_writes_bounded_common_schema(tmp_path, monkeypatch) -> None:
    trace_path = tmp_path / "trajectory-spans.jsonl"
    monkeypatch.setenv("AI_SEARCH_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("AI_SEARCH_TRACE_SAMPLE_RATE", "1.0")

    with trace_span(
        trace_id="trajectory-1",
        component="slime",
        operation="retrieval",
        attributes={"provider_batch_id": "batch-1", "query": "q" * 300},
    ) as span:
        span.set_attributes(latency_ms=1.25)

    event = json.loads(trace_path.read_text(encoding="utf-8"))
    assert event["schema_version"] == 1
    assert event["event"] == "span"
    assert event["trace_id"] == "trajectory-1"
    assert event["component"] == "slime"
    assert event["operation"] == "retrieval"
    assert event["status"] == "ok"
    assert event["duration_ms"] >= 0.0
    assert event["attributes"]["provider_batch_id"] == "batch-1"
    assert len(event["attributes"]["query"]) == 256
    assert event["attributes"]["latency_ms"] == 1.25


def test_trace_sampling_zero_has_no_file(tmp_path, monkeypatch) -> None:
    trace_path = tmp_path / "trajectory-spans.jsonl"
    monkeypatch.setenv("AI_SEARCH_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("AI_SEARCH_TRACE_SAMPLE_RATE", "0")
    with trace_span(
        trace_id="trajectory-1",
        component="current_verl",
        operation="model_generation",
    ):
        pass
    assert not trace_path.exists()


def test_trace_config_rejects_relative_path(monkeypatch) -> None:
    monkeypatch.setenv("AI_SEARCH_TRACE_PATH", "relative.jsonl")
    with pytest.raises(ValueError, match="absolute path"):
        with trace_span(
            trace_id="trajectory-1",
            component="current_verl",
            operation="model_generation",
        ):
            pass
