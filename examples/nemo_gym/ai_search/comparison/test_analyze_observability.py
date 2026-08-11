# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the multi-layer AI-search observability report."""

import json

from analyze_observability import (
    analyze_prometheus,
    analyze_traces,
    parse_training_console,
)


def _write_jsonl(path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_analyze_traces_links_resource_and_retriever_batches(tmp_path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    base = {
        "event": "span",
        "status": "ok",
        "start_unix_ns": 1_000_000_000,
        "end_unix_ns": 1_010_000_000,
        "duration_ms": 10.0,
    }
    _write_jsonl(
        trace_path,
        [
            base
            | {
                "trace_id": "session-1",
                "component": "resource_server",
                "operation": "search",
                "attributes": {"provider_batch_id": "batch-1", "queue_ms": 2.0},
            },
            base
            | {
                "trace_id": "retriever-batch:batch-1",
                "component": "search_r1_e5",
                "operation": "retrieve",
                "attributes": {"provider_batch_id": "batch-1", "encode_ms": 3.0},
            },
            base
            | {
                "trace_id": "session-1",
                "component": "search_r1_agent",
                "operation": "rollout",
                "attributes": {},
            },
            base
            | {
                "trace_id": "retriever-batch:smoke",
                "component": "search_r1_e5",
                "operation": "retrieve",
                "attributes": {"provider_batch_id": "smoke"},
            },
        ],
    )

    report = analyze_traces([trace_path])
    assert report["span_count"] == 3
    assert report["trajectory_count"] == 1
    assert report["provider_batch_links"]["linked_batches"] == 1
    assert report["provider_batch_links"]["retriever_batches"] == 1
    assert report["operations_ms"]["resource_server/search"]["p95"] == 10.0


def test_analyze_prometheus_calculates_interval_deltas(tmp_path) -> None:
    metrics_path = tmp_path / "prometheus.jsonl"
    _write_jsonl(
        metrics_path,
        [
            {
                "endpoint": "e5",
                "scraped_unix_ns": 1_000_000_000,
                "samples": [
                    {
                        "name": "search_r1_e5_requests_total",
                        "labels": {"outcome": "success"},
                        "value": 2.0,
                    },
                    {
                        "name": "search_r1_e5_requests_created",
                        "labels": {"outcome": "success"},
                        "value": 1_000_000_000.0,
                    },
                    {
                        "name": "search_r1_e5_stage_seconds_sum",
                        "labels": {"stage": "encode"},
                        "value": 1.0,
                    },
                    {
                        "name": "search_r1_e5_stage_seconds_count",
                        "labels": {"stage": "encode"},
                        "value": 2.0,
                    },
                ],
            },
            {
                "endpoint": "e5",
                "scraped_unix_ns": 3_000_000_000,
                "samples": [
                    {
                        "name": "search_r1_e5_requests_total",
                        "labels": {"outcome": "success"},
                        "value": 5.0,
                    },
                    {
                        "name": "search_r1_e5_stage_seconds_sum",
                        "labels": {"stage": "encode"},
                        "value": 2.5,
                    },
                    {
                        "name": "search_r1_e5_stage_seconds_count",
                        "labels": {"stage": "encode"},
                        "value": 5.0,
                    },
                ],
            },
        ],
    )

    report = analyze_prometheus([metrics_path])
    request_key = "e5:search_r1_e5_requests_total{outcome=success}"
    histogram_key = "e5:search_r1_e5_stage_seconds{stage=encode}"
    assert report["counter_deltas"][request_key] == 3.0
    assert report["histogram_interval_means"][histogram_key] == {
        "observations": 3.0,
        "mean": 0.5,
    }
    assert not any("_created" in key for key in report["gauge_summaries"])
    assert report["duration_seconds"] == 2.0


def test_analyze_prometheus_separates_interior_and_teardown_errors(tmp_path) -> None:
    metrics_path = tmp_path / "prometheus.jsonl"
    _write_jsonl(
        metrics_path,
        [
            {
                "endpoint": "resource",
                "scraped_unix_ns": 1,
                "scrape_duration_ms": 2.0,
                "samples": [],
            },
            {
                "endpoint": "resource",
                "scraped_unix_ns": 2,
                "error_type": "TimeoutError",
                "samples": [],
            },
            {
                "endpoint": "resource",
                "scraped_unix_ns": 3,
                "scrape_duration_ms": 4.0,
                "samples": [],
            },
            {
                "endpoint": "resource",
                "scraped_unix_ns": 4,
                "error_type": "URLError",
                "samples": [],
            },
        ],
    )

    health = analyze_prometheus([metrics_path])["endpoint_health"]["resource"]
    assert health["successful_snapshots"] == 2
    assert health["interior_error_snapshots"] == 1
    assert health["trailing_error_snapshots"] == 1
    assert health["success_rate"] == 0.5
    assert health["max_interior_consecutive_errors"] == 1
    assert health["error_types"] == {"TimeoutError": 1, "URLError": 1}
    assert health["scrape_duration_ms"]["mean"] == 3.0


def test_parse_training_console_excludes_warmup(tmp_path) -> None:
    console = tmp_path / "console.log"
    console.write_text(
        """
========================= Step 1/2 =========================
Collecting rollouts: 100%|██████████| 40/40 [00:10<00:00, 4.00it/s]
📊 Training Results:
  • Avg Reward: 0.10
  • Mean Generation Length: 200.00
⏱️  Timing:
  • Total step time: 10.00s
  • generation: 4.00s (40.0%)
🔍 Performance Metrics:
    - E2E (Tokens/sec/gpu): 20.00
========================= Step 2/2 =========================
Collecting rollouts: 100%|██████████| 40/40 [00:08<00:00, 5.00it/s]
📊 Training Results:
  • Avg Reward: 0.20
  • Mean Generation Length: 250.00
⏱️  Timing:
  • Total step time: 8.00s
  • generation: 3.00s (37.5%)
🔍 Performance Metrics:
    - E2E (Tokens/sec/gpu): 25.00
""",
        encoding="utf-8",
    )

    report = parse_training_console(console, warmup_steps=1)
    assert report["step_count"] == 2
    assert report["steady_step_count"] == 1
    assert report["timing_seconds"]["total_step_time"]["mean"] == 8.0
    assert report["throughput"]["E2E (Tokens/sec/gpu)"]["mean"] == 25.0
    assert report["results"]["avg_reward"]["mean"] == 0.2
    assert report["results"]["mean_generation_length"]["mean"] == 250.0
    assert report["work"]["completed_trajectories"]["mean"] == 40.0
    assert report["steps"][1]["requested_trajectories"] == 40
