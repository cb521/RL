# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the multi-layer AI-search observability report."""

import json

import pytest

from analyze_observability import (
    analyze_gpu_samples,
    analyze_host_samples,
    analyze_prometheus,
    analyze_traces,
    parse_framework_training_console,
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


def test_parse_original_verl_console_normalizes_step_metrics(tmp_path) -> None:
    console = tmp_path / "original.log"
    console.write_text(
        "\n".join(
            (
                "step:1 - critic/rewards/mean:0.1 - response_length/mean:100 "
                "- timing_s/gen:4 - timing_s/policy_logprob:1 - timing_s/ref:1 "
                "- timing_s/adv:0.5 - timing_s/reward:0.2 "
                "- timing_s/advantage:0.1 - timing_s/update_actor:3 "
                "- timing_s/step:10 - perf/total_num_tokens:8000 "
                "- perf/throughput:100",
                "step:2 - critic/rewards/mean:0.2 - response_length/mean:120 "
                "- timing_s/gen:3 - timing_s/policy_logprob:1 - timing_s/ref:1 "
                "- timing_s/adv:0.4 - timing_s/reward:0.2 "
                "- timing_s/advantage:0.1 - timing_s/update_actor:2 "
                "- timing_s/step:8 - perf/total_num_tokens:8000 "
                "- perf/throughput:125",
            )
        ),
        encoding="utf-8",
    )

    report = parse_framework_training_console(
        console,
        framework="original",
        warmup_steps=1,
        trajectories_per_step=40,
    )
    assert report["steady_step_count"] == 1
    assert report["timing_seconds"]["total_step_time"]["mean"] == 8.0
    assert report["timing_seconds"]["generation_and_agent"]["mean"] == 3.0
    assert report["timing_seconds"]["postprocessing_reward_advantage"]["mean"] == 0.4
    assert report["throughput"]["E2E (Samples/sec)"]["mean"] == 5.0
    assert report["throughput"]["E2E (Tokens/sec)"]["mean"] == 1000.0
    assert report["results"]["avg_reward"]["mean"] == 0.2


def test_parse_current_verl_console_maps_advantage_without_double_counting(
    tmp_path,
) -> None:
    console = tmp_path / "current.log"
    console.write_text(
        "step:1 - timing_s/gen:4 - timing_s/reward:0.2 "
        "- timing_s/old_log_prob:1 - timing_s/ref:1 - timing_s/adv:0.3 "
        "- timing_s/update_actor:2 - timing_s/step:9 - perf/total_num_tokens:7200 "
        "- perf/throughput:100\n",
        encoding="utf-8",
    )

    report = parse_framework_training_console(
        console,
        framework="current",
        warmup_steps=0,
        trajectories_per_step=40,
    )
    assert report["timing_seconds"]["advantage_calculation"]["mean"] == 0.3
    assert "postprocessing_reward_advantage" not in report["timing_seconds"]
    assert report["timing_seconds"]["policy_logprobs"]["mean"] == 1.0


def test_parse_slime_console_requires_rollout_and_update_evidence(tmp_path) -> None:
    console = tmp_path / "slime.log"
    console.write_text(
        "\n".join(
            (
                "perf 0: {'rollout/response_len/mean': 100, "
                "'perf/rollout_time': 4, 'perf/tokens_per_gpu_per_sec': 20}",
                "perf 0: {'perf/train_wait_time': 5, 'perf/train_time': 5, "
                "'perf/step_time': 10, 'perf/log_probs_time': 1, "
                "'perf/ref_log_probs_time': 1, 'perf/actor_train_time': 2, "
                "'perf/actor_train_tok_per_s': 30}",
                "perf 1: {'rollout/response_len/mean': 120, "
                "'perf/rollout_time': 3, 'perf/tokens_per_gpu_per_sec': 25}",
                "perf 1: {'perf/train_wait_time': 4, 'perf/train_time': 4, "
                "'perf/step_time': 8, 'perf/log_probs_time': 0.8, "
                "'perf/ref_log_probs_time': 0.9, 'perf/actor_train_time': 1.8, "
                "'perf/actor_train_tok_per_s': 35}",
                "perf 2: {'perf/rollout_time': 1}",
            )
        ),
        encoding="utf-8",
    )

    report = parse_framework_training_console(
        console,
        framework="slime",
        warmup_steps=1,
        trajectories_per_step=40,
    )
    assert report["step_count"] == 2
    assert report["steady_step_count"] == 1
    assert report["timing_seconds"]["total_step_time"]["mean"] == 8.0
    assert report["timing_seconds"]["generation_and_agent"]["mean"] == 3.0
    assert report["throughput"]["E2E (Samples/sec)"]["mean"] == 5.0
    assert report["results"]["mean_generation_length"]["mean"] == 120.0


def test_analyze_gpu_samples_reports_utilization_memory_and_energy(tmp_path) -> None:
    samples = tmp_path / "gpu.csv"
    samples.write_text(
        "timestamp,index,uuid,memory_used_mib,memory_total_mib,gpu_util_percent,"
        "memory_util_percent,power_watts,temperature_c,sm_clock_mhz,"
        "memory_clock_mhz,pstate\n"
        "2026/08/11 10:00:00.000,0,GPU-a,1000,80000,50,10,100,60,1500,2000,P0\n"
        "2026/08/11 10:00:01.000,0,GPU-a,2000,80000,100,20,200,70,1600,2000,P0\n"
        "2026/08/11 10:00:00.000,1,GPU-b,3000,80000,0,5,50,50,1400,2000,P0\n"
        "2026/08/11 10:00:01.000,1,GPU-b,4000,80000,90,15,50,55,1500,2000,P0\n",
        encoding="utf-8",
    )

    report = analyze_gpu_samples([samples])
    assert report["gpu_count"] == 2
    assert report["duration_seconds"] == 1.0
    assert report["all_gpus"]["memory_used_mib"]["max"] == 4000.0
    assert report["all_gpus"]["gpu_util_percent"]["mean"] == 60.0
    assert report["energy"]["estimated_total_wh"] == pytest.approx(200 / 3600)
    assert report["per_gpu"]["GPU-a"]["high_utilization_sample_fraction"] == 0.5


def test_analyze_host_samples_sums_only_monotonic_network_deltas(tmp_path) -> None:
    samples = tmp_path / "host.csv"
    samples.write_text(
        "timestamp,mem_total_kib,mem_available_kib,load_1m,rx_bytes,tx_bytes\n"
        "2026-08-11T10:00:00-07:00,1000,800,1.0,100,200\n"
        "2026-08-11T10:00:01-07:00,1000,700,2.0,150,260\n"
        "2026-08-11T10:00:02-07:00,1000,600,3.0,20,30\n",
        encoding="utf-8",
    )

    report = analyze_host_samples([samples])
    assert report["duration_seconds"] == 2.0
    assert report["metrics"]["mem_used_kib"]["mean"] == 300.0
    assert report["metrics"]["load_1m"]["p95"] == pytest.approx(2.9)
    assert report["network_counter_deltas"] == {
        "available": True,
        "rx_bytes": 50.0,
        "tx_bytes": 60.0,
    }
