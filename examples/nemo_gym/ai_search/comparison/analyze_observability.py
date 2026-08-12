# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Merge training, Prometheus, and trajectory traces into one breakdown report."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


_ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_STEP_PATTERN = re.compile(r"=+ Step (\d+)/(\d+) =+")
_TIMING_PATTERN = re.compile(
    r"^\s*•\s+(.+?):\s+([0-9]+(?:\.[0-9]+)?)s(?:\s+\([^)]+\))?\s*$"
)
_THROUGHPUT_PATTERN = re.compile(r"^\s*-\s+(.+?):\s+([0-9]+(?:\.[0-9]+)?)\s*$")
_RESULT_PATTERN = re.compile(
    r"^\s*•\s+(.+?):\s+([-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)\s*$"
)
_ROLLOUT_PATTERN = re.compile(
    r"Collecting rollouts:\s+100%.*?\|\s*([0-9]+)/([0-9]+)(?:\s|$)"
)
_VERL_STEP_PATTERN = re.compile(r"(?:^|\s)step:(\d+)\s+-")
_SLIME_PERF_PATTERN = re.compile(r"perf (\d+):\s*\{(.*)\}")
_DICT_NUMBER_PATTERN = re.compile(
    r"['\"]([^'\"]+)['\"]:\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)

_VERL_TIMING_NAMES = {
    "step": "total_step_time",
    "gen": "generation_and_agent",
    "policy_logprob": "policy_logprobs",
    "old_log_prob": "policy_logprobs",
    "ref": "reference_logprobs",
    "reward": "reward_calculation",
    "advantage": "advantage_calculation",
    "adv": "postprocessing_reward_advantage",
    "update_actor": "policy_training",
    "update_weights": "policy_to_engine_synchronization",
}

_SLIME_TIMING_NAMES = {
    "step_time": "total_step_time",
    "rollout_time": "generation_and_agent",
    "log_probs_time": "policy_logprobs",
    "ref_log_probs_time": "reference_logprobs",
    "actor_train_time": "policy_training",
    "train_wait_time": "trainer_wait_inclusive",
    "train_time": "trainer_work_inclusive",
    "data_preprocess_time": "data_preprocess",
    "update_weights_time": "policy_to_engine_synchronization",
}

_WORK_NAMES = (
    "completed_trajectories",
    "requested_trajectories",
    "generated_tokens",
    "observation_tokens",
    "response_tokens",
    "processed_tokens",
    "search_count",
    "search_errors",
    "invalid_actions",
    "trajectory_wall_seconds_mean",
    "trajectory_wall_seconds_max",
)


def _add_canonical_throughput(
    step: dict[str, Any], *, training_gpus: int
) -> None:
    """Compute rates only from explicitly classified work counters."""
    total_seconds = step["timing_seconds"].get("total_step_time")
    if not isinstance(total_seconds, (int, float)) or total_seconds <= 0:
        return
    throughput = step["throughput"]
    completed = step.get("completed_trajectories")
    if isinstance(completed, (int, float)):
        throughput["E2E (Samples/sec)"] = completed / total_seconds
    token_rates = {
        "generated_tokens": "E2E Model-generated (Tokens/sec)",
        "response_tokens": "E2E Response-sequence (Tokens/sec)",
        "processed_tokens": "E2E Processed-sequence (Tokens/sec)",
    }
    for work_name, rate_name in token_rates.items():
        value = step.get(work_name)
        if not isinstance(value, (int, float)):
            continue
        rate = value / total_seconds
        throughput[rate_name] = rate
        throughput[f"{rate_name[:-1]}/gpu)"] = rate / training_gpus

    generated = step.get("generated_tokens")
    generation_seconds = step["timing_seconds"].get("generation_and_agent")
    if generation_seconds is None:
        generation_seconds = step["timing_seconds"].get("generation")
    if (
        isinstance(generated, (int, float))
        and isinstance(generation_seconds, (int, float))
        and generation_seconds > 0
    ):
        throughput["Generation+agent Model-generated (Tokens/sec)"] = (
            generated / generation_seconds
        )


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    """Return the same bounded summary for every latency and throughput series."""
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "p50": _percentile(finite, 50),
        "p95": _percentile(finite, 95),
        "max": max(finite),
    }


def _read_jsonl(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSONL at {path}:{line_number}"
                    ) from error
                if not isinstance(value, dict):
                    raise ValueError(f"Expected an object at {path}:{line_number}")
                yield value


def _inside_window(
    timestamp_seconds: float,
    window: tuple[float, float] | None,
) -> bool:
    return window is None or window[0] <= timestamp_seconds <= window[1]


def _merge_intervals(
    intervals: Iterable[tuple[int, int]],
) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError(f"Trace interval ends before it starts: {start}, {end}")
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _interval_duration_ms(intervals: Iterable[tuple[int, int]]) -> float:
    return sum(end - start for start, end in _merge_intervals(intervals)) / 1e6


def _interval_intersection_ms(
    first: Iterable[tuple[int, int]], second: Iterable[tuple[int, int]]
) -> float:
    first_merged = _merge_intervals(first)
    second_merged = _merge_intervals(second)
    first_index = 0
    second_index = 0
    overlap_ns = 0
    while first_index < len(first_merged) and second_index < len(second_merged):
        first_start, first_end = first_merged[first_index]
        second_start, second_end = second_merged[second_index]
        overlap_ns += max(
            0,
            min(first_end, second_end) - max(first_start, second_start),
        )
        if first_end <= second_end:
            first_index += 1
        else:
            second_index += 1
    return overlap_ns / 1e6


def analyze_traces(
    paths: list[Path],
    *,
    window: tuple[float, float] | None = None,
) -> dict[str, Any]:
    events = [
        event
        for event in _read_jsonl(paths)
        if event.get("event") == "span"
        and _inside_window(int(event["start_unix_ns"]) / 1_000_000_000.0, window)
        and _inside_window(int(event["end_unix_ns"]) / 1_000_000_000.0, window)
    ]
    operation_durations: dict[str, list[float]] = defaultdict(list)
    operation_intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    attribute_values: dict[str, list[float]] = defaultdict(list)
    trace_bounds: dict[str, list[tuple[int, int]]] = defaultdict(list)
    resource_batch_ids = {
        str(attributes["provider_batch_id"])
        for event in events
        if event.get("component") == "resource_server"
        and isinstance((attributes := event.get("attributes")), dict)
        and isinstance(attributes.get("provider_batch_id"), str)
    }
    retriever_batch_ids: set[str] = set()
    trajectory_trace_ids: set[str] = set()
    status_counts: dict[str, int] = defaultdict(int)
    span_count = 0
    all_intervals: list[tuple[int, int]] = []
    model_intervals: list[tuple[int, int]] = []
    retrieval_intervals: list[tuple[int, int]] = []

    for event in events:
        component = str(event.get("component", "unknown"))
        attributes = event.get("attributes")
        batch_id = (
            attributes.get("provider_batch_id")
            if isinstance(attributes, dict)
            else None
        )
        # A long-lived retriever trace can contain smoke probes and spans from
        # earlier framework runs. When resource spans are present, retain only
        # retriever work linked to this run's provider batches.
        if (
            component == "search_r1_e5"
            and resource_batch_ids
            and batch_id not in resource_batch_ids
        ):
            continue
        span_count += 1
        operation = str(event.get("operation", "unknown"))
        operation_key = f"{component}/{operation}"
        operation_durations[operation_key].append(
            float(event["duration_ms"])
        )
        interval = (int(event["start_unix_ns"]), int(event["end_unix_ns"]))
        operation_intervals[operation_key].append(interval)
        all_intervals.append(interval)
        if operation in {"model_generate", "model_generation"}:
            model_intervals.append(interval)
        if (
            operation
            in {"retrieval", "retrieval_http", "search", "search_tool", "retrieve"}
            or component in {"resource_server", "search_r1_e5"}
        ):
            retrieval_intervals.append(interval)
        if component == "search_r1_agent" and operation == "rollout":
            trajectory_trace_ids.add(str(event.get("trace_id", "")))
        status_counts[str(event.get("status", "unknown"))] += 1
        trace_id = str(event.get("trace_id", ""))
        trace_bounds[trace_id].append(
            (int(event["start_unix_ns"]), int(event["end_unix_ns"]))
        )
        if not isinstance(attributes, dict):
            continue
        for name, value in attributes.items():
            if name.endswith("_ms") and isinstance(value, (int, float)):
                attribute_values[f"{component}/{operation}/{name}"].append(float(value))
        if not isinstance(batch_id, str):
            continue
        if component == "search_r1_e5":
            retriever_batch_ids.add(batch_id)

    trajectory_wall_ms = []
    for trace_id in sorted(trajectory_trace_ids):
        bounds = trace_bounds[trace_id]
        if not bounds:
            continue
        trajectory_wall_ms.append(
            (max(end for _, end in bounds) - min(start for start, _ in bounds))
            / 1_000_000.0
        )

    window_duration_ms = (
        (window[1] - window[0]) * 1000.0 if window is not None else None
    )
    operation_activity = {}
    for name, intervals in sorted(operation_intervals.items()):
        wall_union_ms = _interval_duration_ms(intervals)
        inclusive_sum_ms = sum(operation_durations[name])
        operation_activity[name] = {
            "inclusive_span_sum_ms": inclusive_sum_ms,
            "wall_union_ms": wall_union_ms,
            "mean_concurrency_when_active": (
                inclusive_sum_ms / wall_union_ms if wall_union_ms else 0.0
            ),
            "measurement_window_coverage_fraction": (
                wall_union_ms / window_duration_ms
                if window_duration_ms is not None and window_duration_ms > 0
                else None
            ),
        }
    model_union_ms = _interval_duration_ms(model_intervals)
    retrieval_union_ms = _interval_duration_ms(retrieval_intervals)
    any_union_ms = _interval_duration_ms(all_intervals)
    model_retrieval_overlap_ms = _interval_intersection_ms(
        model_intervals, retrieval_intervals
    )

    return {
        "span_count": span_count,
        "trace_count": len(trace_bounds),
        "trajectory_count": len(trajectory_wall_ms),
        "status_counts": dict(sorted(status_counts.items())),
        "trajectory_wall_ms": summarize(trajectory_wall_ms),
        "operations_ms": {
            name: summarize(values)
            for name, values in sorted(operation_durations.items())
        },
        "operation_activity": operation_activity,
        "timeline_activity": {
            "model_generation_wall_union_ms": model_union_ms,
            "retrieval_wall_union_ms": retrieval_union_ms,
            "model_retrieval_overlap_ms": model_retrieval_overlap_ms,
            "any_traced_activity_wall_union_ms": any_union_ms,
            "measurement_window_ms": window_duration_ms,
            "interpretation": (
                "Wall unions collapse concurrent trajectory spans. Model and "
                "retrieval unions can overlap and are inclusive evidence, not "
                "additive critical-path buckets. Clean traces may be sampled."
            ),
        },
        "stage_attributes_ms": {
            name: summarize(values) for name, values in sorted(attribute_values.items())
        },
        "provider_batch_links": {
            "resource_batches": len(resource_batch_ids),
            "retriever_batches": len(retriever_batch_ids),
            "linked_batches": len(resource_batch_ids & retriever_batch_ids),
        },
    }


def _series_key(name: str, labels: dict[str, str]) -> str:
    if not labels:
        return name
    rendered = ",".join(f"{key}={labels[key]}" for key in sorted(labels))
    return f"{name}{{{rendered}}}"


def analyze_prometheus(
    paths: list[Path],
    *,
    window: tuple[float, float] | None = None,
) -> dict[str, Any]:
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    endpoints: set[str] = set()
    endpoint_outcomes: dict[str, list[bool]] = defaultdict(list)
    endpoint_error_types: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    scrape_durations_ms: dict[str, list[float]] = defaultdict(list)
    error_snapshots = 0
    snapshot_count = 0

    for snapshot in _read_jsonl(paths):
        timestamp = int(snapshot["scraped_unix_ns"])
        if not _inside_window(timestamp / 1_000_000_000.0, window):
            continue
        snapshot_count += 1
        endpoint = str(snapshot.get("endpoint", "unknown"))
        endpoints.add(endpoint)
        if "error_type" in snapshot:
            error_snapshots += 1
            endpoint_outcomes[endpoint].append(False)
            endpoint_error_types[endpoint][str(snapshot["error_type"])] += 1
            continue
        endpoint_outcomes[endpoint].append(True)
        if "scrape_duration_ms" in snapshot:
            scrape_durations_ms[endpoint].append(float(snapshot["scrape_duration_ms"]))
        raw_samples = snapshot.get("samples", [])
        if not isinstance(raw_samples, list):
            continue
        for sample in raw_samples:
            if not isinstance(sample, dict) or sample.get("value") is None:
                continue
            name = str(sample["name"])
            if name.endswith("_created"):
                continue
            labels = sample.get("labels", {})
            if not isinstance(labels, dict):
                labels = {}
            key = f"{endpoint}:{_series_key(name, labels)}"
            series[key].append((timestamp, float(sample["value"])))

    counter_deltas: dict[str, float] = {}
    gauge_summaries: dict[str, dict[str, float | int]] = {}
    histogram_parts: dict[str, dict[str, float]] = defaultdict(dict)
    for key, points in sorted(series.items()):
        points.sort()
        values = [value for _, value in points]
        if key.endswith("_total") or "_total{" in key:
            counter_deltas[key] = max(0.0, values[-1] - values[0])
        elif key.endswith("_sum") or "_sum{" in key:
            base = key.replace("_sum{", "{") if "_sum{" in key else key[:-4]
            histogram_parts[base]["sum"] = max(0.0, values[-1] - values[0])
        elif key.endswith("_count") or "_count{" in key:
            base = key.replace("_count{", "{") if "_count{" in key else key[:-6]
            histogram_parts[base]["count"] = max(0.0, values[-1] - values[0])
        elif "_bucket" not in key:
            gauge_summaries[key] = summarize(values)

    histogram_means = {}
    for key, parts in sorted(histogram_parts.items()):
        count = parts.get("count", 0.0)
        histogram_means[key] = {
            "observations": count,
            "mean": parts.get("sum", 0.0) / count if count else 0.0,
        }

    timestamps = [timestamp for points in series.values() for timestamp, _ in points]
    duration_seconds = (
        (max(timestamps) - min(timestamps)) / 1_000_000_000.0 if timestamps else 0.0
    )
    endpoint_health = {}
    for endpoint in sorted(endpoints):
        outcomes = endpoint_outcomes[endpoint]
        successful = sum(outcomes)
        last_success = max(
            (index for index, outcome in enumerate(outcomes) if outcome), default=-1
        )
        interior_errors = sum(
            not outcome
            for index, outcome in enumerate(outcomes)
            if index < last_success
        )
        trailing_errors = sum(
            not outcome
            for index, outcome in enumerate(outcomes)
            if index > last_success
        )
        longest_error_run = 0
        longest_interior_error_run = 0
        current_error_run = 0
        for index, outcome in enumerate(outcomes):
            if outcome:
                current_error_run = 0
                continue
            current_error_run += 1
            longest_error_run = max(longest_error_run, current_error_run)
            if index < last_success:
                longest_interior_error_run = max(
                    longest_interior_error_run, current_error_run
                )

        endpoint_health[endpoint] = {
            "snapshot_count": len(outcomes),
            "successful_snapshots": successful,
            "error_snapshots": len(outcomes) - successful,
            "interior_error_snapshots": interior_errors,
            "trailing_error_snapshots": trailing_errors,
            "max_consecutive_errors": longest_error_run,
            "max_interior_consecutive_errors": longest_interior_error_run,
            "success_rate": successful / len(outcomes) if outcomes else 0.0,
            "error_types": dict(sorted(endpoint_error_types[endpoint].items())),
            "scrape_duration_ms": summarize(scrape_durations_ms[endpoint]),
        }
    return {
        "snapshot_count": snapshot_count,
        "error_snapshots": error_snapshots,
        "endpoints": sorted(endpoints),
        "endpoint_health": endpoint_health,
        "duration_seconds": duration_seconds,
        "counter_deltas": counter_deltas,
        "gauge_summaries": gauge_summaries,
        "histogram_interval_means": histogram_means,
    }


def _parse_timestamp(value: str) -> float:
    normalized = value.strip().replace("Z", "+00:00")
    try:
        return dt.datetime.fromisoformat(normalized).timestamp()
    except ValueError:
        pass
    for pattern in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(normalized, pattern).timestamp()
        except ValueError:
            continue
    raise ValueError(f"Unsupported sample timestamp: {value!r}")


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value.strip())
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _integrate_power(points: list[tuple[float, float]]) -> dict[str, float | int]:
    points.sort()
    energy_joules = 0.0
    integrated_seconds = 0.0
    skipped_gap_seconds = 0.0
    interval_count = 0
    for (first_time, first_power), (second_time, second_power) in zip(
        points, points[1:]
    ):
        elapsed = second_time - first_time
        if elapsed <= 0:
            continue
        # The collector samples once per second. Do not invent energy during a
        # long monitoring outage; retain the uncovered gap explicitly.
        if elapsed > 5.0:
            skipped_gap_seconds += elapsed
            continue
        energy_joules += (first_power + second_power) * 0.5 * elapsed
        integrated_seconds += elapsed
        interval_count += 1
    observed_seconds = points[-1][0] - points[0][0] if len(points) > 1 else 0.0
    return {
        "energy_wh": energy_joules / 3600.0,
        "integrated_seconds": integrated_seconds,
        "observed_seconds": observed_seconds,
        "skipped_gap_seconds": skipped_gap_seconds,
        "coverage_fraction": (
            integrated_seconds / observed_seconds if observed_seconds else 0.0
        ),
        "interval_count": interval_count,
    }


def analyze_gpu_samples(
    paths: list[Path],
    *,
    window: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Summarize one-second nvidia-smi samples without hiding missing values."""
    metric_columns = (
        "memory_used_mib",
        "memory_total_mib",
        "gpu_util_percent",
        "memory_util_percent",
        "power_watts",
        "temperature_c",
        "sm_clock_mhz",
        "memory_clock_mhz",
    )
    per_gpu_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    per_gpu_power: dict[str, list[tuple[float, float]]] = defaultdict(list)
    timestamps: list[float] = []
    row_count = 0
    missing_columns: set[str] = set()

    for path in paths:
        with path.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            fieldnames = set(reader.fieldnames or ())
            required = {"timestamp", "index"}
            missing_required = required.difference(fieldnames)
            if missing_required:
                raise ValueError(
                    f"GPU sample CSV {path} is missing {sorted(missing_required)}"
                )
            missing_columns.update(set(metric_columns).difference(fieldnames))
            if "uuid" not in fieldnames:
                missing_columns.add("uuid")
            for row in reader:
                timestamp = _parse_timestamp(row["timestamp"])
                if not _inside_window(timestamp, window):
                    continue
                uuid = (row.get("uuid") or "").strip()
                gpu = uuid or (
                    f"index:{row['index'].strip()}:"
                    f"{(row.get('name') or 'unknown').strip()}"
                )
                timestamps.append(timestamp)
                row_count += 1
                for column in metric_columns:
                    value = _parse_float(row.get(column))
                    if value is None:
                        continue
                    per_gpu_values[gpu][column].append(value)
                    if column == "power_watts":
                        per_gpu_power[gpu].append((timestamp, value))

    per_gpu = {}
    all_values: dict[str, list[float]] = defaultdict(list)
    total_energy_wh = 0.0
    total_integrated_seconds = 0.0
    for gpu in sorted(per_gpu_values):
        summaries = {
            name: summarize(values)
            for name, values in sorted(per_gpu_values[gpu].items())
        }
        for name, values in per_gpu_values[gpu].items():
            all_values[name].extend(values)
        power = _integrate_power(per_gpu_power[gpu])
        total_energy_wh += float(power["energy_wh"])
        total_integrated_seconds += float(power["integrated_seconds"])
        util_values = per_gpu_values[gpu].get("gpu_util_percent", [])
        per_gpu[gpu] = {
            "metrics": summaries,
            "power_integration": power,
            "active_sample_fraction": (
                sum(value > 0 for value in util_values) / len(util_values)
                if util_values
                else 0.0
            ),
            "high_utilization_sample_fraction": (
                sum(value >= 90 for value in util_values) / len(util_values)
                if util_values
                else 0.0
            ),
        }

    duration_seconds = max(timestamps) - min(timestamps) if timestamps else 0.0
    return {
        "row_count": row_count,
        "gpu_count": len(per_gpu),
        "missing_columns": sorted(missing_columns),
        "duration_seconds": duration_seconds,
        "all_gpus": {
            name: summarize(values) for name, values in sorted(all_values.items())
        },
        "energy": {
            "estimated_total_wh": total_energy_wh,
            "integrated_gpu_seconds": total_integrated_seconds,
            "method": "per-GPU trapezoidal integration; gaps over five seconds excluded",
        },
        "per_gpu": per_gpu,
    }


def analyze_host_samples(
    paths: list[Path],
    *,
    window: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Summarize host memory, load, and monotonic network-counter deltas."""
    metric_columns = ("mem_total_kib", "mem_available_kib", "load_1m")
    values: dict[str, list[float]] = defaultdict(list)
    timestamps: list[float] = []
    network_points: list[tuple[float, float, float]] = []
    row_count = 0
    network_columns_available = True
    for path in paths:
        with path.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            fieldnames = set(reader.fieldnames or ())
            required = {"timestamp", *metric_columns}
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(f"Host sample CSV {path} is missing {sorted(missing)}")
            has_network = {"rx_bytes", "tx_bytes"}.issubset(fieldnames)
            network_columns_available &= has_network
            for row in reader:
                timestamp = _parse_timestamp(row["timestamp"])
                if not _inside_window(timestamp, window):
                    continue
                parsed = {
                    column: _parse_float(row.get(column)) for column in metric_columns
                }
                if any(parsed[column] is None for column in metric_columns):
                    continue
                timestamps.append(timestamp)
                row_count += 1
                for column in metric_columns:
                    values[column].append(float(parsed[column]))
                values["mem_used_kib"].append(
                    float(parsed["mem_total_kib"] - parsed["mem_available_kib"])
                )
                rx_bytes = _parse_float(row.get("rx_bytes")) if has_network else None
                tx_bytes = _parse_float(row.get("tx_bytes")) if has_network else None
                if rx_bytes is not None and tx_bytes is not None:
                    network_points.append(
                        (
                            timestamp,
                            rx_bytes,
                            tx_bytes,
                        )
                    )

    network_points.sort()
    rx_bytes = 0.0
    tx_bytes = 0.0
    for (_, first_rx, first_tx), (_, second_rx, second_tx) in zip(
        network_points, network_points[1:]
    ):
        rx_bytes += max(0.0, second_rx - first_rx)
        tx_bytes += max(0.0, second_tx - first_tx)
    return {
        "row_count": row_count,
        "duration_seconds": max(timestamps) - min(timestamps) if timestamps else 0.0,
        "metrics": {
            name: summarize(series) for name, series in sorted(values.items())
        },
        "network_counter_deltas": {
            "available": network_columns_available,
            "rx_bytes": rx_bytes if network_columns_available else None,
            "tx_bytes": tx_bytes if network_columns_available else None,
        },
    }


def parse_training_console(
    path: Path, warmup_steps: int, *, training_gpus: int = 8
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    in_performance_metrics = False

    with path.open(encoding="utf-8", errors="replace") as source:
        for raw_line in source:
            line = _ANSI_PATTERN.sub("", raw_line.rstrip())
            step_match = _STEP_PATTERN.search(line)
            if step_match:
                if current is not None:
                    steps.append(current)
                current = {
                    "step": int(step_match.group(1)),
                    "max_steps": int(step_match.group(2)),
                    "timing_seconds": {},
                    "throughput": {},
                    "native_throughput": {},
                    "results": {},
                    "work_sources": {},
                }
                in_performance_metrics = False
                continue
            if current is None:
                continue
            if "Performance Metrics:" in line:
                in_performance_metrics = True
                continue
            rollout_match = _ROLLOUT_PATTERN.search(line)
            if rollout_match:
                current["completed_trajectories"] = int(rollout_match.group(1))
                current["requested_trajectories"] = int(rollout_match.group(2))
                continue
            timing_match = _TIMING_PATTERN.match(line)
            if timing_match:
                name = timing_match.group(1).strip().lower().replace(" ", "_")
                current["timing_seconds"][name] = float(timing_match.group(2))
                continue
            if in_performance_metrics:
                throughput_match = _THROUGHPUT_PATTERN.match(line)
                if throughput_match:
                    name = throughput_match.group(1).strip()
                    current["native_throughput"][name] = float(
                        throughput_match.group(2)
                    )
                    continue
            result_match = _RESULT_PATTERN.match(line)
            if result_match:
                name = result_match.group(1).strip().lower().replace(" ", "_")
                current["results"][name] = float(result_match.group(2))

    if current is not None:
        steps.append(current)
    for step in steps:
        completed = step.get("completed_trajectories")
        mean_generated = step["results"].get("mean_generation_length")
        if isinstance(completed, (int, float)) and isinstance(
            mean_generated, (int, float)
        ):
            step["generated_tokens"] = mean_generated * completed
            step["work_sources"]["generated_tokens"] = (
                "Mean Generation Length x completed trajectories"
            )

        mean_observation = step["results"].get("mean_observation_length")
        if isinstance(completed, (int, float)) and isinstance(
            mean_observation, (int, float)
        ):
            step["observation_tokens"] = mean_observation * completed
            step["work_sources"]["observation_tokens"] = (
                "Mean Observation Length x completed trajectories"
            )

        mean_response = step["results"].get("mean_response_sequence_length")
        if isinstance(completed, (int, float)) and isinstance(
            mean_response, (int, float)
        ):
            step["response_tokens"] = mean_response * completed
            step["work_sources"]["response_tokens"] = (
                "Mean Response Sequence Length x completed trajectories"
            )
        elif "generated_tokens" in step and "observation_tokens" in step:
            step["response_tokens"] = (
                step["generated_tokens"] + step["observation_tokens"]
            )
            step["work_sources"]["response_tokens"] = (
                "generated_tokens + observation_tokens"
            )

        if {
            "generated_tokens",
            "observation_tokens",
            "response_tokens",
        }.issubset(step) and not math.isclose(
            step["generated_tokens"] + step["observation_tokens"],
            step["response_tokens"],
            rel_tol=1e-6,
            abs_tol=1.0,
        ):
            raise ValueError(
                f"NeMo step {step['step']} has inconsistent response-token work"
            )

        mean_total = step["results"].get("mean_total_tokens_per_sample")
        if isinstance(completed, (int, float)) and isinstance(
            mean_total, (int, float)
        ):
            step["processed_tokens"] = mean_total * completed
            step["work_sources"]["processed_tokens"] = (
                "Mean Total Tokens per Sample x completed trajectories"
            )
        else:
            total_seconds = step["timing_seconds"].get("total_step_time")
            aggregate_rate = step["native_throughput"].get("E2E (Tokens/sec)")
            per_gpu_rate = step["native_throughput"].get(
                "E2E (Tokens/sec/gpu)"
            )
            if isinstance(total_seconds, (int, float)) and total_seconds > 0:
                if isinstance(aggregate_rate, (int, float)):
                    step["processed_tokens"] = aggregate_rate * total_seconds
                    step["work_sources"]["processed_tokens"] = (
                        "rounded native E2E token rate x rounded step time"
                    )
                elif isinstance(per_gpu_rate, (int, float)):
                    step["processed_tokens"] = (
                        per_gpu_rate * training_gpus * total_seconds
                    )
                    step["work_sources"]["processed_tokens"] = (
                        "rounded native per-GPU E2E token rate x GPU count x "
                        "rounded step time"
                    )
        _add_canonical_throughput(step, training_gpus=training_gpus)
    report = _summarize_training_steps(steps, warmup_steps=warmup_steps)
    report["token_semantics"] = {
        "generated_tokens": "model-emitted tokens; exact up to printed mean precision",
        "processed_tokens": (
            "unpadded training-sequence tokens used by NeMo's native throughput"
        ),
        "observation_tokens": "environment/retrieval tokens excluded from policy loss",
        "response_tokens": "model-generated plus environment/retrieval tokens",
        "native_throughput": (
            "verbatim framework labels retained separately and never compared as "
            "model-generated throughput"
        ),
    }
    return report


def _summarize_training_steps(
    steps: list[dict[str, Any]], warmup_steps: int
) -> dict[str, Any]:
    steady_steps = [step for step in steps if step["step"] > warmup_steps]
    timing_names = sorted(
        {name for step in steady_steps for name in step["timing_seconds"]}
    )
    throughput_names = sorted(
        {name for step in steady_steps for name in step["throughput"]}
    )
    result_names = sorted({name for step in steady_steps for name in step["results"]})
    work_names = tuple(
        name
        for name in _WORK_NAMES
        if any(name in step for step in steady_steps)
    )
    native_throughput_names = sorted(
        {
            name
            for step in steady_steps
            for name in step.get("native_throughput", {})
        }
    )
    return {
        "step_count": len(steps),
        "warmup_steps_excluded": warmup_steps,
        "steady_step_count": len(steady_steps),
        "timing_seconds": {
            name: summarize(
                step["timing_seconds"][name]
                for step in steady_steps
                if name in step["timing_seconds"]
            )
            for name in timing_names
        },
        "throughput": {
            name: summarize(
                step["throughput"][name]
                for step in steady_steps
                if name in step["throughput"]
            )
            for name in throughput_names
        },
        "native_throughput": {
            name: summarize(
                step["native_throughput"][name]
                for step in steady_steps
                if name in step.get("native_throughput", {})
            )
            for name in native_throughput_names
        },
        "results": {
            name: summarize(
                step["results"][name]
                for step in steady_steps
                if name in step["results"]
            )
            for name in result_names
        },
        "work": {
            name: summarize(step[name] for step in steady_steps if name in step)
            for name in work_names
        },
        "steps": steps,
    }


def _parse_dash_metrics(line: str, match: re.Match[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for field in line[match.end() :].split(" - "):
        key, separator, raw_value = field.rpartition(":")
        if not separator or not key:
            continue
        try:
            metrics[key.strip()] = float(raw_value.strip())
        except ValueError:
            continue
    return metrics


def _standard_results(metrics: dict[str, float]) -> dict[str, float]:
    result_keys = {
        "avg_reward": (
            "critic/rewards/mean",
            "reward/mean",
            "rollout/rewards/mean",
        ),
        "mean_generation_length": (
            "response_length/mean",
            "response_len/mean",
            "rollout/response_len/mean",
        ),
    }
    results = {}
    for output_name, candidates in result_keys.items():
        for candidate in candidates:
            if candidate in metrics:
                results[output_name] = metrics[candidate]
                break
    return results


def parse_verl_training_console(
    path: Path,
    *,
    framework: str,
    warmup_steps: int,
    trajectories_per_step: int,
    training_gpus: int = 8,
) -> dict[str, Any]:
    """Parse original or current veRL's console logger into the common schema."""
    steps_by_id: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8", errors="replace") as source:
        for raw_line in source:
            line = _ANSI_PATTERN.sub("", raw_line.rstrip())
            match = _VERL_STEP_PATTERN.search(line)
            if match is None:
                continue
            step_id = int(match.group(1))
            metrics = _parse_dash_metrics(line, match)
            if "timing_s/step" not in metrics or "timing_s/update_actor" not in metrics:
                continue
            timing_seconds = {}
            for key, value in metrics.items():
                if not key.startswith("timing_s/"):
                    continue
                native_name = key.removeprefix("timing_s/")
                normalized_name = _VERL_TIMING_NAMES.get(native_name, key)
                if framework == "current" and native_name == "adv":
                    normalized_name = "advantage_calculation"
                timing_seconds[normalized_name] = value
            total_seconds = timing_seconds["total_step_time"]
            throughput: dict[str, float] = {}
            native_throughput: dict[str, float] = {}
            results = _standard_results(metrics)
            mean_response_length = results.pop("mean_generation_length", None)
            if mean_response_length is not None:
                results["mean_response_sequence_length"] = mean_response_length
            work_sources: dict[str, str] = {}
            work: dict[str, float] = {}

            if framework == "original" and "state_tokens/total" in metrics:
                work["generated_tokens"] = metrics["state_tokens/total"]
                work_sources["generated_tokens"] = "native state_tokens/total"
            elif framework == "current" and "work/generated_tokens" in metrics:
                work["generated_tokens"] = metrics["work/generated_tokens"]
                work_sources["generated_tokens"] = (
                    "measurement-only TransferQueue scalar sum"
                )

            if framework == "current" and "work/observation_tokens" in metrics:
                work["observation_tokens"] = metrics["work/observation_tokens"]
                work_sources["observation_tokens"] = (
                    "measurement-only TransferQueue scalar sum"
                )

            if mean_response_length is not None:
                work["response_tokens"] = (
                    mean_response_length * trajectories_per_step
                )
                work_sources["response_tokens"] = (
                    "native response_length/mean x completed trajectories"
                )
            elif {
                "generated_tokens",
                "observation_tokens",
            }.issubset(work):
                work["response_tokens"] = (
                    work["generated_tokens"] + work["observation_tokens"]
                )
                work_sources["response_tokens"] = (
                    "generated_tokens + observation_tokens"
                )

            if (
                "observation_tokens" not in work
                and "response_tokens" in work
                and "generated_tokens" in work
            ):
                observation_tokens = (
                    work["response_tokens"] - work["generated_tokens"]
                )
                if observation_tokens >= 0:
                    work["observation_tokens"] = observation_tokens
                    work_sources["observation_tokens"] = (
                        "response_tokens - generated_tokens"
                    )

            if {
                "generated_tokens",
                "observation_tokens",
                "response_tokens",
            }.issubset(work) and not math.isclose(
                work["generated_tokens"] + work["observation_tokens"],
                work["response_tokens"],
                rel_tol=1e-6,
                abs_tol=1.0,
            ):
                raise ValueError(
                    f"veRL step {step_id} has inconsistent response-token work"
                )

            if "perf/total_num_tokens" in metrics:
                total_tokens = metrics["perf/total_num_tokens"]
                work["processed_tokens"] = total_tokens
                work_sources["processed_tokens"] = (
                    "native perf/total_num_tokens (prompt + response)"
                )
                if "perf/throughput" in metrics:
                    native_throughput["perf/throughput"] = metrics[
                        "perf/throughput"
                    ]

            metric_work_names = {
                "search_count": "work/search_count",
                "search_errors": "work/search_errors",
                "invalid_actions": "work/invalid_actions",
                "trajectory_wall_seconds_mean": (
                    "work/trajectory_wall_seconds_mean"
                ),
                "trajectory_wall_seconds_max": "work/trajectory_wall_seconds_max",
            }
            if framework == "original":
                metric_work_names["search_count"] = "env/search_calls"
            for output_name, metric_name in metric_work_names.items():
                if metric_name in metrics:
                    work[output_name] = metrics[metric_name]
                    work_sources[output_name] = f"native {metric_name}"

            if "generated_tokens" in work:
                results["mean_generation_length"] = (
                    work["generated_tokens"] / trajectories_per_step
                )

            step = {
                "step": step_id,
                "max_steps": 0,
                "completed_trajectories": trajectories_per_step,
                "requested_trajectories": trajectories_per_step,
                "timing_seconds": timing_seconds,
                "throughput": throughput,
                "native_throughput": native_throughput,
                "results": results,
                "work_sources": work_sources,
                "native_metrics": metrics,
            }
            step.update(work)
            _add_canonical_throughput(step, training_gpus=training_gpus)
            steps_by_id[step_id] = step

    steps = [steps_by_id[step_id] for step_id in sorted(steps_by_id)]
    max_steps = max(steps_by_id, default=0)
    for step in steps:
        step["max_steps"] = max_steps
    report = _summarize_training_steps(steps, warmup_steps=warmup_steps)
    report["timing_semantics"] = (
        "Mapped timing_s fields are native veRL timers. The total step is the "
        "end-to-end boundary; nested or concurrent timers must not be summed."
    )
    report["token_semantics"] = {
        "generated_tokens": (
            "loss-mask/state tokens for original Search-R1; adapter scalar sum "
            "for current veRL"
        ),
        "observation_tokens": "retrieved information tokens excluded from policy loss",
        "response_tokens": "generated plus retrieved-observation response tokens",
        "processed_tokens": (
            "native perf/total_num_tokens: prompt plus response tokens; missing "
            "when the older fork does not emit it"
        ),
        "native_throughput": (
            "verbatim perf/throughput retained separately as processed tokens/s/GPU"
        ),
    }
    return report


def parse_slime_training_console(
    path: Path,
    *,
    warmup_steps: int,
    trajectories_per_step: int,
    training_gpus: int = 8,
) -> dict[str, Any]:
    """Parse slime rollout and actor perf dictionaries into the common schema."""
    metrics_by_step: dict[int, dict[str, float]] = defaultdict(dict)
    evidence_by_step: dict[int, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8", errors="replace") as source:
        for raw_line in source:
            line = _ANSI_PATTERN.sub("", raw_line.rstrip())
            match = _SLIME_PERF_PATTERN.search(line)
            if match is None:
                continue
            step_id = int(match.group(1)) + 1
            metrics = {
                name: float(value)
                for name, value in _DICT_NUMBER_PATTERN.findall(match.group(2))
            }
            metrics_by_step[step_id].update(metrics)
            if "perf/rollout_time" in metrics:
                evidence_by_step[step_id].add("rollout")
            if "perf/actor_train_time" in metrics:
                evidence_by_step[step_id].add("update")

    steps = []
    max_steps = max(metrics_by_step, default=0)
    for step_id in sorted(metrics_by_step):
        metrics = metrics_by_step[step_id]
        if evidence_by_step[step_id] != {"rollout", "update"}:
            continue
        timing_seconds = {
            _SLIME_TIMING_NAMES.get(key.removeprefix("perf/"), key): value
            for key, value in metrics.items()
            if key.startswith("perf/")
            and key.removeprefix("perf/") in _SLIME_TIMING_NAMES
        }
        total_seconds = timing_seconds["total_step_time"]
        throughput: dict[str, float] = {}
        native_throughput: dict[str, float] = {}
        results = _standard_results(metrics)
        work_sources: dict[str, str] = {}
        work: dict[str, float] = {}

        rollout_seconds = timing_seconds["generation_and_agent"]
        effective_rate = metrics.get("perf/effective_tokens_per_gpu_per_sec")
        if effective_rate is not None:
            work["generated_tokens"] = (
                effective_rate * rollout_seconds * training_gpus
            )
            work_sources["generated_tokens"] = (
                "native effective token rate x rollout time x rollout GPUs"
            )
            native_throughput[
                "perf/effective_tokens_per_gpu_per_sec"
            ] = effective_rate
            mean_generated = results.get("mean_generation_length")
            if isinstance(mean_generated, (int, float)) and not math.isclose(
                work["generated_tokens"],
                mean_generated * trajectories_per_step,
                rel_tol=1e-6,
                abs_tol=1.0,
            ):
                raise ValueError(
                    f"slime step {step_id} token rate does not match its "
                    "effective response mean and GPU count"
                )
        elif "mean_generation_length" in results:
            work["generated_tokens"] = (
                results["mean_generation_length"] * trajectories_per_step
            )
            work_sources["generated_tokens"] = (
                "native effective response mean x completed trajectories"
            )

        if "perf/tokens_per_gpu_per_sec" in metrics:
            full_response_rate = metrics["perf/tokens_per_gpu_per_sec"]
            work["response_tokens"] = (
                full_response_rate * rollout_seconds * training_gpus
            )
            work_sources["response_tokens"] = (
                "native full-response token rate x rollout time x rollout GPUs"
            )
            native_throughput["perf/tokens_per_gpu_per_sec"] = full_response_rate

        if "generated_tokens" in work and "response_tokens" in work:
            observation_tokens = work["response_tokens"] - work["generated_tokens"]
            if observation_tokens >= -1e-6:
                work["observation_tokens"] = max(0.0, observation_tokens)
                work_sources["observation_tokens"] = (
                    "response_tokens - generated_tokens"
                )

        if "perf/actor_train_tok_per_s" in metrics:
            native_throughput["perf/actor_train_tok_per_s"] = metrics[
                "perf/actor_train_tok_per_s"
            ]
        step = {
            "step": step_id,
            "max_steps": max_steps,
            "completed_trajectories": trajectories_per_step,
            "requested_trajectories": trajectories_per_step,
            "timing_seconds": timing_seconds,
            "throughput": throughput,
            "native_throughput": native_throughput,
            "results": results,
            "work_sources": work_sources,
            "native_metrics": metrics,
        }
        step.update(work)
        if "generated_tokens" in work:
            step["results"]["mean_generation_length"] = (
                work["generated_tokens"] / trajectories_per_step
            )
        _add_canonical_throughput(step, training_gpus=training_gpus)
        steps.append(step)

    report = _summarize_training_steps(steps, warmup_steps=warmup_steps)
    report["timing_semantics"] = (
        "total_step_time is slime's train_wait plus train boundary. "
        "generation_and_agent is nested in trainer_wait_inclusive, while "
        "log-probability and actor timers are nested in trainer_work_inclusive; "
        "inclusive fields must not be summed."
    )
    report["token_semantics"] = {
        "generated_tokens": (
            "effective response tokens, which are model generated and trainable"
        ),
        "observation_tokens": (
            "full SGLang response tokens minus effective response tokens"
        ),
        "response_tokens": "model-generated plus injected observation tokens",
        "processed_tokens": (
            "missing: actor_train_tok_per_s is retained as a native stage rate "
            "because its aggregation scope is not an end-to-end work counter"
        ),
        "native_throughput": (
            "native rollout-stage and actor-stage rates retained separately"
        ),
    }
    return report


def parse_framework_training_console(
    path: Path,
    *,
    framework: str,
    warmup_steps: int,
    trajectories_per_step: int | None,
    training_gpus: int = 8,
) -> dict[str, Any]:
    """Dispatch console parsing while keeping one output schema."""
    if framework == "nemo":
        return parse_training_console(
            path, warmup_steps=warmup_steps, training_gpus=training_gpus
        )
    if trajectories_per_step is None or trajectories_per_step < 1:
        raise ValueError(
            "Non-NeMo console parsing requires a positive trajectories_per_step"
        )
    if framework in {"original", "current"}:
        return parse_verl_training_console(
            path,
            framework=framework,
            warmup_steps=warmup_steps,
            trajectories_per_step=trajectories_per_step,
            training_gpus=training_gpus,
        )
    if framework == "slime":
        return parse_slime_training_console(
            path,
            warmup_steps=warmup_steps,
            trajectories_per_step=trajectories_per_step,
            training_gpus=training_gpus,
        )
    raise ValueError(f"Unsupported framework: {framework}")


def analyze_step_timestamps(
    path: Path,
    *,
    framework: str,
    warmup_steps: int,
) -> dict[str, Any]:
    """Find native step boundaries in a runner-produced timestamped console."""
    step_starts: dict[int, float] = {}
    step_completions: dict[int, float] = {}
    current_nemo_step: int | None = None
    with path.open(encoding="utf-8", errors="replace") as source:
        for line_number, raw_line in enumerate(source, start=1):
            raw_timestamp, separator, raw_message = raw_line.rstrip("\n").partition(
                "\t"
            )
            if not separator:
                raise ValueError(
                    f"Timestamped console line {line_number} has no tab separator"
                )
            try:
                timestamp = float(raw_timestamp)
            except ValueError as error:
                raise ValueError(
                    f"Invalid console timestamp at {path}:{line_number}"
                ) from error
            if not math.isfinite(timestamp):
                raise ValueError(
                    f"Non-finite console timestamp at {path}:{line_number}"
                )
            message = _ANSI_PATTERN.sub("", raw_message)

            if framework == "nemo":
                match = _STEP_PATTERN.search(message)
                if match is not None:
                    current_nemo_step = int(match.group(1))
                    step_starts.setdefault(current_nemo_step, timestamp)
                    continue
                if current_nemo_step is not None and "Training FLOPS:" in message:
                    step_completions[current_nemo_step] = timestamp
                continue

            if framework in {"original", "current"}:
                match = _VERL_STEP_PATTERN.search(message)
                if match is not None and "timing_s/update_actor:" in message:
                    step_completions[int(match.group(1))] = timestamp
                continue

            if framework == "slime":
                match = _SLIME_PERF_PATTERN.search(message)
                if match is not None and "perf/actor_train_time" in message:
                    step_completions[int(match.group(1)) + 1] = timestamp
                continue

            raise ValueError(f"Unsupported framework: {framework}")

    completion_steps = sorted(step_completions)
    steady_steps = [step for step in completion_steps if step > warmup_steps]
    report: dict[str, Any] = {
        "timestamp_source": str(path),
        "step_starts_unix_seconds": {
            str(step): timestamp for step, timestamp in sorted(step_starts.items())
        },
        "step_completions_unix_seconds": {
            str(step): timestamp
            for step, timestamp in sorted(step_completions.items())
        },
        "steady_step_ids": steady_steps,
        "available": False,
        "start_unix_seconds": None,
        "end_unix_seconds": None,
    }
    if not steady_steps:
        report["unavailable_reason"] = "no completed post-warmup steps"
        return report

    first_steady = steady_steps[0]
    if framework == "nemo":
        start = step_starts.get(first_steady)
        start_semantics = "first steady NeMo step header"
    else:
        start = step_completions.get(first_steady - 1)
        start_semantics = "preceding native step-completion metric"
    if start is None:
        report["unavailable_reason"] = (
            f"no start boundary for steady step {first_steady}"
        )
        return report

    end = step_completions[steady_steps[-1]]
    if end <= start:
        report["unavailable_reason"] = "steady window is not positive"
        return report

    report |= {
        "available": True,
        "start_unix_seconds": start,
        "end_unix_seconds": end,
        "duration_seconds": end - start,
        "start_semantics": start_semantics,
        "end_semantics": "last steady native step-completion metric",
        "sample_filter_semantics": "closed interval; spans must be fully contained",
    }
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, action="append", default=[])
    parser.add_argument("--prometheus", type=Path, action="append", default=[])
    parser.add_argument("--gpu-samples", type=Path, action="append", default=[])
    parser.add_argument("--host-samples", type=Path, action="append", default=[])
    parser.add_argument("--console", type=Path)
    parser.add_argument("--timestamped-console", type=Path)
    parser.add_argument(
        "--framework",
        choices=("nemo", "original", "current", "slime"),
        default="nemo",
    )
    parser.add_argument("--trajectories-per-step", type=int)
    parser.add_argument(
        "--training-gpus",
        type=int,
        default=8,
        help="Physical GPUs used by the fixed four-way protocol (default: 8)",
    )
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if (
        not args.trace
        and not args.prometheus
        and not args.gpu_samples
        and not args.host_samples
        and args.console is None
    ):
        raise ValueError(
            "Provide at least one trace, Prometheus, resource sample, or console input"
        )
    if args.warmup_steps < 0:
        raise ValueError("Warmup steps cannot be negative")
    if args.training_gpus < 1:
        raise ValueError("Training GPUs must be positive")
    for path in [
        *args.trace,
        *args.prometheus,
        *args.gpu_samples,
        *args.host_samples,
        args.console,
        args.timestamped_console,
    ]:
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)

    if args.timestamped_console is not None and args.console is None:
        raise ValueError("--timestamped-console requires --console")

    report: dict[str, Any] = {"schema_version": 2}
    if args.trace:
        report["trajectory_trace"] = analyze_traces(args.trace)
    if args.prometheus:
        report["prometheus"] = analyze_prometheus(args.prometheus)
    if args.gpu_samples:
        report["gpu_resources"] = analyze_gpu_samples(args.gpu_samples)
    if args.host_samples:
        report["host_resources"] = analyze_host_samples(args.host_samples)
    if args.console is not None:
        report["framework"] = args.framework
        report["training_gpus"] = args.training_gpus
        report["training"] = parse_framework_training_console(
            args.console,
            framework=args.framework,
            warmup_steps=args.warmup_steps,
            trajectories_per_step=args.trajectories_per_step,
            training_gpus=args.training_gpus,
        )
    if args.timestamped_console is not None:
        measurement_window = analyze_step_timestamps(
            args.timestamped_console,
            framework=args.framework,
            warmup_steps=args.warmup_steps,
        )
        report["measurement_window"] = measurement_window
        if measurement_window["available"]:
            window = (
                float(measurement_window["start_unix_seconds"]),
                float(measurement_window["end_unix_seconds"]),
            )
            steady_state: dict[str, Any] = {}
            if args.trace:
                steady_state["trajectory_trace"] = analyze_traces(
                    args.trace, window=window
                )
            if args.prometheus:
                steady_state["prometheus"] = analyze_prometheus(
                    args.prometheus, window=window
                )
            if args.gpu_samples:
                steady_state["gpu_resources"] = analyze_gpu_samples(
                    args.gpu_samples, window=window
                )
            if args.host_samples:
                steady_state["host_resources"] = analyze_host_samples(
                    args.host_samples, window=window
                )
            report["steady_state_evidence"] = steady_state
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
