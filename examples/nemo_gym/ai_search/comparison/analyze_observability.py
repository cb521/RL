# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Merge training, Prometheus, and trajectory traces into one breakdown report."""

from __future__ import annotations

import argparse
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


def analyze_traces(paths: list[Path]) -> dict[str, Any]:
    events = [event for event in _read_jsonl(paths) if event.get("event") == "span"]
    operation_durations: dict[str, list[float]] = defaultdict(list)
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
        operation_durations[f"{component}/{operation}"].append(
            float(event["duration_ms"])
        )
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


def analyze_prometheus(paths: list[Path]) -> dict[str, Any]:
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
        timestamp = int(snapshot["scraped_unix_ns"])
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


def parse_training_console(path: Path, warmup_steps: int) -> dict[str, Any]:
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
                }
                in_performance_metrics = False
                continue
            if current is None:
                continue
            if "Performance Metrics:" in line:
                in_performance_metrics = True
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
                    current["throughput"][name] = float(throughput_match.group(2))

    if current is not None:
        steps.append(current)
    steady_steps = [step for step in steps if step["step"] > warmup_steps]
    timing_names = sorted(
        {name for step in steady_steps for name in step["timing_seconds"]}
    )
    throughput_names = sorted(
        {name for step in steady_steps for name in step["throughput"]}
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
        "steps": steps,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, action="append", default=[])
    parser.add_argument("--prometheus", type=Path, action="append", default=[])
    parser.add_argument("--console", type=Path)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.trace and not args.prometheus and args.console is None:
        raise ValueError("Provide at least one trace, Prometheus, or console input")
    if args.warmup_steps < 0:
        raise ValueError("Warmup steps cannot be negative")
    for path in [*args.trace, *args.prometheus, args.console]:
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)

    report: dict[str, Any] = {"schema_version": 1}
    if args.trace:
        report["trajectory_trace"] = analyze_traces(args.trace)
    if args.prometheus:
        report["prometheus"] = analyze_prometheus(args.prometheus)
    if args.console is not None:
        report["training"] = parse_training_console(
            args.console, warmup_steps=args.warmup_steps
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
