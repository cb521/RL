# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare paired baseline and clean four-way AI-search measurement reports."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable


_WORK_FIELDS = (
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
_FIXED_WORK_FIELDS = (
    "completed_trajectories",
    "requested_trajectories",
    "generated_tokens",
    "observation_tokens",
    "response_tokens",
    "processed_tokens",
    "search_count",
    "search_errors",
    "invalid_actions",
)


def _summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "stdev": 0.0}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _paired_summary(
    baseline_steps: list[dict[str, Any]],
    clean_steps: list[dict[str, Any]],
    value: Callable[[dict[str, Any]], float],
) -> dict[str, Any]:
    baseline_values = [float(value(step)) for step in baseline_steps]
    clean_values = [float(value(step)) for step in clean_steps]
    deltas = [
        clean - baseline for baseline, clean in zip(baseline_values, clean_values)
    ]
    relative = [
        (clean / baseline - 1.0) * 100.0
        for baseline, clean in zip(baseline_values, clean_values)
        if baseline != 0.0
    ]
    baseline_mean = statistics.fmean(baseline_values)
    clean_mean = statistics.fmean(clean_values)
    return {
        "baseline": _summarize(baseline_values),
        "clean": _summarize(clean_values),
        "paired_delta": _summarize(deltas),
        "paired_relative_percent": _summarize(relative),
        "mean_relative_percent": (
            (clean_mean / baseline_mean - 1.0) * 100.0 if baseline_mean != 0.0 else None
        ),
    }


def _steady_steps(training: dict[str, Any]) -> list[dict[str, Any]]:
    warmup = int(training["warmup_steps_excluded"])
    steps = [step for step in training["steps"] if int(step["step"]) > warmup]
    if len(steps) != int(training["steady_step_count"]):
        raise ValueError("Steady-step count does not match the report metadata")
    return steps


def _compare_group(
    baseline_steps: list[dict[str, Any]],
    clean_steps: list[dict[str, Any]],
    group: str,
) -> dict[str, Any]:
    baseline_names = {name for step in baseline_steps for name in step.get(group, {})}
    clean_names = {name for step in clean_steps for name in step.get(group, {})}
    if baseline_names != clean_names:
        raise ValueError(
            f"{group} metric sets differ: baseline={sorted(baseline_names)}, "
            f"clean={sorted(clean_names)}"
        )
    return {
        name: _paired_summary(
            baseline_steps,
            clean_steps,
            lambda step, name=name: step[group][name],
        )
        for name in sorted(baseline_names)
    }


def compare_reports(baseline: dict[str, Any], clean: dict[str, Any]) -> dict[str, Any]:
    """Return paired timing, throughput, result, and work deltas."""
    baseline_framework = baseline.get("framework")
    clean_framework = clean.get("framework")
    if baseline_framework != clean_framework:
        raise ValueError(
            "Baseline and clean reports use different frameworks: "
            f"baseline={baseline_framework!r}, clean={clean_framework!r}"
        )
    baseline_training = baseline["training"]
    clean_training = clean["training"]
    if (
        baseline_training["warmup_steps_excluded"]
        != clean_training["warmup_steps_excluded"]
    ):
        raise ValueError("Baseline and clean reports use different warm-up windows")

    baseline_steps = _steady_steps(baseline_training)
    clean_steps = _steady_steps(clean_training)
    baseline_ids = [int(step["step"]) for step in baseline_steps]
    clean_ids = [int(step["step"]) for step in clean_steps]
    if baseline_ids != clean_ids:
        raise ValueError(
            f"Measured step IDs differ: baseline={baseline_ids}, clean={clean_ids}"
        )

    baseline_work_fields = {
        name for name in _WORK_FIELDS if any(name in step for step in baseline_steps)
    }
    clean_work_fields = {
        name for name in _WORK_FIELDS if any(name in step for step in clean_steps)
    }
    if baseline_work_fields != clean_work_fields:
        raise ValueError(
            "Work metric sets differ: "
            f"baseline={sorted(baseline_work_fields)}, "
            f"clean={sorted(clean_work_fields)}"
        )
    work = {}
    for name in sorted(baseline_work_fields):
        if not all(name in step for step in [*baseline_steps, *clean_steps]):
            raise ValueError(f"Missing required work field {name!r}")
        work[name] = _paired_summary(
            baseline_steps, clean_steps, lambda step, name=name: step[name]
        )

    completed_match = all(
        baseline_step["completed_trajectories"]
        == clean_step["completed_trajectories"]
        == baseline_step["requested_trajectories"]
        == clean_step["requested_trajectories"]
        for baseline_step, clean_step in zip(baseline_steps, clean_steps)
    )
    fixed_work_match = {
        name: all(
            math.isclose(
                float(baseline_step[name]),
                float(clean_step[name]),
                rel_tol=1e-9,
                abs_tol=1e-6,
            )
            for baseline_step, clean_step in zip(baseline_steps, clean_steps)
        )
        for name in _FIXED_WORK_FIELDS
        if name in baseline_work_fields
    }
    report = {
        "schema_version": 1,
        "framework": baseline_framework,
        "measured_step_ids": baseline_ids,
        "validity": {
            "completed_trajectory_counts_match": completed_match,
            "fixed_work_matches": fixed_work_match,
            "all_reported_fixed_work_matches": all(fixed_work_match.values()),
            "paired_step_count": len(baseline_ids),
        },
        "timing_seconds": _compare_group(baseline_steps, clean_steps, "timing_seconds"),
        "throughput": _compare_group(baseline_steps, clean_steps, "throughput"),
        "results": _compare_group(baseline_steps, clean_steps, "results"),
        "work": work,
    }
    if not completed_match:
        raise ValueError(
            "Baseline and clean runs did not complete identical trajectory work"
        )
    total_step = report["timing_seconds"].get("total_step_time")
    if total_step is None or not math.isfinite(total_step["baseline"]["mean"]):
        raise ValueError("Reports do not contain a finite total-step comparison")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_reports(
        json.loads(args.baseline.read_text()), json.loads(args.clean.read_text())
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
