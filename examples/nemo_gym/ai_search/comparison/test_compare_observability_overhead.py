# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for paired instrumentation-overhead analysis."""

import pytest

from compare_observability_overhead import compare_reports


def _report(total_times, generation_times, throughputs):
    steps = []
    for step, total, generation, throughput in zip(
        (1, 2, 3, 4), total_times, generation_times, throughputs
    ):
        steps.append(
            {
                "step": step,
                "max_steps": 4,
                "completed_trajectories": 40,
                "requested_trajectories": 40,
                "generated_tokens": 8_000,
                "observation_tokens": 1_000,
                "timing_seconds": {
                    "total_step_time": total,
                    "generation": generation,
                },
                "throughput": {"E2E (Tokens/sec)": throughput},
                "results": {"mean_generation_length": 200.0},
            }
        )
    return {
        "training": {
            "warmup_steps_excluded": 1,
            "steady_step_count": 3,
            "steps": steps,
        }
    }


def test_compare_reports_calculates_paired_overhead() -> None:
    baseline = _report([11, 10, 10, 10], [5, 4, 4, 4], [90, 100, 100, 100])
    clean = _report([12, 11, 11, 11], [6, 5, 5, 5], [85, 90, 90, 90])

    report = compare_reports(baseline, clean)

    assert report["measured_step_ids"] == [2, 3, 4]
    assert report["validity"]["completed_trajectory_counts_match"] is True
    assert report["validity"]["all_reported_fixed_work_matches"] is True
    assert report["work"]["generated_tokens"]["baseline"]["mean"] == 8_000
    assert report["timing_seconds"]["total_step_time"][
        "mean_relative_percent"
    ] == pytest.approx(10.0)
    assert report["throughput"]["E2E (Tokens/sec)"][
        "mean_relative_percent"
    ] == pytest.approx(-10.0)


def test_compare_reports_rejects_mismatched_work() -> None:
    baseline = _report([11, 10, 10, 10], [5, 4, 4, 4], [90, 100, 100, 100])
    clean = _report([12, 11, 11, 11], [6, 5, 5, 5], [85, 90, 90, 90])
    clean["training"]["steps"][2]["completed_trajectories"] = 39

    with pytest.raises(ValueError, match="identical trajectory work"):
        compare_reports(baseline, clean)


def test_compare_reports_rejects_mismatched_frameworks() -> None:
    baseline = _report([11, 10, 10, 10], [5, 4, 4, 4], [90, 100, 100, 100])
    clean = _report([12, 11, 11, 11], [6, 5, 5, 5], [85, 90, 90, 90])
    baseline["framework"] = "nemo"
    clean["framework"] = "slime"

    with pytest.raises(ValueError, match="different frameworks"):
        compare_reports(baseline, clean)


def test_compare_reports_marks_token_work_mismatch() -> None:
    baseline = _report([11, 10, 10, 10], [5, 4, 4, 4], [90, 100, 100, 100])
    clean = _report([12, 11, 11, 11], [6, 5, 5, 5], [85, 90, 90, 90])
    clean["training"]["steps"][2]["generated_tokens"] = 7_999

    report = compare_reports(baseline, clean)

    assert report["validity"]["completed_trajectory_counts_match"] is True
    assert report["validity"]["fixed_work_matches"]["generated_tokens"] is False
    assert report["validity"]["all_reported_fixed_work_matches"] is False
