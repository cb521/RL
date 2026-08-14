# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for compact Nsight Systems report aggregation."""

from pathlib import Path

import pytest

from analyze_nsys import aggregate_tables, worker_kind


def test_worker_kind_uses_stable_roles() -> None:
    assert worker_kind(Path("dtensor_policy_worker_2:3_10.nsys-rep")) == "policy"
    assert worker_kind(Path("vllm_generation_worker_2:3_11.nsys-rep")) == "generation"
    assert (
        worker_kind(Path("original_search_r1_worker_456.nsys-rep"))
        == "original_hybrid_actor_rollout_ref"
    )
    assert (
        worker_kind(Path("slime_actor_789.nsys-rep"))
        == "slime_training_actor"
    )
    assert (
        worker_kind(Path("worker_process_123.1.nsys-rep"))
        == "verl_worker_unspecified"
    )
    assert worker_kind(Path("unknown_12.nsys-rep")) == "other"


def test_aggregate_tables_sums_workers_and_keeps_report_shares_separate() -> None:
    worker_one = {
        "cuda_api_sum": [
            {"Name": "cudaLaunchKernel", "Total Time (ns)": 30, "Num Calls": 2},
            {"Name": "cudaMemcpy", "Total Time (ns)": 10, "Num Calls": 1},
        ],
        "cuda_gpu_kern_sum": [{"Name": "gemm", "Total Time (ns)": 60, "Instances": 2}],
    }
    worker_two = {
        "cuda_api_sum": [
            {"Name": "cudaLaunchKernel", "Total Time (ns)": 50, "Num Calls": 3}
        ],
        "cuda_gpu_kern_sum": [
            {"Name": "gemm", "Total Time (ns)": 20, "Instances": 1},
            {"Name": "reduce", "Total Time (ns)": 20, "Instances": 1},
        ],
    }

    result = aggregate_tables([worker_one, worker_two], top=5)
    assert result["cuda_api_sum"][0] == {
        "name": "cudaLaunchKernel",
        "total_time_ns": 80.0,
        "count": 5,
        "report_share_pct": pytest.approx(80.0 / 90.0 * 100.0),
    }
    assert result["cuda_gpu_kern_sum"][0]["name"] == "gemm"
    assert result["cuda_gpu_kern_sum"][0]["total_time_ns"] == 80.0
    assert result["cuda_gpu_kern_sum"][0]["report_share_pct"] == 80.0


def test_aggregate_kernel_execution_weights_api_and_queue_means() -> None:
    tables = {
        "cuda_kern_exec_sum": [
            {
                "API Name": "cudaLaunchKernel",
                "Kernel Name": "gemm",
                "Count": 4,
                "QCount": 3,
                "AAvg (ns)": 5,
                "QAvg (ns)": 7,
                "KAvg (ns)": 11,
            }
        ]
    }
    row = aggregate_tables([tables], top=1)["cuda_kern_exec_sum"][0]
    assert row["total_time_ns"] == 44.0
    assert row["api_time_ns"] == 20.0
    assert row["positive_queue_time_ns"] == 21.0
