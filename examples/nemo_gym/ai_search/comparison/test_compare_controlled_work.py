# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from compare_controlled_work import compare_steps


def _write_trace(path: Path, framework: str, *, changed: bool = False) -> None:
    rows = []
    timestamp = 1_000
    for step in range(1, 3):
        for trajectory in range(2):
            trace_id = f"{framework}-{step}-{trajectory}"
            if framework == "current":
                rows.extend(
                    [
                        {
                            "component": "current_verl",
                            "operation": "model_generation",
                            "trace_id": trace_id,
                            "start_unix_ns": timestamp,
                            "attributes": {
                                "turn": 1,
                                "input_tokens": 10 + trajectory,
                                "output_tokens": 3,
                            },
                        },
                        {
                            "component": "current_verl",
                            "operation": "retrieval",
                            "trace_id": trace_id,
                            "start_unix_ns": timestamp + 10,
                            "attributes": {
                                "turn": 1,
                                "query_characters": 7 + trajectory,
                            },
                        },
                        {
                            "component": "current_verl",
                            "operation": "model_generation",
                            "trace_id": trace_id,
                            "start_unix_ns": timestamp + 20,
                            "attributes": {
                                "turn": 2,
                                "input_tokens": 20 + trajectory,
                                "output_tokens": 4,
                            },
                        },
                    ]
                )
            else:
                rows.extend(
                    [
                        {
                            "component": "search_r1_agent",
                            "operation": "model_generate",
                            "trace_id": trace_id,
                            "start_unix_ns": timestamp,
                            "attributes": {
                                "turn_index": 0,
                                "input_tokens": 10 + trajectory,
                                "output_tokens": 3,
                            },
                        },
                        {
                            "component": "search_r1_agent",
                            "operation": "search_tool",
                            "trace_id": trace_id,
                            "start_unix_ns": timestamp + 10,
                            "attributes": {
                                "turn_index": 0,
                                "query_chars": 7 + trajectory,
                            },
                        },
                        {
                            "component": "search_r1_agent",
                            "operation": "model_generate",
                            "trace_id": trace_id,
                            "start_unix_ns": timestamp + 20,
                            "attributes": {
                                "turn_index": 1,
                                "input_tokens": 20 + trajectory,
                                "output_tokens": 5 if changed and step == 2 else 4,
                            },
                        },
                    ]
                )
            timestamp += 100
        timestamp += 1_000
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_compare_steps_accepts_identical_cross_framework_work(tmp_path: Path) -> None:
    current = tmp_path / "current.jsonl"
    nemo = tmp_path / "nemo.jsonl"
    _write_trace(current, "current")
    _write_trace(nemo, "nemo")

    report = compare_steps(
        left_trace=current,
        left_framework="current",
        left_step=2,
        right_trace=nemo,
        right_framework="nemo",
        right_step=2,
        trajectories_per_step=2,
    )

    assert report["exact_match"] is True
    assert report["left_summary"]["generated_tokens"] == 14
    assert report["left_summary"]["searches"] == 2


def test_compare_steps_reports_per_trajectory_mismatch(tmp_path: Path) -> None:
    current = tmp_path / "current.jsonl"
    nemo = tmp_path / "nemo.jsonl"
    _write_trace(current, "current")
    _write_trace(nemo, "nemo", changed=True)

    report = compare_steps(
        left_trace=current,
        left_framework="current",
        left_step=2,
        right_trace=nemo,
        right_framework="nemo",
        right_step=2,
        trajectories_per_step=2,
    )

    assert report["exact_match"] is False
    assert report["differences"]["scalars"]["generated_tokens"]["delta"] == 2
    assert report["differences"]["trajectory_signatures_missing_on_right"]
    assert report["differences"]["trajectory_signatures_extra_on_right"]
