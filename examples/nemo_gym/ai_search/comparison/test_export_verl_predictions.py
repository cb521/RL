# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for conversion of current-veRL validation dumps."""

import json

import pytest

from export_verl_predictions import convert_record, export_predictions


def _dump_record(example_id="test_0"):
    return {
        "input": "question",
        "output": "<answer>Paris</answer>",
        "gts": {"target": ["Paris"]},
        "score": 1.0,
        "step": 500,
        "example_id": example_id,
        "data_source": "nq",
        "question": "What is the capital of France?",
        "golden_answers": ["Paris"],
    }


def test_convert_verl_record_requires_stable_evidence() -> None:
    converted = convert_record(_dump_record())
    assert converted["example_id"] == "test_0"
    assert converted["response"] == "<answer>Paris</answer>"
    assert converted["status"] == "unknown"
    assert converted["native_reward"] == 1.0


def test_convert_verl_record_preserves_agent_loop_diagnostics() -> None:
    diagnostics = {
        "status": "completed",
        "search_count": 1,
        "turn_count": 2,
        "natural_termination": True,
        "provider_batch_ids": ["batch-1"],
        "search_errors": 0,
        "generated_tokens": 42,
        "observation_tokens": 17,
        "wall_time_seconds": 1.25,
    }
    converted = convert_record(_dump_record() | diagnostics)
    for name, value in diagnostics.items():
        assert converted[name] == value


def test_convert_verl_record_rejects_ground_truth_mismatch() -> None:
    value = _dump_record() | {"gts": {"target": ["London"]}}
    with pytest.raises(ValueError, match="disagrees"):
        convert_record(value)


def test_export_verl_predictions_rejects_duplicate_ids(tmp_path) -> None:
    source = tmp_path / "500.jsonl"
    source.write_text(
        json.dumps(_dump_record()) + "\n" + json.dumps(_dump_record()) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate example_id"):
        export_predictions([source], tmp_path / "predictions.jsonl")


def test_export_verl_predictions_writes_common_jsonl(tmp_path) -> None:
    source = tmp_path / "500.jsonl"
    source.write_text(json.dumps(_dump_record()) + "\n", encoding="utf-8")
    output = tmp_path / "predictions.jsonl"
    assert export_predictions([source], output) == 1
    assert json.loads(output.read_text(encoding="utf-8"))["data_source"] == "nq"
