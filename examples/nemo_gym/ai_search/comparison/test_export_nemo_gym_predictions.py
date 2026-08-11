# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the NeMo Gym common-schema prediction exporter."""

import json

import pytest

from export_nemo_gym_predictions import (
    convert_result,
    export_predictions,
    response_text,
)


def _full_result(source_id="test_0"):
    return {
        "source_id": source_id,
        "data_source": "nq",
        "question": "What is the capital of France?",
        "answers": ["Paris"],
        "response": {
            "status": "completed",
            "error": None,
            "incomplete_details": None,
            "usage": {"output_tokens": 10},
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "ignored"}],
                    "generation_str": "<search>capital France</search>",
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": "<information>Paris is the capital.</information>",
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "<answer>Paris</answer>"}
                    ],
                },
            ],
        },
        "reward": 1.0,
        "exact_match": 1.0,
        "answer_f1": 1.0,
        "num_search_calls": 1,
        "duplicate_query_rate": 0.0,
        "search_error_count": 0,
    }


def test_response_text_preserves_actions_and_observations() -> None:
    assert response_text(_full_result()) == (
        "<search>capital France</search>"
        "<information>Paris is the capital.</information>"
        "<answer>Paris</answer>"
    )


def test_convert_result_exports_common_and_native_evidence() -> None:
    record = convert_result(_full_result())
    assert record["example_id"] == "test_0"
    assert record["golden_answers"] == ["Paris"]
    assert record["generated_tokens"] == 10
    assert record["search_errors"] == 0
    assert record["native_exact_match"] == 1.0


def test_convert_result_requires_stable_source_id() -> None:
    full_result = _full_result()
    del full_result["source_id"]
    with pytest.raises(ValueError, match="source_id"):
        convert_result(full_result)


def test_export_predictions_rejects_duplicate_ids(tmp_path) -> None:
    source = tmp_path / "trajectories.jsonl"
    source.write_text(
        json.dumps(_full_result()) + "\n" + json.dumps(_full_result()) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate example_id"):
        export_predictions([source], tmp_path / "predictions.jsonl")
