# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the current-veRL and slime evaluation hooks."""

import asyncio
import json
from enum import Enum
from types import SimpleNamespace

import pytest

from framework_eval_adapters import (
    slime_log_eval_rollout_data,
    slime_reward,
    verl_compute_score,
)


def _metadata(example_id="test_0"):
    return {
        "example_id": example_id,
        "data_source": "nq",
        "question": "What is the capital of France?",
        "golden_answers": ["Paris"],
    }


def test_verl_reward_preserves_identity_and_scores_last_answer() -> None:
    result = verl_compute_score(
        data_source="nq",
        solution_str="<answer>London</answer><answer>The Paris.</answer>",
        ground_truth={"target": ["Paris"]},
        extra_info=_metadata(),
    )
    assert result == {
        "score": 1.0,
        "example_id": "test_0",
        "data_source": "nq",
        "question": "What is the capital of France?",
        "golden_answers": ["Paris"],
    }


def test_verl_reward_rejects_unprepared_rows() -> None:
    with pytest.raises(ValueError, match="example_id"):
        verl_compute_score(
            data_source="nq",
            solution_str="<answer>Paris</answer>",
            ground_truth={"target": ["Paris"]},
            extra_info={},
        )


def test_slime_reward_scores_response_without_prompt_workaround() -> None:
    sample = SimpleNamespace(
        response="<answer>Paris</answer>",
        label={"ground_truth": {"target": ["Paris"]}},
    )
    assert asyncio.run(slime_reward(None, sample)) == 1.0


def test_reward_distinguishes_missing_tag_from_normalized_empty_answer() -> None:
    sample = SimpleNamespace(
        response="no final tag",
        label={"ground_truth": {"target": ["the"]}},
    )
    assert asyncio.run(slime_reward(None, sample)) == 0.0
    sample.response = "<answer>The</answer>"
    assert asyncio.run(slime_reward(None, sample)) == 1.0


class _Status(Enum):
    COMPLETED = "completed"


def _slime_sample(example_id="test_0"):
    return SimpleNamespace(
        metadata=_metadata(example_id),
        label={"ground_truth": {"target": ["Paris"]}},
        response=(
            "<search>capital France</search>"
            "<information>Paris</information><answer>Paris</answer>"
        ),
        response_length=7,
        loss_mask=[1, 1, 0, 0, 0, 1, 1],
        status=_Status.COMPLETED,
        reward=1.0,
    )


def test_slime_eval_hook_writes_atomic_common_jsonl(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SEARCH_R1_COMMON_EVAL_DIR", str(tmp_path))
    args = SimpleNamespace(eval_reward_key=None, reward_key=None)
    should_skip_native_logging = slime_log_eval_rollout_data(
        50,
        args,
        {"official_test": {"samples": [_slime_sample()], "rewards": [1.0]}},
        {},
    )
    assert should_skip_native_logging is False
    record = json.loads((tmp_path / "50.jsonl").read_text(encoding="utf-8"))
    assert record["example_id"] == "test_0"
    assert record["data_source"] == "nq"
    assert record["native_dataset"] == "official_test"
    assert record["status"] == "completed"
    assert record["generated_tokens"] == 4
    assert record["observation_tokens"] == 3


def test_slime_eval_hook_rejects_duplicate_question_ids(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("SEARCH_R1_COMMON_EVAL_DIR", str(tmp_path))
    args = SimpleNamespace(eval_reward_key=None, reward_key=None)
    with pytest.raises(ValueError, match="Duplicate example_id"):
        slime_log_eval_rollout_data(
            50,
            args,
            {
                "nq": {
                    "samples": [_slime_sample(), _slime_sample()],
                    "rewards": [1.0, 1.0],
                }
            },
            {},
        )
