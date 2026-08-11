# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the original Search-R1 streaming validation writer."""

import json

import numpy as np

from original_search_r1_export import OriginalSearchR1ValidationWriter


class _Tokenizer:
    def decode(self, values, skip_special_tokens=True):
        assert skip_special_tokens is True
        return "".join(chr(int(value)) for value in values)


class _Item:
    def __init__(self, batch, non_tensor_batch):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch


class _Batch:
    def __init__(self, response):
        response_ids = np.array([[ord(character) for character in response]])
        prompt_ids = np.array([[1, 2]])
        response_mask = np.ones_like(response_ids)
        observation_tokens = 3
        response_info_mask = np.ones_like(response_ids)
        response_info_mask[:, 2 : 2 + observation_tokens] = 0
        self.batch = {
            "prompts": prompt_ids,
            "responses": response_ids,
            "attention_mask": np.concatenate(
                [np.ones_like(prompt_ids), response_mask], axis=1
            ),
            "info_mask": np.concatenate(
                [np.ones_like(prompt_ids), response_info_mask], axis=1
            ),
        }
        self._fields = {
            "id": "test_0",
            "data_source": "nq",
            "question": "Capital of France?",
            "reward_model": {
                "style": "rule",
                "ground_truth": {"target": np.array(["Paris"], dtype=object)},
            },
        }

    def __getitem__(self, index):
        return _Item(
            {name: value[index] for name, value in self.batch.items()},
            self._fields,
        )


def test_original_writer_publishes_only_on_finish(tmp_path) -> None:
    output = tmp_path / "predictions.jsonl"
    response = "<answer>Paris</answer>"
    writer = OriginalSearchR1ValidationWriter(output, _Tokenizer())
    writer.write_batch(
        _Batch(response),
        np.array([[1.0]]),
        {
            "active_mask": [False],
            "observability_trace_ids": ["trace-1"],
            "provider_batch_ids": [["batch-1", "batch-2"]],
        },
    )
    assert not output.exists()
    assert output.with_suffix(".jsonl.partial").is_file()

    writer.finish()
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["example_id"] == "nq:test_0"
    assert record["response"] == response
    assert record["status"] == "completed"
    assert record["generated_tokens"] == len(response) - 3
    assert record["observation_tokens"] == 3
    assert record["native_reward"] == 1.0
    assert record["observability_trace_id"] == "trace-1"
    assert record["provider_batch_ids"] == ["batch-1", "batch-2"]


def test_disabled_original_writer_is_a_noop() -> None:
    writer = OriginalSearchR1ValidationWriter(None, _Tokenizer())
    writer.write_batch(None, None, None)
    writer.finish()
