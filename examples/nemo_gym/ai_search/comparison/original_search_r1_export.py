# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming common-schema writer for the original Search-R1 validation loop."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_INVALID_ACTION_TEXT = "My previous action is invalid."


def _scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _answers(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        for key in ("target", "ground_truth", "golden_answers"):
            if key in value:
                return _answers(value[key])
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        value = [value]
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(answer, str) and answer for answer in value)
    ):
        raise ValueError("reward_model ground truth must contain string answers")
    return value


def _integer_sum(value: Any, name: str) -> int:
    result = _scalar(value.sum())
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise ValueError(f"{name} sum must be numeric")
    integer = int(result)
    if integer < 0 or integer != result:
        raise ValueError(f"{name} sum must be a non-negative integer")
    return integer


class OriginalSearchR1ValidationWriter:
    """Write batches to a partial file and publish only after validation finishes."""

    def __init__(self, output: str | Path | None, tokenizer: Any):
        self.output = Path(output) if output else None
        self.tokenizer = tokenizer
        self.seen: set[str] = set()
        self.count = 0
        self.partial = (
            self.output.with_suffix(self.output.suffix + ".partial")
            if self.output is not None
            else None
        )
        if self.partial is not None:
            self.output.parent.mkdir(parents=True, exist_ok=True)
            self.partial.write_text("", encoding="utf-8")

    @property
    def enabled(self) -> bool:
        return self.output is not None

    def write_batch(
        self,
        batch: Any,
        reward_tensor: Any,
        generation_meta: Mapping[str, Any] | None,
    ) -> None:
        """Append one driver-side validation batch."""
        if not self.enabled:
            return
        assert self.partial is not None
        generation_meta = generation_meta or {}
        batch_size = int(batch.batch["responses"].shape[0])
        active = generation_meta.get("active_mask", [False] * batch_size)
        if hasattr(active, "tolist"):
            active = active.tolist()
        if not isinstance(active, list) or len(active) != batch_size:
            raise ValueError("generation active_mask does not match validation batch")

        records: list[dict[str, Any]] = []
        for index in range(batch_size):
            item = batch[index]
            fields = item.non_tensor_batch
            source_id = fields.get("id")
            if source_id is None or not str(source_id):
                raise ValueError("validation row is missing a stable id")
            data_source = fields.get("data_source")
            question = fields.get("question")
            if not isinstance(data_source, str) or not data_source:
                raise ValueError(f"validation row {source_id!r} has no data_source")
            example_id = f"{data_source}:{source_id}"
            if example_id in self.seen:
                raise ValueError(f"Duplicate validation id {example_id!r}")
            self.seen.add(example_id)
            if not isinstance(question, str) or not question:
                raise ValueError(f"validation row {example_id!r} has no question")
            answers = _answers(fields.get("reward_model"))

            prompt_length = int(item.batch["prompts"].shape[-1])
            response_attention = item.batch["attention_mask"][prompt_length:]
            response_tokens = _integer_sum(
                response_attention, "response attention mask"
            )
            response_ids = item.batch["responses"][:response_tokens]
            response = self.tokenizer.decode(
                response_ids, skip_special_tokens=True
            )
            if not isinstance(response, str):
                raise ValueError("tokenizer.decode must return a string")

            if "info_mask" in item.batch:
                generated_tokens = _integer_sum(
                    item.batch["info_mask"][prompt_length:], "response info mask"
                )
                observation_tokens = response_tokens - generated_tokens
            else:
                generated_tokens = response_tokens
                observation_tokens = 0
            if observation_tokens < 0:
                raise ValueError("info_mask contains more tokens than attention_mask")

            native_reward = _scalar(reward_tensor[index].sum())
            record = {
                "example_id": example_id,
                "data_source": data_source,
                "golden_answers": answers,
                "question": question,
                "response": response,
                "status": "unfinished" if bool(active[index]) else "completed",
                "invalid_actions": response.count(_INVALID_ACTION_TEXT),
                "generated_tokens": generated_tokens,
                "observation_tokens": observation_tokens,
                "native_reward": native_reward,
            }
            records.append(record)

        with self.partial.open("a", encoding="utf-8") as destination:
            for record in records:
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.count += len(records)

    def finish(self) -> None:
        """Atomically publish a non-empty completed validation artifact."""
        if not self.enabled:
            return
        assert self.output is not None and self.partial is not None
        if self.count == 0:
            raise ValueError("No original Search-R1 validation rows were exported")
        os.replace(self.partial, self.output)
        print(f"Exported {self.count} common Search-R1 predictions to {self.output}")
