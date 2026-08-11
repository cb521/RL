# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime reward and export hooks for aligned veRL and slime evaluation."""

from __future__ import annotations

import json
import os
import re
import string
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_INVALID_ACTION_TEXT = "My previous action is invalid."
_SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def normalize_answer(value: str) -> str:
    """Apply Search-R1's case, punctuation, article, and whitespace rules."""
    lowered = value.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def _as_answers(value: Any, field_name: str) -> list[str]:
    if isinstance(value, Mapping):
        for key in ("golden_answers", "target", "ground_truth"):
            if key in value:
                return _as_answers(value[key], f"{field_name}.{key}")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, str):
        value = [value]
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(answer, str) and answer for answer in value)
    ):
        raise ValueError(f"{field_name} must contain a non-empty string list")
    return value


def _last_answer(response: str) -> str | None:
    matches = list(_ANSWER_PATTERN.finditer(response))
    return matches[-1].group(1).strip() if matches else None


def _exact_match(response: str, answers: list[str]) -> float:
    answer = _last_answer(response)
    if answer is None:
        return 0.0
    prediction = normalize_answer(answer)
    return float(
        any(prediction == normalize_answer(answer) for answer in answers)
    )


def _required_metadata(extra_info: Any) -> dict[str, Any]:
    if not isinstance(extra_info, Mapping):
        raise ValueError("extra_info must be a mapping")
    metadata = dict(extra_info)
    for name in ("example_id", "data_source", "question", "golden_answers"):
        if name not in metadata:
            raise ValueError(
                f"extra_info.{name} is required; use prepare_four_way_eval_data.py"
            )
    if not str(metadata["example_id"]):
        raise ValueError("extra_info.example_id cannot be empty")
    if not isinstance(metadata["data_source"], str) or not metadata["data_source"]:
        raise ValueError("extra_info.data_source must be a non-empty string")
    if not isinstance(metadata["question"], str) or not metadata["question"]:
        raise ValueError("extra_info.question must be a non-empty string")
    metadata["golden_answers"] = _as_answers(
        metadata["golden_answers"], "extra_info.golden_answers"
    )
    return metadata


def verl_compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Any,
    **_: Any,
) -> dict[str, Any]:
    """Current-veRL custom reward preserving stable per-question evidence."""
    if not isinstance(solution_str, str):
        raise ValueError("solution_str must be a string")
    metadata = _required_metadata(extra_info)
    answers = _as_answers(ground_truth, "ground_truth")
    if data_source != metadata["data_source"]:
        raise ValueError(
            "data_source disagrees with prepared metadata: "
            f"{data_source!r} != {metadata['data_source']!r}"
        )
    if answers != metadata["golden_answers"]:
        raise ValueError("ground_truth disagrees with prepared golden_answers")

    return {
        "score": _exact_match(solution_str, answers),
        "example_id": str(metadata["example_id"]),
        "data_source": data_source,
        "question": metadata["question"],
        "golden_answers": answers,
    }


async def slime_reward(_: Any, sample: Any, **__: Any) -> float:
    """Score slime's generated response directly, without prepending its prompt."""
    response = getattr(sample, "response", None)
    if not isinstance(response, str):
        raise ValueError("sample.response must be a string")
    answers = _as_answers(getattr(sample, "label", None), "sample.label")
    return _exact_match(response, answers)


def _plain_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _sample_status(sample: Any) -> str:
    status = getattr(sample, "status", None)
    status = getattr(status, "value", status)
    if status is None:
        return "unknown"
    if not isinstance(status, str) or not status:
        raise ValueError("sample.status must be a non-empty string or enum")
    return status


def _native_reward(args: Any, sample: Any) -> Any:
    reward = getattr(sample, "reward", None)
    if not isinstance(reward, Mapping):
        return _plain_scalar(reward)
    reward_key = getattr(args, "eval_reward_key", None) or getattr(
        args, "reward_key", None
    )
    if reward_key:
        if reward_key not in reward:
            raise ValueError(f"sample.reward is missing configured key {reward_key!r}")
        return _plain_scalar(reward[reward_key])
    if "score" in reward:
        return _plain_scalar(reward["score"])
    raise ValueError("dict-valued sample.reward requires an evaluation reward key")


def _slime_prediction(args: Any, dataset_name: str, sample: Any) -> dict[str, Any]:
    metadata = _required_metadata(getattr(sample, "metadata", None))
    answers = _as_answers(getattr(sample, "label", None), "sample.label")
    if answers != metadata["golden_answers"]:
        raise ValueError("sample.label disagrees with prepared golden_answers")

    response = getattr(sample, "response", None)
    if not isinstance(response, str):
        raise ValueError("sample.response must be a string")
    response_length = _plain_scalar(getattr(sample, "response_length", None))
    if isinstance(response_length, bool) or not isinstance(response_length, int):
        raise ValueError("sample.response_length must be an integer")
    if response_length < 0:
        raise ValueError("sample.response_length cannot be negative")

    record: dict[str, Any] = {
        "example_id": str(metadata["example_id"]),
        "data_source": metadata["data_source"],
        "native_dataset": dataset_name,
        "golden_answers": answers,
        "question": metadata["question"],
        "response": response,
        "status": _sample_status(sample),
        "invalid_actions": response.count(_INVALID_ACTION_TEXT),
        "native_reward": _native_reward(args, sample),
    }

    loss_mask = getattr(sample, "loss_mask", None)
    if loss_mask is not None:
        if hasattr(loss_mask, "tolist"):
            loss_mask = loss_mask.tolist()
        if not isinstance(loss_mask, list) or len(loss_mask) != response_length:
            raise ValueError("sample.loss_mask must align with sample.response_length")
        mask_values = [_plain_scalar(value) for value in loss_mask]
        if any(value not in (0, 1, False, True) for value in mask_values):
            raise ValueError("sample.loss_mask values must be binary")
        generated_tokens = sum(int(value) for value in mask_values)
        record["generated_tokens"] = generated_tokens
        record["observation_tokens"] = response_length - generated_tokens

    search_errors = metadata.get("search_errors")
    if search_errors is not None:
        search_errors = _plain_scalar(search_errors)
        if (
            isinstance(search_errors, bool)
            or not isinstance(search_errors, int)
            or search_errors < 0
        ):
            raise ValueError("metadata.search_errors must be a non-negative integer")
        record["search_errors"] = search_errors
    return {key: value for key, value in record.items() if value is not None}


def slime_log_eval_rollout_data(
    rollout_id: Any,
    args: Any,
    data: Any,
    extra_metrics: Any,
) -> bool:
    """Write one atomic common-schema JSONL from slime's public eval hook."""
    del extra_metrics
    output_dir_value = os.environ.get("SEARCH_R1_COMMON_EVAL_DIR")
    if not output_dir_value:
        raise ValueError(
            "SEARCH_R1_COMMON_EVAL_DIR must be set when the common eval hook is used"
        )
    if not isinstance(data, Mapping) or not data:
        raise ValueError("slime eval data must be a non-empty mapping")

    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    for dataset_name, dataset_result in data.items():
        if not isinstance(dataset_name, str) or not dataset_name:
            raise ValueError("slime evaluation dataset names must be non-empty strings")
        if not isinstance(dataset_result, Mapping):
            raise ValueError(f"slime result for {dataset_name!r} must be a mapping")
        samples = dataset_result.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"slime result for {dataset_name!r} has no samples")
        for sample in samples:
            record = _slime_prediction(args, dataset_name, sample)
            example_id = record["example_id"]
            if example_id in seen:
                raise ValueError(
                    f"Duplicate example_id {example_id!r}; aligned final evaluation "
                    "requires one sample per question"
                )
            seen.add(example_id)
            records.append(record)

    safe_rollout_id = _SAFE_FILENAME_PATTERN.sub("_", str(rollout_id)).strip("._")
    if not safe_rollout_id:
        raise ValueError("rollout_id does not contain a safe filename component")
    output_dir = Path(output_dir_value)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{safe_rollout_id}.jsonl"
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        for record in records:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, output)
    print(f"Exported {len(records)} common Search-R1 predictions to {output}")

    # Preserve slime's native metric logging after the artifact has been written.
    return False
