# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export NeMo Gym trajectory collection rows to the common quality schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


_INVALID_ACTION_TEXT = "My previous action is invalid."


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        text = content.get("text")
        return text if isinstance(text, str) else ""
    if isinstance(content, list):
        return "".join(_content_text(item) for item in content)
    return ""


def response_text(full_result: dict[str, Any]) -> str:
    """Reconstruct generated actions and injected observations in order."""
    response = full_result.get("response")
    if not isinstance(response, dict):
        raise ValueError("full_result.response must be an object")
    output = response.get("output")
    if not isinstance(output, list):
        raise ValueError("full_result.response.output must be a list")

    chunks: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        generation = item.get("generation_str")
        if isinstance(generation, str):
            chunks.append(generation)
            continue
        if item.get("type") == "message" or item.get("role") in {
            "assistant",
            "user",
        }:
            chunks.append(_content_text(item.get("content")))
    return "".join(chunks)


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def convert_result(full_result: dict[str, Any]) -> dict[str, Any]:
    """Convert one complete NeMo Gym verify result."""
    source_id = full_result.get("source_id")
    data_source = full_result.get("data_source")
    answers = full_result.get("answers")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("full_result.source_id must be a non-empty string")
    if not isinstance(data_source, str) or not data_source:
        raise ValueError("full_result.data_source must be a non-empty string")
    if (
        not isinstance(answers, list)
        or not answers
        or not all(isinstance(answer, str) and answer for answer in answers)
    ):
        raise ValueError("full_result.answers must be a non-empty string list")

    response = full_result.get("response")
    if not isinstance(response, dict):
        raise ValueError("full_result.response must be an object")
    text = response_text(full_result)
    response_status = response.get("status", "completed")
    if response.get("error") is not None:
        status = "error"
    elif response.get("incomplete_details") is not None:
        status = "truncated"
    elif isinstance(response_status, str) and response_status:
        status = response_status
    else:
        status = "completed"

    usage = response.get("usage")
    generated_tokens = None
    if isinstance(usage, dict) and usage.get("output_tokens") is not None:
        generated_tokens = _nonnegative_int(
            usage["output_tokens"], "response.usage.output_tokens"
        )

    search_errors = full_result.get("search_error_count", 0)
    search_errors = _nonnegative_int(search_errors, "search_error_count")
    record = {
        "example_id": source_id,
        "data_source": data_source,
        "golden_answers": answers,
        "question": full_result.get("question"),
        "response": text,
        "status": status,
        "search_errors": search_errors,
        "invalid_actions": text.count(_INVALID_ACTION_TEXT),
        "generated_tokens": generated_tokens,
        "native_reward": full_result.get("reward"),
        "native_exact_match": full_result.get("exact_match"),
        "native_answer_f1": full_result.get("answer_f1"),
        "native_num_search_calls": full_result.get("num_search_calls"),
        "native_duplicate_query_rate": full_result.get("duplicate_query_rate"),
    }
    return {key: value for key, value in record.items() if value is not None}


def read_results(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    """Yield full-result objects from NeMo Gym trajectory JSONL files."""
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
                if not isinstance(value, dict):
                    raise ValueError(f"Expected an object at {path}:{line_number}")
                yield value


def export_predictions(paths: list[Path], output: Path) -> int:
    """Convert files while rejecting duplicate evaluation example IDs."""
    seen: set[str] = set()
    records = []
    for full_result in read_results(paths):
        record = convert_result(full_result)
        example_id = record["example_id"]
        if example_id in seen:
            raise ValueError(
                f"Duplicate example_id {example_id!r}; final evaluation requires "
                "one rollout per question"
            )
        seen.add(example_id)
        records.append(record)
    if not records:
        raise ValueError("No trajectory results were found")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as destination:
        for record in records:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    for path in args.input:
        if not path.is_file():
            raise FileNotFoundError(path)
    count = export_predictions(args.input, args.output)
    print(f"{args.output} ({count} predictions)")


if __name__ == "__main__":
    main()
