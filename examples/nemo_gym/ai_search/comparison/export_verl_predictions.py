# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert current-veRL validation dumps to the common Search-R1 schema."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


_INVALID_ACTION_TEXT = "My previous action is invalid."


def _as_answers(value: Any, field_name: str) -> list[str]:
    if isinstance(value, Mapping):
        for key in ("golden_answers", "target", "ground_truth"):
            if key in value:
                return _as_answers(value[key], f"{field_name}.{key}")
    if isinstance(value, str):
        value = [value]
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(answer, str) and answer for answer in value)
    ):
        raise ValueError(f"{field_name} must contain a non-empty string list")
    return value


def convert_record(value: dict[str, Any]) -> dict[str, Any]:
    """Convert one veRL generation-dump object without row-order fallbacks."""
    example_id = value.get("example_id")
    if example_id is None or not str(example_id):
        raise ValueError(
            "example_id is required; run the aligned reward adapter and prepared data"
        )
    data_source = value.get("data_source")
    if not isinstance(data_source, str) or not data_source:
        raise ValueError("data_source must be a non-empty string")
    response = value.get("output")
    if not isinstance(response, str):
        raise ValueError("output must be a string")

    answers = _as_answers(value.get("golden_answers"), "golden_answers")
    if value.get("gts") is not None:
        dump_answers = _as_answers(value["gts"], "gts")
        if answers != dump_answers:
            raise ValueError("golden_answers disagrees with the veRL dump ground truth")

    status = value.get("status", "unknown")
    if not isinstance(status, str) or not status:
        raise ValueError("status must be a non-empty string")
    record: dict[str, Any] = {
        "example_id": str(example_id),
        "data_source": data_source,
        "golden_answers": answers,
        "response": response,
        "status": status,
        "invalid_actions": value.get(
            "invalid_actions", response.count(_INVALID_ACTION_TEXT)
        ),
        "native_reward": value.get("score"),
    }
    for name in (
        "question",
        "search_errors",
        "search_count",
        "turn_count",
        "natural_termination",
        "provider_batch_ids",
        "generated_tokens",
        "observation_tokens",
        "wall_time_seconds",
    ):
        if value.get(name) is not None:
            record[name] = value[name]
    return {key: item for key, item in record.items() if item is not None}


def _read_jsonl(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
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
    """Convert veRL dumps atomically and reject duplicate evaluation IDs."""
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in _read_jsonl(paths):
        record = convert_record(value)
        example_id = record["example_id"]
        if example_id in seen:
            raise ValueError(f"Duplicate example_id {example_id!r}")
        seen.add(example_id)
        records.append(record)
    if not records:
        raise ValueError("No veRL validation records were found")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        for record in records:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, output)
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
