# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the fixed two-step dataset used for equal-work performance comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


# These prompts produced stable greedy Search-R1 actions across the frozen
# current-veRL and NeMo-vLLM runtimes. This is a calibration workload for
# equal-work timing, not a representative accuracy or population sample.
SOURCE_POSITIONS = (0, 1, 2, 4, 6, 8, 9, 12)
STEPS = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain_list(value: Any, field: str) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise ValueError(f"{field} must be list-valued")
    return value


def _messages(value: Any, field: str) -> list[dict[str, str]]:
    normalized = []
    for message in _plain_list(value, field):
        if not isinstance(message, dict):
            raise ValueError(f"{field} entries must be objects")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"{field} messages require text role and content")
        normalized.append({"role": role, "content": content})
    return normalized


def _answers(value: Any, field: str) -> list[str]:
    answers = [str(answer) for answer in _plain_list(value, field)]
    if not answers:
        raise ValueError(f"{field} must not be empty")
    return answers


def _read_selected_jsonl(path: Path) -> list[dict[str, Any]]:
    selected = {}
    wanted = set(SOURCE_POSITIONS)
    with path.open(encoding="utf-8") as source:
        for position, line in enumerate(source):
            if position in wanted:
                selected[position] = json.loads(line)
            if position >= SOURCE_POSITIONS[-1]:
                break
    missing = sorted(wanted.difference(selected))
    if missing:
        raise ValueError(f"{path} is missing source positions {missing}")
    return [selected[position] for position in SOURCE_POSITIONS]


def _semantic_row(
    current_row: dict[str, Any], nemo_row: dict[str, Any], position: int
) -> dict[str, Any]:
    current_question = str(current_row["question"])
    nemo_question = str(nemo_row["question"])
    current_answers = _answers(
        current_row["golden_answers"], "current golden_answers"
    )
    nemo_answers = _answers(nemo_row["answers"], "NeMo answers")
    current_messages = _messages(current_row["prompt"], "current prompt")
    try:
        nemo_input = nemo_row["responses_create_params"]["input"]
    except (KeyError, TypeError) as error:
        raise ValueError("NeMo row is missing responses_create_params.input") from error
    nemo_messages = _messages(nemo_input, "NeMo input")
    mismatches = {}
    if current_question != nemo_question:
        mismatches["question"] = [current_question, nemo_question]
    if current_answers != nemo_answers:
        mismatches["answers"] = [current_answers, nemo_answers]
    if current_messages != nemo_messages:
        mismatches["messages"] = [current_messages, nemo_messages]
    if mismatches:
        raise ValueError(
            f"Cross-framework semantic mismatch at source position {position}: "
            f"{mismatches}"
        )
    return {
        "source_position": position,
        "question": current_question,
        "answers": current_answers,
        "messages": current_messages,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare(
    current_source: Path,
    nemo_source: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Create identical ordered prompt views for two warmup/timed outer steps."""
    current_frame = pd.read_parquet(current_source)
    if len(current_frame) <= SOURCE_POSITIONS[-1]:
        raise ValueError(
            f"{current_source} has {len(current_frame)} rows, but source position "
            f"{SOURCE_POSITIONS[-1]} is required"
        )
    current_selected = current_frame.iloc[list(SOURCE_POSITIONS)].copy()
    nemo_selected = _read_selected_jsonl(nemo_source)
    current_records = current_selected.to_dict(orient="records")
    semantics = [
        _semantic_row(current_row, nemo_row, position)
        for current_row, nemo_row, position in zip(
            current_records, nemo_selected, SOURCE_POSITIONS, strict=True
        )
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    current_output = output_dir / "current-train.parquet"
    nemo_output = output_dir / "nemo-train.jsonl"
    manifest_output = output_dir / "manifest.json"

    pd.concat([current_selected] * STEPS, ignore_index=True).to_parquet(
        current_output, index=False
    )
    _write_jsonl(nemo_output, nemo_selected * STEPS)

    semantic_bytes = json.dumps(
        semantics, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    manifest = {
        "schema_version": 1,
        "purpose": "cross-framework equal-work performance calibration",
        "selection_policy": "fixed-cross-vllm-stable-greedy-search-r1-v1",
        "representative_population_sample": False,
        "source_positions": list(SOURCE_POSITIONS),
        "prompts_per_step": len(SOURCE_POSITIONS),
        "steps": STEPS,
        "rows_per_view": len(SOURCE_POSITIONS) * STEPS,
        "semantic_rows_sha256": hashlib.sha256(semantic_bytes).hexdigest(),
        "sources": {
            "current": {
                "file_name": current_source.name,
                "sha256": _sha256(current_source),
            },
            "nemo": {
                "file_name": nemo_source.name,
                "sha256": _sha256(nemo_source),
            },
        },
        "outputs": {
            "current": {
                "file_name": current_output.name,
                "sha256": _sha256(current_output),
            },
            "nemo": {
                "file_name": nemo_output.name,
                "sha256": _sha256(nemo_output),
            },
        },
    }
    manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-source", type=Path, required=True)
    parser.add_argument("--nemo-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(
        args.current_source.resolve(),
        args.nemo_source.resolve(),
        args.output_dir.resolve(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
