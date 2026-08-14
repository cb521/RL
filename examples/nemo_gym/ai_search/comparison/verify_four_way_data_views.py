# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that NeMo JSONL and external-framework Parquet are row-identical."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterable, Mapping
from itertools import zip_longest
from pathlib import Path
from typing import Any


_COLUMNS = (
    "id",
    "data_source",
    "question",
    "golden_answers",
    "prompt",
    "reward_model",
    "extra_info",
)
_MISSING = object()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"Blank JSONL row at {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield row


def _parquet_rows(path: Path) -> Iterable[dict[str, Any]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    missing = sorted(set(_COLUMNS).difference(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    for batch in parquet.iter_batches(columns=list(_COLUMNS), batch_size=4096):
        yield from batch.to_pylist()


def _require_string(value: Any, field: str, row_number: int) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"row {row_number}: {field} must be a non-empty string")
    return value


def _require_answers(value: Any, field: str, row_number: int) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(answer, str) and answer for answer in value)
    ):
        raise ValueError(f"row {row_number}: {field} must be a non-empty string list")
    return value


def _reward_answers(reward_model: Any, row_number: int) -> list[str]:
    if not isinstance(reward_model, Mapping):
        raise ValueError(f"row {row_number}: reward_model must be an object")
    ground_truth = reward_model.get("ground_truth")
    if not isinstance(ground_truth, Mapping):
        raise ValueError(f"row {row_number}: reward_model.ground_truth is invalid")
    return _require_answers(
        ground_truth.get("target"),
        "reward_model.ground_truth.target",
        row_number,
    )


def verify_views(jsonl_path: Path, parquet_path: Path) -> dict[str, Any]:
    """Fail on any row-order or model/scorer-visible semantic mismatch."""
    semantic_digest = hashlib.sha256()
    id_digest = hashlib.sha256()
    source_counts: Counter[str] = Counter()
    seen: set[str] = set()
    row_count = 0

    for row_number, (json_row, parquet_row) in enumerate(
        zip_longest(
            _jsonl_rows(jsonl_path),
            _parquet_rows(parquet_path),
            fillvalue=_MISSING,
        ),
        start=1,
    ):
        if json_row is _MISSING or parquet_row is _MISSING:
            raise ValueError(
                f"row count mismatch between {jsonl_path} and {parquet_path} "
                f"at row {row_number}"
            )
        assert isinstance(json_row, dict) and isinstance(parquet_row, dict)
        data_source = _require_string(
            parquet_row.get("data_source"), "Parquet data_source", row_number
        )
        raw_id = _require_string(parquet_row.get("id"), "Parquet id", row_number)
        example_id = f"{data_source}:{raw_id}"
        if example_id in seen:
            raise ValueError(f"row {row_number}: duplicate example ID {example_id!r}")
        seen.add(example_id)

        extra_info = parquet_row.get("extra_info")
        if not isinstance(extra_info, Mapping):
            raise ValueError(f"row {row_number}: Parquet extra_info is invalid")
        question = _require_string(
            parquet_row.get("question"), "Parquet question", row_number
        )
        answers = _require_answers(
            parquet_row.get("golden_answers"),
            "Parquet golden_answers",
            row_number,
        )
        prompt = parquet_row.get("prompt")
        if not isinstance(prompt, list) or not prompt:
            raise ValueError(f"row {row_number}: Parquet prompt must be a list")
        reward_answers = _reward_answers(parquet_row.get("reward_model"), row_number)
        responses_params = json_row.get("responses_create_params")
        if not isinstance(responses_params, Mapping):
            raise ValueError(
                f"row {row_number}: JSONL responses_create_params is invalid"
            )

        comparisons = {
            "example_id": (json_row.get("source_id"), example_id),
            "extra_info.example_id": (extra_info.get("example_id"), example_id),
            "data_source": (json_row.get("data_source"), data_source),
            "extra_info.data_source": (extra_info.get("data_source"), data_source),
            "question": (json_row.get("question"), question),
            "extra_info.question": (extra_info.get("question"), question),
            "answers": (json_row.get("answers"), answers),
            "extra_info.golden_answers": (
                extra_info.get("golden_answers"),
                answers,
            ),
            "reward answers": (reward_answers, answers),
            "prompt": (responses_params.get("input"), prompt),
        }
        for field, (json_value, parquet_value) in comparisons.items():
            if _canonical_bytes(json_value) != _canonical_bytes(parquet_value):
                raise ValueError(f"row {row_number}: mismatched {field}")

        semantic_row = {
            "example_id": example_id,
            "data_source": data_source,
            "question": question,
            "answers": answers,
            "prompt": prompt,
        }
        encoded = _canonical_bytes(semantic_row)
        semantic_digest.update(len(encoded).to_bytes(8, "big"))
        semantic_digest.update(encoded)
        id_digest.update(len(example_id.encode()).to_bytes(8, "big"))
        id_digest.update(example_id.encode())
        source_counts[data_source] += 1
        row_count += 1

    if row_count == 0:
        raise ValueError("The aligned data views are empty")
    return {
        "schema_version": 1,
        "contract": "four-way-search-r1-data-view-equivalence-v1",
        "jsonl": str(jsonl_path.resolve()),
        "jsonl_sha256": _file_sha256(jsonl_path),
        "parquet": str(parquet_path.resolve()),
        "parquet_sha256": _file_sha256(parquet_path),
        "rows": row_count,
        "data_sources": dict(sorted(source_counts.items())),
        "id_order_sha256": id_digest.hexdigest(),
        "semantic_rows_sha256": semantic_digest.hexdigest(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    for path in (args.jsonl, args.parquet):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = verify_views(args.jsonl, args.parquet)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
