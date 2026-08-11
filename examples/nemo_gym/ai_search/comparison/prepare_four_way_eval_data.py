# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare one metadata-preserving validation Parquet for all four frameworks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd


_REQUIRED_COLUMNS = {
    "id",
    "question",
    "golden_answers",
    "data_source",
    "prompt",
    "reward_model",
    "extra_info",
}
_EVIDENCE_FIELDS = ("example_id", "data_source", "question", "golden_answers")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float):
        if not math.isfinite(value):
            return {"non_finite_float": repr(value)}
        # PyArrow promotes nullable integer fields inside heterogeneous metadata
        # structs to floats on a read/write round trip. Treat integral values as
        # the same JSON number while still detecting an actual value change.
        if value.is_integer():
            return int(value)
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _rows_sha256(frame: pd.DataFrame, columns: list[str]) -> str:
    digest = hashlib.sha256()
    for values in frame[columns].itertuples(index=False, name=None):
        encoded = _canonical_bytes(dict(zip(columns, values, strict=True)))
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _answers(value: Any) -> list[str]:
    value = _jsonable(value)
    if isinstance(value, str):
        value = [value]
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(answer, str) and answer for answer in value)
    ):
        raise ValueError("golden_answers must contain a non-empty string list")
    return value


def _example_id(data_source: Any, source_id: Any) -> str:
    """Namespace official per-source IDs, which are not globally unique."""
    data_source = str(data_source)
    source_id = str(source_id)
    if not data_source or not source_id:
        raise ValueError("data_source and id values must be non-empty")
    return f"{data_source}:{source_id}"


def _enriched_extra_info(row: pd.Series) -> dict[str, Any]:
    original = row["extra_info"]
    if original is None or (isinstance(original, float) and math.isnan(original)):
        extra_info: dict[str, Any] = {}
    elif isinstance(original, Mapping):
        extra_info = {str(key): _jsonable(value) for key, value in original.items()}
    else:
        raise ValueError("extra_info must be a mapping or null")

    expected = {
        "example_id": _example_id(row["data_source"], row["id"]),
        "data_source": str(row["data_source"]),
        "question": str(row["question"]),
        "golden_answers": _answers(row["golden_answers"]),
    }
    for key, value in expected.items():
        if key in extra_info and _canonical_bytes(extra_info[key]) != _canonical_bytes(
            value
        ):
            raise ValueError(f"extra_info.{key} conflicts with the source column")
        extra_info[key] = value
    return extra_info


def _validate_enrichment(source: pd.DataFrame, prepared: pd.DataFrame) -> None:
    if list(source.columns) != list(prepared.columns):
        raise ValueError("Parquet columns changed during preparation")
    if len(source) != len(prepared):
        raise ValueError("Parquet row count changed during preparation")
    passthrough_columns = [name for name in source.columns if name != "extra_info"]
    if _rows_sha256(source, passthrough_columns) != _rows_sha256(
        prepared, passthrough_columns
    ):
        raise ValueError("A field other than extra_info changed during preparation")

    for row_number, (source_info, prepared_row) in enumerate(
        zip(source["extra_info"], prepared.to_dict(orient="records"), strict=True)
    ):
        source_info = (
            {}
            if source_info is None
            or (isinstance(source_info, float) and math.isnan(source_info))
            else _jsonable(source_info)
        )
        if not isinstance(source_info, Mapping):
            raise ValueError(f"source extra_info at row {row_number} is not a mapping")
        prepared_info = _jsonable(prepared_row["extra_info"])
        if not isinstance(prepared_info, Mapping):
            raise ValueError(f"prepared extra_info at row {row_number} is not a mapping")
        for key, value in source_info.items():
            if key not in prepared_info or _canonical_bytes(
                prepared_info[key]
            ) != _canonical_bytes(value):
                raise ValueError(
                    f"source extra_info.{key} changed at row {row_number}"
                )
        expected = {
            "example_id": _example_id(
                prepared_row["data_source"], prepared_row["id"]
            ),
            "data_source": str(prepared_row["data_source"]),
            "question": str(prepared_row["question"]),
            "golden_answers": _answers(prepared_row["golden_answers"]),
        }
        for key, value in expected.items():
            if _canonical_bytes(prepared_info.get(key)) != _canonical_bytes(value):
                raise ValueError(f"prepared extra_info.{key} is invalid at row {row_number}")


def prepare(
    source: Path,
    destination: Path,
    manifest_path: Path,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Write and verify an aligned validation file plus its evidence manifest."""
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination must be different paths")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive when provided")

    frame = pd.read_parquet(source)
    missing = sorted(_REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    if max_rows is not None:
        frame = frame.head(max_rows).copy()
    example_ids = [
        _example_id(source_name, source_id)
        for source_name, source_id in zip(
            frame["data_source"], frame["id"], strict=True
        )
    ]
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("data_source:id values must be globally unique")

    prepared = frame.copy(deep=True)
    prepared["extra_info"] = [
        _enriched_extra_info(row) for _, row in frame.iterrows()
    ]
    passthrough_columns = [name for name in frame.columns if name != "extra_info"]
    source_semantic_sha256 = _rows_sha256(frame, passthrough_columns)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    prepared.to_parquet(temporary, index=False)
    round_trip = pd.read_parquet(temporary)
    _validate_enrichment(frame, round_trip)
    os.replace(temporary, destination)

    source_counts = {
        str(name): int(count)
        for name, count in frame["data_source"].value_counts(sort=False).items()
    }
    manifest = {
        "schema_version": 1,
        "contract": "four-way-search-r1-evaluation-v1",
        "source": str(source.resolve()),
        "source_file_sha256": _file_sha256(source),
        "destination": str(destination.resolve()),
        "destination_file_sha256": _file_sha256(destination),
        "rows": len(frame),
        "max_rows": max_rows,
        "data_sources": source_counts,
        "passthrough_columns": passthrough_columns,
        "passthrough_rows_sha256": source_semantic_sha256,
        "id_order_sha256": _rows_sha256(frame, ["id"]),
        "example_id_scheme": "{data_source}:{id}",
        "prompt_sha256": _rows_sha256(frame, ["prompt"]),
        "ground_truth_sha256": _rows_sha256(
            frame, ["golden_answers", "reward_model"]
        ),
        "added_extra_info_fields": list(_EVIDENCE_FIELDS),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    manifest_temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(manifest_temporary, manifest_path)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--max-rows", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    manifest_path = args.manifest or args.output.with_suffix(
        args.output.suffix + ".manifest.json"
    )
    manifest = prepare(
        args.source.resolve(),
        args.output.resolve(),
        manifest_path.resolve(),
        args.max_rows,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
