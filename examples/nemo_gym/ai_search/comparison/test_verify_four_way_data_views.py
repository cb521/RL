# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for cross-framework Search-R1 data-view equivalence."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from verify_four_way_data_views import verify_views


def _row(source_id: str, question: str, answer: str) -> tuple[dict, dict]:
    data_source, raw_id = source_id.split(":", maxsplit=1)
    prompt = [{"role": "user", "content": f"Question: {question}\n"}]
    json_row = {
        "source_id": source_id,
        "data_source": data_source,
        "question": question,
        "answers": [answer],
        "responses_create_params": {"input": prompt},
    }
    parquet_row = {
        "id": raw_id,
        "data_source": data_source,
        "question": question,
        "golden_answers": [answer],
        "prompt": prompt,
        "reward_model": {
            "ground_truth": {"target": [answer]},
            "style": "rule",
        },
        "extra_info": {
            "example_id": source_id,
            "data_source": data_source,
            "question": question,
            "golden_answers": [answer],
        },
    }
    return json_row, parquet_row


def _write_views(tmp_path, rows):
    jsonl_path = tmp_path / "data.jsonl"
    parquet_path = tmp_path / "data.parquet"
    jsonl_path.write_text(
        "".join(json.dumps(row[0]) + "\n" for row in rows), encoding="utf-8"
    )
    pq.write_table(pa.Table.from_pylist([row[1] for row in rows]), parquet_path)
    return jsonl_path, parquet_path


def test_verify_views_accepts_identical_ordered_semantics(tmp_path) -> None:
    paths = _write_views(
        tmp_path,
        [
            _row("nq:test_0", "First?", "One"),
            _row("hotpotqa:test_0", "Second?", "Two"),
        ],
    )
    manifest = verify_views(*paths)
    assert manifest["rows"] == 2
    assert manifest["data_sources"] == {"hotpotqa": 1, "nq": 1}
    assert len(manifest["semantic_rows_sha256"]) == 64


def test_verify_views_rejects_prompt_or_order_mismatch(tmp_path) -> None:
    rows = [
        _row("nq:test_0", "First?", "One"),
        _row("nq:test_1", "Second?", "Two"),
    ]
    jsonl_path, parquet_path = _write_views(tmp_path, rows)
    table = pq.read_table(parquet_path)
    pq.write_table(table.take(pa.array([1, 0])), parquet_path)
    with pytest.raises(ValueError, match="mismatched"):
        verify_views(jsonl_path, parquet_path)
