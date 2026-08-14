# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pandas as pd
import pytest

from prepare_controlled_work_data import SOURCE_POSITIONS, prepare


def _current_row(position: int) -> dict:
    return {
        "id": str(position),
        "question": f"Question {position}?",
        "golden_answers": [f"Answer {position}"],
        "data_source": "fixture",
        "prompt": [{"role": "user", "content": f"Prompt {position}"}],
        "extra_info": {"index": position},
    }


def _nemo_row(position: int) -> dict:
    return {
        "question": f"Question {position}?",
        "answers": [f"Answer {position}"],
        "responses_create_params": {
            "input": [{"role": "user", "content": f"Prompt {position}"}]
        },
    }


def _write_sources(tmp_path):
    current_source = tmp_path / "train.parquet"
    nemo_source = tmp_path / "train.jsonl"
    row_count = SOURCE_POSITIONS[-1] + 1
    pd.DataFrame([_current_row(index) for index in range(row_count)]).to_parquet(
        current_source, index=False
    )
    nemo_source.write_text(
        "".join(json.dumps(_nemo_row(index)) + "\n" for index in range(row_count)),
        encoding="utf-8",
    )
    return current_source, nemo_source


def test_prepare_repeats_same_eight_aligned_prompts_for_two_steps(tmp_path) -> None:
    current_source, nemo_source = _write_sources(tmp_path)
    output_dir = tmp_path / "controlled"

    manifest = prepare(current_source, nemo_source, output_dir)

    current = pd.read_parquet(output_dir / "current-train.parquet")
    nemo = [
        json.loads(line)
        for line in (output_dir / "nemo-train.jsonl").read_text().splitlines()
    ]
    expected_questions = [f"Question {index}?" for index in SOURCE_POSITIONS]
    assert current["question"].tolist() == expected_questions * 2
    assert [row["question"] for row in nemo] == expected_questions * 2
    assert manifest["prompts_per_step"] == 8
    assert manifest["steps"] == 2
    assert manifest["rows_per_view"] == 16
    assert manifest["representative_population_sample"] is False
    assert len(manifest["semantic_rows_sha256"]) == 64


def test_prepare_rejects_cross_framework_prompt_mismatch(tmp_path) -> None:
    current_source, nemo_source = _write_sources(tmp_path)
    rows = [json.loads(line) for line in nemo_source.read_text().splitlines()]
    rows[SOURCE_POSITIONS[0]]["responses_create_params"]["input"][0][
        "content"
    ] = "different prompt"
    nemo_source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="semantic mismatch"):
        prepare(current_source, nemo_source, tmp_path / "controlled")
