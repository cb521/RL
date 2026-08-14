# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the metadata-only four-framework validation conversion."""

import numpy as np
import pandas as pd
import pytest

from prepare_four_way_eval_data import prepare


def _frame(ids=("test_0", "test_1")) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": list(ids),
            "question": ["Capital of France?", "Capital of England?"],
            "golden_answers": [
                np.array(["Paris"], dtype=object),
                np.array(["London"], dtype=object),
            ],
            "data_source": ["nq", "nq"],
            "prompt": [
                np.array([{"role": "user", "content": "Q1"}], dtype=object),
                np.array([{"role": "user", "content": "Q2"}], dtype=object),
            ],
            "reward_model": [
                {"style": "rule", "ground_truth": {"target": ["Paris"]}},
                {"style": "rule", "ground_truth": {"target": ["London"]}},
            ],
            "extra_info": [
                {"index": 0, "split": "test"},
                {"index": 1, "split": "test"},
            ],
        }
    )


def test_prepare_preserves_all_non_metadata_fields(tmp_path) -> None:
    source = tmp_path / "source.parquet"
    output = tmp_path / "prepared.parquet"
    manifest_path = tmp_path / "manifest.json"
    _frame().to_parquet(source, index=False)

    manifest = prepare(source, output, manifest_path)
    prepared = pd.read_parquet(output)
    assert manifest["rows"] == 2
    assert manifest["data_sources"] == {"nq": 2}
    assert prepared.iloc[0]["prompt"].tolist() == [
        {"role": "user", "content": "Q1"}
    ]
    assert prepared.iloc[0]["extra_info"] == {
        "index": 0,
        "split": "test",
        "example_id": "nq:test_0",
        "data_source": "nq",
        "question": "Capital of France?",
        "golden_answers": ["Paris"],
    }
    assert manifest_path.is_file()


def test_prepare_allows_per_source_ids_but_rejects_composite_duplicates(
    tmp_path,
) -> None:
    frame = _frame(("duplicate", "duplicate"))
    frame.loc[1, "data_source"] = "hotpotqa"
    source = tmp_path / "source.parquet"
    frame.to_parquet(source, index=False)
    manifest = prepare(
        source,
        tmp_path / "prepared-ok.parquet",
        tmp_path / "manifest-ok.json",
    )
    assert manifest["rows"] == 2

    frame.loc[1, "data_source"] = "nq"
    frame.to_parquet(source, index=False)
    with pytest.raises(ValueError, match="unique"):
        prepare(
            source,
            tmp_path / "prepared.parquet",
            tmp_path / "manifest.json",
        )
