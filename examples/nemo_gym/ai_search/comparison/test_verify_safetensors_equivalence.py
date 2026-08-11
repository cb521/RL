# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for exact Search-R1 checkpoint round-trip verification."""

import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from verify_safetensors_equivalence import verify_equivalence


def _write_checkpoint(directory, shards) -> None:
    directory.mkdir()
    weight_map = {}
    for index, tensors in enumerate(shards, start=1):
        filename = f"model-{index:05d}-of-{len(shards):05d}.safetensors"
        save_file(tensors, directory / filename)
        weight_map.update(dict.fromkeys(tensors, filename))
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8"
    )


def test_verify_equivalence_ignores_shard_layout(tmp_path) -> None:
    tensors = {
        "model.embed_tokens.weight": np.arange(24, dtype=np.float32).reshape(6, 4),
        "model.layers.0.input_layernorm.weight": np.arange(4, dtype=np.float16),
    }
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_checkpoint(reference, [tensors])
    _write_checkpoint(
        candidate,
        [
            {"model.layers.0.input_layernorm.weight": tensors["model.layers.0.input_layernorm.weight"]},
            {"model.embed_tokens.weight": tensors["model.embed_tokens.weight"]},
        ],
    )

    report = verify_equivalence(reference, candidate)
    assert report["bitwise_equal"] is True
    assert report["tensor_count"] == 2
    assert report["parameter_count"] == 28
    assert len(report["semantic_tensor_sha256"]) == 64


def test_verify_equivalence_rejects_value_change(tmp_path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_checkpoint(reference, [{"weight": np.array([1.0, 2.0], dtype=np.float32)}])
    _write_checkpoint(candidate, [{"weight": np.array([1.0, 3.0], dtype=np.float32)}])

    with pytest.raises(ValueError, match="differs at byte"):
        verify_equivalence(reference, candidate)


def test_verify_equivalence_rejects_key_change(tmp_path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_checkpoint(reference, [{"weight": np.ones(2, dtype=np.float32)}])
    _write_checkpoint(candidate, [{"other": np.ones(2, dtype=np.float32)}])

    with pytest.raises(ValueError, match="Tensor key mismatch"):
        verify_equivalence(reference, candidate)
