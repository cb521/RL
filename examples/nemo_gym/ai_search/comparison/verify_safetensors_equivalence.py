# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify two sharded Safetensors checkpoints are bitwise tensor-equivalent."""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import struct
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO


_CHUNK_BYTES = 8 * 1024 * 1024
_MAX_HEADER_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class _TensorLocation:
    path: Path
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int

    @property
    def byte_count(self) -> int:
        return self.end - self.start


def _parse_tensor_file(path: Path) -> dict[str, _TensorLocation]:
    size = path.stat().st_size
    if size < 10:
        raise ValueError(f"Safetensors file is too small: {path}")
    with path.open("rb") as source:
        encoded_length = source.read(8)
        header_length = struct.unpack("<Q", encoded_length)[0]
        if header_length < 2 or header_length > _MAX_HEADER_BYTES:
            raise ValueError(f"Invalid Safetensors header length in {path}")
        data_start = 8 + header_length
        if data_start > size:
            raise ValueError(f"Truncated Safetensors header in {path}")
        try:
            header = json.loads(source.read(header_length))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid Safetensors header in {path}") from error

    if not isinstance(header, dict):
        raise ValueError(f"Safetensors header must be an object: {path}")
    tensors: dict[str, _TensorLocation] = {}
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(metadata, dict):
            raise ValueError(f"Invalid tensor metadata in {path}")
        dtype = metadata.get("dtype")
        shape = metadata.get("shape")
        offsets = metadata.get("data_offsets")
        if (
            not isinstance(dtype, str)
            or not isinstance(shape, list)
            or not all(isinstance(dimension, int) and dimension >= 0 for dimension in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(offset, int) and offset >= 0 for offset in offsets)
        ):
            raise ValueError(f"Invalid metadata for tensor {name!r} in {path}")
        relative_start, relative_end = offsets
        start = data_start + relative_start
        end = data_start + relative_end
        if relative_end < relative_start or end > size:
            raise ValueError(f"Invalid offsets for tensor {name!r} in {path}")
        tensors[name] = _TensorLocation(
            path=path,
            dtype=dtype,
            shape=tuple(shape),
            start=start,
            end=end,
        )
    return tensors


def _index_weight_map(directory: Path) -> dict[str, str] | None:
    index_path = directory / "model.safetensors.index.json"
    if not index_path.is_file():
        return None
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
    if not isinstance(weight_map, dict) or not all(
        isinstance(name, str) and isinstance(filename, str)
        for name, filename in weight_map.items()
    ):
        raise ValueError(f"Invalid weight_map in {index_path}")
    return weight_map


def _checkpoint_tensors(directory: Path) -> dict[str, _TensorLocation]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No Safetensors files found under {directory}")

    tensors: dict[str, _TensorLocation] = {}
    for path in files:
        for name, location in _parse_tensor_file(path).items():
            if name in tensors:
                raise ValueError(f"Duplicate tensor {name!r} under {directory}")
            tensors[name] = location

    weight_map = _index_weight_map(directory)
    if weight_map is not None:
        if set(weight_map) != set(tensors):
            missing = sorted(set(weight_map).difference(tensors))[:5]
            extra = sorted(set(tensors).difference(weight_map))[:5]
            raise ValueError(
                f"Safetensors index/header mismatch under {directory}: "
                f"missing={missing}, extra={extra}"
            )
        for name, filename in weight_map.items():
            if tensors[name].path.name != filename:
                raise ValueError(
                    f"Safetensors index maps {name!r} to {filename!r}, but it is in "
                    f"{tensors[name].path.name!r}"
                )
    return tensors


def _open_mmaps(
    stack: ExitStack, locations: dict[str, _TensorLocation]
) -> dict[Path, mmap.mmap]:
    mapped: dict[Path, mmap.mmap] = {}
    for path in sorted({location.path for location in locations.values()}):
        source: BinaryIO = stack.enter_context(path.open("rb"))
        mapped[path] = stack.enter_context(
            mmap.mmap(source.fileno(), length=0, access=mmap.ACCESS_READ)
        )
    return mapped


def _chunks(mapped: mmap.mmap, start: int, end: int) -> Iterator[bytes]:
    for offset in range(start, end, _CHUNK_BYTES):
        yield mapped[offset : min(offset + _CHUNK_BYTES, end)]


def _shape_elements(shape: tuple[int, ...]) -> int:
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements


def verify_equivalence(reference_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    """Return a manifest or fail on the first tensor metadata/value mismatch."""
    reference = _checkpoint_tensors(reference_dir)
    candidate = _checkpoint_tensors(candidate_dir)
    if set(reference) != set(candidate):
        missing = sorted(set(reference).difference(candidate))[:10]
        extra = sorted(set(candidate).difference(reference))[:10]
        raise ValueError(f"Tensor key mismatch: missing={missing}, extra={extra}")

    digest = hashlib.sha256()
    parameter_count = 0
    tensor_bytes = 0
    with ExitStack() as stack:
        reference_maps = _open_mmaps(stack, reference)
        candidate_maps = _open_mmaps(stack, candidate)
        for name in sorted(reference):
            expected = reference[name]
            actual = candidate[name]
            if expected.dtype != actual.dtype:
                raise ValueError(
                    f"Tensor {name!r} dtype mismatch: {expected.dtype} != {actual.dtype}"
                )
            if expected.shape != actual.shape:
                raise ValueError(
                    f"Tensor {name!r} shape mismatch: {expected.shape} != {actual.shape}"
                )
            if expected.byte_count != actual.byte_count:
                raise ValueError(
                    f"Tensor {name!r} byte-count mismatch: "
                    f"{expected.byte_count} != {actual.byte_count}"
                )

            digest.update(len(name.encode()).to_bytes(8, "big"))
            digest.update(name.encode())
            digest.update(expected.dtype.encode())
            digest.update(json.dumps(expected.shape).encode())
            expected_chunks = _chunks(
                reference_maps[expected.path], expected.start, expected.end
            )
            actual_chunks = _chunks(
                candidate_maps[actual.path], actual.start, actual.end
            )
            compared = 0
            for expected_chunk, actual_chunk in zip(
                expected_chunks, actual_chunks, strict=True
            ):
                if expected_chunk != actual_chunk:
                    first_offset = next(
                        index
                        for index, values in enumerate(
                            zip(expected_chunk, actual_chunk, strict=True)
                        )
                        if values[0] != values[1]
                    )
                    raise ValueError(
                        f"Tensor {name!r} differs at byte {compared + first_offset}"
                    )
                digest.update(expected_chunk)
                compared += len(expected_chunk)
            if compared != expected.byte_count:
                raise AssertionError(f"Incomplete byte comparison for tensor {name!r}")
            parameter_count += _shape_elements(expected.shape)
            tensor_bytes += expected.byte_count

    return {
        "schema_version": 1,
        "contract": "search-r1-safetensors-bitwise-equivalence-v1",
        "reference_dir": str(reference_dir.resolve()),
        "candidate_dir": str(candidate_dir.resolve()),
        "tensor_count": len(reference),
        "parameter_count": parameter_count,
        "tensor_bytes": tensor_bytes,
        "semantic_tensor_sha256": digest.hexdigest(),
        "bitwise_equal": True,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = verify_equivalence(args.reference, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
