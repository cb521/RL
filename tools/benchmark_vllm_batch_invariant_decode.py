# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tune vLLM's fixed batch-invariant BF16 GEMM tile for Search-R1.

The benchmark uses the four backbone linear shapes from Qwen2.5-7B and live
decode row counts 1-10. It compares 16/32/64-row tiles with vLLM's 128-row
default, rejects candidates whose BF16 output is not bit-identical, and also
prices representative prefill rows because Dynamo bakes one fixed tile into
the symbolic-M backbone. The emitted JSON is suitable for the Search-R1
optimization ledger.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from nemo_rl.models.generation.vllm.patches import (
    BATCH_INVARIANT_BLOCK_SIZE_M_ENV,
    _patch_vllm_batch_invariant_block_size,
)


_TILES = (128, 64, 32, 16)
_DECODE_ROWS = tuple(range(1, 11))
# The controlled trace has 10 concurrent requests per generation engine. The
# five turns contain about 1.25K, 6.4K, 3.1K, 4.3K, and 5.7K input tokens per
# engine if each turn coalesces fully; these round-number rows span that live
# range and the configured 8,192-token scheduler ceiling.
_PREFILL_ROWS = (1024, 2048, 4096, 6144, 8192)
_LINEAR_SHAPES = (
    ("qkv_proj", 3584, 4608),
    ("o_proj", 3584, 3584),
    ("gate_up_proj", 3584, 37888),
    ("down_proj", 18944, 3584),
)
_NUM_LAYERS = 28


class _Logger:
    """Minimal logger accepted by the source-patch helper."""

    def info(self, message: str, *args: object) -> None:
        print("INFO " + (message % args if args else message))

    def warning(self, message: str, *args: object) -> None:
        print("WARNING " + (message % args if args else message))


def _time_kernel_ms(fn, *, warmup: int, iterations: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return min(samples)


def _benchmark(
    *,
    warmup: int,
    iterations: int,
    repeats: int,
    prefill_warmup: int,
    prefill_iterations: int,
    prefill_repeats: int,
    decode_forwards: int,
    prefill_forwards: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not _patch_vllm_batch_invariant_block_size(_Logger()):
        raise RuntimeError("vLLM batch-invariant source patch did not apply")

    # Import after installing the source hook so this process executes it.
    from vllm.model_executor.layers.batch_invariant import matmul_persistent

    torch.manual_seed(17)
    device = torch.device("cuda")
    details: list[dict[str, Any]] = []
    exact_by_tile = {tile: True for tile in _TILES}
    rows_by_phase = {"decode": _DECODE_ROWS, "prefill": _PREFILL_ROWS}
    kernel_ms_by_tile_by_row = {
        tile: {row: 0.0 for rows in rows_by_phase.values() for row in rows}
        for tile in _TILES
    }

    for shape_name, k, n in _LINEAR_SHAPES:
        # vLLM passes weight.t(): logical [K, N] with column-major strides.
        weight = torch.randn((n, k), device=device, dtype=torch.bfloat16)
        b = weight.t()
        a_full = torch.randn(
            (max(_PREFILL_ROWS), k), device=device, dtype=torch.bfloat16
        )

        for phase, rows in rows_by_phase.items():
            phase_warmup = warmup if phase == "decode" else prefill_warmup
            phase_iterations = iterations if phase == "decode" else prefill_iterations
            phase_repeats = repeats if phase == "decode" else prefill_repeats

            for m in rows:
                a = a_full[:m]
                os.environ[BATCH_INVARIANT_BLOCK_SIZE_M_ENV] = "128"
                reference = matmul_persistent(a, b)
                torch.cuda.synchronize()

                for tile in _TILES:
                    os.environ[BATCH_INVARIANT_BLOCK_SIZE_M_ENV] = str(tile)
                    candidate = matmul_persistent(a, b)
                    torch.cuda.synchronize()
                    bit_exact = torch.equal(reference, candidate)
                    exact_by_tile[tile] &= bit_exact
                    max_abs_diff = float(
                        (reference.float() - candidate.float()).abs().max().item()
                    )
                    elapsed_ms = _time_kernel_ms(
                        lambda current_a=a, current_b=b: matmul_persistent(
                            current_a, current_b
                        ),
                        warmup=phase_warmup,
                        iterations=phase_iterations,
                        repeats=phase_repeats,
                    )
                    kernel_ms_by_tile_by_row[tile][m] += elapsed_ms
                    details.append(
                        {
                            "phase": phase,
                            "shape": shape_name,
                            "m": m,
                            "k": k,
                            "n": n,
                            "tile": tile,
                            "bit_exact_vs_128": bit_exact,
                            "max_abs_diff": max_abs_diff,
                            "kernel_ms": elapsed_ms,
                        }
                    )

        del a_full, b, weight, reference, candidate
        torch.cuda.empty_cache()

    estimated_forward_ms_by_tile_by_row = {
        tile: {
            row: kernel_ms * _NUM_LAYERS
            for row, kernel_ms in kernel_ms_by_tile_by_row[tile].items()
        }
        for tile in _TILES
    }
    decode_forward_ms = {
        tile: sum(
            estimated_forward_ms_by_tile_by_row[tile][row] for row in _DECODE_ROWS
        )
        / len(_DECODE_ROWS)
        for tile in _TILES
    }
    prefill_forward_ms = {
        tile: sum(
            estimated_forward_ms_by_tile_by_row[tile][row] for row in _PREFILL_ROWS
        )
        / len(_PREFILL_ROWS)
        for tile in _TILES
    }
    workload_ms = {
        tile: decode_forward_ms[tile] * decode_forwards
        + prefill_forward_ms[tile] * prefill_forwards
        for tile in _TILES
    }
    exact_tiles = [tile for tile in _TILES if exact_by_tile[tile]]
    winner = min(exact_tiles, key=workload_ms.__getitem__)
    baseline_ms = workload_ms[128]
    winner_ms = workload_ms[winner]

    return {
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "decode_rows": list(_DECODE_ROWS),
        "prefill_rows": list(_PREFILL_ROWS),
        "tiles": list(_TILES),
        "decode_timing": {
            "warmup": warmup,
            "iterations": iterations,
            "repeats": repeats,
        },
        "prefill_timing": {
            "warmup": prefill_warmup,
            "iterations": prefill_iterations,
            "repeats": prefill_repeats,
        },
        "num_layers": _NUM_LAYERS,
        "workload_weights": {
            "decode_forwards": decode_forwards,
            "prefill_forwards": prefill_forwards,
            "prefill_rows_are_uniformly_weighted": True,
        },
        "exact_by_tile": {str(k): v for k, v in exact_by_tile.items()},
        "estimated_decode_backbone_linear_ms_by_tile": {
            str(k): v for k, v in decode_forward_ms.items()
        },
        "estimated_prefill_backbone_linear_ms_by_tile": {
            str(k): v for k, v in prefill_forward_ms.items()
        },
        "estimated_backbone_linear_ms_by_tile_by_row": {
            str(tile): {str(row): value for row, value in row_values.items()}
            for tile, row_values in estimated_forward_ms_by_tile_by_row.items()
        },
        "estimated_weighted_workload_ms_by_tile": {
            str(k): v for k, v in workload_ms.items()
        },
        "winner": winner,
        "winner_speedup_vs_128": baseline_ms / winner_ms,
        "winner_decode_speedup_vs_128": decode_forward_ms[128]
        / decode_forward_ms[winner],
        "winner_prefill_time_ratio_vs_128": prefill_forward_ms[winner]
        / prefill_forward_ms[128],
        "winner_saves_ms_per_weighted_workload": baseline_ms - winner_ms,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prefill-warmup", type=int, default=2)
    parser.add_argument("--prefill-iterations", type=int, default=5)
    parser.add_argument("--prefill-repeats", type=int, default=2)
    parser.add_argument("--decode-forwards", type=int, default=1085)
    parser.add_argument("--prefill-forwards", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        min(
            args.warmup,
            args.iterations,
            args.repeats,
            args.prefill_warmup,
            args.prefill_iterations,
            args.prefill_repeats,
            args.decode_forwards,
            args.prefill_forwards,
        )
        <= 0
    ):
        parser.error("timing parameters and workload weights must all be positive")

    result = _benchmark(
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
        prefill_warmup=args.prefill_warmup,
        prefill_iterations=args.prefill_iterations,
        prefill_repeats=args.prefill_repeats,
        decode_forwards=args.decode_forwards,
        prefill_forwards=args.prefill_forwards,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    if result["winner"] == 128 or result["winner_speedup_vs_128"] <= 1.0:
        raise SystemExit("No bit-exact fixed tile beat the weighted 128-row baseline")


if __name__ == "__main__":
    main()
