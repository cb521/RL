# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert Nsight Systems worker reports into a compact JSON breakdown."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REPORTS = (
    "cuda_api_sum",
    "cuda_gpu_kern_sum",
    "cuda_gpu_mem_time_sum",
    "cuda_kern_exec_sum",
    "nvtx_sum",
    "nvtx_gpu_proj_sum",
)

_LABEL_KEYS = ("Name", "Operation", "Range", "Kernel Name", "API Name")
_COUNT_KEYS = ("Num Calls", "Instances", "Count", "Range Instances")


def worker_kind(path: Path) -> str:
    """Map a Ray Nsight filename to a stable implementation role."""
    name = path.name.lower()
    if "policy" in name:
        return "policy"
    if "vllm" in name or "generation" in name:
        return "generation"
    return "other"


def _row_label(report: str, row: dict[str, Any]) -> str:
    if report == "cuda_kern_exec_sum":
        api_name = row.get("API Name")
        kernel_name = row.get("Kernel Name")
        if api_name is not None and kernel_name is not None:
            return f"{api_name} -> {kernel_name}"
    for key in _LABEL_KEYS:
        value = row.get(key)
        if value is not None:
            return str(value)
    return "unknown"


def _numeric(row: dict[str, Any], keys: Iterable[str]) -> float:
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return 0.0


def _row_count(row: dict[str, Any]) -> int:
    return int(_numeric(row, _COUNT_KEYS))


def _row_duration_ns(report: str, row: dict[str, Any]) -> float:
    if report == "cuda_kern_exec_sum":
        return _numeric(row, ("KAvg (ns)",)) * _row_count(row)
    return _numeric(row, ("Total Time (ns)", "Total Proj Time (ns)"))


def aggregate_tables(
    tables_by_worker: Iterable[dict[str, list[dict[str, Any]]]],
    *,
    top: int,
) -> dict[str, list[dict[str, Any]]]:
    """Aggregate report-local totals across workers without mixing time domains."""
    totals: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    for tables in tables_by_worker:
        for report, rows in tables.items():
            for row in rows:
                label = _row_label(report, row)
                aggregate = totals[report][label]
                aggregate["total_time_ns"] += _row_duration_ns(report, row)
                aggregate["count"] += _row_count(row)
                if report == "cuda_kern_exec_sum":
                    count = _row_count(row)
                    queue_count = int(_numeric(row, ("QCount",)))
                    aggregate["api_time_ns"] += _numeric(row, ("AAvg (ns)",)) * count
                    aggregate["queue_time_ns"] += (
                        _numeric(row, ("QAvg (ns)",)) * queue_count
                    )

    result: dict[str, list[dict[str, Any]]] = {}
    for report, entries in sorted(totals.items()):
        report_total = sum(entry["total_time_ns"] for entry in entries.values())
        rows = []
        for label, values in entries.items():
            row: dict[str, Any] = {
                "name": label,
                "total_time_ns": values["total_time_ns"],
                "count": int(values["count"]),
                "report_share_pct": (
                    100.0 * values["total_time_ns"] / report_total
                    if report_total
                    else 0.0
                ),
            }
            if report == "cuda_kern_exec_sum":
                row["api_time_ns"] = values["api_time_ns"]
                row["positive_queue_time_ns"] = values["queue_time_ns"]
            rows.append(row)
        rows.sort(key=lambda row: (-row["total_time_ns"], row["name"]))
        result[report] = rows[:top]
    return result


def _run_stats(
    nsys: str,
    input_path: Path,
    temporary_root: Path | None,
) -> dict[str, list[dict[str, Any]]]:
    with tempfile.TemporaryDirectory(
        prefix="nemo-search-r1-nsys-",
        dir=temporary_root,
    ) as temporary_directory:
        temporary_path = Path(temporary_directory)
        output_base = temporary_path / "stats"
        sqlite_path = temporary_path / "report.sqlite"
        command = [
            nsys,
            "stats",
            f"--report={','.join(REPORTS)}",
            f"--format={','.join('json' for _ in REPORTS)}",
            f"--output={','.join(str(output_base) for _ in REPORTS)}",
            f"--sqlite={sqlite_path}",
            "--timeunit=nsec",
            "--force-overwrite=true",
            str(input_path),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(
                f"nsys stats failed for {input_path} with exit code "
                f"{completed.returncode}: {details}"
            )

        tables: dict[str, list[dict[str, Any]]] = {}
        for report in REPORTS:
            report_path = Path(f"{output_base}_{report}.json")
            if not report_path.exists() or report_path.stat().st_size == 0:
                tables[report] = []
                continue
            value = json.loads(report_path.read_text(encoding="utf-8"))
            if not isinstance(value, list) or not all(
                isinstance(row, dict) for row in value
            ):
                raise ValueError(f"Unexpected {report} output in {report_path}")
            tables[report] = value
        return tables


def analyze_reports(
    inputs: list[Path],
    *,
    nsys: str,
    top: int,
) -> dict[str, Any]:
    """Run fixed Nsight reports and return bounded per-worker and aggregate data."""
    if top < 1:
        raise ValueError("top must be positive")
    if len(set(inputs)) != len(inputs):
        raise ValueError("Duplicate Nsight input paths are not allowed")
    for input_path in inputs:
        if not input_path.is_file() or input_path.stat().st_size == 0:
            raise ValueError(f"Nsight report is missing or empty: {input_path}")

    resolved_nsys = shutil.which(nsys) if os.sep not in nsys else nsys
    if not resolved_nsys or not Path(resolved_nsys).is_file():
        raise FileNotFoundError(f"Nsight Systems executable was not found: {nsys}")
    version = subprocess.run(
        [resolved_nsys, "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    version_text = next(
        (
            line.strip()
            for line in (version.stdout + "\n" + version.stderr).splitlines()
            if "NVIDIA Nsight Systems version" in line
        ),
        "unknown",
    )

    temporary_root_value = os.environ.get("NSYS_TMPDIR")
    temporary_root = Path(temporary_root_value) if temporary_root_value else None
    if temporary_root is not None:
        temporary_root.mkdir(parents=True, exist_ok=True)

    complete_tables = []
    workers = []
    kind_counts: dict[str, int] = defaultdict(int)
    for input_path in inputs:
        tables = _run_stats(resolved_nsys, input_path, temporary_root)
        complete_tables.append(tables)
        kind = worker_kind(input_path)
        kind_counts[kind] += 1
        workers.append(
            {
                "input": str(input_path),
                "filename": input_path.name,
                "size_bytes": input_path.stat().st_size,
                "worker_kind": kind,
                "empty_reports": [
                    report for report, rows in tables.items() if not rows
                ],
                "reports": {
                    report: sorted(
                        rows,
                        key=lambda row: -_row_duration_ns(report, row),
                    )[:top]
                    for report, rows in tables.items()
                },
            }
        )

    return {
        "schema_version": 1,
        "nsys_version": version_text,
        "report_count": len(inputs),
        "worker_kinds": dict(sorted(kind_counts.items())),
        "reports_requested": list(REPORTS),
        "top_rows_per_report": top,
        "workers": workers,
        "aggregate": aggregate_tables(complete_tables, top=top),
        "interpretation": (
            "Percentages are local to each Nsight report. CUDA, NVTX, and "
            "worker totals overlap and must not be added to end-to-end wall time."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--nsys", default="nsys")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = analyze_reports(args.input, nsys=args.nsys, top=args.top)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
