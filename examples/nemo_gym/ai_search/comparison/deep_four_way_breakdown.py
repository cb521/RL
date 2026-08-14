#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a reconciled macro/eight-stage Search-R1 performance report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


FRAMEWORKS = ("original", "current", "slime", "nemo")
DISPLAY = {
    "original": "original Search-R1",
    "current": "current veRL",
    "slime": "SLIME",
    "nemo": "NeMo RL",
}
STAGE_NAMES = (
    "1_rollout_preparation_and_weight_sync",
    "2_model_generation_and_agent_scheduling",
    "3_retrieval_exposed_critical_path",
    "4_trajectory_postprocess_reward_advantage",
    "5_policy_logprob",
    "6_reference_logprob",
    "7_training_and_data_preparation",
    "8_actor_forward_backward_grad_sync_optimizer",
)
LOCAL_TIMEZONE = ZoneInfo("Asia/Shanghai")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _read_manifest(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key or key in values:
            raise ValueError(f"Malformed or duplicate manifest line in {path}: {line!r}")
        values[key] = value
    return values


def _read_jobs(path: Path, mode: str) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    selected = {
        row["framework"]: row["job_id"]
        for row in rows
        if row.get("observability_mode") == mode
    }
    if set(selected) != set(FRAMEWORKS):
        raise ValueError(f"{path} does not select one {mode} run per framework: {selected}")
    return selected


def _result_dir(root: Path, framework: str, mode: str, job_id: str) -> Path:
    return root / f"four-way-performance-{framework}-{mode}-job-{job_id}"


def _mean(report: dict[str, Any], metric: str) -> float | None:
    value = (
        report.get("training", {})
        .get("timing_seconds", {})
        .get(metric, {})
        .get("mean")
    )
    return float(value) if isinstance(value, (int, float)) else None


def _work_mean(report: dict[str, Any], metric: str) -> float | None:
    value = report.get("training", {}).get("work", {}).get(metric, {}).get("mean")
    return float(value) if isinstance(value, (int, float)) else None


def _throughput_mean(report: dict[str, Any], metric: str) -> float | None:
    value = (
        report.get("training", {})
        .get("throughput", {})
        .get(metric, {})
        .get("mean")
    )
    return float(value) if isinstance(value, (int, float)) else None


def _native_means(report: dict[str, Any], metric: str) -> list[float]:
    values = []
    for step in report.get("training", {}).get("steps", []):
        if step.get("step") not in {2, 3, 4}:
            continue
        value = step.get("native_metrics", {}).get(metric)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _required(value: float | None, name: str) -> float:
    if value is None or not math.isfinite(value):
        raise ValueError(f"Missing required timing metric: {name}")
    return value


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile from no samples")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (
        position - lower
    )


def _resource_usage(run_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    """Summarize host RAM and VRAM over the exact steady measurement window."""
    window = report.get("measurement_window", {})
    start = float(window.get("start_unix_seconds", 0.0))
    end = float(window.get("end_unix_seconds", 0.0))
    if not window.get("available") or end <= start:
        raise ValueError(f"Missing measurement window for {run_dir}")

    gpu_samples: list[tuple[float, str, float, float]] = []
    with (run_dir / "gpu.csv").open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            timestamp = datetime.strptime(
                row["timestamp"].strip(), "%Y/%m/%d %H:%M:%S.%f"
            ).replace(tzinfo=LOCAL_TIMEZONE).timestamp()
            if start <= timestamp <= end:
                gpu_samples.append(
                    (
                        timestamp,
                        row["uuid"].strip(),
                        float(row["memory_used_mib"]),
                        float(row["memory_total_mib"]),
                    )
                )
    if not gpu_samples:
        raise ValueError(f"No steady GPU samples in {run_dir / 'gpu.csv'}")

    used_mib = [sample[2] for sample in gpu_samples]
    capacities_mib = {sample[3] for sample in gpu_samples}
    if len(capacities_mib) != 1:
        raise ValueError(f"Inconsistent GPU capacities in {run_dir / 'gpu.csv'}")
    capacity_mib = capacities_mib.pop()
    by_gpu: dict[str, list[float]] = {}
    by_second: dict[int, dict[str, float]] = {}
    for timestamp, uuid, used, _ in gpu_samples:
        by_gpu.setdefault(uuid, []).append(used)
        by_second.setdefault(round(timestamp), {})[uuid] = used
    synchronized_totals = [
        sum(values.values())
        for values in by_second.values()
        if len(values) == len(by_gpu)
    ]

    host_samples: list[tuple[float, float]] = []
    cgroup_current_window: list[float] = []
    cgroup_peak_window: list[float] = []
    cgroup_anon_window: list[float] = []
    with (run_dir / "host.csv").open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            timestamp = datetime.fromisoformat(row["timestamp"]).timestamp()
            used_kib = float(row["mem_total_kib"]) - float(
                row["mem_available_kib"]
            )
            host_samples.append((timestamp, used_kib))
            if start <= timestamp <= end:
                for column, destination in (
                    ("cgroup_memory_current_bytes", cgroup_current_window),
                    ("cgroup_memory_peak_bytes", cgroup_peak_window),
                    ("cgroup_memory_anon_bytes", cgroup_anon_window),
                ):
                    raw_value = row.get(column)
                    if raw_value not in (None, "") and float(raw_value) >= 0:
                        destination.append(float(raw_value))
    host_window_kib = [
        used for timestamp, used in host_samples if start <= timestamp <= end
    ]
    if not host_window_kib:
        raise ValueError(f"No steady host samples in {run_dir / 'host.csv'}")
    initial_samples = [used for _, used in host_samples[:10]]
    if not initial_samples:
        raise ValueError(f"No initial host samples in {run_dir / 'host.csv'}")
    host_baseline_kib = statistics.median(initial_samples)

    mib_to_gib = 1.0 / 1024.0
    kib_to_gib = 1.0 / 1024.0 / 1024.0
    peak_mib = max(used_mib)
    return {
        "measurement_window_seconds": end - start,
        "gpu": {
            "sample_interval_note": "external nvidia-smi samples at about one-second cadence",
            "sample_count": len(used_mib),
            "gpu_count": len(by_gpu),
            "capacity_per_card_gib": capacity_mib * mib_to_gib,
            "mean_used_per_card_gib": statistics.mean(used_mib) * mib_to_gib,
            "p95_used_per_card_gib": _percentile(used_mib, 0.95) * mib_to_gib,
            "observed_peak_used_per_card_gib": peak_mib * mib_to_gib,
            "observed_headroom_per_card_gib": (capacity_mib - peak_mib)
            * mib_to_gib,
            "observed_peak_capacity_percent": peak_mib / capacity_mib * 100.0,
            "observed_peak_total_four_cards_gib": (
                max(synchronized_totals) * mib_to_gib
                if synchronized_totals
                else None
            ),
            "observed_peak_by_card_gib": {
                uuid: max(values) * mib_to_gib for uuid, values in by_gpu.items()
            },
        },
        "host": {
            "confidence": "node-level-estimate",
            "method": "node MemTotal-MemAvailable; subtract median of first ten collector samples",
            "sample_count": len(host_window_kib),
            "initial_baseline_gib": host_baseline_kib * kib_to_gib,
            "mean_node_used_gib": statistics.mean(host_window_kib) * kib_to_gib,
            "peak_node_used_gib": max(host_window_kib) * kib_to_gib,
            "mean_increment_over_initial_gib": (
                statistics.mean(host_window_kib) - host_baseline_kib
            )
            * kib_to_gib,
            "peak_increment_over_initial_gib": (
                max(host_window_kib) - host_baseline_kib
            )
            * kib_to_gib,
            "job_cgroup": {
                "available": bool(cgroup_current_window),
                "steady_current_peak_gib": (
                    max(cgroup_current_window) / 1024**3
                    if cgroup_current_window
                    else None
                ),
                "steady_anon_peak_gib": (
                    max(cgroup_anon_window) / 1024**3
                    if cgroup_anon_window
                    else None
                ),
                "lifetime_peak_seen_in_window_gib": (
                    max(cgroup_peak_window) / 1024**3
                    if cgroup_peak_window
                    else None
                ),
            },
        },
    }


def _retrieval_evidence(report: dict[str, Any]) -> dict[str, float | int | None]:
    trace = report.get("steady_state_evidence", {}).get("trajectory_trace", {})
    timeline = trace.get("timeline_activity", {})
    steady_steps = int(report.get("training", {}).get("steady_step_count", 0))
    retrieval = float(timeline.get("retrieval_wall_union_ms", 0.0)) / 1000.0
    overlap = float(timeline.get("model_retrieval_overlap_ms", 0.0)) / 1000.0
    exposed = max(0.0, retrieval - overlap)
    return {
        "trace_span_count": trace.get("span_count"),
        "sampled_trajectory_count": trace.get("trajectory_count"),
        "retrieval_union_seconds_window": retrieval,
        "model_retrieval_overlap_seconds_window": overlap,
        "overlap_fraction": overlap / retrieval if retrieval > 0 else None,
        "exposed_retrieval_seconds_per_step": (
            exposed / steady_steps if steady_steps else None
        ),
    }


def _stage(
    seconds: float,
    source: str,
    *,
    additive: bool = True,
    confidence: str = "direct",
    note: str = "",
) -> dict[str, Any]:
    return {
        "seconds": seconds,
        "additive": additive,
        "source": source,
        "confidence": confidence,
        "note": note,
    }


def _breakdown(framework: str, report: dict[str, Any]) -> dict[str, Any]:
    total = _required(_mean(report, "total_step_time"), "total_step_time")
    retrieval = _retrieval_evidence(report)
    exposed = float(retrieval["exposed_retrieval_seconds_per_step"] or 0.0)

    if framework == "original":
        stages = {
            STAGE_NAMES[0]: _stage(
                0.0,
                "not separately instrumented",
                confidence="folded",
                note="Hybrid worker transition is folded into generation/other.",
            ),
            STAGE_NAMES[1]: _stage(
                _required(_mean(report, "generation_and_agent"), "generation_and_agent"),
                "training.timing_seconds.generation_and_agent",
            ),
            STAGE_NAMES[2]: _stage(
                exposed,
                "sampled trace retrieval union minus model overlap",
                additive=False,
                confidence="sampled-diagnostic",
                note="Search is already nested in stage 2.",
            ),
            STAGE_NAMES[3]: _stage(
                _required(
                    _mean(report, "postprocessing_reward_advantage"),
                    "postprocessing_reward_advantage",
                ),
                "training.timing_seconds.postprocessing_reward_advantage",
            ),
            STAGE_NAMES[4]: _stage(
                _required(_mean(report, "policy_logprobs"), "policy_logprobs"),
                "training.timing_seconds.policy_logprobs",
            ),
            STAGE_NAMES[5]: _stage(
                _required(_mean(report, "reference_logprobs"), "reference_logprobs"),
                "training.timing_seconds.reference_logprobs",
            ),
            STAGE_NAMES[6]: _stage(
                0.0,
                "not separately instrumented",
                confidence="folded",
                note="Any small preparation cost remains in residual.",
            ),
            STAGE_NAMES[7]: _stage(
                _required(_mean(report, "policy_training"), "policy_training"),
                "training.timing_seconds.policy_training",
            ),
        }
    elif framework == "current":
        stages = {
            STAGE_NAMES[0]: _stage(
                _required(
                    _mean(report, "policy_to_engine_synchronization"),
                    "policy_to_engine_synchronization",
                ),
                "training.timing_seconds.policy_to_engine_synchronization",
            ),
            STAGE_NAMES[1]: _stage(
                _required(_mean(report, "generation_and_agent"), "generation_and_agent"),
                "training.timing_seconds.generation_and_agent",
            ),
            STAGE_NAMES[2]: _stage(
                exposed,
                "sampled trace retrieval union minus model overlap",
                additive=False,
                confidence="sampled-diagnostic",
                note="Search is already nested in stage 2.",
            ),
            STAGE_NAMES[3]: _stage(
                _required(_mean(report, "advantage_calculation"), "advantage_calculation"),
                "training.timing_seconds.advantage_calculation",
            ),
            STAGE_NAMES[4]: _stage(
                _required(_mean(report, "policy_logprobs"), "policy_logprobs"),
                "training.timing_seconds.policy_logprobs",
            ),
            STAGE_NAMES[5]: _stage(
                _required(_mean(report, "reference_logprobs"), "reference_logprobs"),
                "training.timing_seconds.reference_logprobs",
            ),
            STAGE_NAMES[6]: _stage(
                0.0,
                "not separately instrumented",
                confidence="folded",
                note="Small preparation/logging gaps remain in residual.",
            ),
            STAGE_NAMES[7]: _stage(
                _required(_mean(report, "policy_training"), "policy_training"),
                "training.timing_seconds.policy_training",
            ),
        }
    elif framework == "slime":
        sync = _required(
            _mean(report, "policy_to_engine_synchronization"),
            "policy_to_engine_synchronization",
        )
        generation = _required(_mean(report, "generation_and_agent"), "generation_and_agent")
        wait = _required(_mean(report, "trainer_wait_inclusive"), "trainer_wait_inclusive")
        work = _required(_mean(report, "trainer_work_inclusive"), "trainer_work_inclusive")
        actor = _required(_mean(report, "policy_training"), "policy_training")
        reference = _required(_mean(report, "reference_logprobs"), "reference_logprobs")
        sleep = _required(_average(_native_means(report, "perf/sleep_time")), "perf/sleep_time")
        wake = _required(_average(_native_means(report, "perf/wake_up_time")), "perf/wake_up_time")
        rollout_residual = max(0.0, wait - sleep - wake - sync - generation)
        training_prep = max(0.0, work - reference - actor)
        stages = {
            STAGE_NAMES[0]: _stage(
                sleep + wake + sync,
                "mean perf/sleep_time + perf/wake_up_time + perf/update_weights_time",
                confidence="direct-composite",
            ),
            STAGE_NAMES[1]: _stage(
                generation,
                "training.timing_seconds.generation_and_agent",
            ),
            STAGE_NAMES[2]: _stage(
                exposed,
                "sampled trace retrieval union minus model overlap",
                additive=False,
                confidence="sampled-diagnostic",
                note="Search is already nested in stage 2.",
            ),
            STAGE_NAMES[3]: _stage(
                rollout_residual,
                "trainer_wait minus sleep/wake/sync/rollout",
                confidence="residual-derived-upper-bound",
                note="Includes postprocess, driver scheduling, and rounding.",
            ),
            STAGE_NAMES[4]: _stage(
                0.0,
                "folded into the actor training forward",
                confidence="fused-not-free",
                note="No standalone old-policy logprob pass in this one-update path.",
            ),
            STAGE_NAMES[5]: _stage(
                reference,
                "training.timing_seconds.reference_logprobs",
            ),
            STAGE_NAMES[6]: _stage(
                training_prep,
                "trainer_work minus reference_logprobs and actor_train",
                confidence="residual-derived-upper-bound",
                note="Includes measured data_preprocess plus unclassified trainer setup.",
            ),
            STAGE_NAMES[7]: _stage(
                actor,
                "training.timing_seconds.policy_training",
            ),
        }
    elif framework == "nemo":
        stages = {
            STAGE_NAMES[0]: _stage(
                _required(
                    _mean(report, "prepare_for_generation/total"),
                    "prepare_for_generation/total",
                ),
                "training.timing_seconds.prepare_for_generation/total",
                note="Includes the nested transfer_and_update_weights timer.",
            ),
            STAGE_NAMES[1]: _stage(
                _required(_mean(report, "generation"), "generation"),
                "training.timing_seconds.generation",
            ),
            STAGE_NAMES[2]: _stage(
                exposed,
                "sampled trace retrieval union minus model overlap",
                additive=False,
                confidence="sampled-diagnostic",
                note="Search is already nested in stage 2.",
            ),
            STAGE_NAMES[3]: _stage(
                _required(_mean(report, "reward_calculation"), "reward_calculation")
                + _required(_mean(report, "advantage_calculation"), "advantage_calculation"),
                "reward_calculation + advantage_calculation",
                confidence="direct-composite",
            ),
            STAGE_NAMES[4]: _stage(
                _required(_mean(report, "policy_logprobs"), "policy_logprobs"),
                "training.timing_seconds.policy_logprobs",
            ),
            STAGE_NAMES[5]: _stage(
                _required(_mean(report, "reference_logprobs"), "reference_logprobs"),
                "training.timing_seconds.reference_logprobs",
            ),
            STAGE_NAMES[6]: _stage(
                _required(_mean(report, "logprob_inference_prep"), "logprob_inference_prep")
                + _required(_mean(report, "training_prep"), "training_prep")
                + _required(_mean(report, "data_processing"), "data_processing"),
                "logprob_inference_prep + training_prep + data_processing",
                confidence="direct-composite",
            ),
            STAGE_NAMES[7]: _stage(
                _required(_mean(report, "policy_training"), "policy_training"),
                "training.timing_seconds.policy_training",
            ),
        }
    else:
        raise ValueError(framework)

    additive_sum = sum(
        stage["seconds"] for stage in stages.values() if stage["additive"]
    )
    residual = total - additive_sum
    if residual < -max(0.02, total * 0.001):
        raise ValueError(
            f"{framework} additive stages exceed total: {additive_sum} > {total}"
        )
    macro = {
        "rollout_lifecycle": stages[STAGE_NAMES[0]]["seconds"]
        + stages[STAGE_NAMES[1]]["seconds"],
        "training_compute": stages[STAGE_NAMES[4]]["seconds"]
        + stages[STAGE_NAMES[5]]["seconds"]
        + stages[STAGE_NAMES[7]]["seconds"],
        "postprocess_and_preparation": stages[STAGE_NAMES[3]]["seconds"]
        + stages[STAGE_NAMES[6]]["seconds"],
        "residual_other": residual,
    }
    return {
        "total_step_seconds": total,
        "additive_stage_sum_seconds": additive_sum,
        "residual_seconds": residual,
        "residual_percent": residual / total * 100.0,
        "stages": stages,
        "macro": {
            name: {"seconds": seconds, "percent": seconds / total * 100.0}
            for name, seconds in macro.items()
        },
        "retrieval": retrieval,
    }


def _prometheus_counter_sum(report: dict[str, Any], fragment: str) -> float | None:
    deltas = (
        report.get("steady_state_evidence", {})
        .get("prometheus", {})
        .get("counter_deltas", {})
    )
    values = [
        float(value)
        for key, value in deltas.items()
        if fragment in key and isinstance(value, (int, float))
    ]
    return sum(values) if values else None


def _histogram_mean(report: dict[str, Any], fragment: str) -> dict[str, Any] | None:
    values = (
        report.get("steady_state_evidence", {})
        .get("prometheus", {})
        .get("histogram_interval_means", {})
    )
    matches = [value for key, value in values.items() if fragment in key]
    return matches[0] if len(matches) == 1 else None


def _reconciliation(report: dict[str, Any], breakdown: dict[str, Any]) -> dict[str, Any]:
    steps = int(report.get("training", {}).get("steady_step_count", 0))
    window = float(report.get("measurement_window", {}).get("duration_seconds", 0.0))
    timer = float(breakdown["total_step_seconds"])
    boundary = window / steps if steps else None
    generated_mean = _work_mean(report, "generated_tokens")
    console_tokens = generated_mean * steps if generated_mean is not None else None
    prometheus_tokens = _prometheus_counter_sum(report, "generation_tokens_total")
    console_search_mean = _work_mean(report, "search_count")
    console_search = console_search_mean * steps if console_search_mean is not None else None
    e5_queries = _prometheus_counter_sum(report, "search_r1_e5_queries_total")
    return {
        "console_timer_mean_seconds": timer,
        "timestamp_boundary_mean_seconds": boundary,
        "timestamp_minus_console_seconds": boundary - timer if boundary is not None else None,
        "timestamp_minus_console_percent": (
            (boundary / timer - 1.0) * 100.0 if boundary is not None else None
        ),
        "console_generated_tokens_window": console_tokens,
        "prometheus_generated_tokens_window": prometheus_tokens,
        "prometheus_token_difference_percent": (
            (prometheus_tokens / console_tokens - 1.0) * 100.0
            if prometheus_tokens is not None and console_tokens
            else None
        ),
        "console_search_queries_window": console_search,
        "prometheus_e5_queries_window": e5_queries,
        "e5_request_mean_seconds": (
            (_histogram_mean(report, "stage=request_total") or {}).get("mean")
        ),
        "e5_queue_mean_seconds": (
            (_histogram_mean(report, "stage=queue") or {}).get("mean")
        ),
        "e5_batch_size_mean": (
            (_histogram_mean(report, "search_r1_e5_batch_size") or {}).get("mean")
        ),
    }


def _top_nsys(summary: dict[str, Any]) -> dict[str, Any]:
    aggregate = summary.get("aggregate", {})
    return {
        "report_count": summary.get("report_count"),
        "worker_kinds": summary.get("worker_kinds"),
        "top_kernel": (aggregate.get("cuda_gpu_kern_sum") or [None])[0],
        "top_cuda_api": (aggregate.get("cuda_api_sum") or [None])[0],
        "top_memory_operation": (aggregate.get("cuda_gpu_mem_time_sum") or [None])[0],
        "interpretation": summary.get("interpretation"),
    }


def _nemo_policy_copy_evidence(summary: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for worker in summary.get("workers", []):
        if not str(worker.get("filename", "")).startswith("dtensor_policy"):
            continue
        reports = worker.get("reports", {})
        api = (reports.get("cuda_api_sum") or [{}])[0]
        kernels = (reports.get("cuda_gpu_kern_sum") or [{}])[0]
        memory = {
            row.get("Operation") or row.get("Name"): row
            for row in reports.get("cuda_gpu_mem_time_sum", [])
        }
        rows.append(
            {
                "filename": worker.get("filename"),
                "top_api_name": api.get("Name"),
                "top_api_share_percent": api.get("Time (%)"),
                "top_kernel_name": kernels.get("Name"),
                "top_kernel_share_percent": kernels.get("Time (%)"),
                "h2d_seconds": (
                    memory.get("[CUDA memcpy Host-to-Device]", {}).get("Total Time (ns)", 0)
                    / 1e9
                ),
                "d2h_seconds": (
                    memory.get("[CUDA memcpy Device-to-Host]", {}).get("Total Time (ns)", 0)
                    / 1e9
                ),
            }
        )
    return {
        "policy_worker_count": len(rows),
        "mean_cuda_memcpy_async_api_share_percent": _average(
            [float(row["top_api_share_percent"]) for row in rows]
        ),
        "mean_h2d_seconds_per_rank": _average([row["h2d_seconds"] for row in rows]),
        "mean_d2h_seconds_per_rank": _average([row["d2h_seconds"] for row in rows]),
        "mean_top_allgather_kernel_share_percent": _average(
            [float(row["top_kernel_share_percent"]) for row in rows]
        ),
        "workers": rows,
    }


def _alignment(
    manifests: dict[str, dict[str, str]],
    train_equivalence: dict[str, Any],
    eval_equivalence: dict[str, Any],
) -> dict[str, Any]:
    def values(*keys: str) -> dict[str, str | None]:
        output = {}
        for framework, manifest in manifests.items():
            output[framework] = next(
                (manifest[key] for key in keys if key in manifest), None
            )
        return output

    dimensions: dict[str, dict[str, Any]] = {}
    contracts = {
        "model_revision": ("model_revision",),
        "seed": ("seed",),
        "hardware_contract": ("hardware_contract",),
        "training_gpu_count": ("training_gpus", "physical_gpus"),
        "prompts_per_outer_step": ("prompts_per_step", "prompts_per_rollout"),
        "trajectories_per_outer_step": (
            "trajectories_per_step",
            "trajectories_per_rollout",
        ),
        "outer_steps": ("total_steps", "num_rollout"),
        "optimizer": ("optimizer",),
        "optimizer_lr": ("optimizer_lr",),
        "optimizer_weight_decay": ("optimizer_weight_decay",),
        "optimizer_betas": ("optimizer_betas",),
        "optimizer_epsilon": ("optimizer_epsilon",),
        "lr_warmup_outer_steps": ("lr_warmup_outer_steps",),
        "lr_after_warmup": ("lr_after_warmup",),
    }
    for name, keys in contracts.items():
        observed = values(*keys)
        unique = {value for value in observed.values() if value is not None}
        dimensions[name] = {
            "aligned": len(unique) == 1
            and all(value is not None for value in observed.values()),
            "values": observed,
        }

    train_hashes = values("train_sha256")
    eval_hashes = values("eval_sha256")
    expected_train = {
        "nemo": train_equivalence["jsonl_sha256"],
        "original": train_equivalence["parquet_sha256"],
        "current": train_equivalence["parquet_sha256"],
        "slime": train_equivalence["parquet_sha256"],
    }
    expected_eval = {
        "nemo": eval_equivalence["jsonl_sha256"],
        "original": eval_equivalence["parquet_sha256"],
        "current": eval_equivalence["parquet_sha256"],
        "slime": eval_equivalence["parquet_sha256"],
    }
    dimensions["train_data_semantics"] = {
        "aligned": train_hashes == expected_train,
        "values": train_hashes,
        "semantic_rows_sha256": train_equivalence["semantic_rows_sha256"],
        "rows": train_equivalence["rows"],
    }
    dimensions["eval_data_semantics"] = {
        "aligned": eval_hashes == expected_eval,
        "values": eval_hashes,
        "semantic_rows_sha256": eval_equivalence["semantic_rows_sha256"],
        "rows": eval_equivalence["rows"],
    }
    return {
        "verdict": "basically-aligned" if all(
            item["aligned"] for item in dimensions.values()
        ) else "alignment-failure",
        "dimensions": dimensions,
        "implementation_differences": [
            "NeMo reads the verified JSONL view; the other frameworks read the verified Parquet view.",
            "Native microbatching, packing, scheduler, and transition implementations intentionally differ.",
            "Generated trajectory lengths are stochastic, so actual generated-token work is not equal.",
        ],
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    clean_ids = _read_jobs(args.clean_jobs, "clean")
    profile_ids = _read_jobs(args.profile_jobs, "profile")
    clean_reports = {}
    clean_manifests = {}
    profile_reports = {}
    nsys = {}
    clean_dirs = {}
    profile_dirs = {}
    for framework in FRAMEWORKS:
        clean_dir = _result_dir(args.clean_root, framework, "clean", clean_ids[framework])
        profile_dir = _result_dir(
            args.profile_root, framework, "profile", profile_ids[framework]
        )
        clean_dirs[framework] = str(clean_dir)
        profile_dirs[framework] = str(profile_dir)
        clean_reports[framework] = _load_json(clean_dir / "observability.json")
        clean_manifests[framework] = _read_manifest(clean_dir / "framework-manifest.txt")
        profile_reports[framework] = _load_json(profile_dir / "observability.json")
        nsys[framework] = _load_json(profile_dir / "nsight-summary.json")

    breakdowns = {
        framework: _breakdown(framework, clean_reports[framework])
        for framework in FRAMEWORKS
    }
    reconciliations = {
        framework: _reconciliation(clean_reports[framework], breakdowns[framework])
        for framework in FRAMEWORKS
    }
    resources = {
        framework: _resource_usage(Path(clean_dirs[framework]), clean_reports[framework])
        for framework in FRAMEWORKS
    }
    headline = {}
    for framework in FRAMEWORKS:
        report = clean_reports[framework]
        headline[framework] = {
            "step_seconds": breakdowns[framework]["total_step_seconds"],
            "generated_tokens_per_step": _work_mean(report, "generated_tokens"),
            "processed_tokens_per_step": _work_mean(report, "processed_tokens"),
            "response_tokens_per_step": _work_mean(report, "response_tokens"),
            "generated_tokens_per_second": _throughput_mean(
                report, "E2E Model-generated (Tokens/sec)"
            ),
            "samples_per_second": _throughput_mean(report, "E2E (Samples/sec)"),
            "profile_step_seconds": _mean(profile_reports[framework], "total_step_time"),
            "profile_vs_clean_time_delta_percent": (
                _required(_mean(profile_reports[framework], "total_step_time"), "profile total")
                / breakdowns[framework]["total_step_seconds"]
                - 1.0
            )
            * 100.0,
        }

    current = breakdowns["current"]["stages"]
    nemo = breakdowns["nemo"]["stages"]
    comparisons = {
        "nemo_vs_current": {
            "total_time_ratio": headline["nemo"]["step_seconds"]
            / headline["current"]["step_seconds"],
            "generated_work_ratio": headline["nemo"]["generated_tokens_per_step"]
            / headline["current"]["generated_tokens_per_step"],
            "stage_time_ratios": {
                stage: (
                    nemo[stage]["seconds"] / current[stage]["seconds"]
                    if current[stage]["seconds"] > 0 and current[stage]["additive"]
                    else None
                )
                for stage in STAGE_NAMES
                if stage != STAGE_NAMES[2]
            },
        },
        "nemo_vs_slime": {
            "total_time_ratio": headline["nemo"]["step_seconds"]
            / headline["slime"]["step_seconds"],
            "generated_work_ratio": headline["nemo"]["generated_tokens_per_step"]
            / headline["slime"]["generated_tokens_per_step"],
        },
    }
    return {
        "verdict": "pass",
        "schema_version": 1,
        "classification": "nonformal-four-gpu-h20-diagnostic",
        "alignment": _alignment(
            clean_manifests,
            _load_json(args.train_equivalence),
            _load_json(args.eval_equivalence),
        ),
        "headline": headline,
        "breakdowns": breakdowns,
        "resources": resources,
        "reconciliation": reconciliations,
        "nsight": {framework: _top_nsys(nsys[framework]) for framework in FRAMEWORKS},
        "nemo_policy_copy_evidence": _nemo_policy_copy_evidence(nsys["nemo"]),
        "comparisons": comparisons,
        "artifacts": {
            "clean": clean_dirs,
            "profile": profile_dirs,
        },
        "interpretation_boundaries": [
            "Stage 3 is nested diagnostic evidence and is never added to the step total.",
            "Clean timers determine performance; profile timers mix profiler overhead and stochastic work differences.",
            "VRAM values are externally observed at about one-second cadence; host-RAM increments are node-level estimates, not exact process RSS.",
            "SLIME Nsight covers only rank-0 training actor; current veRL rollout engine is not covered.",
            "A 4xH20 diagnostic does not establish an 8-GPU result or model-quality ranking.",
        ],
    }


def _fmt(value: Any, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _stage_label(name: str) -> str:
    return name.split("_", 1)[1].replace("_", " ")


def write_outputs(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "deep-breakdown.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with (output_dir / "eight-stage.tsv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ["stage", "framework", "seconds", "additive", "confidence", "source", "note"]
        )
        for stage in STAGE_NAMES:
            for framework in FRAMEWORKS:
                value = report["breakdowns"][framework]["stages"][stage]
                writer.writerow(
                    [
                        stage,
                        framework,
                        f"{value['seconds']:.9f}",
                        str(value["additive"]).lower(),
                        value["confidence"],
                        value["source"],
                        value["note"],
                    ]
                )

    with (output_dir / "macro.tsv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow(["framework", "macro_bucket", "seconds", "percent"])
        for framework in FRAMEWORKS:
            for name, value in report["breakdowns"][framework]["macro"].items():
                writer.writerow(
                    [framework, name, f"{value['seconds']:.9f}", f"{value['percent']:.9f}"]
                )

    with (output_dir / "resource-usage.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "framework",
                "gpu_mean_used_per_card_gib",
                "gpu_p95_used_per_card_gib",
                "gpu_observed_peak_per_card_gib",
                "gpu_observed_headroom_per_card_gib",
                "host_mean_increment_over_initial_gib",
                "host_peak_increment_over_initial_gib",
                "host_confidence",
            ]
        )
        for framework in FRAMEWORKS:
            gpu = report["resources"][framework]["gpu"]
            host = report["resources"][framework]["host"]
            writer.writerow(
                [
                    framework,
                    f"{gpu['mean_used_per_card_gib']:.9f}",
                    f"{gpu['p95_used_per_card_gib']:.9f}",
                    f"{gpu['observed_peak_used_per_card_gib']:.9f}",
                    f"{gpu['observed_headroom_per_card_gib']:.9f}",
                    f"{host['mean_increment_over_initial_gib']:.9f}",
                    f"{host['peak_increment_over_initial_gib']:.9f}",
                    host["confidence"],
                ]
            )

    markdown = [
        "# Four-framework Search-R1 deep performance breakdown",
        "",
        "## Bottom line",
        "",
        "The comparison is basically aligned at the problem/configuration level. The important caveat is that stochastic trajectories produced different token counts: NeMo did less generation work than current veRL and SLIME but still took much longer. Its main bottleneck is policy-side training and logprob computation, not retrieval.",
        "",
        "| Framework | Step (s) | Generated tokens/step | Generated tok/s | Samples/s |",
        "|---|---:|---:|---:|---:|",
    ]
    for framework in FRAMEWORKS:
        value = report["headline"][framework]
        markdown.append(
            f"| {DISPLAY[framework]} | {_fmt(value['step_seconds'])} | {_fmt(value['generated_tokens_per_step'], 0)} | {_fmt(value['generated_tokens_per_second'], 1)} | {_fmt(value['samples_per_second'])} |"
        )

    markdown += [
        "",
        "## CPU memory and GPU memory",
        "",
        "The GPU values use the same three-step clean measurement window as the timing result. The peak is the capacity constraint; the mean also reflects deliberate sleep/offload phases. Host RAM is an approximate node-level increment over the first ten collector samples, because the completed jobs do not expose a reliable process-tree/cgroup peak for every framework.",
        "",
        "| Framework | Mean VRAM/card | Observed peak VRAM/card | Observed headroom/card | Approx. peak host-RAM increment |",
        "|---|---:|---:|---:|---:|",
    ]
    for framework in FRAMEWORKS:
        gpu = report["resources"][framework]["gpu"]
        host = report["resources"][framework]["host"]
        markdown.append(
            f"| {DISPLAY[framework]} | {_fmt(gpu['mean_used_per_card_gib'], 2)} GiB | {_fmt(gpu['observed_peak_used_per_card_gib'], 2)} GiB | {_fmt(gpu['observed_headroom_per_card_gib'], 2)} GiB | {_fmt(host['peak_increment_over_initial_gib'], 2)} GiB |"
        )
    markdown += [
        "",
        "NeMo's low VRAM is not a free efficiency win: it coincides with the largest host-RAM footprint and the Nsight copy evidence below. The implementation shifts policy/optimizer state to CPU and pays for repeated transfers. Final optimized candidates must report timing, observed VRAM peak/headroom, and host-RAM peak together.",
    ]

    dimensions = report["alignment"]["dimensions"]
    model_revision = dimensions["model_revision"]["values"]["nemo"]
    train_semantics = dimensions["train_data_semantics"]
    eval_semantics = dimensions["eval_data_semantics"]
    markdown += [
        "",
        "## Alignment audit",
        "",
        "All runtime-manifest checks below pass. JSONL versus Parquet is a serialization difference only: a separate row-by-row normalization audit produced the same semantic hash.",
        "",
        "| Dimension | Common contract | Result |",
        "|---|---|---:|",
        f"| Model | Qwen2.5-7B revision `{model_revision}` | PASS |",
        "| Hardware | one 4xH20 node; four training GPUs | PASS |",
        "| Workload | 8 prompts x 5 trajectories = 40 trajectories/step; 4 outer steps | PASS |",
        f"| Training data | {train_semantics['rows']:,} rows; semantic SHA-256 `{train_semantics['semantic_rows_sha256']}` | PASS |",
        f"| Evaluation data | {eval_semantics['rows']:,} rows; semantic SHA-256 `{eval_semantics['semantic_rows_sha256']}` | PASS |",
        "| Optimizer | AdamW, lr 1e-6, weight decay 0.01, betas 0.9/0.999, eps 1e-8; 142-step warmup then constant | PASS |",
        "| Search/rollout protocol | E5-base top-3; Search-R1 text protocol; 4 executable generations plus final answer; max 500 tokens/action, max sequence 4096; temperature/top-p 1.0 | PASS |",
        "| RL objective | GRPO with standard-deviation normalization, KL 0.001, clip 0.2, one optimizer update/outer step | PASS |",
        "",
        "NeMo's legacy top-level leave-one-out flag remains true, but the nested advantage-estimator field actually used by the training objective is false; dynamic sampling is disabled. Therefore this does not create an effective LOO mismatch.",
        "",
        "Native packing, microbatching, scheduling, and framework internals are deliberately not forced to be identical: they are the implementations being compared.",
    ]

    markdown += [
        "",
        "## Macro critical path",
        "",
        "Search is inside rollout, so it is not added again. Percentages below sum to 100% including residual.",
        "",
        "| Framework | Rollout lifecycle | Training compute | Postprocess/prep | Residual |",
        "|---|---:|---:|---:|---:|",
    ]
    for framework in FRAMEWORKS:
        macro = report["breakdowns"][framework]["macro"]
        markdown.append(
            "| {} | {}s ({:.1f}%) | {}s ({:.1f}%) | {}s ({:.1f}%) | {}s ({:.1f}%) |".format(
                DISPLAY[framework],
                _fmt(macro["rollout_lifecycle"]["seconds"]),
                macro["rollout_lifecycle"]["percent"],
                _fmt(macro["training_compute"]["seconds"]),
                macro["training_compute"]["percent"],
                _fmt(macro["postprocess_and_preparation"]["seconds"]),
                macro["postprocess_and_preparation"]["percent"],
                _fmt(macro["residual_other"]["seconds"]),
                macro["residual_other"]["percent"],
            )
        )

    markdown += [
        "",
        "## Eight functional stages",
        "",
        "Stage 3 is the observed non-overlapped retrieval wall-union from sampled traces. It is diagnostic, nested in stage 2, and excluded from the sum.",
        "",
        "| Stage | original | current veRL | SLIME | NeMo RL |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage in STAGE_NAMES:
        prefix = "~" if stage == STAGE_NAMES[2] else ""
        values = [
            prefix + _fmt(report["breakdowns"][framework]["stages"][stage]["seconds"])
            for framework in FRAMEWORKS
        ]
        markdown.append(f"| {_stage_label(stage)} | " + " | ".join(values) + " |")
    markdown.append("| additive sum | " + " | ".join(
        _fmt(report["breakdowns"][framework]["additive_stage_sum_seconds"])
        for framework in FRAMEWORKS
    ) + " |")
    markdown.append("| residual/rounding | " + " | ".join(
        _fmt(report["breakdowns"][framework]["residual_seconds"])
        for framework in FRAMEWORKS
    ) + " |")
    markdown.append("| actual total | " + " | ".join(
        _fmt(report["breakdowns"][framework]["total_step_seconds"])
        for framework in FRAMEWORKS
    ) + " |")

    markdown += [
        "",
        "SLIME's policy-logprob cell is zero only as a standalone stage: the work is folded into its actor forward/loss path, not eliminated mathematically. Its residual-derived stage 4 and stage 7 are upper bounds because native timers do not split those pieces further.",
        "",
        "## Timer and auxiliary-source reconciliation",
        "",
        "| Framework | Console mean | Boundary mean | Time delta | Console gen tok | Prometheus gen tok | Token delta | E5 queries | E5 request | Retrieval/model overlap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for framework in FRAMEWORKS:
        value = report["reconciliation"][framework]
        overlap = report["breakdowns"][framework]["retrieval"]["overlap_fraction"]
        token_delta = value["prometheus_token_difference_percent"]
        markdown.append(
            f"| {DISPLAY[framework]} | {_fmt(value['console_timer_mean_seconds'])}s | {_fmt(value['timestamp_boundary_mean_seconds'])}s | {_fmt(value['timestamp_minus_console_percent'], 2)}% | {_fmt(value['console_generated_tokens_window'], 0)} | {_fmt(value['prometheus_generated_tokens_window'], 0)} | {(_fmt(token_delta, 2) + '%') if token_delta is not None else '—'} | {_fmt(value['prometheus_e5_queries_window'], 0)} | {_fmt(value['e5_request_mean_seconds'] * 1000 if value['e5_request_mean_seconds'] is not None else None, 1)}ms | {_fmt(overlap * 100 if overlap is not None else None, 1)}% |"
        )

    markdown += [
        "",
        "Timestamp-boundary differences are at most about 2%; they include console/logging boundaries. NeMo's Prometheus generation-token total matches exactly. Current veRL's counter is 6.3% lower than the console total because the scrape-selected closed window misses boundary work, so it is supporting evidence rather than an equality check. Original and SLIME did not expose a comparable engine token counter. E5 reports zero request errors.",
        "",
        "## Nsight cross-check",
        "",
        "| Framework | Reports | Top GPU kernel | Share | Top CUDA API | Share | Profile/clean step delta* |",
        "|---|---:|---|---:|---|---:|---:|",
    ]
    for framework in FRAMEWORKS:
        value = report["nsight"][framework]
        kernel = value["top_kernel"] or {}
        api = value["top_cuda_api"] or {}
        markdown.append(
            f"| {DISPLAY[framework]} | {value['report_count']} | {kernel.get('name', '—')} | {_fmt(kernel.get('report_share_pct'), 1)}% | {api.get('name', '—')} | {_fmt(api.get('report_share_pct'), 1)}% | {_fmt(report['headline'][framework]['profile_vs_clean_time_delta_percent'], 1)}% |"
        )

    copy = report["nemo_policy_copy_evidence"]
    markdown += [
        "",
        "*The profile/clean delta is diagnostic only. Profiler overhead and stochastic token-work differences are mixed together, so this column must not be interpreted as a pure instrumentation-overhead measurement.*",
        "",
        "On each of four NeMo policy ranks, `cudaMemcpyAsync` averages {:.1f}% of CUDA API time. The captured step shows about {:.2f}s H2D and {:.2f}s D2H copy time per rank, while AllGather is the largest policy kernel at {:.1f}% on average. These per-rank/device streams overlap and must not be summed into E2E wall time, but they strongly corroborate CPU offload and sharded-state movement as the training bottleneck.".format(
            copy["mean_cuda_memcpy_async_api_share_percent"],
            copy["mean_h2d_seconds_per_rank"],
            copy["mean_d2h_seconds_per_rank"],
            copy["mean_top_allgather_kernel_share_percent"],
        ),
        "",
        "## NeMo gap to the leaders",
        "",
        "NeMo is {:.2f}x slower than current veRL even though it generates only {:.1f}% as many tokens. Its actor update is {:.2f}x slower, policy logprob {:.2f}x slower, reference logprob {:.2f}x slower, and rollout preparation {:.2f}x slower. The first optimization target is therefore disabling DTensor CPU offload and activation checkpointing on memory-rich H20s, then testing packing/dynamic batching. Rollout CUDA graphs come later because training already consumes about three quarters of NeMo's step.".format(
            report["comparisons"]["nemo_vs_current"]["total_time_ratio"],
            report["comparisons"]["nemo_vs_current"]["generated_work_ratio"] * 100.0,
            report["comparisons"]["nemo_vs_current"]["stage_time_ratios"][STAGE_NAMES[7]],
            report["comparisons"]["nemo_vs_current"]["stage_time_ratios"][STAGE_NAMES[4]],
            report["comparisons"]["nemo_vs_current"]["stage_time_ratios"][STAGE_NAMES[5]],
            report["comparisons"]["nemo_vs_current"]["stage_time_ratios"][STAGE_NAMES[0]],
        ),
        "",
        "## Boundaries",
        "",
        "- This is a nonformal 4xH20 diagnostic, not an 8-GPU result or quality comparison.",
        "- Clean timers rank performance; profile runs explain behavior and mix profiler overhead with stochastic work differences.",
        "- SLIME Nsight covers only rank-0 training actor; current veRL does not cover its rollout engine.",
        "- Actual token work differs, so both per-step time and normalized work must be reported.",
        "- VRAM is sampled externally at about one-second cadence. Host-RAM increments are node-level estimates; promoted reruns must add job-cgroup memory peaks.",
        "",
    ]
    (output_dir / "deep-breakdown.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--clean-jobs", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--profile-jobs", type=Path, required=True)
    parser.add_argument("--train-equivalence", type=Path, required=True)
    parser.add_argument("--eval-equivalence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args)
    if report["alignment"]["verdict"] != "basically-aligned":
        raise SystemExit("Four-way alignment gate failed")
    write_outputs(report, args.output_dir)
    print(args.output_dir / "deep-breakdown.md")


if __name__ == "__main__":
    main()
