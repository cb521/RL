# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local metrics and common spans for the original Search-R1 fork."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from framework_observability import TraceSpan, trace_span


@dataclass(frozen=True)
class TraceIdentity:
    """Stable identity for one rollout in an original Search-R1 outer step."""

    trace_id: str
    example_id: str
    rollout_index: int


def _items(values: Any, *, expected: int, field: str) -> list[Any] | None:
    if values is None:
        return None
    try:
        result = list(values)
    except TypeError as error:
        raise ValueError(f"{field} must be iterable") from error
    if len(result) != expected:
        raise ValueError(f"{field} has {len(result)} rows, expected {expected}")
    return result


def build_trace_identities(
    *,
    global_step: int,
    batch_size: int,
    extra_info: Iterable[Any] | None = None,
    indexes: Iterable[Any] | None = None,
    phase: str = "train",
) -> list[TraceIdentity]:
    """Build deterministic, duplicate-safe IDs after ``n_agent`` expansion."""
    if isinstance(global_step, bool) or not isinstance(global_step, int):
        raise ValueError("global_step must be an integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 0:
        raise ValueError("batch_size must be a non-negative integer")
    if not phase or any(character.isspace() for character in phase):
        raise ValueError("phase must be a non-empty token")

    extra_rows = _items(extra_info, expected=batch_size, field="extra_info")
    index_rows = _items(indexes, expected=batch_size, field="indexes")
    occurrences: dict[str, int] = defaultdict(int)
    identities: list[TraceIdentity] = []
    for position in range(batch_size):
        example_id: Any = None
        if extra_rows is not None and isinstance(extra_rows[position], Mapping):
            example_id = extra_rows[position].get("example_id")
        if example_id is None and index_rows is not None:
            example_id = index_rows[position]
        if example_id is None:
            example_id = f"position-{position}"
        example_id = str(example_id)
        rollout_index = occurrences[example_id]
        occurrences[example_id] += 1
        digest = hashlib.blake2b(
            f"{phase}\0{global_step}\0{example_id}\0{rollout_index}".encode(),
            digest_size=12,
        ).hexdigest()
        identities.append(
            TraceIdentity(
                trace_id=f"original-search-r1:{digest}",
                example_id=example_id,
                rollout_index=rollout_index,
            )
        )
    return identities


@contextmanager
def trace_identity_batch(
    identities: Iterable[TraceIdentity],
    *,
    component: str,
    operation: str,
    attributes: Mapping[str, Any] | None = None,
):
    """Open one same-boundary span per trajectory in a batched operation."""
    with ExitStack() as stack:
        spans: list[TraceSpan] = []
        for identity in identities:
            span_attributes = {
                "example_id": identity.example_id,
                "rollout_index": identity.rollout_index,
                **dict(attributes or {}),
            }
            spans.append(
                stack.enter_context(
                    trace_span(
                        trace_id=identity.trace_id,
                        component=component,
                        operation=operation,
                        attributes=span_attributes,
                    )
                )
            )
        yield spans


def new_provider_batch_id() -> str:
    """Return an opaque ID shared by the caller and E5 service trace."""
    return uuid.uuid4().hex


def _absolute_output(name: str) -> Path:
    raw_path = os.environ.get(name)
    if not raw_path:
        raise ValueError(f"{name} must be set for the local Search-R1 logger")
    path = Path(raw_path)
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path: {raw_path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _float_scalar(value: Any) -> float | None:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


class OriginalSearchR1LocalLogger:
    """Mirror reduced driver metrics to JSONL, TensorBoard and Prometheus."""

    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        config: Any = None,
        *,
        writer_factory: Any = None,
        prometheus_module: Any = None,
    ) -> None:
        del config
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.metrics_path = _absolute_output("AI_SEARCH_METRICS_PATH")
        tensorboard_dir = _absolute_output("AI_SEARCH_TENSORBOARD_DIR")
        tensorboard_dir.mkdir(parents=True, exist_ok=True)

        if writer_factory is None:
            from torch.utils.tensorboard import SummaryWriter

            writer_factory = SummaryWriter
        self.writer = writer_factory(log_dir=str(tensorboard_dir))

        self._latest = None
        self._completed_steps = None
        self._step_seconds = None
        raw_port = os.environ.get("AI_SEARCH_PROMETHEUS_PORT")
        if raw_port:
            try:
                port = int(raw_port)
            except ValueError as error:
                raise ValueError("AI_SEARCH_PROMETHEUS_PORT must be an integer") from error
            if not 1 <= port <= 65535:
                raise ValueError("AI_SEARCH_PROMETHEUS_PORT must be in [1, 65535]")
            if prometheus_module is None:
                import prometheus_client as prometheus_module
            registry = prometheus_module.CollectorRegistry()
            self._latest = prometheus_module.Gauge(
                "search_r1_step_metric",
                "Latest reduced original Search-R1 scalar metric.",
                ["name"],
                registry=registry,
            )
            self._completed_steps = prometheus_module.Counter(
                "search_r1_completed_outer_steps_total",
                "Completed original Search-R1 optimizer outer steps.",
                registry=registry,
            )
            self._step_seconds = prometheus_module.Histogram(
                "search_r1_outer_step_seconds",
                "Original Search-R1 end-to-end outer-step latency.",
                buckets=(30, 60, 90, 120, 180, 240, 360, 600, 900),
                registry=registry,
            )
            prometheus_module.start_http_server(port, registry=registry)

    def log(self, data: Mapping[str, Any], step: int) -> None:
        """Record only finite scalar metrics, preserving the raw common artifact."""
        scalars = {
            str(name): scalar
            for name, value in data.items()
            if (scalar := _float_scalar(value)) is not None
        }
        event = {
            "schema_version": 1,
            "event": "step_metrics",
            "framework": "original-search-r1",
            "project": self.project_name,
            "experiment": self.experiment_name,
            "step": int(step),
            "time_unix_ns": time.time_ns(),
            "metrics": scalars,
        }
        payload = (json.dumps(event, sort_keys=True) + "\n").encode()
        descriptor = os.open(
            self.metrics_path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            os.write(descriptor, payload)
        finally:
            os.close(descriptor)

        for name, value in scalars.items():
            self.writer.add_scalar(name, value, global_step=step)
            if self._latest is not None:
                self._latest.labels(name=name[:200]).set(value)
        step_seconds = scalars.get("timing_s/step")
        if step_seconds is not None:
            if self._completed_steps is not None:
                self._completed_steps.inc()
            if self._step_seconds is not None:
                self._step_seconds.observe(step_seconds)
        self.writer.flush()

    def close(self) -> None:
        """Flush and close the TensorBoard writer."""
        self.writer.flush()
        self.writer.close()
