# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Low-overhead common JSONL spans for external Search-R1 frameworks."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from pathlib import Path
from types import TracebackType
from typing import Any


_TRACE_PATH_ENV = "AI_SEARCH_TRACE_PATH"
_TRACE_SAMPLE_RATE_ENV = "AI_SEARCH_TRACE_SAMPLE_RATE"


def _trace_config() -> tuple[Path, float] | None:
    raw_path = os.environ.get(_TRACE_PATH_ENV)
    if not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        raise ValueError(f"{_TRACE_PATH_ENV} must be an absolute path: {raw_path}")
    if not path.parent.is_dir():
        raise ValueError(
            f"Parent directory for {_TRACE_PATH_ENV} does not exist: {path.parent}"
        )
    raw_sample_rate = os.environ.get(_TRACE_SAMPLE_RATE_ENV, "1.0")
    try:
        sample_rate = float(raw_sample_rate)
    except ValueError as error:
        raise ValueError(
            f"{_TRACE_SAMPLE_RATE_ENV} must be a number in [0, 1], not "
            f"{raw_sample_rate!r}"
        ) from error
    if not 0.0 <= sample_rate <= 1.0:
        raise ValueError(
            f"{_TRACE_SAMPLE_RATE_ENV} must be in [0, 1], not {sample_rate}"
        )
    return path, sample_rate


def _sampled(trace_id: str, sample_rate: float) -> bool:
    if sample_rate <= 0.0:
        return False
    if sample_rate >= 1.0:
        return True
    digest = hashlib.blake2b(trace_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big") / float(2**64) < sample_rate


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:256]
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value[:32]]
    return str(value)[:256]


class TraceSpan:
    """Append one bounded span when its trajectory is sampled."""

    def __init__(
        self,
        *,
        trace_id: str | None,
        component: str,
        operation: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        self.trace_id = trace_id
        self.component = component
        self.operation = operation
        self.attributes = dict(attributes or {})
        self._path: Path | None = None
        self._start_unix_ns = 0
        self._start_perf_ns = 0

    def __enter__(self) -> TraceSpan:
        config = _trace_config()
        if config is None or self.trace_id is None:
            return self
        path, sample_rate = config
        if not _sampled(self.trace_id, sample_rate):
            return self
        self._path = path
        self._start_unix_ns = time.time_ns()
        self._start_perf_ns = time.perf_counter_ns()
        return self

    def set_attributes(self, **attributes: Any) -> None:
        """Attach bounded scalar evidence before the span closes."""
        self.attributes.update(attributes)

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exception, traceback
        if self._path is None or self.trace_id is None:
            return False
        end_unix_ns = time.time_ns()
        duration_ns = time.perf_counter_ns() - self._start_perf_ns
        event = {
            "schema_version": 1,
            "event": "span",
            "trace_id": self.trace_id,
            "component": self.component,
            "operation": self.operation,
            "status": "error" if exception_type is not None else "ok",
            "error_type": exception_type.__name__ if exception_type else None,
            "start_unix_ns": self._start_unix_ns,
            "end_unix_ns": end_unix_ns,
            "duration_ms": duration_ns / 1_000_000.0,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "attributes": {
                key: _json_value(value) for key, value in self.attributes.items()
            },
        }
        payload = (json.dumps(event, sort_keys=True) + "\n").encode()
        descriptor = os.open(
            self._path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            os.write(descriptor, payload)
        finally:
            os.close(descriptor)
        return False


def trace_span(
    *,
    trace_id: str | None,
    component: str,
    operation: str,
    attributes: dict[str, Any] | None = None,
) -> TraceSpan:
    """Create a disabled-by-default common span context manager."""
    return TraceSpan(
        trace_id=trace_id,
        component=component,
        operation=operation,
        attributes=attributes,
    )
