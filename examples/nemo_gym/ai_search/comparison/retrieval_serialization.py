# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concurrency guard for a shared single-GPU retrieval index."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager


class SerializedRequestGate:
    """Run one retrieval request at a time and report queue wait time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    @contextmanager
    def enter(self) -> Iterator[float]:
        """Acquire the request gate and yield seconds spent waiting for it."""
        queued_at = time.perf_counter()
        with self._lock:
            yield time.perf_counter() - queued_at
