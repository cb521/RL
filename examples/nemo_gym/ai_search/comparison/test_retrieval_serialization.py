# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for serialized access to a shared single-GPU retrieval index."""

import threading

from retrieval_serialization import SerializedRequestGate


def test_waiter_cannot_enter_until_active_request_exits() -> None:
    gate = SerializedRequestGate()
    waiter_started = threading.Event()
    waiter_entered = threading.Event()
    waits: list[float] = []

    def wait_for_gate() -> None:
        waiter_started.set()
        with gate.enter() as wait_seconds:
            waits.append(wait_seconds)
            waiter_entered.set()

    with gate.enter():
        waiter = threading.Thread(target=wait_for_gate)
        waiter.start()
        assert waiter_started.wait(timeout=1)
        assert not waiter_entered.wait(timeout=0.05)

    waiter.join(timeout=1)
    assert not waiter.is_alive()
    assert waiter_entered.is_set()
    assert waits[0] >= 0.05

