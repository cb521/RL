# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the original Search-R1 observability bridge."""

import json
from types import SimpleNamespace

from original_search_r1_observability import (
    OriginalSearchR1LocalLogger,
    build_trace_identities,
    trace_identity_batch,
)


class _Writer:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.scalars = []
        self.flushes = 0
        self.closed = False

    def add_scalar(self, name, value, global_step):
        self.scalars.append((name, value, global_step))

    def flush(self):
        self.flushes += 1

    def close(self):
        self.closed = True


def test_trace_identities_distinguish_five_rollouts_and_steps(tmp_path, monkeypatch):
    rows = [{"example_id": "nq:7"}] * 5 + [{"example_id": "hotpotqa:2"}] * 5
    identities = build_trace_identities(
        global_step=3,
        batch_size=10,
        extra_info=rows,
    )
    assert [identity.rollout_index for identity in identities] == list(range(5)) * 2
    assert len({identity.trace_id for identity in identities}) == 10
    assert {
        identity.example_id for identity in identities
    } == {"nq:7", "hotpotqa:2"}
    next_step = build_trace_identities(
        global_step=4,
        batch_size=1,
        extra_info=rows[:1],
    )
    assert next_step[0].trace_id != identities[0].trace_id

    trace_path = tmp_path / "trajectory.jsonl"
    monkeypatch.setenv("AI_SEARCH_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("AI_SEARCH_TRACE_SAMPLE_RATE", "1")
    with trace_identity_batch(
        identities[:2],
        component="original_search_r1",
        operation="model_generation",
        attributes={"turn": 1, "generation_batch_id": "batch-1"},
    ) as spans:
        spans[0].set_attributes(output_tokens=7)
        spans[1].set_attributes(output_tokens=9)
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert len(events) == 2
    assert {event["trace_id"] for event in events} == {
        identity.trace_id for identity in identities[:2]
    }
    assert events[0]["attributes"]["generation_batch_id"] == "batch-1"


def test_local_logger_writes_raw_and_tensorboard_metrics(tmp_path, monkeypatch):
    metrics_path = tmp_path / "metrics.jsonl"
    tensorboard_dir = tmp_path / "tensorboard"
    monkeypatch.setenv("AI_SEARCH_METRICS_PATH", str(metrics_path))
    monkeypatch.setenv("AI_SEARCH_TENSORBOARD_DIR", str(tensorboard_dir))
    monkeypatch.delenv("AI_SEARCH_PROMETHEUS_PORT", raising=False)

    writers = []

    def writer_factory(log_dir):
        writer = _Writer(log_dir)
        writers.append(writer)
        return writer

    logger = OriginalSearchR1LocalLogger(
        "search-r1-four-way",
        "original-performance",
        writer_factory=writer_factory,
    )
    logger.log(
        {
            "timing_s/step": 12.5,
            "env/search_calls": 17,
            "ignored": [1, 2],
            "not_finite": float("nan"),
        },
        step=2,
    )
    logger.close()

    event = json.loads(metrics_path.read_text())
    assert event["framework"] == "original-search-r1"
    assert event["step"] == 2
    assert event["metrics"] == {
        "env/search_calls": 17.0,
        "timing_s/step": 12.5,
    }
    assert writers[0].scalars == [
        ("timing_s/step", 12.5, 2),
        ("env/search_calls", 17.0, 2),
    ]
    assert writers[0].flushes == 2
    assert writers[0].closed is True


def test_local_logger_starts_bounded_prometheus_endpoint(tmp_path, monkeypatch):
    metrics_path = tmp_path / "metrics.jsonl"
    tensorboard_dir = tmp_path / "tensorboard"
    monkeypatch.setenv("AI_SEARCH_METRICS_PATH", str(metrics_path))
    monkeypatch.setenv("AI_SEARCH_TENSORBOARD_DIR", str(tensorboard_dir))
    monkeypatch.setenv("AI_SEARCH_PROMETHEUS_PORT", "9123")

    calls = []

    class Metric:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

        def labels(self, **labels):
            calls.append(("labels", labels))
            return self

        def set(self, value):
            calls.append(("set", value))

        def inc(self):
            calls.append(("inc",))

        def observe(self, value):
            calls.append(("observe", value))

    prometheus = SimpleNamespace(
        CollectorRegistry=lambda: "registry",
        Gauge=Metric,
        Counter=Metric,
        Histogram=Metric,
        start_http_server=lambda port, registry: calls.append(
            ("server", port, registry)
        ),
    )
    logger = OriginalSearchR1LocalLogger(
        "project",
        "experiment",
        writer_factory=_Writer,
        prometheus_module=prometheus,
    )
    logger.log({"timing_s/step": 45.0}, step=1)
    assert ("server", 9123, "registry") in calls
    assert ("inc",) in calls
    assert ("observe", 45.0) in calls
