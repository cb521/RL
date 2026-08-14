# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded-cardinality Prometheus metrics for the AI-search resource server."""

from __future__ import annotations

from typing import Protocol

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    make_asgi_app,
)


class SearchTimings(Protocol):
    """Timing fields emitted by all AI-search retrieval providers."""

    queue_ms: float
    encode_ms: float
    index_ms: float
    fetch_ms: float
    total_ms: float
    batch_size: int


_LATENCY_BUCKETS_MS = (0.1, 0.25, 0.5, 1, 2.5, 5, 10, 25, 50, 100, 250, 500, 1000, 5000)
_BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


class SearchPrometheusMetrics:
    """Own an isolated registry so tests and co-located servers never collide."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "nemo_ai_search_requests_total",
            "AI-search resource requests by outcome.",
            ("provider", "outcome"),
            registry=self.registry,
        )
        self.rejections = Counter(
            "nemo_ai_search_rejections_total",
            "AI-search requests rejected before retrieval.",
            ("reason",),
            registry=self.registry,
        )
        self.inflight = Gauge(
            "nemo_ai_search_requests_inflight",
            "AI-search requests currently waiting or executing.",
            registry=self.registry,
        )
        self.queue_depth = Gauge(
            "nemo_ai_search_queue_depth",
            "Queries waiting in the resource-server microbatch queue.",
            registry=self.registry,
        )
        self.active_batch_size = Gauge(
            "nemo_ai_search_active_batch_size",
            "Queries in the currently executing retrieval batch.",
            registry=self.registry,
        )
        self.active_sessions = Gauge(
            "nemo_ai_search_active_sessions",
            "Seeded rollout sessions awaiting verification.",
            registry=self.registry,
        )
        self.search_calls_per_session = Histogram(
            "nemo_ai_search_calls_per_session",
            "Search calls made by a completed rollout session.",
            buckets=(0, 1, 2, 3, 4, 5, 8, 16),
            registry=self.registry,
        )
        self.batch_size = Histogram(
            "nemo_ai_search_batch_size",
            "Queries executed together by the retrieval provider.",
            buckets=_BATCH_BUCKETS,
            registry=self.registry,
        )
        self.latency_ms = Histogram(
            "nemo_ai_search_stage_latency_ms",
            "AI-search latency by resource-server stage.",
            ("provider", "stage"),
            buckets=_LATENCY_BUCKETS_MS,
            registry=self.registry,
        )
        for outcome in ("success", "rejected", "provider_error"):
            self.requests.labels(provider=self.provider, outcome=outcome)
        for reason in ("budget_exhausted", "query_too_long", "top_k_too_large"):
            self.rejections.labels(reason=reason)
        for stage in ("queue", "encode", "index", "fetch", "provider_total"):
            self.latency_ms.labels(provider=self.provider, stage=stage)

    def asgi_app(self):
        """Return an ASGI application exposing only this server's registry."""
        return make_asgi_app(registry=self.registry)

    def observe_success(self, timings: SearchTimings) -> None:
        self.requests.labels(provider=self.provider, outcome="success").inc()
        for stage, value in (
            ("queue", timings.queue_ms),
            ("encode", timings.encode_ms),
            ("index", timings.index_ms),
            ("fetch", timings.fetch_ms),
            ("provider_total", timings.total_ms),
        ):
            self.latency_ms.labels(provider=self.provider, stage=stage).observe(value)

    def observe_active_batch_size(self, batch_size: int) -> None:
        """Update the live gauge and count each provider batch exactly once."""
        self.active_batch_size.set(batch_size)
        if batch_size > 0:
            self.batch_size.observe(batch_size)

    def observe_rejection(self, reason: str) -> None:
        self.requests.labels(provider=self.provider, outcome="rejected").inc()
        self.rejections.labels(reason=reason).inc()

    def observe_provider_error(self) -> None:
        self.requests.labels(provider=self.provider, outcome="provider_error").inc()
