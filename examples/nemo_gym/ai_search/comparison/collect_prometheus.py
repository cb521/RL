# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Poll selected Prometheus endpoints into timestamped, machine-readable JSONL."""

from __future__ import annotations

import argparse
import json
import math
import re
import signal
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Pattern

from prometheus_client.parser import text_string_to_metric_families


DEFAULT_INCLUDE = r"^(nemo_ai_search_|search_r1_e5_|vllm:|ray_node_)"


@dataclass(frozen=True)
class Endpoint:
    name: str
    url: str


def parse_endpoint(raw: str) -> Endpoint:
    """Parse ``NAME=URL`` and reject ambiguous or unsupported inputs."""
    name, separator, url = raw.partition("=")
    if not separator or not name or not url:
        raise ValueError(f"Endpoint must use NAME=URL syntax, not {raw!r}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError(f"Endpoint name contains unsupported characters: {name!r}")
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"Endpoint URL must be HTTP(S), not {url!r}")
    return Endpoint(name=name, url=url)


def parse_metrics(text: str, include: Pattern[str]) -> list[dict[str, Any]]:
    """Return matching samples while preserving their Prometheus labels."""
    samples: list[dict[str, Any]] = []
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name.endswith("_created") or include.search(sample.name) is None:
                continue
            value = float(sample.value)
            samples.append(
                {
                    "name": sample.name,
                    "labels": dict(sorted(sample.labels.items())),
                    "value": value if math.isfinite(value) else None,
                }
            )
    return samples


def fetch(
    endpoint: Endpoint, include: Pattern[str], timeout_s: float
) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint.url, headers={"User-Agent": "nemo-rl-ai-search-observer/1"}
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        text = response.read().decode("utf-8")
    return {
        "schema_version": 1,
        "endpoint": endpoint.name,
        "url": endpoint.url,
        "scraped_unix_ns": time.time_ns(),
        "scrape_duration_ms": (time.perf_counter() - started) * 1000.0,
        "samples": parse_metrics(text, include),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint", action="append", required=True, metavar="NAME=URL"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument("--ready-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--duration-seconds", type=float, default=0.0)
    parser.add_argument("--max-consecutive-errors", type=int, default=3)
    parser.add_argument("--include", default=DEFAULT_INCLUDE)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    endpoints = [parse_endpoint(raw) for raw in args.endpoint]
    if args.interval_seconds <= 0 or args.timeout_seconds <= 0:
        raise ValueError("Polling interval and HTTP timeout must be positive")
    if args.ready_timeout_seconds <= 0:
        raise ValueError("Readiness timeout must be positive")
    if args.duration_seconds < 0:
        raise ValueError("Duration cannot be negative")
    if args.max_consecutive_errors <= 0:
        raise ValueError("Maximum consecutive errors must be positive")
    include = re.compile(args.include)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    stopped = False

    def stop(_signal_number, _frame) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    ready_deadline = time.monotonic() + args.ready_timeout_seconds
    pending = {endpoint.name: endpoint for endpoint in endpoints}
    while pending and not stopped:
        for name, endpoint in list(pending.items()):
            try:
                fetch(endpoint, include, args.timeout_seconds)
            except Exception:
                continue
            pending.pop(name)
        if pending:
            if time.monotonic() >= ready_deadline:
                names = ", ".join(sorted(pending))
                raise RuntimeError(
                    f"Prometheus endpoints did not become ready: {names}"
                )
            time.sleep(min(1.0, args.interval_seconds))

    started = time.monotonic()
    consecutive_errors = {endpoint.name: 0 for endpoint in endpoints}
    with args.output.open("a", encoding="utf-8") as output:
        while not stopped:
            loop_started = time.monotonic()
            for endpoint in endpoints:
                if stopped:
                    break
                try:
                    snapshot = fetch(endpoint, include, args.timeout_seconds)
                except Exception as error:
                    # A parent process commonly stops the collector after the
                    # instrumented service has begun teardown. Do not turn an
                    # intentional SIGTERM into a synthetic endpoint failure.
                    if stopped:
                        break
                    consecutive_errors[endpoint.name] += 1
                    snapshot = {
                        "schema_version": 1,
                        "endpoint": endpoint.name,
                        "url": endpoint.url,
                        "scraped_unix_ns": time.time_ns(),
                        "error_type": type(error).__name__,
                        "samples": [],
                    }
                    if consecutive_errors[endpoint.name] >= args.max_consecutive_errors:
                        raise RuntimeError(
                            f"Prometheus endpoint {endpoint.name!r} failed "
                            f"{consecutive_errors[endpoint.name]} consecutive scrapes"
                        ) from error
                else:
                    consecutive_errors[endpoint.name] = 0
                output.write(json.dumps(snapshot, sort_keys=True) + "\n")
                output.flush()

            if stopped:
                break
            if args.duration_seconds and (
                time.monotonic() - started >= args.duration_seconds
            ):
                break
            sleep_seconds = args.interval_seconds - (time.monotonic() - loop_started)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
