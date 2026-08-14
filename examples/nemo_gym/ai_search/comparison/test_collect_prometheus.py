# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the bounded Prometheus snapshot collector."""

import re

import pytest

from collect_prometheus import parse_endpoint, parse_metrics


def test_parse_endpoint_requires_named_http_url() -> None:
    endpoint = parse_endpoint("e5=http://127.0.0.1:8000/metrics")
    assert endpoint.name == "e5"
    assert endpoint.url.endswith("/metrics")

    with pytest.raises(ValueError, match="NAME=URL"):
        parse_endpoint("http://127.0.0.1:8000/metrics")
    with pytest.raises(ValueError, match="HTTP"):
        parse_endpoint("e5=file:///tmp/metrics")


def test_parse_metrics_filters_names_and_preserves_labels() -> None:
    payload = """
# HELP search_r1_e5_requests_total requests
# TYPE search_r1_e5_requests_total counter
search_r1_e5_requests_total{outcome="success"} 7
# HELP search_r1_e5_requests_created creation timestamp
# TYPE search_r1_e5_requests_created gauge
search_r1_e5_requests_created{outcome="success"} 1234567890
# HELP unrelated_total ignored
# TYPE unrelated_total counter
unrelated_total 99
"""
    samples = parse_metrics(payload, re.compile(r"^search_r1_e5_"))
    assert samples == [
        {
            "name": "search_r1_e5_requests_total",
            "labels": {"outcome": "success"},
            "value": 7.0,
        }
    ]
