# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for native rollout Prometheus endpoint discovery."""

from discover_prometheus_endpoints import discover_endpoints


def test_discovers_nemo_vllm_metrics_urls() -> None:
    endpoints = discover_endpoints(
        "nemo",
        "Reserved 2 vLLM server URLs: "
        "['http://10.0.0.1:4101/v1', 'http://10.0.0.1:4102/v1']",
    )
    assert [(endpoint.name, endpoint.url) for endpoint in endpoints] == [
        ("nemo-vllm-0", "http://10.0.0.1:4101/metrics"),
        ("nemo-vllm-1", "http://10.0.0.1:4102/metrics"),
    ]


def test_discovers_current_verl_vllm_metrics_urls() -> None:
    endpoints = discover_endpoints(
        "current",
        "LLMServerManager: ['10.0.0.2:4201', 'http://10.0.0.2:4202']",
    )
    assert [endpoint.url for endpoint in endpoints] == [
        "http://10.0.0.2:4201/metrics",
        "http://10.0.0.2:4202/metrics",
    ]


def test_discovers_slime_router_prometheus_port() -> None:
    endpoints = discover_endpoints(
        "slime",
        "\n".join(
            [
                "Ports for engine 2: {'host': '10.0.0.3', 'port': 15032, "
                "'nccl_port': 15033, 'dist_init_addr': '10.0.0.3:15034'}",
                "Ports for engine 0: {'host': '10.0.0.3', 'port': 15000, "
                "'nccl_port': 15001, 'dist_init_addr': '10.0.0.3:15002'}",
                "Router launched at 10.0.0.3:3300, Prometheus port: 4400",
            ]
        ),
    )
    assert [(endpoint.name, endpoint.url) for endpoint in endpoints] == [
        ("slime-sglang-router", "http://10.0.0.3:4400/metrics"),
        ("slime-sglang-engine-0", "http://10.0.0.3:15000/metrics"),
        ("slime-sglang-engine-2", "http://10.0.0.3:15032/metrics"),
    ]


def test_returns_empty_until_dynamic_endpoint_is_announced() -> None:
    assert discover_endpoints("nemo", "model loading") == []
    assert discover_endpoints("current", "model loading") == []
    assert discover_endpoints("slime", "model loading") == []


def test_original_driver_endpoint_is_fixed_and_bounded() -> None:
    endpoints = discover_endpoints("original", "", original_port=19108)
    assert endpoints[0].url == "http://127.0.0.1:19108/metrics"
