# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover framework-native rollout Prometheus endpoints from console output."""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


_ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_NEMO_PATTERN = re.compile(r"Reserved \d+ vLLM server URLs:\s*(\[[^\n]+\])")
_CURRENT_VERL_PATTERN = re.compile(r"LLMServerManager:\s*(\[[^\n]+\])")
_SLIME_PATTERN = re.compile(
    r"Router launched at\s+([^,\s]+),\s+Prometheus port:\s*(\d+)"
)
_SLIME_ENGINE_PATTERN = re.compile(r"Ports for engine\s+(\d+):\s*(\{[^\n]+\})")


@dataclass(frozen=True)
class Endpoint:
    name: str
    url: str


def _parse_address(raw: str) -> tuple[str, str, int]:
    value = raw if "://" in raw else f"http://{raw}"
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise ValueError(f"Unsupported metrics server address: {raw!r}")
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError(f"Invalid metrics server port: {raw!r}") from error
    if port is None:
        raise ValueError(f"Metrics server address has no port: {raw!r}")
    return parsed.scheme, parsed.hostname, port


def _metrics_url(raw: str, *, port_override: int | None = None) -> str:
    scheme, hostname, port = _parse_address(raw)
    if port_override is not None:
        port = port_override
    rendered_host = f"[{hostname}]" if ":" in hostname else hostname
    return f"{scheme}://{rendered_host}:{port}/metrics"


def _literal_addresses(raw: str) -> list[str]:
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"Malformed server-address list: {raw!r}") from error
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) for item in value)
    ):
        raise ValueError(f"Expected a non-empty string-address list: {value!r}")
    return value


def discover_endpoints(
    framework: str,
    console_text: str,
    *,
    original_port: int = 9108,
) -> list[Endpoint]:
    """Return stable endpoint names and URLs once a framework has announced them."""
    text = _ANSI_PATTERN.sub("", console_text)
    if framework == "original":
        if not 1 <= original_port <= 65535:
            raise ValueError("original_port must be in [1, 65535]")
        return [
            Endpoint(
                name="original-driver",
                url=f"http://127.0.0.1:{original_port}/metrics",
            )
        ]

    if framework == "nemo":
        matches = _NEMO_PATTERN.findall(text)
        addresses = _literal_addresses(matches[-1]) if matches else []
        prefix = "nemo-vllm"
    elif framework == "current":
        matches = _CURRENT_VERL_PATTERN.findall(text)
        addresses = _literal_addresses(matches[-1]) if matches else []
        prefix = "current-vllm"
    elif framework == "slime":
        matches = _SLIME_PATTERN.findall(text)
        if not matches:
            return []
        router_address, prometheus_port = matches[-1]
        endpoints = [
            Endpoint(
                name="slime-sglang-router",
                url=_metrics_url(
                    router_address,
                    port_override=int(prometheus_port),
                ),
            )
        ]
        engine_addresses: dict[int, str] = {}
        for rank_text, mapping_text in _SLIME_ENGINE_PATTERN.findall(text):
            try:
                mapping = ast.literal_eval(mapping_text)
            except (SyntaxError, ValueError) as error:
                raise ValueError(
                    f"Malformed slime engine-address mapping: {mapping_text!r}"
                ) from error
            if not isinstance(mapping, dict):
                raise ValueError(
                    f"Expected a slime engine-address mapping: {mapping!r}"
                )
            host = mapping.get("host")
            port = mapping.get("port")
            if not isinstance(host, str) or not isinstance(port, int):
                raise ValueError(
                    "slime engine-address mapping requires string host and integer port: "
                    f"{mapping!r}"
                )
            engine_addresses[int(rank_text)] = f"{host}:{port}"
        endpoints.extend(
            Endpoint(
                name=f"slime-sglang-engine-{rank}",
                url=_metrics_url(engine_addresses[rank]),
            )
            for rank in sorted(engine_addresses)
        )
        return endpoints
    else:
        raise ValueError(f"Unsupported framework: {framework}")

    endpoints = []
    seen_urls = set()
    for address in addresses:
        url = _metrics_url(address)
        if url in seen_urls:
            continue
        seen_urls.add(url)
        endpoints.append(Endpoint(name=f"{prefix}-{len(endpoints)}", url=url))
    return endpoints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--framework", choices=("nemo", "original", "current", "slime"), required=True
    )
    parser.add_argument("--console", type=Path, required=True)
    parser.add_argument("--original-port", type=int, default=9108)
    args = parser.parse_args()
    endpoints = discover_endpoints(
        args.framework,
        args.console.read_text(encoding="utf-8", errors="replace"),
        original_port=args.original_port,
    )
    if not endpoints:
        raise SystemExit(3)
    for endpoint in endpoints:
        print(f"{endpoint.name}={endpoint.url}")


if __name__ == "__main__":
    main()
